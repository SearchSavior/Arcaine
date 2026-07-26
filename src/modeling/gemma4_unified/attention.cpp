#include "attention.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/rope.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/layers/attention_batched.hpp"   // batched_attention, scores conversions, ...
#include <cmath>
#include <climits>
#include <optional>
#include <vector>
#include <stdexcept>




// ---------------------------------------------------------------------------
// Sliding attention
// ---------------------------------------------------------------------------
void sliding_attention_forward(
    GpuEngine& ctx,
    const SlidingAttnWeights& w,
    bf16* hidden,
    bf16* tmp,
    LayerKvCache& kv,
    int seq_len, int past_len,
    const TextConfig& cfg,
    const int32_t* block_ids
) {
    auto& q = ctx.queue;
    int H    = cfg.hidden_size;
    int nq   = cfg.num_attn_heads;      // 16
    int nkv  = cfg.num_kv_heads;        // 8
    int hd   = cfg.head_dim;            // 256
    int win  = cfg.sliding_window;      // 1024

    GpuBuffer<bf16> Q((size_t)seq_len * nq * hd, q);
    proj_matmul(w.q_proj, hidden, seq_len, H, nq * hd, Q.data(), ctx);

    GpuBuffer<bf16> K_raw((size_t)seq_len * nkv * hd, q);
    proj_matmul(w.k_proj, hidden, seq_len, H, nkv * hd, K_raw.data(), ctx);

    GpuBuffer<bf16> V_raw((size_t)seq_len * nkv * hd, q);
    proj_matmul(w.v_proj, hidden, seq_len, H, nkv * hd, V_raw.data(), ctx);

    rms_norm(q, Q.data(),     w.q_norm.data(), Q.data(),     seq_len * nq,  hd, cfg.rms_norm_eps);
    rms_norm(q, K_raw.data(), w.k_norm.data(), K_raw.data(), seq_len * nkv, hd, cfg.rms_norm_eps);
    rms_norm_no_scale(q, V_raw.data(), V_raw.data(), seq_len * nkv, hd, cfg.rms_norm_eps);

    apply_rope(q, Q.data(), K_raw.data(),
               seq_len, past_len, nq, nkv, hd,
               cfg.sliding_rope.rope_theta,
               cfg.sliding_rope.partial_rotary_factor);

    // Append K and V into KV cache (in-order queue ensures ordering).
    {
        size_t row  = (size_t)nkv * hd;
        size_t koff = (size_t)past_len * row;
        q.memcpy(kv.k.data() + koff, K_raw.data(), seq_len * row * sizeof(bf16));
        q.memcpy(kv.v.data() + koff, V_raw.data(), seq_len * row * sizeof(bf16));
        kv.filled = past_len + seq_len;
    }

    // Decode (seq_len == 1): use the general masked attention path on the
    // primary KV cache (kv.k / kv.v). The dedicated sliding_decode_attention
    // kernel that read the transposed k_decode/v_decode buffers is unreliable
    // once the context length exceeds the sliding window (filled > win): the
    // strided batched matmul over those buffers produces incoherent scores at
    // kv_len > 1024 that drive the model into a degenerate token-repetition
    // loop, and decode throughput collapses ~8x. The general path shares the
    // exact kv.k/kv.v + fill_causal_mask machinery used for prefill (verified
    // correct at any length) and stays coherent. For a single decode query
    // the score matrix is only (nq * filled), so the cost is modest.
    if (seq_len == 1) {
        auto attn_ctx = batched_attention(ctx,
            Q.data(), seq_len, nq, hd,
            kv.k.data(), kv.v.data(), kv.filled, nkv, hd,
            past_len, /*sliding_window=*/win,
            /*scale=*/1.0f,            // Q/K are RMSNorm-ed, so no 1/sqrt(d) factor
            /*skip_mask=*/false,
            /*block_ids=*/nullptr);     // text decode: no vision blocks
        proj_matmul(w.o_proj, attn_ctx.data(), seq_len, nq * hd, H, tmp, ctx);
        return;
    }

    // Prefill: do NOT tail-truncate the KV cache. The decode truncation above is
    // only valid because the lone decode query is at the cache tail. During
    // prefill the queries span the whole chunk, so an early query (e.g. position
    // 0 of a 1812-token prompt) must attend to early KV positions that a
    // tail-truncation to the last `win` entries would drop. Worse, when the
    // chunk is longer than `win`, the earliest queries would have NO surviving
    // KV within the window, producing a fully -inf mask row → NaN softmax →
    // corrupt hidden states and collapse of the model into a degenerate token
    // loop. Pass the full accumulated KV and the real `sliding_window` so
    // fill_causal_mask applies the per-query window
    // (kv_global < q_global - win + 1 → -inf) correctly.
    auto attn_ctx = batched_attention(ctx,
        Q.data(), seq_len, nq, hd,
        kv.k.data(), kv.v.data(), kv.filled, nkv, hd,
        past_len, /*sliding_window=*/win,
        /*scale=*/1.0f,            // Q/K are RMSNorm-ed, so no 1/sqrt(d) factor
        /*skip_mask=*/false,
        block_ids);

    proj_matmul(w.o_proj, attn_ctx.data(), seq_len, nq * hd, H, tmp, ctx);
}

// ---------------------------------------------------------------------------
// Full attention (K=V architecture)
// ---------------------------------------------------------------------------
void full_attention_forward(
    GpuEngine& ctx,
    const FullAttnWeights& w,
    bf16* hidden,
    bf16* tmp,
    LayerKvCache& kv,
    int seq_len, int past_len,
    const TextConfig& cfg,
    const int32_t* block_ids
) {
    auto& q = ctx.queue;
    int H    = cfg.hidden_size;
    int nq   = cfg.num_attn_heads;       // 16
    int nkv  = cfg.num_global_kv_heads;  // 1
    int hd   = cfg.global_head_dim;      // 512

    GpuBuffer<bf16> Q((size_t)seq_len * nq * hd, q);
    proj_matmul(w.q_proj, hidden, seq_len, H, nq * hd, Q.data(), ctx);

    GpuBuffer<bf16> K_raw((size_t)seq_len * nkv * hd, q);
    proj_matmul(w.k_proj, hidden, seq_len, H, nkv * hd, K_raw.data(), ctx);

    rms_norm(q, Q.data(), w.q_norm.data(), Q.data(), seq_len * nq, hd, cfg.rms_norm_eps);

    // V = v_norm(K_raw) BEFORE k_norm modifies K_raw
    GpuBuffer<bf16> V((size_t)seq_len * nkv * hd, q);
    q.memcpy(V.data(), K_raw.data(), (size_t)seq_len * nkv * hd * sizeof(bf16));
    rms_norm_no_scale(q, V.data(), V.data(), seq_len * nkv, hd, cfg.rms_norm_eps);

    // K = k_norm(K_raw) + proportional RoPE
    rms_norm(q, K_raw.data(), w.k_norm.data(), K_raw.data(), seq_len * nkv, hd, cfg.rms_norm_eps);
    apply_rope(q, Q.data(), K_raw.data(),
               seq_len, past_len, nq, nkv, hd,
               cfg.full_rope.rope_theta,
               cfg.full_rope.partial_rotary_factor);

    {
        size_t k_row = (size_t)nkv * hd;
        size_t koff  = (size_t)past_len * k_row;
        q.memcpy(kv.k.data() + koff, K_raw.data(), seq_len * k_row * sizeof(bf16));
        q.memcpy(kv.v.data() + koff, V.data(),     seq_len * k_row * sizeof(bf16));
        kv.filled = past_len + seq_len;
    }

    int kv_len = kv.filled;

    // Fast decode path: nkv==1 means all nq Q-heads share the single K/V head.
    // Direct matmul on the contiguous cache avoids the 2× GQA expansion that would
    // otherwise allocate and copy (nq × kv_len × hd) — 268 MB per layer at depth 16k.
    // Cache layout: (kv_len, nkv=1, hd) is contiguous as (kv_len, hd).
    // Queue is in-order so Q buffer is safe to reuse for attention output.
    if (seq_len == 1 && nkv == 1) {
        // scores(nq, kv_len) = Q(nq, hd) @ K(kv_len, hd)^T
        GpuBuffer<bf16> scores_bf16((size_t)nq * kv_len, q);
        matmul_bf16(Q.data(), nq, hd, kv.k.data(), kv_len, scores_bf16.data(), ctx);

        GpuBuffer<float> scores_f32((size_t)nq * kv_len, q);
        scores_bf16_to_f32(q, scores_bf16.data(), scores_f32.data(), (size_t)nq * kv_len);
        softmax_f32(scores_f32.data(), nq, kv_len, ctx);
        f32_to_bf16_buf(q, scores_f32.data(), scores_bf16.data(), (size_t)nq * kv_len);

        // attn_out(nq, hd) = scores(nq, kv_len) @ V(kv_len, hd). Reuse Q (same size).
        matmul_bf16_nn(scores_bf16.data(), nq, kv_len, kv.v.data(), hd, Q.data(), ctx);
        proj_matmul(w.o_proj, Q.data(), 1, nq * hd, H, tmp, ctx);
        return;
    }

    // Prefill (seq_len > 1) or non-unit nkv: general batched path.
    auto attn_ctx = batched_attention(ctx,
        Q.data(), seq_len, nq, hd,
        kv.k.data(), kv.v.data(), kv_len, nkv, hd,
        past_len, INT_MAX,
        /*scale=*/1.0f,            // Q/K are RMSNorm-ed, so no 1/sqrt(d) factor
        /*skip_mask=*/(seq_len == 1),
        block_ids);

    proj_matmul(w.o_proj, attn_ctx.data(), seq_len, nq * hd, H, tmp, ctx);
}

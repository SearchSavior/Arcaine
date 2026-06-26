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



static void append_sliding_decode_layouts(
    sycl::queue& q,
    const bf16* k_src,
    const bf16* v_src,
    bf16* k_dst,
    bf16* v_dst,
    int past_len,
    int seq_len,
    int nkv_heads,
    int head_dim,
    int max_seq
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>((size_t)seq_len, (size_t)nkv_heads, (size_t)head_dim),
            [=](sycl::id<3> id) {
                int t = (int)id[0];
                int kvh = (int)id[1];
                int d = (int)id[2];
                size_t src_off = ((size_t)t * nkv_heads + kvh) * head_dim + d;
                k_dst[((size_t)kvh * max_seq + past_len + t) * head_dim + d] = k_src[src_off];
                v_dst[((size_t)kvh * head_dim + d) * max_seq + past_len + t] = v_src[src_off];
            });
    });
}


static void softmax_bf16_rows_inplace(
    sycl::queue& q,
    bf16* x,
    int rows,
    int cols
) {
    constexpr int WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)rows * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                bf16* row_ptr = x + (size_t)row * cols;

                float maxv = -3.4028234663852886e38f;
                for (int c = lid; c < cols; c += lsz)
                    maxv = sycl::fmax(maxv, bf16_to_float(row_ptr[c]));
                scratch[lid] = maxv;
                it.barrier(sycl::access::fence_space::local_space);

                for (int offset = WG / 2; offset > 0; offset >>= 1) {
                    if (lid < offset)
                        scratch[lid] = sycl::fmax(scratch[lid], scratch[lid + offset]);
                    it.barrier(sycl::access::fence_space::local_space);
                }
                maxv = scratch[0];

                float sum = 0.0f;
                for (int c = lid; c < cols; c += lsz)
                    sum += sycl::exp(bf16_to_float(row_ptr[c]) - maxv);
                scratch[lid] = sum;
                it.barrier(sycl::access::fence_space::local_space);

                for (int offset = WG / 2; offset > 0; offset >>= 1) {
                    if (lid < offset)
                        scratch[lid] += scratch[lid + offset];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float inv_sum = 1.0f / scratch[0];

                for (int c = lid; c < cols; c += lsz) {
                    float p = sycl::exp(bf16_to_float(row_ptr[c]) - maxv) * inv_sum;
                    row_ptr[c] = float_to_bf16(p);
                }
            });
    });
}

static GpuBuffer<bf16> sliding_decode_attention(
    GpuEngine& ctx,
    const bf16* Q_dev,
    const bf16* K_decode_dev,
    const bf16* V_decode_dev,
    int kv_len,
    int kv_start,
    int cache_stride,
    int nq_heads,
    int nkv_heads,
    int head_dim
) {
    auto& q = ctx.queue;
    int gqa_ratio = nq_heads / nkv_heads;

    // scores(b, m, t) = Q(b, m, d) @ K_decode(b, t, d)^T
    GpuBuffer<bf16> scores_bf16((size_t)nq_heads * kv_len, q);
    matmul_bf16_batched_strided(
        Q_dev,
        nkv_heads, gqa_ratio, head_dim,
        (dnnl_dim_t)gqa_ratio * head_dim, head_dim, 1,
        K_decode_dev + (size_t)kv_start * head_dim,
        kv_len,
        (dnnl_dim_t)cache_stride * head_dim, 1, head_dim,
        scores_bf16.data(),
        (dnnl_dim_t)gqa_ratio * kv_len, kv_len, 1,
        ctx);

    softmax_bf16_rows_inplace(q, scores_bf16.data(), nq_heads, kv_len);

    // ctx(b, m, d) = scores(b, m, t) @ V_decode(b, d, t)^T
    GpuBuffer<bf16> ctx_tm((size_t)nq_heads * head_dim, q);
    matmul_bf16_batched_strided(
        scores_bf16.data(),
        nkv_heads, gqa_ratio, kv_len,
        (dnnl_dim_t)gqa_ratio * kv_len, kv_len, 1,
        V_decode_dev + kv_start,
        head_dim,
        (dnnl_dim_t)head_dim * cache_stride, 1, cache_stride,
        ctx_tm.data(),
        (dnnl_dim_t)gqa_ratio * head_dim, head_dim, 1,
        ctx);

    return ctx_tm;
}

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
    matmul_bf16(hidden, seq_len, H, w.q_proj.data(), nq * hd, Q.data(), ctx);

    GpuBuffer<bf16> K_raw((size_t)seq_len * nkv * hd, q);
    matmul_bf16(hidden, seq_len, H, w.k_proj.data(), nkv * hd, K_raw.data(), ctx);

    GpuBuffer<bf16> V_raw((size_t)seq_len * nkv * hd, q);
    matmul_bf16(hidden, seq_len, H, w.v_proj.data(), nkv * hd, V_raw.data(), ctx);

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
        append_sliding_decode_layouts(q, K_raw.data(), V_raw.data(),
                                      kv.k_decode.data(), kv.v_decode.data(),
                                      past_len, seq_len, nkv, hd, kv.max_seq);
        kv.filled = past_len + seq_len;
    }

    // Truncate the KV view to the effective sliding window.
    // Tokens older than `win` positions will be masked to -inf anyway; skipping
    // them eliminates the expand_kv copies and QK^T matmul over masked-out rows.
    // Truncation also guarantees all remaining KV positions are within the window,
    // so the sliding-window part of fill_causal_mask is a no-op → pass INT_MAX.
    // For decode (seq_len==1) the mask is entirely zero after truncation → skip it.
    int eff_kv_len   = std::min(kv.filled, win);
    int eff_kv_start = kv.filled - eff_kv_len;
    size_t kv_row    = (size_t)nkv * hd;
    const bf16* K_ptr = kv.k.data() + (size_t)eff_kv_start * kv_row;
    const bf16* V_ptr = kv.v.data() + (size_t)eff_kv_start * kv_row;

    if (seq_len == 1) {
        auto attn_ctx = sliding_decode_attention(ctx,
            Q.data(), kv.k_decode.data(), kv.v_decode.data(),
            eff_kv_len, eff_kv_start, kv.max_seq, nq, nkv, hd);
        matmul_bf16(attn_ctx.data(), seq_len, nq * hd, w.o_proj.data(), H, tmp, ctx);
        return;
    }

    auto attn_ctx = batched_attention(ctx,
        Q.data(), seq_len, nq, hd,
        K_ptr, V_ptr, eff_kv_len, nkv, hd,
        past_len, /*sliding_window=*/INT_MAX,
        /*scale=*/1.0f,            // Q/K are RMSNorm-ed, so no 1/sqrt(d) factor
        /*skip_mask=*/false,
        block_ids);

    matmul_bf16(attn_ctx.data(), seq_len, nq * hd, w.o_proj.data(), H, tmp, ctx);
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
    matmul_bf16(hidden, seq_len, H, w.q_proj.data(), nq * hd, Q.data(), ctx);

    GpuBuffer<bf16> K_raw((size_t)seq_len * nkv * hd, q);
    matmul_bf16(hidden, seq_len, H, w.k_proj.data(), nkv * hd, K_raw.data(), ctx);

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
        matmul_bf16(Q.data(), 1, nq * hd, w.o_proj.data(), H, tmp, ctx);
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

    matmul_bf16(attn_ctx.data(), seq_len, nq * hd, w.o_proj.data(), H, tmp, ctx);
}

#pragma once
//
// Qwen3.5-MoE full-attention forward (Qwen3_5MoeAttention) + per-layer KV cache.
//
// Reference: reference/transformers/.../modeling_qwen3_5_moe.py lines 646-720.
//   q_proj: H -> nq*head_dim*2 (8192); the 2nd half *per head* is the sigmoid
//           output gate (DEFINITIVELY sigmoid, not swish — config's
//           output_gate_type="swish" is vestigial; reference line 717 uses
//           torch.sigmoid). Q and gate are interleaved per head, deinterleaved
//           by split_q_gate.
//   q_norm / k_norm: per-head RMSNorm (1+w); w has +1 baked at load time so the
//           plain `rms_norm` kernel (w*x*rsqrt) is reused directly.
//   apply_qwen_rope: HF rotate_half + partial rotary (rotary_dim = 64 of 256).
//   GQA 16:2, scale = 1/sqrt(head_dim) = 1/16, standard KV cache, no sliding
//           window (sliding_window = INT_MAX -> pure causal).
//   attn_output *= sigmoid(gate)  (elementwise per (s, h, j))   [line 717]
//   o_proj: nq*head_dim (4096) -> H (2048).
//
// All NVFP4 projections reuse the shared NVFP4 matmul path (runtime/quantization); the only
// Qwen-specific device kernels are split_q_gate (deinterleave) plus the
// kernels.hpp helpers (apply_qwen_rope, mul_sigmoid_inplace). Numerical
// validation vs the HF reference is Phase 6 (deferred).
//

#include <climits>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "../../runtime/gpu/buffer.hpp"
#include "../../runtime/gpu/engine.hpp"
#include "../../runtime/quantization/nvfp4.hpp"               // matmul_nvfp4, Nvfp4Linear
#include "runtime/kernels/rms_norm.hpp"       // rms_norm (plain w*x; +1 baked)
#include "kernels/attention_batched.hpp" // batched_attention (GQA)

#include "config.hpp"
#include "weights.hpp"   // QwenFullAttn
#include "kernels.hpp"    // apply_qwen_rope, mul_sigmoid_inplace

using namespace qwen35moe_kernels;

// ---------------------------------------------------------------------------
// KV cache for the full-attention layers (10 of 40). Time-major
// (kv_len, nkv, hd) to match batched_attention's K/V layout. Only allocated
// for is_full_attention[l]==true layers; the rest stay empty (unused).
// ---------------------------------------------------------------------------
struct QwenKvLayer {
    GpuBuffer<bf16> k;   // (max_seq, nkv, hd)
    GpuBuffer<bf16> v;   // (max_seq, nkv, hd)
    int filled  = 0;
    int max_seq = 0;
};

struct QwenKvCache {
    std::vector<QwenKvLayer> layers;   // size == num_hidden_layers, indexed by global layer idx
    int nkv = 0, hd = 0;

    void init(const QwenConfig& cfg, int max_seq_len, sycl::queue& q) {
        nkv = cfg.num_key_value_heads;
        hd  = cfg.head_dim;
        layers.resize(cfg.num_hidden_layers);
        for (int l = 0; l < cfg.num_hidden_layers; ++l) {
            if (!cfg.is_full_attn(l)) continue;
            auto& lkv = layers[l];
            lkv.max_seq = max_seq_len;
            size_t n = (size_t)max_seq_len * nkv * hd;
            lkv.k = GpuBuffer<bf16>(n, q);
            lkv.v = GpuBuffer<bf16>(n, q);
        }
    }
    void reset() { for (auto& lkv : layers) lkv.filled = 0; }
};

// ---------------------------------------------------------------------------
// Deinterleave q_proj's [S, nq, 2*hd] output into query [S, nq, hd] (first
// half of each head) and gate [S, nq, hd] (second half). Per-head interleaving
// matches the reference chunk(dim=-1) on a (S, nq, 2*head_dim) view:
//   qproj[s, h*(2*hd) + j]       -> query[s, h*hd + j]   (j in [0, hd))
//   qproj[s, h*(2*hd) + hd + j]   -> gate[s,  h*hd + j]
// Both outputs land in the (seq, nq, hd) time-major layout batched_attention
// expects for Q, and the flat (seq, nq*hd) layout mul_sigmoid_inplace/o_proj use.
// ---------------------------------------------------------------------------
inline void split_q_gate(sycl::queue& q, const bf16* qproj, bf16* query, bf16* gate,
                         int S, int nq, int hd) {
    int total = S * nq * hd;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int j = (int)gid[0] % hd;
            int h = ((int)gid[0] / hd) % nq;
            int s = (int)gid[0] / (nq * hd);
            size_t src = ((size_t)s * nq + h) * (2 * hd) + j;
            size_t dst = ((size_t)s * nq + h) * hd + j;
            query[dst] = qproj[src];
            gate[dst]  = qproj[src + (size_t)hd];
        });
    });
}

// ---------------------------------------------------------------------------
// Grow-only scratch for the per-layer projection intermediates. The full
// attention forward runs once per full-attention layer per call; per-call USM
// alloc/free would dominate launch overhead at decode (S=1). Guarded by the
// batched-attention scratch mutex (single lock ordering, held by
// qwen_full_attention_forward for its whole body).
// ---------------------------------------------------------------------------
namespace qwen_full_attn_detail {
struct Scratch {
    GpuBuffer<bf16> qproj;  // [S, nq*2*hd]
    GpuBuffer<bf16> query;  // [S, nq*hd]
    GpuBuffer<bf16> gate;   // [S, nq*hd]
    GpuBuffer<bf16> k;      // [S, nkv*hd]
    GpuBuffer<bf16> v;      // [S, nkv*hd]
    size_t q_cap = 0;       // rows S covered for qproj/query/gate
    size_t kv_cap = 0;      // rows S covered for k/v
};
inline Scratch& scratch()
{
    static Scratch s;
    return s;
}
} // namespace qwen_full_attn_detail

// ---------------------------------------------------------------------------
// Full-attention forward for one full-attention layer.
//   hidden: [S, H]  (already input_layernorm-ed by the caller)
//   out:    [S, H]  (caller-allocated; attention output, PRE-residual)
// past_len: number of cached tokens before this call (0 for the first prefill).
// ---------------------------------------------------------------------------
inline void qwen_full_attention_forward(
    GpuEngine& ctx,
    const QwenFullAttn& w,
    QwenKvLayer& kv,
    const bf16* hidden,
    bf16* out,
    int seq_len, int past_len,
    const QwenConfig& cfg
) {
    auto& q   = ctx.queue;
    int H     = cfg.hidden_size;            // 2048
    int nq    = cfg.num_attention_heads;    // 16
    int nkv   = cfg.num_key_value_heads;    // 2
    int hd    = cfg.head_dim;              // 256
    int rdim  = cfg.rotary_dim();          // 64
    float theta = cfg.rope.rope_theta;
    float scale = 1.0f / std::sqrt((float)hd);   // 1/sqrt(256) = 1/16

    if (kv.filled + seq_len > kv.max_seq)
        throw std::runtime_error("Qwen KV cache overflow");

    // Held for the whole forward: guards the projection scratch below and the
    // batched_attention scratch whose ctx_tm pointer is consumed by
    // mul_sigmoid_inplace / o_proj at the end of this function.
    std::lock_guard<std::mutex> lock(qwen_attn_batched_detail::scratch_mutex());
    auto& sc = qwen_full_attn_detail::scratch();
    if (sc.q_cap < (size_t)seq_len) {
        sc.qproj = GpuBuffer<bf16>((size_t)seq_len * nq * 2 * hd, q);
        sc.query = GpuBuffer<bf16>((size_t)seq_len * nq * hd, q);
        sc.gate  = GpuBuffer<bf16>((size_t)seq_len * nq * hd, q);
        sc.q_cap = (size_t)seq_len;
    }
    if (sc.kv_cap < (size_t)seq_len) {
        sc.k = GpuBuffer<bf16>((size_t)seq_len * nkv * hd, q);
        sc.v = GpuBuffer<bf16>((size_t)seq_len * nkv * hd, q);
        sc.kv_cap = (size_t)seq_len;
    }

    // q_proj: [S, H] -> [S, nq*2*hd] (8192), then deinterleave into Q | gate.
    qwen_matmul_proj(hidden, seq_len, H, w.q_proj, sc.qproj.data(), ctx);
    split_q_gate(q, sc.qproj.data(), sc.query.data(), sc.gate.data(), seq_len, nq, hd);

    // k_proj / v_proj: [S, H] -> [S, nkv*hd] (512). No v_norm in Qwen3.5.
    qwen_matmul_proj(hidden, seq_len, H, w.k_proj, sc.k.data(), ctx);
    qwen_matmul_proj(hidden, seq_len, H, w.v_proj, sc.v.data(), ctx);

    // Per-head RMSNorm of Q and K (weights have +1 baked -> plain rms_norm).
    rms_norm(q, sc.query.data(), w.q_norm.data(), sc.query.data(), seq_len * nq, hd, cfg.rms_norm_eps);
    rms_norm(q, sc.k.data(),     w.k_norm.data(), sc.k.data(),     seq_len * nkv, hd, cfg.rms_norm_eps);

    // Partial RoPE (rotary_dim=64) applied to Q and K.
    apply_qwen_rope(q, sc.query.data(), sc.k.data(), seq_len, past_len, nq, nkv, hd, rdim, theta);

    // Append K, V to the per-layer cache (time-major (kv_len, nkv, hd)).
    {
        size_t row   = (size_t)nkv * hd;
        size_t off   = (size_t)past_len * row;
        size_t bytes = (size_t)seq_len * row * sizeof(bf16);
        q.memcpy(kv.k.data() + off, sc.k.data(), bytes);
        q.memcpy(kv.v.data() + off, sc.v.data(), bytes);
        kv.filled = past_len + seq_len;
    }

    int kv_len = kv.filled;
    // GQA attention: no sliding window (INT_MAX => pure causal); for decode
    // (seq_len==1) the causal mask is all-zero so it can be skipped.
    bf16* attn = batched_attention(ctx,
        sc.query.data(), seq_len, nq, hd,
        kv.k.data(), kv.v.data(), kv_len, nkv, hd,
        past_len, INT_MAX,
        scale, /*skip_mask=*/(seq_len == 1));

    // attn is [S, nq, hd] time-major == [S, nq*hd] flat; gate has the same layout.
    // attn_output *= sigmoid(gate), elementwise per (s, h, j).  (ref line 717)
    mul_sigmoid_inplace(q, attn, sc.gate.data(), (size_t)seq_len * nq * hd);

    // o_proj: [S, nq*hd] (4096) -> [S, H] (2048).
    qwen_matmul_proj(attn, seq_len, nq * hd, w.o_proj, out, ctx);
}

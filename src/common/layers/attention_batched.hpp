#pragma once

#include <climits>
#include <cstdint>
#include <optional>

#include "../gpu/ops.hpp"
#include "../kernels/attention_mask.hpp"
#include "attention_layout.hpp"

// ---------------------------------------------------------------------------
// Reusable GQA-aware batched dot-product attention building blocks.
//
// These pieces compose the low-level layout helpers into a full causal SDPA
// pipeline over time-major Q/K/V tensors.
//
//   Q:  (seq, nq_heads, q_head_dim)        — time-major (as projected)
//   K:  (kv_len, nkv_heads, kv_head_dim)   — from cache
//   V:  (kv_len, nkv_heads, kv_head_dim)   — from cache
//
// `scale` multiplies the raw QK^T scores before masking/softmax. Gemma4 passes
// 1.0 (Q and K are RMSNorm-ed per head); architectures using the conventional
// 1/sqrt(head_dim) factor pass that instead.
// ---------------------------------------------------------------------------

// Apply scale + score mask and FP32 conversion in one kernel.
// Use size_t for total/index: nq*seq*kv_len can exceed INT_MAX at long contexts.
inline void apply_mask_f32(
    sycl::queue& q,
    const bf16* scores_bf16, float* scores_f32,
    const float* mask,
    int nq, int seq, int kv_len,
    float scale
) {
    size_t total = (size_t)nq * seq * kv_len;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            size_t i  = gid[0];
            int    sq = (int)((i / (size_t)kv_len) % (size_t)seq);
            int    kv = (int)(i % (size_t)kv_len);
            scores_f32[i] = bf16_to_float(scores_bf16[i]) * scale + mask[sq * kv_len + kv];
        });
    });
}

// Scale + BF16->FP32 without mask (mask is all-zeros).
inline void scores_bf16_to_f32(
    sycl::queue& q,
    const bf16* src, float* dst, size_t n,
    float scale = 1.0f
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            dst[gid[0]] = bf16_to_float(src[gid[0]]) * scale;
        });
    });
}

// Convert FP32 softmax output back to BF16.
inline void f32_to_bf16_buf(sycl::queue& q, const float* src, bf16* dst, size_t n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            size_t i = id[0];
            dst[i] = float_to_bf16(src[i]);
        });
    });
}

// Full GQA batched attention: scores = scale * QK^T, mask, softmax, then @ V.
// Returns context in (seq, nq, q_head_dim) time-major layout, ready for o_proj.
inline GpuBuffer<bf16> batched_attention(
    GpuEngine& ctx,
    const bf16* Q_dev, int seq_len, int nq_heads, int q_head_dim,
    const bf16* K_dev, const bf16* V_dev, int kv_len, int nkv_heads, int kv_head_dim,
    int past_offset,
    int sliding_window,
    float scale = 1.0f,
    bool skip_mask = false,  // true when mask is provably all-zeros (decode after KV truncation)
    const int32_t* block_ids = nullptr
) {
    auto& q = ctx.queue;

    // 1. Causal mask: (seq, kv_len)
    std::optional<GpuBuffer<float>> mask_opt;
    if (!skip_mask) {
        mask_opt.emplace((size_t)seq_len * kv_len, q);
        if (block_ids) {
            fill_causal_mask_with_block_ids(q, mask_opt->data(), block_ids,
                                            seq_len, kv_len, past_offset, sliding_window);
        } else {
            fill_causal_mask(q, mask_opt->data(), seq_len, kv_len, past_offset, sliding_window);
        }
    }

    // 2. Q: time-major (seq, nq, hd) -> head-major (nq, seq, hd)
    GpuBuffer<bf16> Q_hm = transpose_q(q, Q_dev, seq_len, nq_heads, q_head_dim, q);

    // 3. K and V: time-major (kv_len, nkv, hd) -> head-major (nq, kv_len, hd) + GQA expansion
    GpuBuffer<bf16> K_exp = expand_kv(q, K_dev, kv_len, nkv_heads, nq_heads, kv_head_dim, q);
    GpuBuffer<bf16> V_exp = expand_kv(q, V_dev, kv_len, nkv_heads, nq_heads, kv_head_dim, q);

    // 4. Scores: (nq, seq, kv_len) = Q_hm @ K_exp^T
    GpuBuffer<bf16> scores_bf16((size_t)nq_heads * seq_len * kv_len, q);
    matmul_bf16_batched(Q_hm.data(), nq_heads, seq_len, q_head_dim,
                        K_exp.data(), kv_len, /*transpose_W=*/true,
                        scores_bf16.data(), ctx);

    // 5. Scale + mask + BF16->FP32
    GpuBuffer<float> scores_f32((size_t)nq_heads * seq_len * kv_len, q);
    if (skip_mask) {
        scores_bf16_to_f32(q, scores_bf16.data(), scores_f32.data(),
                           (size_t)nq_heads * seq_len * kv_len, scale);
    } else {
        apply_mask_f32(q, scores_bf16.data(), scores_f32.data(), mask_opt->data(),
                       nq_heads, seq_len, kv_len, scale);
    }

    // 6. Softmax over kv_len: treat as (nq*seq, kv_len)
    softmax_f32(scores_f32.data(), nq_heads * seq_len, kv_len, ctx);

    // 7. FP32->BF16 (reuse scores_bf16 buffer)
    f32_to_bf16_buf(q, scores_f32.data(), scores_bf16.data(),
                    (size_t)nq_heads * seq_len * kv_len);

    // 8. Context: (nq, seq, q_hd) = scores @ V_exp
    GpuBuffer<bf16> ctx_hm((size_t)nq_heads * seq_len * q_head_dim, q);
    matmul_bf16_batched(scores_bf16.data(), nq_heads, seq_len, kv_len,
                        V_exp.data(), q_head_dim, /*transpose_W=*/false,
                        ctx_hm.data(), ctx);

    // 9. Scatter back to (seq, nq, q_hd) time-major for o_proj
    GpuBuffer<bf16> ctx_tm((size_t)seq_len * nq_heads * q_head_dim, q);
    scatter_ctx(q, ctx_hm.data(), ctx_tm.data(), nq_heads, seq_len, q_head_dim);

    return ctx_tm;
}

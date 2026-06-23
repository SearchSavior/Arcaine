#pragma once
#include <sycl/sycl.hpp>
#include "../gpu/buffer.hpp"

// RMSNorm: out[row, d] = x[row, d] / rms(x[row, :]) * weight[d]
// x and out may alias (in-place ok).
// If weight is nullptr, uses identity scale (v_norm).
inline void rms_norm(
    sycl::queue& q,
    const bf16* x,
    const bf16* weight,   // (H,) — may be nullptr
    bf16* out,
    int seq_len, int H,
    float eps
) {
    size_t local_size = std::min(256, H);
    while (local_size & (local_size - 1)) local_size--;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local_size, h);
        bool has_weight = (weight != nullptr);

        h.parallel_for(
            sycl::nd_range<1>(seq_len * local_size, local_size),
            [=](sycl::nd_item<1> it) {
                int tok = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                const bf16* xrow = x   + tok * H;
                bf16*       orow = out + tok * H;

                float ss = 0.0f;
                for (int d = lid; d < H; d += lsz) { float v = bf16_to_float(xrow[d]); ss += v*v; }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz>>1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid+s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float rms_inv = sycl::rsqrt(lmem[0] / float(H) + eps);
                for (int d = lid; d < H; d += lsz) {
                    float v = bf16_to_float(xrow[d]) * rms_inv;
                    if (has_weight) v *= bf16_to_float(weight[d]);
                    orow[d] = float_to_bf16(v);
                }
            });
    });
}

// v_norm: unit RMS, no learned scale.
inline void rms_norm_no_scale(
    sycl::queue& q, const bf16* x, bf16* out, int seq_len, int H, float eps
) {
    rms_norm(q, x, nullptr, out, seq_len, H, eps);
}

// ---------------------------------------------------------------------------
// Fused: residual[row] += rms_norm(x[row], w1)
//        out[row]       = rms_norm(residual[row], w2)
//
// Replaces: rms_norm(x,w1,tmp) + add_inplace(residual,tmp) + rms_norm(residual,w2,out)
// One kernel, two reduction passes, zero extra buffers.
// ---------------------------------------------------------------------------
inline void rms_norm_add_rms_norm(
    sycl::queue& q,
    const bf16* x,       // attn_out / ffn_out delta
    const bf16* w1,      // post_attn_ln / post_ffn_ln weights
    bf16* residual,      // hidden state, updated in-place
    const bf16* w2,      // pre_ffn_ln / next norm weights
    bf16* out,           // normed output for next sub-layer
    int seq_len, int H, float eps
) {
    size_t local_size = std::min(256, H);
    while (local_size & (local_size - 1)) local_size--;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local_size, h);

        h.parallel_for(
            sycl::nd_range<1>(seq_len * local_size, local_size),
            [=](sycl::nd_item<1> it) {
                int tok = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                const bf16* xrow = x        + tok * H;
                bf16*       rrow = residual  + tok * H;
                bf16*       orow = out       + tok * H;

                // --- Pass 1: reduce x for norm1 ---
                float ss = 0.0f;
                for (int d = lid; d < H; d += lsz) { float v = bf16_to_float(xrow[d]); ss += v*v; }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz>>1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid+s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float rms_inv1 = sycl::rsqrt(lmem[0] / float(H) + eps);

                // --- Pass 2: apply norm1 + residual add, collect ss2 for norm2 ---
                ss = 0.0f;
                for (int d = lid; d < H; d += lsz) {
                    float xn = bf16_to_float(xrow[d]) * rms_inv1 * bf16_to_float(w1[d]);
                    float r  = bf16_to_float(rrow[d]) + xn;
                    rrow[d]  = float_to_bf16(r);
                    ss += r * r;
                }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz>>1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid+s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float rms_inv2 = sycl::rsqrt(lmem[0] / float(H) + eps);

                // --- Pass 3: apply norm2 to updated residual → out ---
                for (int d = lid; d < H; d += lsz) {
                    float r = bf16_to_float(rrow[d]);
                    orow[d] = float_to_bf16(r * rms_inv2 * bf16_to_float(w2[d]));
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Fused: residual[row] = (residual[row] + rms_norm(x[row], weight)) * scalar
//
// Replaces: rms_norm(x,w,tmp) + add_inplace(residual,tmp) + scale_inplace(residual,scalar)
// One kernel, one reduction pass.
// ---------------------------------------------------------------------------
inline void rms_norm_add_scale(
    sycl::queue& q,
    const bf16* x,
    const bf16* weight,
    bf16* residual,    // updated in-place
    float scalar,
    int seq_len, int H, float eps
) {
    size_t local_size = std::min(256, H);
    while (local_size & (local_size - 1)) local_size--;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local_size, h);

        h.parallel_for(
            sycl::nd_range<1>(seq_len * local_size, local_size),
            [=](sycl::nd_item<1> it) {
                int tok = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                const bf16* xrow = x        + tok * H;
                bf16*       rrow = residual  + tok * H;

                float ss = 0.0f;
                for (int d = lid; d < H; d += lsz) { float v = bf16_to_float(xrow[d]); ss += v*v; }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz>>1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid+s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float rms_inv = sycl::rsqrt(lmem[0] / float(H) + eps);

                for (int d = lid; d < H; d += lsz) {
                    float xn = bf16_to_float(xrow[d]) * rms_inv * bf16_to_float(weight[d]);
                    float r  = (bf16_to_float(rrow[d]) + xn) * scalar;
                    rrow[d]  = float_to_bf16(r);
                }
            });
    });
}

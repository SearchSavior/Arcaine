#pragma once
//
// qwen3_5 — model-local Qwen-family kernels (activations, gated RMSNorm, L2
// norm) consumed by this module's operators. Wrapped in a model-local namespace
// so global-scope inline functions with identical names in other models do not
// ODR-merge at link time; divergent bodies would then crash at runtime (see
// AGENTS.md model isolation).
// Math is transcribed from the HF transformers Qwen3.5-MoE reference implementation.
//

#include <sycl/sycl.hpp>
#include "runtime/gpu/buffer.hpp"
#include <algorithm>
#include <cmath>
#include "runtime/profiling/launch_prof.hpp"

namespace qwen35_kernels {

// x[i] = sigmoid(x[i]) = 1 / (1 + exp(-x)).
inline void sigmoid_inplace(sycl::queue& q, bf16* x, int n) {
    auto _ev = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float v = bf16_to_float(x[id[0]]);
            x[id[0]] = float_to_bf16(1.0f / (1.0f + sycl::exp(-v)));
        });
    });
    launchprof::record("qwen35_sigmoid_inplace", _ev);
}

// SwiGLU from stacked (seq, 2*inter) layout into compact (seq, inter) output.
//   gate_up[tok, dim]       = gate proj  (dim < inter)
//   gate_up[tok, inter+dim] = up   proj  (dim < inter)
//   out[tok, dim] = silu(gate_up[tok, dim]) * gate_up[tok, inter + dim]
inline void swiglu_strided(sycl::queue& q, const bf16* gate_up, bf16* out,
                           int seq, int inter) {
    int total = seq * inter;
    auto _ev = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int tok = gid[0] / inter;
            int dim = gid[0] % inter;
            float g = bf16_to_float(gate_up[tok * 2 * inter + dim]);
            float u = bf16_to_float(gate_up[tok * 2 * inter + inter + dim]);
            out[tok * inter + dim] = float_to_bf16((g / (1.0f + sycl::exp(-g))) * u);
        });
    });
    launchprof::record("qwen35_swiglu_strided", _ev);
}

// Full-attention output gate: a[i] *= sigmoid(gate[i]).  (modeling line 717)
inline void mul_sigmoid_inplace(sycl::queue& q, bf16* a, const bf16* gate, int n) {
    auto _ev = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float g = bf16_to_float(gate[id[0]]);
            float sig = 1.0f / (1.0f + sycl::exp(-g));
            a[id[0]] = float_to_bf16(bf16_to_float(a[id[0]]) * sig);
        });
    });
    launchprof::record("qwen35_mul_sigmoid_inplace", _ev);
}

// Gated RMSNorm (Qwen3_5MoeRMSNormGated, linear-attn output norm).  (lines 192-201)
//   out[n, d] = (weight[d] * rmsnorm_D(x[n, :])) * silu(z[n, d])
// weight is ONES-init, PLAIN scale (no +1). Reduction is the MEAN of squares.
inline void gated_rmsnorm(
    sycl::queue& q,
    const bf16* x, const bf16* z, const bf16* weight,
    bf16* out, int N, int D, float eps
) {
    size_t local_size = static_cast<size_t>(std::min(256, D));
    while (local_size & (local_size - 1)) local_size--;
    auto _ev = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local_size, h);
        h.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
            [=](sycl::nd_item<1> it) {
                int n   = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                const bf16* xrow = x + static_cast<size_t>(n) * D;
                const bf16* zrow = z + static_cast<size_t>(n) * D;
                bf16*       orow = out + static_cast<size_t>(n) * D;
                float ss = 0.0f;
                for (int d = lid; d < D; d += lsz) { float v = bf16_to_float(xrow[d]); ss += v * v; }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz >> 1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid + s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float rms_inv = sycl::rsqrt(lmem[0] / static_cast<float>(D) + eps);
                for (int d = lid; d < D; d += lsz) {
                    float v = bf16_to_float(xrow[d]) * rms_inv * bf16_to_float(weight[d]);
                    float g = bf16_to_float(zrow[d]);
                    orow[d] = float_to_bf16(v * (g / (1.0f + sycl::exp(-g))));
                }
            });
    });
    launchprof::record("qwen35_gated_rmsnorm", _ev);
}

// L2 normalization (Gated DeltaNet q/k, FLA-style).  (lines 239-242)
//   out[n, d] = x[n, d] * rsqrt(SUM_d(x^2) + eps)   (SUM, not mean; no weight)
inline void l2norm(sycl::queue& q, const bf16* x, bf16* out, int N, int D, float eps) {
    size_t local_size = static_cast<size_t>(std::min(256, D));
    while (local_size & (local_size - 1)) local_size--;
    auto _ev = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local_size, h);
        h.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
            [=](sycl::nd_item<1> it) {
                int n   = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                const bf16* xrow = x + static_cast<size_t>(n) * D;
                bf16*       orow = out + static_cast<size_t>(n) * D;
                float ss = 0.0f;
                for (int d = lid; d < D; d += lsz) { float v = bf16_to_float(xrow[d]); ss += v * v; }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz >> 1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid + s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float inv_norm = sycl::rsqrt(lmem[0] + eps);
                for (int d = lid; d < D; d += lsz)
                    orow[d] = float_to_bf16(bf16_to_float(xrow[d]) * inv_norm);
            });
    });
    launchprof::record("qwen35_l2norm", _ev);
}

} // namespace qwen35_kernels

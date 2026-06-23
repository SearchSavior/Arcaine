#pragma once
// Fused FFN prenorm kernels for the DiffusionGemma hot path.
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "../../common/gpu/buffer.hpp"

// ---------------------------------------------------------------------------
// F2 — triple pre-norm.  All three FFN inputs normalize the SAME hidden row,
// so one RMS reduction serves all:
//   x1 = h*rms_inv * w1                    (dense-MLP pre-norm)
//   x2 = h*rms_inv * w2                    (MoE expert pre-norm)
//   rn = h*rms_inv * router_scale * c      (router input; scaleless norm + scale)
// ---------------------------------------------------------------------------
inline void fused_triple_prenorm(
    sycl::queue& q,
    const bf16* hidden,
    const bf16* w1, const bf16* w2, const bf16* router_scale, float c,
    bf16* x1, bf16* x2, bf16* rn,
    int seq, int H, float eps)
{
    size_t local = std::min(256, H);
    while (local & (local - 1)) local--;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local, h);
        h.parallel_for(
            sycl::nd_range<1>(seq * local, local),
            [=](sycl::nd_item<1> it) {
                int tok = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                const bf16* hr = hidden + (size_t)tok * H;

                float ss = 0.0f;
                for (int d = lid; d < H; d += lsz) { float v = bf16_to_float(hr[d]); ss += v * v; }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = (int)lsz >> 1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid + s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float rms_inv = sycl::rsqrt(lmem[0] / float(H) + eps);

                for (int d = lid; d < H; d += lsz) {
                    float n = bf16_to_float(hr[d]) * rms_inv;
                    x1[(size_t)tok * H + d] = float_to_bf16(n * bf16_to_float(w1[d]));
                    x2[(size_t)tok * H + d] = float_to_bf16(n * bf16_to_float(w2[d]));
                    rn[(size_t)tok * H + d] = float_to_bf16(n * bf16_to_float(router_scale[d]) * c);
                }
            });
    });
}

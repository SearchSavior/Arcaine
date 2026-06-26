#pragma once
// Fused FFN postnorm kernels for the DiffusionGemma hot path.
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "../../../common/gpu/buffer.hpp"

// ---------------------------------------------------------------------------
// F3 — dual post-norm combine:
//   hidden = (hidden + norm(norm(mlp_out,w1) + norm(moe_out,w2), w3)) * scalar
// Three reductions in one kernel; removes 3 temps + 6 launches.
// ---------------------------------------------------------------------------
inline void fused_dual_postnorm(
    sycl::queue& q,
    const bf16* mlp_out, const bf16* w1,
    const bf16* moe_out, const bf16* w2,
    const bf16* w3,
    bf16* hidden,         // residual, updated in place
    float scalar,
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
                const bf16* a = mlp_out + (size_t)tok * H;
                const bf16* b = moe_out + (size_t)tok * H;
                bf16* hr = hidden + (size_t)tok * H;

                auto reduce = [&](float v) {
                    lmem[lid] = v;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int s = (int)lsz >> 1; s > 0; s >>= 1) {
                        if (lid < s) lmem[lid] += lmem[lid + s];
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    float r = lmem[0];
                    it.barrier(sycl::access::fence_space::local_space);
                    return r;
                };

                float ssa = 0.0f, ssb = 0.0f;
                for (int d = lid; d < H; d += lsz) {
                    float va = bf16_to_float(a[d]); ssa += va * va;
                    float vb = bf16_to_float(b[d]); ssb += vb * vb;
                }
                float ra = sycl::rsqrt(reduce(ssa) / float(H) + eps);
                float rb = sycl::rsqrt(reduce(ssb) / float(H) + eps);

                // s = norm(a,w1) + norm(b,w2); reduce s for the combine norm.
                // Recomputed in the final pass (cheaper than an H-sized temp).
                float sss = 0.0f;
                for (int d = lid; d < H; d += lsz) {
                    float s = bf16_to_float(a[d]) * ra * bf16_to_float(w1[d])
                            + bf16_to_float(b[d]) * rb * bf16_to_float(w2[d]);
                    sss += s * s;
                }
                float rs = sycl::rsqrt(reduce(sss) / float(H) + eps);

                for (int d = lid; d < H; d += lsz) {
                    float s = bf16_to_float(a[d]) * ra * bf16_to_float(w1[d])
                            + bf16_to_float(b[d]) * rb * bf16_to_float(w2[d]);
                    float out = (bf16_to_float(hr[d]) + s * rs * bf16_to_float(w3[d])) * scalar;
                    hr[d] = float_to_bf16(out);
                }
            });
    });
}

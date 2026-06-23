#pragma once
#include <sycl/sycl.hpp>
#include "../gpu/buffer.hpp"

// Standard LayerNorm with learned weight and bias (vision pipeline only).
// Normalizes over last dim D. x/out shape: (N, D).
inline void layer_norm(
    sycl::queue& q,
    const bf16* x,
    const bf16* weight,  // (D,)
    const bf16* bias,    // (D,)
    bf16* out,
    int N, int D,
    float eps
) {
    size_t local_size = std::min(256, D);
    while (local_size & (local_size - 1)) local_size--;

    q.submit([&](sycl::handler& h) {
        // Two slots: [0..lsz) = sum, [lsz..2*lsz) = sum-of-squares
        sycl::local_accessor<float, 1> lmem(local_size * 2, h);

        h.parallel_for(
            sycl::nd_range<1>(N * local_size, local_size),
            [=](sycl::nd_item<1> it) {
                int tok = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);

                const bf16* xrow = x   + tok * D;
                bf16*       orow = out + tok * D;

                float s = 0.0f, ss = 0.0f;
                for (int d = lid; d < D; d += lsz) {
                    float v = bf16_to_float(xrow[d]);
                    s  += v;
                    ss += v * v;
                }
                lmem[lid]       = s;
                lmem[lsz + lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);

                for (int stride = lsz >> 1; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        lmem[lid]       += lmem[lid + stride];
                        lmem[lsz + lid] += lmem[lsz + lid + stride];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                float mean    = lmem[0] / float(D);
                float var     = lmem[lsz] / float(D) - mean * mean;
                float std_inv = sycl::rsqrt(var + eps);

                for (int d = lid; d < D; d += lsz) {
                    float v = (bf16_to_float(xrow[d]) - mean) * std_inv;
                    v = v * bf16_to_float(weight[d]) + bf16_to_float(bias[d]);
                    orow[d] = float_to_bf16(v);
                }
            });
    }).wait();
}

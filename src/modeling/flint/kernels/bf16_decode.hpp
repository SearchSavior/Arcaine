#pragma once
#include <cstdlib>
#include <cstring>
#include "../weights.hpp"

namespace flint::decode_vec {
inline int width() {
    const char* p = std::getenv("ARCAINE_FLINT_DECODE_VEC");
    if (!p) return 0;
    if (!std::strcmp(p, "1") || !std::strcmp(p, "4")) return 4;
    if (!std::strcmp(p, "8")) return 8;
    if (!std::strcmp(p, "16")) return 16;
    return 0;
}
template<int V>
inline void gemv(sycl::queue& q, const QTensor& t, const bf16* input,
                 float* output, const int* active = nullptr) {
    const int K = t.k, N = t.n;
    const auto* weights = t.dense;
    q.parallel_for(sycl::nd_range<1>(size_t((N+7)/8) * 256, 256),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int row = int(it.get_group(0)) * 8 + int(it.get_local_id(0)) / 32;
            if (row >= N || (active && !*active)) return;
            const int lane = int(it.get_local_id(0)) % 32;
            const auto sg = it.get_sub_group();
            float accum[V] = {};
            for (int k0 = lane * V; k0 < K; k0 += 32 * V) {
                sycl::vec<uint32_t, V/2> a, w;
                a.load(0, reinterpret_cast<const uint32_t*>(input + k0));
                w.load(0, reinterpret_cast<const uint32_t*>(weights + size_t(row) * K + k0));
#pragma unroll
                for (int j = 0; j < V/2; ++j) {
                    const uint32_t av = a[j], wv = w[j];
                    accum[2*j] = sycl::fma(sycl::bit_cast<float>(av << 16),
                        sycl::bit_cast<float>(wv << 16), accum[2*j]);
                    accum[2*j+1] = sycl::fma(sycl::bit_cast<float>(av & 0xffff0000u),
                        sycl::bit_cast<float>(wv & 0xffff0000u), accum[2*j+1]);
                }
            }
            float sum = accum[0];
#pragma unroll
            for (int j = 1; j < V; ++j) sum += accum[j];
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<>());
            if (!lane) output[row] = float(bf16(sum));
        });
}
inline bool launch(sycl::queue& q, const QTensor& t, const bf16* input,
                   float* output, const int* active = nullptr) {
    const int v = width();
    if (t.format != WeightFormat::BF16 || !v || t.k % (32*v)) return false;
    if (v == 4) gemv<4>(q,t,input,output,active);
    else if (v == 8) gemv<8>(q,t,input,output,active);
    else gemv<16>(q,t,input,output,active);
    return true;
}
}

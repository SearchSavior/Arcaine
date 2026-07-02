// src/nvfp4_vec_dequant_probe.cpp
// Validate vectorized (ESIMD simd) arithmetic e2m1 dequant vs scalar nvfp4_e2m1_fast.
// e2m1 nibble (4-bit): sign(1) exp(2) mant(1). magnitudes {0,0.5,1,1.5,2,3,4,6}.
// Arithmetic: two_pow = 0.5*float(1<<e); absval = (e==0)?(0.5*m):(two_pow*(1+0.5*m));
//             val = s ? -absval : absval.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include "common/gpu/engine.hpp"
#include "common/gpu/buffer.hpp"
// NOTE: do NOT include nvfp4.hpp here — its inline v1/v2 kernels call
// __spirv_SubgroupMatrixMultiplyAccumulateINTEL, which leaks into the probe's
// ESIMD device module and breaks llvm-spirv (RequiresExtension
// SPV_INTEL_subgroup_matrix_multiply_accumulate). The vectorized helpers below
// are copied verbatim from nvfp4.hpp's esimd_vec namespace and are self-
// contained arithmetic, so the probe can validate them in isolation.

namespace esimd = sycl::ext::intel::esimd;

namespace esimd_vec {
namespace esimd_local = sycl::ext::intel::esimd;
template <int N>
inline esimd_local::simd<float, N> e2m1(esimd_local::simd<uint16_t, N> nib) {
    auto s = nib >> 3;
    auto e = (nib >> 1) & esimd_local::simd<uint16_t, N>(3);
    auto m = nib & esimd_local::simd<uint16_t, N>(1);
    auto one = esimd_local::simd<uint16_t, N>(1);
    auto one_shl_e = one << e;
    auto two_pow = esimd_local::convert<float>(one_shl_e) * 0.5f;
    auto m_f = esimd_local::convert<float>(m);
    auto normal = two_pow + two_pow * 0.5f * m_f;
    auto subnorm = 0.5f * m_f;
    auto absval = normal;
    absval.merge(subnorm, e == esimd_local::simd<uint16_t, N>(0));
    auto val = absval;
    val.merge(-absval, s != esimd_local::simd<uint16_t, N>(0));
    return val;
}
template <int N>
inline esimd_local::simd<float, N> e4m3(esimd_local::simd<uint16_t, N> b) {
    auto s = b >> 7;
    auto exp = (b >> 3) & esimd_local::simd<uint16_t, N>(0x0f);
    auto mant = b & esimd_local::simd<uint16_t, N>(0x07);
    auto one = esimd_local::simd<uint16_t, N>(1);
    auto one_shl_e = one << exp;
    auto two_pow = esimd_local::convert<float>(one_shl_e) * (1.0f / 128.0f);
    auto mant_f = esimd_local::convert<float>(mant) * (1.0f / 8.0f);
    auto normal = two_pow * (1.0f + mant_f);
    auto subnorm = esimd_local::convert<float>(mant) * (1.0f / 512.0f);
    auto absval = normal;
    absval.merge(subnorm, exp == esimd_local::simd<uint16_t, N>(0));
    auto val = absval;
    val.merge(-absval, s != esimd_local::simd<uint16_t, N>(0));
    return val;
}
}  // namespace esimd_vec

static inline float nvfp4_e2m1_fast(uint8_t bits) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (bits & 8) ? -mag[bits & 7] : mag[bits & 7];
}

static inline float nvfp4_e4m3_fast(uint8_t b) {
    uint32_t exp = (b >> 3) & 0x0f;
    uint32_t mant = b & 0x07;
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    if (exp == 0) return sign * (float)mant * (1.0f / 512.0f);
    uint32_t bits = ((uint32_t)(b & 0x80) << 24) |
                    ((exp - 7 + 127) << 23) |
                    (mant << 20);
    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

int main() {
    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    constexpr int N = 256;
    std::vector<uint8_t> nib(N);
    for (int i = 0; i < N; ++i) nib[i] = (uint8_t)(i & 0x0f);

    GpuBuffer<uint8_t> nib_dev(N, q); nib_dev.upload(nib.data(), N);
    GpuBuffer<float>    out_dev(N, q);
    const uint8_t* np = nib_dev.data();
    float* op = out_dev.data();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            (void)i;
            // vectorized dequant of all N nibbles in one work-item
            esimd::simd<uint16_t, N> nib;
            #pragma unroll
            for (int k = 0; k < N; ++k) nib[k] = np[k];
            esimd::simd<uint16_t, N> s  = nib >> 3;
            esimd::simd<uint16_t, N> e  = (nib >> 1) & 3;
            esimd::simd<uint16_t, N> m  = nib & 1;
            esimd::simd<uint16_t, N> one(1);
            esimd::simd<uint16_t, N> one_shl_e = one << e;          // 1,2,4,8
            esimd::simd<float, N> two_pow = esimd::convert<float>(one_shl_e) * 0.5f; // 0.5,1,2,4
            esimd::simd<float, N> m_f = esimd::convert<float>(m);
            esimd::simd<float, N> normal = two_pow + two_pow * 0.5f * m_f; // two_pow*(1+0.5*m)
            esimd::simd<float, N> subnorm = 0.5f * m_f;
            esimd::simd<float, N> absval = normal;
            auto e0 = (e == 0);                                     // simd_mask<N>
            absval.merge(subnorm, e0);                               // e0? subnorm : normal
            esimd::simd<float, N> val = absval;
            auto sm = (s != 0);
            val.merge(-absval, sm);                                 // s? -absval : absval
            #pragma unroll
            for (int k = 0; k < N; ++k) op[k] = val[k];
        });
    });
    q.wait();

    std::vector<float> host(N);
    out_dev.download(host.data(), N);
    double maxerr = 0; int arg = -1;
    for (int k = 0; k < N; ++k) {
        double ref = nvfp4_e2m1_fast(nib[k]);
        double d = std::fabs(host[k] - ref);
        if (d > maxerr) { maxerr = d; arg = k; }
    }
    std::printf("[vec-e2m1] N=%d max_abs=%.6g at nib=%d (vec=%.4g ref=%.4g)\n",
                N, maxerr, nib[arg], host[arg], (double)nvfp4_e2m1_fast(nib[arg]));
    // dump all 16 unique values
    std::printf("[vec-e2m1] ");
    for (int b = 0; b < 16; ++b) std::printf("%d:%.3f ", b, host[b]);
    std::printf("\n[ref-e2m1]  ");
    for (int b = 0; b < 16; ++b) std::printf("%d:%.3f ", b, (double)nvfp4_e2m1_fast(b));
    std::printf("\n");

    // ---- e4m3: validate vectorized dequant over all 256 byte codes ----
    GpuBuffer<uint8_t> e4_dev(256, q);
    {
        std::vector<uint8_t> v(256);
        for (int i = 0; i < 256; ++i) v[i] = (uint8_t)i;
        e4_dev.upload(v.data(), 256);
    }
    GpuBuffer<float> e4out_dev(256, q);
    const uint8_t* e4p = e4_dev.data();
    float* e4op = e4out_dev.data();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            esimd::simd<uint16_t, 256> b;
            #pragma unroll
            for (int i = 0; i < 256; ++i) b[i] = e4p[i];
            auto f = esimd_vec::e4m3<256>(b);
            #pragma unroll
            for (int i = 0; i < 256; ++i) e4op[i] = f[i];
        });
    });
    q.wait();
    std::vector<float> e4host(256);
    e4out_dev.download(e4host.data(), 256);
    double e4maxerr = 0; int e4arg = -1;
    for (int i = 0; i < 256; ++i) {
        double ref = nvfp4_e4m3_fast((uint8_t)i);
        double d = std::fabs(e4host[i] - ref);
        if (d > e4maxerr) { e4maxerr = d; e4arg = i; }
    }
    std::printf("[vec-e4m3] 256 codes max_abs=%.6g at byte=%d (vec=%.6g ref=%.6g)\n",
                e4maxerr, e4arg, e4host[e4arg], (double)nvfp4_e4m3_fast((uint8_t)e4arg));
    std::printf("[vec-e4m3] samples:");
    for (int i : {0, 1, 7, 8, 0x10, 0x7e, 0x80, 0xff})
        std::printf(" %d:%.4g", i, e4host[i]);
    std::printf("\n[ref-e4m3]  samples:");
    for (int i : {0, 1, 7, 8, 0x10, 0x7e, 0x80, 0xff})
        std::printf(" %d:%.4g", i, (double)nvfp4_e4m3_fast((uint8_t)i));
    std::printf("\n");
    return 0;
}

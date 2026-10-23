// Replicate the prefill_xmx fold for one work-group tile and dump facc.
#include <cstdio>
#include <cmath>
#include <vector>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using dpas_t = xmx::dpas_argument_type;
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int SD = 8, RC = 8, ES = 16, AF = RC * 32, BF = 32 * ES / 2, RES = RC * ES;
constexpr int M = 8, N = 16, K = 64, G = K / 64;

int main() {
    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                  sycl::property::queue::in_order{});
    std::vector<int8_t> a((size_t)M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) a[m * K + k] = (int8_t)((m * 7 + k * 3) % 61 - 30);
    std::vector<uint8_t> w((size_t)N * K / 2);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; k += 2) {
            int c0 = (n * 5 + k * 11) % 16, c1 = (n * 5 + (k + 1) * 11) % 16;
            w[(size_t)n * (K / 2) + k / 2] = (uint8_t)(c0 | (c1 << 4));
        }
    std::vector<bf16> s((size_t)N * G), b((size_t)N * G);
    for (int n = 0; n < N; ++n) {
        s[(size_t)n * G] = bf16(0.01f + 0.001f * (float)(n % 7));
        b[(size_t)n * G] = bf16(-0.002f + 0.0005f * (float)(n % 5));
    }
    std::vector<float> xs((size_t)M * G);
    std::vector<int> xi((size_t)M * G);
    for (int m = 0; m < M; ++m) {
        xs[(size_t)m * G] = 0.02f + 0.001f * (float)m;
        int t = 0;
        for (int k = 0; k < K; ++k) t += a[(size_t)m * K + k];
        xi[(size_t)m * G] = t;
    }
    int8_t* d_a = sycl::malloc_device<int8_t>((size_t)M * K, q);
    uint8_t* d_w = sycl::malloc_device<uint8_t>((size_t)N * K / 2, q);
    bf16* d_s = sycl::malloc_device<bf16>((size_t)N * G, q);
    bf16* d_b = sycl::malloc_device<bf16>((size_t)N * G, q);
    float* d_xs = sycl::malloc_device<float>((size_t)M * G, q);
    int* d_xi = sycl::malloc_device<int>((size_t)M * G, q);
    float* d_c = sycl::malloc_device<float>(RES, q);
    q.memcpy(d_a, a.data(), a.size());
    q.memcpy(d_w, w.data(), w.size());
    q.memcpy(d_s, s.data(), s.size() * 2);
    q.memcpy(d_b, b.data(), b.size() * 2);
    q.memcpy(d_xs, xs.data(), xs.size() * 4);
    q.memcpy(d_xi, xi.data(), xi.size() * 4);
    q.fill(d_c, 0.f, RES).wait();
    const int64_t* Aq64 = reinterpret_cast<const int64_t*>(d_a);
    const int32_t* wbase = reinterpret_cast<const int32_t*>(d_w);
    const int16_t* s16 = reinterpret_cast<const int16_t*>(d_s);
    const int16_t* b16 = reinterpret_cast<const int16_t*>(d_b);
    q.parallel_for(sycl::nd_range<1>(16, 16),
                   sycl::ext::oneapi::experimental::properties{
                       sycl::ext::oneapi::experimental::sub_group_size<16>},
                   [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                       esimd::simd<uint32_t, 32> e8 = esimd::simd<uint32_t, 32>(0, 1);
                       esimd::simd<uint32_t, RES> e128 = esimd::simd<uint32_t, RES>(0, 1);
                       esimd::simd<uint32_t, 32> tok = e8 / 4u;
                       esimd::simd<uint32_t, 32> aoff =
                           tok * (uint32_t)K + 16u * ((e8 / 2u) & 1u) + 8u * (e8 % 2u);
                       esimd::simd<uint32_t, BF / 4> eB = esimd::simd<uint32_t, BF / 4>(0, 1);
                       esimd::simd<uint32_t, BF / 4> boff =
                           (eB % 16u) * (uint32_t)(K / 2) + (eB / 16u) * 4u;
                       esimd::simd<uint32_t, RES> soff =
                           (e128 % 16u) * (uint32_t)(G * 2);
                       esimd::simd<uint32_t, RES> xoff =
                           (e128 / 16u) * (uint32_t)(G * 4);
                       esimd::simd<float, RES> facc = 0.0f;
                       for (int g = 0; g < G; ++g) {
                           esimd::simd<int, RES> ai = 0;
#pragma unroll
                           for (int half = 0; half < 2; ++half) {
                               esimd::simd<int64_t, AF / 8> araw =
                                   esimd::gather<int64_t, AF / 8>(
                                       Aq64, aoff + (uint32_t)(g * 64 + half * 32));
                               esimd::simd<int, BF / 4> braw = esimd::gather<int, BF / 4>(
                                   wbase, boff + (uint32_t)(g * 32 + half * 16));
                               esimd::simd<int8_t, AF> a8 = araw.template bit_cast_view<int8_t>();
                               esimd::simd<uint8_t, BF> b8 = braw.template bit_cast_view<uint8_t>();
                               ai = xmx::dpas<SD, RC, int, int, uint8_t, int8_t,
                                              dpas_t::u4, dpas_t::s8, RES, BF, AF>(ai, b8, a8);
                           }
                           esimd::simd<int16_t, RES> sraw =
                               esimd::gather<int16_t, RES>(s16, soff + (uint32_t)(g * 2));
                           esimd::simd<int16_t, RES> braw =
                               esimd::gather<int16_t, RES>(b16, soff + (uint32_t)(g * 2));
                           esimd::simd<bf16, RES> sbf = sraw.template bit_cast_view<bf16>();
                           esimd::simd<bf16, RES> bbf = braw.template bit_cast_view<bf16>();
                           esimd::simd<float, RES> sv = esimd::simd<float, RES>(sbf);
                           esimd::simd<float, RES> bv = esimd::simd<float, RES>(bbf);
                           esimd::simd<float, RES> xsv =
                               esimd::gather<float, RES>(d_xs, xoff + (uint32_t)(g * 4));
                           esimd::simd<float, RES> xiv = esimd::convert<float>(
                               esimd::gather<int, RES>(d_xi, xoff + (uint32_t)(g * 4)));
                           esimd::simd<float, RES> t = esimd::convert<float>(ai) * sv;
                           t = t + xiv * bv;
                           facc = facc + t * xsv;
                       }
                       esimd::block_store<float, RES>(d_c, 0, facc);
                   });
    q.wait();
    std::vector<float> c(RES);
    q.memcpy(c.data(), d_c, RES * 4).wait();
    auto s4 = [](int cc) { return cc; };
    int errs = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float gold = 0;
            for (int g = 0; g < G; ++g) {
                int ai = 0;
                for (int k = g * 64; k < g * 64 + 64; ++k) {
                    int cc = (k & 1) ? (w[(size_t)n * (K / 2) + k / 2] >> 4) & 15
                                     : w[(size_t)n * (K / 2) + k / 2] & 15;
                    ai += (int)a[(size_t)m * K + k] * s4(cc);
                }
                gold += (float(s[(size_t)n * G + g]) * ai +
                         float(b[(size_t)n * G + g]) * xi[(size_t)m * G + g]) *
                        xs[(size_t)m * G + g];
            }
            float got = c[m * ES + n];
            if (std::fabs(got - gold) > 1e-4f * (1.f + std::fabs(gold)) && errs++ < 8)
                std::printf("c[%d][%d]=%g gold=%g\n", m, n, got, gold);
        }
    std::printf("fold errs=%d\n", errs);
}

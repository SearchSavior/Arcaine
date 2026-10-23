// Replicate the prefill_xmx fragment gathers for one atom and dump them.
#include <cstdio>
#include <vector>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using dpas_t = xmx::dpas_argument_type;
constexpr int SD = 8, RC = 8, ES = 16, AF = RC * 32, BF = 32 * ES / 2, RES = RC * ES;

int main() {
    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                  sycl::property::queue::in_order{});
    const int K = 64, M = 8, N = 16;
    // activations int8 [M][K]
    std::vector<int8_t> a((size_t)M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) a[m * K + k] = (int8_t)((m * 7 + k * 3) % 61 - 30);
    // weights int4 packed [N][K/2]
    std::vector<uint8_t> w((size_t)N * K / 2);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; k += 2) {
            int c0 = (n * 5 + k * 11) % 16, c1 = (n * 5 + (k + 1) * 11) % 16;
            w[(size_t)n * (K / 2) + k / 2] = (uint8_t)(c0 | (c1 << 4));
        }
    int8_t* d_a = sycl::malloc_device<int8_t>((size_t)M * K, q);
    uint8_t* d_w = sycl::malloc_device<uint8_t>((size_t)N * K / 2, q);
    int* d_ai = sycl::malloc_device<int>(RES, q);
    int8_t* d_af = sycl::malloc_device<int8_t>(AF, q);
    uint8_t* d_bf = sycl::malloc_device<uint8_t>(BF, q);
    q.memcpy(d_a, a.data(), a.size());
    q.memcpy(d_w, w.data(), w.size());
    q.fill(d_ai, 0, RES).wait();
    const int64_t* Aq64 = reinterpret_cast<const int64_t*>(d_a);
    const int32_t* wbase = reinterpret_cast<const int32_t*>(d_w);
    q.parallel_for(sycl::nd_range<1>(16, 16),
                   sycl::ext::oneapi::experimental::properties{
                       sycl::ext::oneapi::experimental::sub_group_size<16>},
                   [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                       esimd::simd<uint32_t, 32> e8 = esimd::simd<uint32_t, 32>(0, 1);
                       // token per int64 element: i/4
                       esimd::simd<uint32_t, 32> tok = e8 / 4u;
                       esimd::simd<uint32_t, 32> aoff =
                           tok * (uint32_t)K + 16u * ((e8 / 2u) & 1u) + 8u * (e8 % 2u);
                       esimd::simd<int64_t, 32> araw =
                           esimd::gather<int64_t, 32>(Aq64, aoff);
                       esimd::simd<int8_t, AF> a8 = araw.template bit_cast_view<int8_t>();
                       esimd::block_store<int8_t, AF>(d_af, 0, a8);
                       esimd::simd<uint32_t, 64> eB = esimd::simd<uint32_t, 64>(0, 1);
                       esimd::simd<uint32_t, 64> boff =
                           (eB % 16u) * (uint32_t)(K / 2) + (eB / 16u) * 4u;
                       esimd::simd<int, 64> braw = esimd::gather<int, 64>(wbase, boff);
                       esimd::simd<uint8_t, BF> b8 = braw.template bit_cast_view<uint8_t>();
                       esimd::block_store<uint8_t, BF>(d_bf, 0, b8);
                       esimd::simd<int, RES> ai = 0;
                       ai = xmx::dpas<SD, RC, int, int, uint8_t, int8_t, dpas_t::s4,
                                      dpas_t::s8, RES, BF, AF>(ai, b8, a8);
                       esimd::block_store<int, RES>(d_ai, 0, ai);
                   });
    q.wait();
    std::vector<int8_t> af(AF);
    std::vector<uint8_t> bf(BF);
    std::vector<int> ai(RES);
    q.memcpy(af.data(), d_af, AF);
    q.memcpy(bf.data(), d_bf, BF);
    q.memcpy(ai.data(), d_ai, RES * 4).wait();
    int aerr = 0, berr = 0, aierr = 0;
    for (int m = 0; m < RC; ++m)
        for (int k = 0; k < 32; ++k)
            if (af[m * 32 + k] != a[m * K + k] && aerr++ < 5)
                std::printf("af[%d*32+%d]=%d want %d\n", m, k, (int)af[m*32+k], (int)a[m*K+k]);
    // fragment half 0 of group 0: k 0..31
    for (int i = 0; i < BF; ++i) {
        int k = ((i / 4) / 16) * 8 + (i % 4) * 2;  // dword=(k/8)*16+n
        int n = (i / 4) % 16;
        int want = w[(size_t)n * (K / 2) + k / 2];
        if (bf[i] != want && berr++ < 5)
            std::printf("bf[%d]=%02x want %02x (n=%d k=%d)\n", i, bf[i], want, n, k);
    }
    auto s4 = [](int c) { return c < 8 ? c : c - 16; };
    for (int m = 0; m < RC; ++m)
        for (int n = 0; n < ES; ++n) {
            int gold = 0;
            for (int k = 0; k < 32; ++k) {
                int c = (k & 1) ? (w[(size_t)n * (K / 2) + k / 2] >> 4) & 15
                                : w[(size_t)n * (K / 2) + k / 2] & 15;
                gold += (int)a[m * K + k] * s4(c);
            }
            if (ai[m * ES + n] != gold && aierr++ < 5)
                std::printf("ai[%d][%d]=%d want %d\n", m, n, ai[m*ES+n], gold);
        }
    std::printf("A frag errs=%d  B frag errs=%d  dpas errs=%d\n", aerr, berr, aierr);
}

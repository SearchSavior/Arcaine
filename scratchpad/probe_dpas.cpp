// Scratch probe: s8xs4 ESIMD dpas fragment layouts + gather/scatter VS>1 +
// gather<int16,128>. Not part of the build.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using dpas_t = xmx::dpas_argument_type;

constexpr int SD = 8, RC = 8, ES = 16, KT = 32;
constexpr int AF = RC * KT;
constexpr int BN = KT * ES / 2;
constexpr int RES = RC * ES;

int main() {
    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                  sycl::property::queue::in_order{});
    std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<int8_t> a((size_t)AF);
    for (int m = 0; m < RC; ++m)
        for (int k = 0; k < KT; ++k)
            a[(size_t)m * KT + k] = (int8_t)((m * 7 + k * 3) % 17 - 8);
    std::vector<uint8_t> b((size_t)BN, 0);
    auto bset = [&](int n, int k, int code) {
        size_t byte = (size_t)(((k / 8) * ES + n) * 4 + (k % 8) / 2);
        if ((k & 1) == 0) b[byte] |= (uint8_t)(code & 0xf);
        else b[byte] |= (uint8_t)((code & 0xf) << 4);
    };
    for (int n = 0; n < ES; ++n)
        for (int k = 0; k < KT; ++k) bset(n, k, (n * 5 + k * 11) % 16);

    int8_t* d_a = sycl::malloc_device<int8_t>(AF, q);
    uint8_t* d_b = sycl::malloc_device<uint8_t>(BN, q);
    int* d_c = sycl::malloc_device<int>(RES, q);
    q.memcpy(d_a, a.data(), AF).wait();
    q.memcpy(d_b, b.data(), BN).wait();
    q.fill(d_c, 0, RES).wait();

    int16_t* d_g = sycl::malloc_device<int16_t>(16 * 32, q);
    std::vector<int16_t> hg(16 * 32);
    for (size_t i = 0; i < hg.size(); ++i) hg[i] = (int16_t)(i * 3 + 1);
    q.memcpy(d_g, hg.data(), hg.size() * 2).wait();
    int16_t* d_o = sycl::malloc_device<int16_t>(128, q);
    q.fill(d_o, (int16_t)0, 128).wait();
    int* d_s = sycl::malloc_device<int>(64, q);
    q.fill(d_s, 0, 64).wait();

    q.parallel_for(sycl::nd_range<1>(16, 16),
                   sycl::ext::oneapi::experimental::properties{
                       sycl::ext::oneapi::experimental::sub_group_size<16>},
                   [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                       esimd::simd<int8_t, AF> av = esimd::block_load<int8_t, AF>(d_a);
                       esimd::simd<uint8_t, BN> bv = esimd::block_load<uint8_t, BN>(d_b);
                       esimd::simd<int, RES> c = 0;
                       c = xmx::dpas<SD, RC, int, int, uint8_t, int8_t,
                                     dpas_t::s4, dpas_t::s8, RES, BN, AF>(c, bv, av);
                       esimd::block_store<int, RES>(d_c, 0, c);

                       esimd::simd<uint32_t, 16> lane = esimd::simd<uint32_t, 16>(0, 1);
                       esimd::simd<int, 64> g4 = esimd::gather<int, 64, 4>(d_s, lane * 64u);
                       esimd::scatter<int, 64, 4>(d_s, lane * 64u, g4 + 1);
                       esimd::simd<uint32_t, 128> e = esimd::simd<uint32_t, 128>(0, 1);
                       esimd::simd<int16_t, 128> g16 =
                           esimd::gather<int16_t, 128>(d_g, (e % 16u) * 2u);
                       esimd::block_store<int16_t, 128>(d_o, 0, g16);
                   });
    q.wait();

    std::vector<int> c(RES);
    q.memcpy(c.data(), d_c, RES * sizeof(int)).wait();
    int errs = 0;
    for (int m = 0; m < RC; ++m) {
        for (int n = 0; n < ES; ++n) {
            int gold = 0;
            for (int k = 0; k < KT; ++k) {
                size_t byte = (size_t)(((k / 8) * ES + n) * 4 + (k % 8) / 2);
                int code = (k & 1) ? ((b[byte] >> 4) & 15) : (b[byte] & 15);
                gold += (int)a[(size_t)m * KT + k] * (code - 8);
            }
            int got = c[(size_t)m * ES + n];
            if (got != gold && errs < 4) {
                std::printf("c[%d][%d]=%d gold=%d\n", m, n, got, gold);
                ++errs;
            }
        }
    }
    std::printf("dpas s8xs4 layout: %s\n", errs ? "FAIL" : "PASS");

    std::vector<int> s(64);
    q.memcpy(s.data(), d_s, 64 * sizeof(int)).wait();
    int serrs = 0;
    for (int i = 0; i < 64; ++i) {
        int want = (16 * (i / 4) + (i % 4)) * 3 + 2;
        if (s[i] != want && serrs++ < 4)
            std::printf("s[%d]=%d want=%d\n", i, s[i], want);
    }
    std::printf("gather/scatter int VS=4: %s\n", serrs ? "FAIL" : "PASS");

    std::vector<int16_t> o(128);
    q.memcpy(o.data(), d_o, 128 * 2).wait();
    int oerrs = 0;
    for (int i = 0; i < 128; ++i) {
        int want = hg[(size_t)(i % 16)];
        if (o[i] != want && oerrs++ < 4)
            std::printf("o[%d]=%d want=%d\n", i, (int)o[i], want);
    }
    std::printf("gather int16 N=128: %s\n", oerrs ? "FAIL" : "PASS");
    return (errs || serrs || oerrs) ? 1 : 0;
}

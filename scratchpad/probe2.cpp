// Probe: dpas s8xs8 and s8xs4 fragment layouts.
#include <cstdio>
#include <vector>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using dpas_t = xmx::dpas_argument_type;
constexpr int SD = 8, RC = 8, ES = 16, KT = 32, AF = RC * KT, RES = RC * ES;

int main() {
    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                  sycl::property::queue::in_order{});
    std::vector<int8_t> a((size_t)AF);
    for (int m = 0; m < RC; ++m)
        for (int k = 0; k < KT; ++k) a[m * KT + k] = (int8_t)((m * 7 + k * 3) % 17 - 8);
    // s8 B: byte idx = ((k/4)*16 + n)*4 + (k%4)
    std::vector<int8_t> b8((size_t)KT * ES, 0);
    for (int n = 0; n < ES; ++n)
        for (int k = 0; k < KT; ++k) b8[((k / 4) * ES + n) * 4 + (k % 4)] = (int8_t)((n * 5 + k * 11) % 19 - 9);
    // s4 B: two candidate nibble orders
    std::vector<uint8_t> b4((size_t)KT * ES / 2, 0);
    std::vector<uint8_t> b4s((size_t)KT * ES / 2, 0);
    for (int n = 0; n < ES; ++n)
        for (int k = 0; k < KT; ++k) {
            int code = (n * 5 + k * 11) % 16;
            size_t byte = (size_t)(((k / 8) * ES + n) * 4 + (k % 8) / 2);
            if ((k & 1) == 0) b4[byte] |= code; else b4[byte] |= code << 4;
            if ((k & 1) == 0) b4s[byte] |= code << 4; else b4s[byte] |= code;
        }
    int8_t* d_a = sycl::malloc_device<int8_t>(AF, q);
    int8_t* d_b8 = sycl::malloc_device<int8_t>(KT * ES, q);
    uint8_t* d_b4 = sycl::malloc_device<uint8_t>(KT * ES / 2, q);
    uint8_t* d_b4s = sycl::malloc_device<uint8_t>(KT * ES / 2, q);
    int* d_c = sycl::malloc_device<int>(2 * RES, q);
    q.memcpy(d_a, a.data(), AF);
    q.memcpy(d_b8, b8.data(), b8.size());
    q.memcpy(d_b4, b4.data(), b4.size());
    q.memcpy(d_b4s, b4s.data(), b4s.size());
    q.fill(d_c, 0, 2 * RES).wait();
    q.parallel_for(sycl::nd_range<1>(16, 16),
                   sycl::ext::oneapi::experimental::properties{
                       sycl::ext::oneapi::experimental::sub_group_size<16>},
                   [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                       esimd::simd<int8_t, AF> av = esimd::block_load<int8_t, AF>(d_a);
                       esimd::simd<int8_t, KT * ES> bv = esimd::block_load<int8_t, KT * ES>(d_b8);
                       esimd::simd<int, RES> c = 0;
                       c = xmx::dpas<SD, RC, int, int, int8_t, int8_t, dpas_t::s8,
                                     dpas_t::s8, RES, KT * ES, AF>(c, bv, av);
                       esimd::block_store<int, RES>(d_c, 0, c);
                       esimd::simd<uint8_t, KT * ES / 2> w4 = esimd::block_load<uint8_t, KT * ES / 2>(d_b4);
                       esimd::simd<int, RES> c4 = 0;
                       c4 = xmx::dpas<SD, RC, int, int, uint8_t, int8_t, dpas_t::s4,
                                      dpas_t::s8, RES, KT * ES / 2, AF>(c4, w4, av);
                       esimd::block_store<int, RES>(d_c + RES, 0, c4);
                   });
    q.wait();
    std::vector<int> c(2 * RES);
    q.memcpy(c.data(), d_c, 2 * RES * sizeof(int)).wait();
    for (int mode = 0; mode < 3; ++mode) {
        int errs = 0;
        for (int m = 0; m < RC; ++m)
            for (int n = 0; n < ES; ++n) {
                int gold = 0;
                for (int k = 0; k < KT; ++k) {
                    int bv;
                    if (mode == 0) bv = b8[((k / 4) * ES + n) * 4 + (k % 4)];
                    else {
                        const std::vector<uint8_t>& s = mode == 1 ? b4 : b4s;
                        size_t byte = (size_t)(((k / 8) * ES + n) * 4 + (k % 8) / 2);
                        int code = (k & 1) ? ((s[byte] >> 4) & 15) : (s[byte] & 15);
                        bv = code - 8;
                    }
                    gold += (int)a[m * KT + k] * bv;
                }
                int got = c[mode == 0 ? m * ES + n : RES + m * ES + n];
                if (got != gold) ++errs;
            }
        std::printf("mode %d: %d/%d mismatch (c[0]=%d)\n", mode, errs, RES, c[mode == 0 ? 0 : RES]);
    }
}

// verify gather/scatter int32 with VS=4 and VS=1 distribution
#include <cstdio>
#include <vector>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
namespace esimd = sycl::ext::intel::esimd;
int main() {
    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                  sycl::property::queue::in_order{});
    // src: 16 lanes x 64 bytes, each byte = lane*64+off
    std::vector<uint8_t> h(16 * 64);
    for (int i = 0; i < 16 * 64; ++i) h[i] = (uint8_t)i;
    uint8_t* d = sycl::malloc_device<uint8_t>(16 * 64, q);
    int* o = sycl::malloc_device<int>(64, q);
    int* o2 = sycl::malloc_device<int>(64, q);
    q.memcpy(d, h.data(), h.size());
    q.fill(o, 0, 64);
    q.fill(o2, 0, 64).wait();
    q.parallel_for(sycl::nd_range<1>(16, 16),
                   sycl::ext::oneapi::experimental::properties{
                       sycl::ext::oneapi::experimental::sub_group_size<16>},
                   [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                       esimd::simd<uint32_t, 16> lane = esimd::simd<uint32_t, 16>(0, 1);
                       esimd::simd<int, 64> g4 = esimd::gather<int, 64, 4>(
                           reinterpret_cast<const int*>(d), lane * 64u + 8u);
                       esimd::block_store<int, 64>(o, 0, g4);
                       esimd::simd<uint32_t, 64> e = esimd::simd<uint32_t, 64>(0, 1);
                       esimd::simd<int, 64> g1 = esimd::gather<int, 64>(
                           reinterpret_cast<const int*>(d), (e % 16u) * 4u + 32u);
                       esimd::block_store<int, 64>(o2, 0, g1);
                   });
    q.wait();
    std::vector<int> r1(64), r2(64);
    q.memcpy(r1.data(), o, 256);
    q.memcpy(r2.data(), o2, 256).wait();
    for (int i = 0; i < 16; ++i) std::printf("g4[%2d]=%08x ", i, r1[i]);
    std::printf("\n");
    for (int i = 0; i < 16; ++i) std::printf("g1[%2d]=%08x ", i, r2[i]);
    std::printf("\n");
    std::printf("expect g4[0]=bytes 8..23 = %02x%02x%02x%02x\n", h[11], h[10], h[9], h[8]);
}

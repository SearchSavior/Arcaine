#include <cstdio>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
namespace esimd = sycl::ext::intel::esimd;
int main() {
    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"));
    for (int wg : {32, 64, 128, 256}) {
        try {
            q.parallel_for(sycl::nd_range<1>(wg, wg),
                sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::sub_group_size<16>},
                [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL { esimd::fence(); }).wait();
            std::printf("wg=%d ok\n", wg);
        } catch (sycl::exception& e) { std::printf("wg=%d FAIL %s\n", wg, e.what()); }
    }
}

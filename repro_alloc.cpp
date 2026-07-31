#include <cstdio>
#include <sycl/sycl.hpp>
int main()
{
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    printf("global_mem: %zu MB\n",
           q.get_device().get_info<sycl::info::device::global_mem_size>() >> 20);
    // Single 1GB allocation
    void* p1 = sycl::malloc_device(1ull << 30, q);
    printf("1GB single alloc: %s\n", p1 ? "OK" : "FAIL");
    // Many 1MB allocations until failure
    size_t n = 0;
    std::vector<void*> ptrs;
    while (true) {
        void* p = sycl::malloc_device(1ull << 20, q);
        if (!p) break;
        ptrs.push_back(p);
        if (++n % 1024 == 0) printf("  %zu MB in 1MB chunks\n", n);
        if (n >= 28000) break;
    }
    printf("1MB chunks before failure: %zu MB\n", n);
    return 0;
}

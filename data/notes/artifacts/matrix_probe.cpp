// Standalone probe: print the joint_matrix hardware combinations the GPU
// reports, so the NVFP4 DPAS kernel can pick a valid (M,N,K) tile shape.
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
    for (auto& dev : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        auto combos = dev.get_info<
            sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        for (auto& c : combos) {
            std::printf("  msize=%u nsize=%u ksize=%u  "
                        "max(M,N,K)=(%u,%u,%u)  atype=%d btype=%d ctype=%d dtype=%d\n",
                        (unsigned)c.msize, (unsigned)c.nsize, (unsigned)c.ksize,
                        (unsigned)c.max_msize, (unsigned)c.max_nsize, (unsigned)c.max_ksize,
                        (int)c.atype, (int)c.btype, (int)c.ctype, (int)c.dtype);
        }
    }
    return 0;
}

// get_device_props
// Reports SYCL device inventory, generic CPU/device info, and Intel GPU metrics.
// Build via the project CMake, then run the emitted binary from
// .skills/check-available-hardware/get_device_props.

#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

// Does any OpenCL GPU device advertise the Intel sub-group matrix-multiply
// (DPAS) builtin extension? This is the path that actually reaches XMX on this
// stack — oneDNN uses it — even when SYCL's aspect::ext_intel_matrix is off.
static bool opencl_advertises_dpas_builtin() {
    for (const auto& dev : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (dev.get_backend() != sycl::backend::opencl) continue;
        for (const auto& ext : dev.get_info<sycl::info::device::extensions>()) {
            if (ext == "cl_intel_subgroup_matrix_multiply_accumulate")
                return true;
        }
    }
    return false;
}

static const char* arch_name(syclex::architecture a) {
    using A = syclex::architecture;
    switch (a) {
        case A::intel_gpu_bmg_g21: return "intel_gpu_bmg_g21";
        case A::intel_gpu_bmg_g31: return "intel_gpu_bmg_g31";
        case A::intel_gpu_lnl_m:   return "intel_gpu_lnl_m";
        case A::intel_gpu_pvc:     return "intel_gpu_pvc";
        case A::intel_gpu_acm_g10: return "intel_gpu_acm_g10";
        case A::intel_gpu_acm_g11: return "intel_gpu_acm_g11";
        case A::intel_gpu_acm_g12: return "intel_gpu_acm_g12";
        default:                   return "other/unmapped";
    }
}

static const char* backend_name(sycl::backend b) {
    switch (b) {
        case sycl::backend::opencl: return "opencl";
        case sycl::backend::ext_oneapi_level_zero: return "level_zero";
        default: return "other";
    }
}

static const char* device_type_name(const sycl::device& dev) {
    if (dev.is_gpu()) return "gpu";
    if (dev.is_cpu()) return "cpu";
    if (dev.is_accelerator()) return "accelerator";
    return "other";
}

template <typename Container>
static std::string join_sizes(const Container& sizes) {
    std::ostringstream oss;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i) oss << " ";
        oss << sizes[i];
    }
    return oss.str();
}

static std::string bool_text(bool value) {
    return value ? "yes" : "no";
}

static void print_sycl_generic_summary(const sycl::device& dev) {
    std::cout << " SYCL Name : " << dev.get_info<sycl::info::device::name>() << "\n";
    std::cout << " SYCL Vendor : " << dev.get_info<sycl::info::device::vendor>() << "\n";
    std::cout << " SYCL Version : " << dev.get_info<sycl::info::device::version>() << "\n";
    std::cout << " SYCL Driver Version : " << dev.get_info<sycl::info::device::driver_version>() << "\n";
    std::cout << " SYCL Vendor ID : " << dev.get_info<sycl::info::device::vendor_id>() << "\n";
    std::cout << " SYCL Max Compute Units : " << dev.get_info<sycl::info::device::max_compute_units>() << "\n";
    std::cout << " SYCL Max Clock Frequency (MHz) : " << dev.get_info<sycl::info::device::max_clock_frequency>() << "\n";
    std::cout << " SYCL Global Memory Size : " << dev.get_info<sycl::info::device::global_mem_size>() << "\n";
    std::cout << " SYCL Local Memory Size : " << dev.get_info<sycl::info::device::local_mem_size>() << "\n";
    std::cout << " SYCL Max Memory Alloc Size : " << dev.get_info<sycl::info::device::max_mem_alloc_size>() << "\n";
    std::cout << " SYCL Max Work-group Size : " << dev.get_info<sycl::info::device::max_work_group_size>() << "\n";
    std::cout << " SYCL Max Work-item Dimensions : " << dev.get_info<sycl::info::device::max_work_item_dimensions>() << "\n";
    std::cout << " SYCL Supported Sub-group Sizes : "
              << join_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()) << "\n";
}

static void print_sycl_aspects(const sycl::device& dev) {
    std::cout << " SYCL Aspects :"
              << " cpu=" << bool_text(dev.has(sycl::aspect::cpu))
              << " gpu=" << bool_text(dev.has(sycl::aspect::gpu))
              << " fp16=" << bool_text(dev.has(sycl::aspect::fp16))
              << " fp64=" << bool_text(dev.has(sycl::aspect::fp64))
              << " usm_shared=" << bool_text(dev.has(sycl::aspect::usm_shared_allocations))
              << " usm_device=" << bool_text(dev.has(sycl::aspect::usm_device_allocations))
              << " legacy_image=" << bool_text(dev.has(sycl::aspect::ext_intel_legacy_image))
              << " matrix=" << bool_text(dev.has(sycl::aspect::ext_intel_matrix))
              << "\n";
}

static void report_intel_gpu_metrics(const sycl::device& dev) {
    auto numSlices = dev.get_info<sycl::ext::intel::info::device::gpu_slices>();
    auto numSubslicesPerSlice = dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
    auto numEUsPerSubslice = dev.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
    auto numThreadsPerEU = dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
    auto global_mem_size = dev.get_info<sycl::info::device::global_mem_size>();
    auto local_mem_size = dev.get_info<sycl::info::device::local_mem_size>();
    auto max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
    auto sub_group_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();

    std::cout << " XeCore count : " << numSlices * numSubslicesPerSlice << "\n";
    std::cout << " Vector Engines per XeCore : " << numEUsPerSubslice << "\n";
    std::cout << " Vector Engine count : " << numSlices * numSubslicesPerSlice * numEUsPerSubslice << "\n";
    std::cout << " Hardware Threads per Vector Engine : " << numThreadsPerEU << "\n";
    std::cout << " Hardware Threads count : "
              << numSlices * numSubslicesPerSlice * numEUsPerSubslice * numThreadsPerEU << "\n";
    std::cout << " GPU Memory Size : " << global_mem_size << "\n";
    std::cout << " Shared Local Memory per Work-group : " << local_mem_size << "\n";
    std::cout << " Max Work-group size : " << max_work_group_size << "\n";
    std::cout << " Supported Sub-group sizes : " << join_sizes(sub_group_sizes) << "\n";

    auto arch = dev.get_info<syclex::info::device::architecture>();
    bool sycl_matrix = dev.has(sycl::aspect::ext_intel_matrix);
    std::cout << " Architecture : " << arch_name(arch) << "\n";
    std::cout << " joint_matrix aspect (ext_intel_matrix) : "
              << (sycl_matrix ? "yes" : "NO") << "\n";
    std::cout << " DPAS via OpenCL matrix-mad builtin : "
              << (opencl_advertises_dpas_builtin() ? "yes" : "no") << "\n";
}

static void report_device(int index, const sycl::device& dev, const std::string& backend_summary = "") {
    std::cout << "Device [index " << index << "]:\n";
    if (backend_summary.empty()) {
        std::cout << " Backend : " << backend_name(dev.get_backend()) << "\n";
    } else {
        std::cout << " Backends : " << backend_summary << "\n";
        std::cout << " Reported via : " << backend_name(dev.get_backend()) << "\n";
    }
    std::cout << " Type : " << device_type_name(dev) << "\n";
    print_sycl_generic_summary(dev);
    print_sycl_aspects(dev);

    if (dev.is_gpu()) {
        report_intel_gpu_metrics(dev);
    }
}

int main() {
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    auto cpu_devices = sycl::device::get_devices(sycl::info::device_type::cpu);

    std::vector<sycl::device> level_zero_gpus;
    std::vector<sycl::device> opencl_gpus;
    for (const auto& dev : gpu_devices) {
        if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            level_zero_gpus.push_back(dev);
        } else if (dev.get_backend() == sycl::backend::opencl) {
            opencl_gpus.push_back(dev);
        }
    }

    const auto physical_gpu_count =
        !level_zero_gpus.empty() ? level_zero_gpus.size() : gpu_devices.size();
    std::cout << "Physical GPU devices : " << physical_gpu_count << "\n";
    std::cout << "SYCL GPU backend entries : " << gpu_devices.size()
              << " (" << level_zero_gpus.size() << " level_zero, "
              << opencl_gpus.size() << " opencl)\n";
    std::cout << "CPU devices : " << cpu_devices.size() << "\n\n";

    struct GpuGroup {
        const sycl::device* report_device;
        size_t level_zero_count;
        size_t opencl_count;
    };

    std::vector<GpuGroup> gpu_groups;
    std::unordered_map<std::string, size_t> gpu_group_by_model;
    for (const auto& dev : gpu_devices) {
        auto device_name = dev.get_info<sycl::info::device::name>();
        auto it = gpu_group_by_model.find(device_name);
        if (it == gpu_group_by_model.end()) {
            gpu_group_by_model.emplace(device_name, gpu_groups.size());
            gpu_groups.push_back(GpuGroup{&dev, 0, 0});
            it = gpu_group_by_model.find(device_name);
        }

        auto& group = gpu_groups[it->second];
        if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ++group.level_zero_count;
            group.report_device = &dev;
        } else if (dev.get_backend() == sycl::backend::opencl) {
            ++group.opencl_count;
        }
    }

    std::cout << "GPU device models : " << gpu_groups.size() << "\n\n";
    if (gpu_groups.empty()) {
        std::cout << "GPU devices : none\n\n";
    } else {
        for (size_t i = 0; i < gpu_groups.size(); ++i) {
            const auto& group = gpu_groups[i];
            std::ostringstream backends;
            backends << group.level_zero_count << " level_zero, "
                     << group.opencl_count << " opencl";
            report_device(static_cast<int>(i), *group.report_device, backends.str());
            std::cout << "\n";
        }
    }

    std::vector<const sycl::device*> unique_cpu_devices;
    std::unordered_set<std::string> seen_cpu_models;
    for (const auto& dev : cpu_devices) {
        auto device_name = dev.get_info<sycl::info::device::name>();
        if (seen_cpu_models.insert(device_name).second) {
            unique_cpu_devices.push_back(&dev);
        }
    }

    std::cout << "CPU devices : " << cpu_devices.size()
              << " discovered, " << unique_cpu_devices.size() << " unique model tables\n";
    if (unique_cpu_devices.empty()) {
        std::cout << " (none)\n\n";
    } else {
        std::cout << "\n";
        const int first_cpu_index = static_cast<int>(gpu_groups.size());
        for (size_t i = 0; i < unique_cpu_devices.size(); ++i) {
            report_device(first_cpu_index + static_cast<int>(i), *unique_cpu_devices[i]);
            std::cout << "\n";
        }
    }
}

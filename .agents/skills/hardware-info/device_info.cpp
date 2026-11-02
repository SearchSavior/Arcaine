#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>

int main() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    // 去重：只保留 Level Zero 后端的设备
    std::vector<sycl::device> unique_devs;
    for (auto& dev : devices) {
        auto platform_name = dev.get_platform().get_info<sycl::info::platform::name>();
        if (platform_name.find("Level-Zero") != std::string::npos) {
            unique_devs.push_back(dev);
        }
    }

    std::cout << "Found " << unique_devs.size() << " GPU(s) (Level-Zero backend)\n\n";

    if (unique_devs.empty()) {
        std::cout << "No GPU found!" << std::endl;
        return 1;
    }

    auto& dev = unique_devs[0];
    int xve_count = dev.get_info<sycl::info::device::max_compute_units>();
    int xe_cores = xve_count / 8;
    int max_clock = dev.get_info<sycl::info::device::max_clock_frequency>();
    size_t global_mem = dev.get_info<sycl::info::device::global_mem_size>();
    size_t global_cache = dev.get_info<sycl::info::device::global_mem_cache_size>();
    size_t local_mem = dev.get_info<sycl::info::device::local_mem_size>();
    auto cache_line = dev.get_info<sycl::info::device::global_mem_cache_line_size>();

    // ============ 基本设备信息 ============
    std::cout << "============================================" << std::endl;
    std::cout << "  Device Info (Card 0, all cards identical)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Name:                 " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Vendor:               " << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Driver Version:       " << dev.get_info<sycl::info::device::driver_version>() << std::endl;
    std::cout << "Max Compute Units:    " << xve_count << " (= XVE count)" << std::endl;
    std::cout << "Xe-cores:             " << xe_cores << " (= XVE / 8)" << std::endl;
    std::cout << "Max Work Group Size:  " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Max Clock Freq(MHz):  " << max_clock << std::endl;

    auto wi_sizes = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
    std::cout << "Max Work Item Sizes:  " << wi_sizes[0] << " x " << wi_sizes[1] << " x " << wi_sizes[2] << std::endl;

    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::cout << "Sub-Group Sizes:      ";
    for (auto s : sg_sizes) std::cout << s << " ";
    std::cout << "(= supported SIMD widths)" << std::endl;
    std::cout << "Max Sub-Groups/WG:    " << dev.get_info<sycl::info::device::max_num_sub_groups>() << std::endl;

    // ============ 内存层级信息 ============
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Memory Hierarchy" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Global Memory:        " << global_mem / (1024*1024) << " MB (VRAM)" << std::endl;
    std::cout << "L2 Cache (shared):    " << global_cache / 1024 << " KB" << std::endl;
    std::cout << "SLM (per Xe-core):    " << local_mem / 1024 << " KB" << std::endl;
    std::cout << "Cache Line Size:      " << cache_line << " bytes" << std::endl;
    std::cout << "Mem Address Bits:     " << dev.get_info<sycl::info::device::address_bits>() << std::endl;

    // ============ 带宽信息（理论值计算） ============
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Bandwidth (Theoretical Peak)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::fixed << std::setprecision(1);

    // --- VRAM 全局内存带宽（从硬件实际读取） ---
    int mem_bus_width_bits = dev.get_info<sycl::ext::intel::info::device::memory_bus_width>();
    int mem_clock_khz = dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();

    // 驱动可能返回无效值，需要回退到已知规格
    // B60 (BMG, device 0xe211): 192-bit GDDR6, 19 Gbps
    if (mem_bus_width_bits <= 64 || mem_clock_khz == 0) {
        std::cout << "[WARN] Driver returned invalid VRAM info, using known B60 specs" << std::endl;
        mem_bus_width_bits = 192;    // B60: 192-bit bus
        mem_clock_khz = 4750000;     // B60: 4750 MHz base → ×4 QDR = 19 Gbps
    }

    double mem_clock_ghz = mem_clock_khz / 1e6;
    double multiplier = (mem_bus_width_bits >= 512) ? 2.0 : 4.0;
    double mem_data_rate_gbps = mem_clock_ghz * multiplier;
    double vram_bw = (double)mem_bus_width_bits / 8.0 * mem_data_rate_gbps;
    
    std::cout << "VRAM Bus Width:       " << mem_bus_width_bits << " bit (from hardware)" << std::endl;
    std::cout << "VRAM Clock Rate:      " << mem_clock_khz / 1000 << " MHz" << std::endl;
    std::cout << "VRAM Data Rate:       " << mem_data_rate_gbps << " Gbps" << std::endl;
    std::cout << "VRAM Bandwidth:       " << vram_bw << " GB/s (per card)" << std::endl;

    // --- L2 Cache 带宽 ---
    // Xe2-HPG: L2 每周期可读 64B/Xe-core (典型值)
    double l2_bw_per_xecore = 64.0 * max_clock * 1e6 / 1e9;
    double l2_bw_total = l2_bw_per_xecore * xe_cores;
    std::cout << "L2 BW (per Xe-core):  " << l2_bw_per_xecore << " GB/s" << std::endl;
    std::cout << "L2 BW (total):        " << l2_bw_total << " GB/s" << std::endl;

    // --- SLM (Shared Local Memory) 带宽 ---
    // Xe2: SLM 每周期 64B read + 32B write / Xe-core (典型值)
    double slm_read_bw = 64.0 * max_clock * 1e6 / 1e9;
    double slm_write_bw = 32.0 * max_clock * 1e6 / 1e9;
    std::cout << "SLM Read BW/Xe-core:  " << slm_read_bw << " GB/s" << std::endl;
    std::cout << "SLM Write BW/Xe-core: " << slm_write_bw << " GB/s" << std::endl;
    std::cout << "SLM Total BW (all):   " << slm_read_bw * xe_cores << " GB/s (read)" << std::endl;

    // --- 寄存器 (GRF) 带宽 ---
    // 每个 XVE 每周期可以从 GRF 读取 3 个 source operands (3x 32B = 96B)
    // 并写回 1 个 destination (32B)，总计 128B/cycle/XVE
    double grf_bw_per_xve = 128.0 * max_clock * 1e6 / 1e9;
    double grf_bw_total = grf_bw_per_xve * xve_count;
    std::cout << "GRF BW (per XVE):     " << grf_bw_per_xve << " GB/s" << std::endl;
    std::cout << "GRF BW (total):       " << grf_bw_total / 1000.0 << " TB/s" << std::endl;

    // ============ 算力信息 ============
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Compute (Theoretical Peak)" << std::endl;
    std::cout << "============================================" << std::endl;

    // FP32: 每 XVE 有 8 个 ALU, FMA = 2 FLOP
    double fp32_tflops = (double)xve_count * 8 * 2 * max_clock * 1e6 / 1e12;
    double fp16_tflops = fp32_tflops * 2;
    double int8_tops = fp32_tflops * 4;
    std::cout << "FP32:                 " << fp32_tflops << " TFLOPS" << std::endl;
    std::cout << "FP16:                 " << fp16_tflops << " TFLOPS" << std::endl;
    std::cout << "INT8:                 " << int8_tops << " TOPS" << std::endl;
    std::cout << "FP32 (8-card total):  " << fp32_tflops * unique_devs.size() << " TFLOPS" << std::endl;

    // ============ Roofline 关键分界点 ============
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Roofline Key Points (FP32)" << std::endl;
    std::cout << "============================================" << std::endl;

    double ridge_vram = fp32_tflops * 1000.0 / vram_bw;
    double ridge_l2 = fp32_tflops * 1000.0 / l2_bw_total;
    double ridge_slm = fp32_tflops * 1000.0 / (slm_read_bw * xe_cores);
    std::cout << "Ridge Point (VRAM):   " << ridge_vram << " FLOP/Byte" << std::endl;
    std::cout << "Ridge Point (L2):     " << ridge_l2 << " FLOP/Byte" << std::endl;
    std::cout << "Ridge Point (SLM):    " << ridge_slm << " FLOP/Byte" << std::endl;
    std::cout << "\n=> Kernel arithmetic intensity > Ridge Point => compute-bound" << std::endl;
    std::cout << "=> Kernel arithmetic intensity < Ridge Point => memory-bound" << std::endl;

    // ============ 内存带宽层级总览 ============
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Memory Bandwidth Hierarchy Summary" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  GRF (Register) :  " << grf_bw_total / 1000.0 << " TB/s" << std::endl;
    std::cout << "  SLM (Local Mem):  " << slm_read_bw * xe_cores / 1000.0 << " TB/s" << std::endl;
    std::cout << "  L2 Cache       :  " << l2_bw_total << " GB/s" << std::endl;
    std::cout << "  VRAM (Global)  :  " << vram_bw << " GB/s" << std::endl;
    std::cout << "  --------------------------------" << std::endl;
    std::cout << "  GRF >> SLM >> L2 >> VRAM (fast to slow)" << std::endl;

    // ============ 架构固定常量 ============
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Architecture Constants (Xe2-HPG)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "GRF per thread:       128 (default) / 256 (large GRF)" << std::endl;
    std::cout << "GRF register width:   32 bytes (256-bit)" << std::endl;
    std::cout << "GRF total per thread: 128 x 32B = 4 KB (default)" << std::endl;
    std::cout << "Threads per XVE:      8 (default) / 4 (large GRF)" << std::endl;
    std::cout << "L1 Cache/Xe-core:     ~192 KB (estimated)" << std::endl;
    std::cout << "Total GPUs:           " << unique_devs.size() << std::endl;

    return 0;
}

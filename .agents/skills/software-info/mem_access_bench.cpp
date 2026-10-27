#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>

// ============================================================
// Memory Access Pattern Benchmark
// Compares: scalar, vec2, vec4, vec8, stride-2, stride-4, random
// ============================================================

constexpr size_t DATA_SIZE = 64 * 1024 * 1024 / sizeof(float); // 64 MB
constexpr int WARMUP = 5;
constexpr int REPEAT = 20;

// Helper: measure kernel bandwidth
template <typename KernelFunc>
double measure_bandwidth(sycl::queue& q, size_t bytes, KernelFunc&& launch) {
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        launch();
    }
    q.wait();

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPEAT; i++) {
        launch();
    }
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - start).count() / REPEAT;
    return (double)bytes / seconds / 1e9; // GB/s
}

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Data size: " << DATA_SIZE * sizeof(float) / (1024 * 1024) << " MB\n\n";

    // Allocate device memory
    float* src = sycl::malloc_device<float>(DATA_SIZE, q);
    float* dst = sycl::malloc_device<float>(DATA_SIZE, q);

    // Initialize source data
    std::vector<float> host_data(DATA_SIZE);
    std::iota(host_data.begin(), host_data.end(), 0.0f);
    q.memcpy(src, host_data.data(), DATA_SIZE * sizeof(float)).wait();
    q.memset(dst, 0, DATA_SIZE * sizeof(float)).wait();

    // Random indices for random access test
    std::vector<uint32_t> host_indices(DATA_SIZE);
    std::mt19937 rng(42);
    for (size_t i = 0; i < DATA_SIZE; i++) {
        host_indices[i] = rng() % DATA_SIZE;
    }
    uint32_t* indices = sycl::malloc_device<uint32_t>(DATA_SIZE, q);
    q.memcpy(indices, host_indices.data(), DATA_SIZE * sizeof(uint32_t)).wait();

    size_t total_bytes = DATA_SIZE * sizeof(float) * 2; // read + write
    double peak_bw = 456.0; // B60 peak VRAM BW

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "=== Memory Access Pattern Benchmark ===\n\n";
    std::cout << std::left << std::setw(25) << "Pattern"
              << std::setw(20) << "Bandwidth (GB/s)"
              << "Efficiency (%)\n";
    std::cout << std::string(60, '-') << "\n";

    // --- 1. Scalar load ---
    {
        size_t N = DATA_SIZE;
        double bw = measure_bandwidth(q, total_bytes, [&]() {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i] = src[i];
            });
        });
        std::cout << std::setw(25) << "Scalar load"
                  << std::setw(20) << bw
                  << bw / peak_bw * 100 << "%\n";
    }

    // --- 2. vec<float,2> load ---
    {
        size_t N = DATA_SIZE / 2;
        auto* src2 = reinterpret_cast<sycl::vec<float,2>*>(src);
        auto* dst2 = reinterpret_cast<sycl::vec<float,2>*>(dst);
        double bw = measure_bandwidth(q, total_bytes, [&]() {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst2[i] = src2[i];
            });
        });
        std::cout << std::setw(25) << "vec<float,2> load"
                  << std::setw(20) << bw
                  << bw / peak_bw * 100 << "%\n";
    }

    // --- 3. vec<float,4> load ---
    {
        size_t N = DATA_SIZE / 4;
        auto* src4 = reinterpret_cast<sycl::vec<float,4>*>(src);
        auto* dst4 = reinterpret_cast<sycl::vec<float,4>*>(dst);
        double bw = measure_bandwidth(q, total_bytes, [&]() {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst4[i] = src4[i];
            });
        });
        std::cout << std::setw(25) << "vec<float,4> load"
                  << std::setw(20) << bw
                  << bw / peak_bw * 100 << "%\n";
    }

    // --- 4. vec<float,8> load ---
    {
        size_t N = DATA_SIZE / 8;
        auto* src8 = reinterpret_cast<sycl::vec<float,8>*>(src);
        auto* dst8 = reinterpret_cast<sycl::vec<float,8>*>(dst);
        double bw = measure_bandwidth(q, total_bytes, [&]() {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst8[i] = src8[i];
            });
        });
        std::cout << std::setw(25) << "vec<float,8> load"
                  << std::setw(20) << bw
                  << bw / peak_bw * 100 << "%\n";
    }

    // --- 5. Stride-2 access ---
    {
        size_t N = DATA_SIZE / 2;
        double bw = measure_bandwidth(q, N * sizeof(float) * 2, [&]() {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i] = src[i * 2];
            });
        });
        std::cout << std::setw(25) << "Stride-2"
                  << std::setw(20) << bw
                  << bw / peak_bw * 100 << "%\n";
    }

    // --- 6. Stride-4 access ---
    {
        size_t N = DATA_SIZE / 4;
        double bw = measure_bandwidth(q, N * sizeof(float) * 2, [&]() {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i] = src[i * 4];
            });
        });
        std::cout << std::setw(25) << "Stride-4"
                  << std::setw(20) << bw
                  << bw / peak_bw * 100 << "%\n";
    }

    // --- 7. Random access ---
    {
        size_t N = DATA_SIZE;
        double bw = measure_bandwidth(q, N * sizeof(float) * 2 + N * sizeof(uint32_t), [&]() {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i] = src[indices[i]];
            });
        });
        std::cout << std::setw(25) << "Random"
                  << std::setw(20) << bw
                  << bw / peak_bw * 100 << "%\n";
    }

    std::cout << "\nPeak VRAM BW (theoretical): " << peak_bw << " GB/s\n";

    sycl::free(src, q);
    sycl::free(dst, q);
    sycl::free(indices, q);
    return 0;
}

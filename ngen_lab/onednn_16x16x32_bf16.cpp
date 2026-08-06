// oneDNN fixed-shape BF16 16x16x32 matmul - comparison harness against the
// ngen-authored kernel in ngen_dpas_8x8_matmul_bf16_16x16x32.cpp.
//
// Same fixed shape, same env knobs, same perf reporting as the ngen harness.
// No CPU parity and no correctness gate: this is a throughput comparison.
//
// Environment knobs (identical to the ngen harness):
//   NGEN_LAB_GROUPS      work-groups / batch tiles per execution  (default 8192)
//   NGEN_LAB_DPAS_ITERS  repeated matmul executions per sample   (default 4096)
//   NGEN_LAB_REPS        timed samples (best kept)               (default 10)
//
// The ngen kernel loads one tile into registers and accumulates iters times;
// oneDNN cannot express register reuse, so the equivalent work is expressed as
// one matmul primitive executed iters times over the same batch of tiles.
// Total FLOPs = 2*M*N*K * groups * iters - identical accounting to the ngen
// harness.
//
// Device selection: ONEAPI_DEVICE_SELECTOR=level_zero:1 (or :0) selects the GPU
// for the SYCL runtime (oneDNN GPU engine index 0 = default SYCL device).
//
// Build: icpx -fsycl -I/opt/onednn/include onednn_16x16x32_bf16.cpp \
//            -L/opt/onednn/lib -ldnnl -o onednn_16x16x32_bf16

#include <dnnl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static uint32_t envU32(const char *name, uint32_t dflt) {
    const char *v = std::getenv(name);
    return v ? uint32_t(std::atoi(v)) : dflt;
}

int main() {
    using namespace dnnl;

    const uint32_t groups = envU32("NGEN_LAB_GROUPS", 8192);
    const uint32_t iters = envU32("NGEN_LAB_DPAS_ITERS", 4096);
    const int reps = int(envU32("NGEN_LAB_REPS", 10));
    constexpr int M = 16, N = 16, K = 32;

    std::printf("onednn %d.%d.%d  gpu engine (SYCL)\n", DNNL_VERSION_MAJOR,
            DNNL_VERSION_MINOR, DNNL_VERSION_PATCH);

    engine eng(engine::kind::gpu, 0);
    stream s(eng);

    // src {groups, M, K} bf16, weights {1, K, N} bf16 (broadcast over batch),
    // dst {groups, M, N} f32.
    const auto Mm = int64_t(groups) * M; // flattened batch: groups tiles of 16 rows
    auto src_md = memory::desc({Mm, K}, memory::data_type::bf16,
            memory::format_tag::any);
    auto wei_md = memory::desc({K, N}, memory::data_type::bf16,
            memory::format_tag::any);
    auto dst_md = memory::desc({Mm, N}, memory::data_type::f32,
            memory::format_tag::any);

    auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md);
    auto prim = matmul(pd);

    memory src(pd.src_desc(), eng), wei(pd.weights_desc(), eng),
            dst(pd.dst_desc(), eng);

    // Host data: random bf16 in [-1, 1).
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<uint16_t> hSrc(Mm * K);
    std::vector<uint16_t> hWei(size_t(K) * N);
    for (auto &v : hSrc) {
        float f = dist(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        u += 0x7FFFu + ((u >> 16) & 1u);
        v = uint16_t(u >> 16);
    }
    for (auto &v : hWei) {
        float f = dist(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        u += 0x7FFFu + ((u >> 16) & 1u);
        v = uint16_t(u >> 16);
    }
    {
        auto *p = src.map_data<uint16_t>();
        std::memcpy(p, hSrc.data(), hSrc.size() * 2);
        src.unmap_data(p);
    }
    {
        auto *p = wei.map_data<uint16_t>();
        std::memcpy(p, hWei.data(), hWei.size() * 2);
        wei.unmap_data(p);
    }

    // Warmup + timed samples: iters primitive executions per sample.
    double best = 1e30;
    for (int r = 0; r < reps + 2; r++) {
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < iters; i++)
            prim.execute(s, {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei},
                                    {DNNL_ARG_DST, dst}});
        s.wait();
        auto t1 = std::chrono::steady_clock::now();
        if (r >= 2)
            best = std::min(best,
                    std::chrono::duration<double>(t1 - t0).count());
    }

    const double flops = 2.0 * double(Mm) * N * K * double(iters);
    std::printf("[perf] onednn groups=%u iters=%u  %8.3f ms  %8.2f GFLOP/s\n",
            groups, iters, best * 1e3, flops / best / 1e9);

    // Light sanity: read back a few dst values (no CPU comparison).
    std::vector<float> out(Mm * N);
    {
        auto *p = dst.map_data<float>();
        std::memcpy(out.data(), p, out.size() * 4);
        dst.unmap_data(p);
    }
    std::printf("[sanity] dst[0..3] = %.4f %.4f %.4f %.4f\n", out[0],
            out[1], out[2], out[3]);
    return 0;
}

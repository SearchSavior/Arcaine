// Qwen3.5-MoE full-attention DECODE (seq==1) kernel benchmark. The op is
// weight-free (q/k/v are activations), so no checkpoint is needed; synthetic
// inputs mirror the production layouts: Q (nq, hd) flat, K/V cache rows
// (kv, nkv, hd) bf16, scale = 1/sqrt(hd), mask all-zero (skip_mask decode).
//   split: batched_attention_baseline  (expand K/V + 2 GEMVs + softmax,
//          ~10 launches, 4+ amplified KV passes)
//   fused: qwen_attn_decode_fused      (flash-decoding + DPAS with GQA
//          reuse, one KV pass, 2 launches; QWEN35_ATTN_DECODE_FUSED=1 in
//          production)
// Numerics are checked against the split path (ctx bf16 out). Never
// end-to-end inference. Registered as `qwen35moe-attention` in kbench.
//
// Run:
//   ./build/arcaine_kbench qwen35moe-attention --kernels split,fused \
//       -d 1024,8192,32768,65536 -n 20

#include "benchmarks/registry.hpp"
#include "benchmarks/util.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "modeling/qwen3_5_moe/kernels/attention_batched.hpp"

using arcaine::bench::aggregate;
using arcaine::bench::parse_int_csv;
using arcaine::bench::split_csv;
using arcaine::bench::Stat;
using namespace qwen35moe_kernels;

namespace {

constexpr int kNQ  = 16;   // query heads
constexpr int kNKV = 2;    // kv heads
constexpr int kHD  = 256;  // head dim
constexpr float kScale = 1.0f / 16.0f;   // 1/sqrt(256)

void usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -d, --d <csv>     decode KV depths to sweep    (default: 1024,4096,8192,16384,32768,65536)\n"
        "  -n, --n <N>       operations per timed sample  (default: 20)\n"
        "  -w, --w <N>       warmup runs per cell         (default: 1)\n"
        "  -r, --r <N>       timed runs per cell          (default: 5)\n"
        "  --kernels <csv>   split,fused                  (default: both)\n"
        "  --device <N>      visible GPU via ZE_AFFINITY_MASK\n"
        "  --seed <S>        synthetic input seed         (default: 42)\n"
        "  --md              emit a markdown table\n",
        program);
}

int run(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string d_csv = "1024,4096,8192,16384,32768,65536";
    std::string kernels_csv = "split,fused";
    std::string device;
    int iterations = 20;
    int warmup = 1;
    int runs = 5;
    unsigned seed = 42;
    bool md = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (arg == "-d" || arg == "--d") d_csv = next();
        else if (arg == "-n" || arg == "--n") iterations = std::stoi(next());
        else if (arg == "-w" || arg == "--w") warmup = std::stoi(next());
        else if (arg == "-r" || arg == "--r") runs = std::stoi(next());
        else if (arg == "--kernels") kernels_csv = next();
        else if (arg == "--device") device = next();
        else if (arg == "--seed") seed = (unsigned)std::stoul(next());
        else if (arg == "--md") md = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", arg.c_str()); return 1; }
    }
    if (!device.empty()) gpu_device_control::apply_device_index(device);
    std::vector<int> depths = parse_int_csv(d_csv);
    std::vector<std::string> kernels = split_csv(kernels_csv);
    if (depths.empty() || kernels.empty() || warmup < 0 || runs <= 0 || iterations <= 0)
        throw std::runtime_error("invalid benchmark arguments");

    auto& context = GpuEngine::get(0);
    auto& queue = context.queue;
    const int max_kv = *std::max_element(depths.begin(), depths.end());

    uint32_t rng = seed;
    auto rand_bf = [&](size_t n, float lo, float hi) {
        std::vector<bf16> v(n);
        for (auto& x : v) {
            rng = rng * 1664525u + 1013904223u;
            float f = lo + (hi - lo) * (float)((rng >> 8) & 0xFFFFFF) / 16777216.0f;
            x = float_to_bf16(f);
        }
        return v;
    };

    std::vector<bf16> q_h = rand_bf((size_t)kNQ * kHD, -1.0f, 1.0f);
    std::vector<bf16> k_h = rand_bf((size_t)max_kv * kNKV * kHD, -1.0f, 1.0f);
    std::vector<bf16> v_h = rand_bf((size_t)max_kv * kNKV * kHD, -1.0f, 1.0f);
    GpuBuffer<bf16> qbuf(q_h.size(), queue), kbuf(k_h.size(), queue),
                    vbuf(v_h.size(), queue);
    qbuf.upload(q_h.data(), q_h.size());
    kbuf.upload(k_h.data(), k_h.size());
    vbuf.upload(v_h.data(), v_h.size());

    // Both variants write into the shared attention scratch (ctx_tm); copy
    // out under the scratch mutex before the next variant runs.
    const size_t out_elems = (size_t)kNQ * kHD;
    GpuBuffer<bf16> outbuf(out_elems, queue);
    std::mutex& mtx = qwen_attn_batched_detail::scratch_mutex();

    std::printf("[bench] qwen35moe attention decode: nq=%d nkv=%d hd=%d | "
                "kernels={", kNQ, kNKV, kHD);
    for (size_t i = 0; i < kernels.size(); ++i)
        std::printf("%s%s", kernels[i].c_str(), i + 1 < kernels.size() ? "," : "");
    std::printf("} | n=%d warmup=%d runs=%d\n", iterations, warmup, runs);
    if (md)
        std::printf("| kernel | kv | runs | mean us | sd us | max abs | max rel |\n"
                    "|---|---:|---:|---:|---:|---:|---:|\n");

    for (int kv : depths) {
        auto call_split = [&] {
            std::lock_guard<std::mutex> lock(mtx);
            bf16* ctx_out = batched_attention_baseline(context,
                qbuf.data(), 1, kNQ, kHD,
                kbuf.data(), vbuf.data(), kv, kNKV, kHD,
                kv - 1, INT_MAX, kScale, /*skip_mask=*/true);
            queue.memcpy(outbuf.data(), ctx_out, out_elems * sizeof(bf16));
            queue.wait();
        };
        auto call_fused = [&] {
            std::lock_guard<std::mutex> lock(mtx);
            bf16* ctx_out = qwen_attn_decode_fused(context,
                qbuf.data(), kNQ, kbuf.data(), vbuf.data(), kv, kNKV, kScale);
            queue.memcpy(outbuf.data(), ctx_out, out_elems * sizeof(bf16));
            queue.wait();
        };
        auto call = [&](const std::string& kernel) {
            if (kernel == "fused") call_fused();
            else if (kernel == "split") call_split();
            else throw std::runtime_error("unknown kernel: " + kernel);
        };

        // Reference: split path at this depth.
        std::vector<bf16> ref(out_elems);
        call_split();
        outbuf.download(ref.data(), out_elems);

        for (const std::string& kernel : kernels) {
            call(kernel);
            std::vector<bf16> actual(out_elems);
            outbuf.download(actual.data(), out_elems);
            float max_abs = 0.0f, max_rel = 0.0f;
            for (size_t i = 0; i < out_elems; ++i) {
                float expected = bf16_to_float(ref[i]);
                float observed = bf16_to_float(actual[i]);
                float error = std::fabs(observed - expected);
                max_abs = std::max(max_abs, error);
                max_rel = std::max(max_rel,
                                   error / std::max(1e-3f, std::fabs(expected)));
            }

            for (int i = 0; i < warmup; ++i)
                for (int s = 0; s < iterations; ++s) call(kernel);
            std::vector<double> samples;
            samples.reserve(runs);
            for (int i = 0; i < runs; ++i) {
                auto start = std::chrono::steady_clock::now();
                for (int s = 0; s < iterations; ++s) call(kernel);
                samples.push_back(
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start).count() /
                    iterations);
            }
            Stat stat = aggregate(samples);
            if (md) {
                std::printf("| %s | %d | %d | %.1f | %.1f | %.6f | %.6f |\n",
                            kernel.c_str(), kv, runs, stat.mean, stat.sd,
                            max_abs, max_rel);
            } else {
                std::printf("qwen35moe_attention_decode kernel=%s kv=%d runs=%d "
                            "mean_us=%.1f sd_us=%.1f max_abs=%.6f max_rel=%.6f\n",
                            kernel.c_str(), kv, runs, stat.mean, stat.sd,
                            max_abs, max_rel);
            }
        }
    }
    return 0;
}

}  // namespace

REGISTER_BENCH("qwen35moe-attention",
    "Qwen3.5-MoE full-attention decode: split chain vs fused flash-decoding DPAS",
    run)

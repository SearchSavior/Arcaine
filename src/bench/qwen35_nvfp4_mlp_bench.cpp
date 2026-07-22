// Qwen3.5 dense MLP kernel benchmark. Loads one real NVFP4 layer and benchmarks
// only gate/up -> SwiGLU -> down, never inference. ARCAINE_QWEN35_NVFP4_DPAS=0/1
// selects the oneDNN baseline or Xe2 DPAS path. Registered as `qwen35-nvfp4-mlp`
// in the unified kernel_bench binary.
//
// Run:
//   ./build/kernel_bench qwen35-nvfp4-mlp [opts]

#include "common/bench/registry.hpp"
#include "common/bench/util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/gpu/buffer.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/nvfp4.hpp"
#include "common/io/quant_loader.hpp"
#include "modeling/qwen3_5_moe/kernels.hpp"

using arcaine::bench::aggregate;
using arcaine::bench::parse_int_csv;
using arcaine::bench::split_csv;
using arcaine::bench::Stat;

namespace {

enum class Kernel { Onednn, OnednnFusedPack, Xe2Dpas, EsimdDecode };

Kernel select_kernel(const std::string& kernel) {
    Kernel selected;
    if (kernel == "onednn" || kernel == "default") {
        selected = Kernel::Onednn;
    } else if (kernel == "onednn-fused-pack" || kernel == "fused-pack") {
        selected = Kernel::OnednnFusedPack;
    } else if (kernel == "xe2-dpas" || kernel == "dpas") {
        selected = Kernel::Xe2Dpas;
    } else if (kernel == "esimd-decode" || kernel == "esimd") {
        selected = Kernel::EsimdDecode;
    } else {
        throw std::runtime_error(
            "unknown kernel '" + kernel +
            "' (use: onednn, onednn-fused-pack, xe2-dpas, esimd-decode)");
    }
    if (::setenv("ARCAINE_QWEN35_NVFP4_DPAS",
                 selected == Kernel::Xe2Dpas ? "1" : "0", 1) != 0)
        throw std::runtime_error("failed to set ARCAINE_QWEN35_NVFP4_DPAS");
    return selected;
}

void usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s --model <dir> [options]\n"
        "  -p, --p <csv>     token counts to sweep           (default: 512)\n"
        "  -n, --n <N>       operations per timed sample     (default: 1)\n"
        "  -w, --w <N>       warmup runs per cell            (default: 1)\n"
        "  -r, --r <N>       timed runs per cell             (default: 5)\n"
        "  --kernels <csv>   onednn,onednn-fused-pack,xe2-dpas,esimd-decode\n"
        "  --layer <N>       real NVFP4 decoder layer       (default: 0)\n"
        "  --device <N>      visible GPU via ZE_AFFINITY_MASK\n"
        "  --seed <S>        synthetic input seed           (default: 42)\n"
        "  --md              emit a markdown table\n",
        program);
}

int run(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model = "/workspace/models/unsloth_Qwen3.6-27B-NVFP4";
    std::string p_csv = "512";
    std::string kernels_csv = "onednn,xe2-dpas";
    std::string device;
    int warmup = 1;
    int runs = 5;
    int iterations = 1;
    int layer = 0;
    unsigned seed = 42;
    bool md = false;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else if (arg == "--model") {
                model = next();
            } else if (arg == "-p" || arg == "--p" || arg == "--tokens") {
                p_csv = next();
            } else if (arg == "-n" || arg == "--n") {
                iterations = std::stoi(next());
            } else if (arg == "-w" || arg == "--w" || arg == "--warmup") {
                warmup = std::stoi(next());
            } else if (arg == "-r" || arg == "--r" || arg == "--iters") {
                runs = std::stoi(next());
            } else if (arg == "--kernels") {
                kernels_csv = next();
            } else if (arg == "--layer") {
                layer = std::stoi(next());
            } else if (arg == "--device") {
                device = next();
            } else if (arg == "--seed") {
                seed = static_cast<unsigned>(std::stoul(next()));
            } else if (arg == "--md") {
                md = true;
            } else {
                std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
                usage(argv[0]);
                return 1;
            }
        }

        std::vector<int> token_counts = parse_int_csv(p_csv);
        std::vector<std::string> kernels = split_csv(kernels_csv);
        if (token_counts.empty() || kernels.empty() || iterations <= 0 ||
            warmup < 0 || runs <= 0 ||
            layer < 0 || layer > 55)
            throw std::runtime_error("invalid benchmark arguments");
        for (int tokens : token_counts)
            if (tokens <= 0) throw std::runtime_error("-p values must be positive");
        if (!device.empty()) gpu_device_control::apply_device_index(device);

        constexpr int H = 5120;
        constexpr int I = 17408;
        int max_tokens = *std::max_element(token_counts.begin(), token_counts.end());
        auto& context = GpuEngine::get(0);
        auto& queue = context.queue;
        ShardedSafetensors checkpoint(model);
        std::string prefix =
            "model.language_model.layers." + std::to_string(layer) + ".mlp.";
        Nvfp4Linear gate_up = upload_nvfp4_linear_pair(
            checkpoint, prefix + "gate_proj", prefix + "up_proj", queue);
        Nvfp4Linear down = upload_nvfp4_linear(checkpoint, prefix + "down_proj", queue);

        std::vector<bf16> host_input((size_t)max_tokens * H);
        uint32_t state = seed;
        for (bf16& element : host_input) {
            state = state * 1664525u + 1013904223u;
            int centered = static_cast<int>((state >> 16) % 257u) - 128;
            element = float_to_bf16(static_cast<float>(centered) / 128.0f);
        }
        GpuBuffer<bf16> input(host_input.size(), queue);
        input.upload(host_input.data(), host_input.size());
        GpuBuffer<bf16> output((size_t)max_tokens * H, queue);
        GpuBuffer<bf16> gate_up_bf16((size_t)max_tokens * 2 * I, queue);
        GpuBuffer<bf16> activation_bf16((size_t)max_tokens * I, queue);
        GpuBuffer<uint8_t> input_packed((size_t)max_tokens * H / 2, queue);
        GpuBuffer<uint8_t> input_scale((size_t)max_tokens * H / 16, queue);
        GpuBuffer<uint8_t> activation_packed((size_t)max_tokens * I / 2, queue);
        GpuBuffer<uint8_t> activation_scale((size_t)max_tokens * I / 16, queue);

        std::printf("[bench] matrix: kernels={%s} p={%s} | n=%d warmup=%d runs=%d | cells=%zu\n",
                    kernels_csv.c_str(), p_csv.c_str(), iterations, warmup, runs,
                    kernels.size() * token_counts.size());
        std::printf("[bench] layer=%d model=%s\n", layer, model.c_str());
        if (md)
            std::printf("| kernel | p | runs | mean ms | sd ms | tokens/s | max abs | max rel |\n"
                        "|---|---:|---:|---:|---:|---:|---:|---:|\n");

        for (const std::string& kernel : kernels) {
            Kernel selected = select_kernel(kernel);
            for (int tokens : token_counts) {
                if (selected == Kernel::EsimdDecode && tokens != 1)
                    throw std::runtime_error(
                        "esimd-decode is an M=1 kernel; benchmark it with --p 1");
                auto run_onednn = [&] {
                    matmul_nvfp4(input.data(), tokens, H, gate_up,
                                 gate_up_bf16.data(), context, input_packed.data(),
                                 input_scale.data());
                    swiglu_strided(queue, gate_up_bf16.data(),
                                  activation_bf16.data(), tokens, I);
                    matmul_nvfp4(activation_bf16.data(), tokens, I, down,
                                 output.data(), context, activation_packed.data(),
                                 activation_scale.data());
                };
                auto run = [&] {
                    if (selected == Kernel::Xe2Dpas) {
                        pack_bf16_to_nvfp4(queue, input.data(), input_packed.data(),
                                           input_scale.data(), tokens, H,
                                           gate_up.input_global_scale);
                        matmul_nvfp4_swiglu_pack_xe2(
                            context, input_packed.data(), input_scale.data(), tokens, H,
                            gate_up, down, activation_packed.data(), activation_scale.data());
                        matmul_nvfp4_packed_xe2(
                            context, activation_packed.data(), activation_scale.data(),
                            tokens, I, down, output.data());
                    } else if (selected == Kernel::EsimdDecode) {
                        pack_bf16_to_nvfp4(queue, input.data(), input_packed.data(),
                                           input_scale.data(), 1, H,
                                           gate_up.input_global_scale);
                        matmul_nvfp4_decode_swiglu_esimd(
                            input_packed.data(), input_scale.data(), H, gate_up,
                            activation_bf16.data(), I, context);
                        pack_bf16_to_nvfp4(
                            queue, activation_bf16.data(), activation_packed.data(),
                            activation_scale.data(), 1, I,
                            down.input_global_scale);
                        matmul_nvfp4_decode_gemv_esimd(
                            activation_packed.data(), activation_scale.data(), I,
                            down, output.data(), context);
                    } else if (selected == Kernel::OnednnFusedPack) {
                        matmul_nvfp4(
                            input.data(), tokens, H, gate_up,
                            gate_up_bf16.data(), context, input_packed.data(),
                            input_scale.data());
                        swiglu_pack_nvfp4(
                            queue, gate_up_bf16.data(), tokens, I,
                            down.input_global_scale, activation_packed.data(),
                            activation_scale.data());
                        matmul_nvfp4_packed(
                            activation_packed.data(), activation_scale.data(),
                            tokens, I, down, output.data(), context);
                    } else {
                        run_onednn();
                    }
                };

                run_onednn();
                queue.wait();
                std::vector<bf16> reference((size_t)tokens * H);
                output.download(reference.data(), reference.size());
                run();
                queue.wait();
                std::vector<bf16> actual(reference.size());
                output.download(actual.data(), actual.size());
                float max_abs = 0.0f, max_rel = 0.0f;
                for (size_t i = 0; i < actual.size(); ++i) {
                    float expected = bf16_to_float(reference[i]);
                    float observed = bf16_to_float(actual[i]);
                    float error = std::fabs(observed - expected);
                    max_abs = std::max(max_abs, error);
                    max_rel = std::max(
                        max_rel, error / std::max(1e-3f, std::fabs(expected)));
                }

                for (int i = 0; i < warmup; ++i)
                    for (int step = 0; step < iterations; ++step) run();
                queue.wait();
                std::vector<double> samples;
                samples.reserve(runs);
                for (int i = 0; i < runs; ++i) {
                    auto start = std::chrono::steady_clock::now();
                    for (int step = 0; step < iterations; ++step) run();
                    queue.wait();
                    samples.push_back(
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count() /
                        iterations);
                }
                Stat stat = aggregate(samples);
                double tokens_per_s = tokens * 1000.0 / stat.mean;
                if (md) {
                    std::printf("| %s | %d | %d | %.3f | %.3f | %.2f | %.6f | %.6f |\n",
                                kernel.c_str(), tokens, runs, stat.mean, stat.sd,
                                tokens_per_s, max_abs, max_rel);
                } else {
                    std::printf("qwen35_nvfp4_mlp kernel=%s layer=%d p=%d runs=%d "
                                "mean_ms=%.3f sd_ms=%.3f tokens_per_s=%.2f "
                                "max_abs=%.6f max_rel=%.6f\n",
                                kernel.c_str(), layer, tokens, runs, stat.mean,
                                stat.sd, tokens_per_s, max_abs, max_rel);
                }
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}

}  // namespace

REGISTER_BENCH("qwen35-nvfp4-mlp",
    "Qwen3.5 dense MLP gate/up->SwiGLU->down (oneDNN / fused-pack / xe2-dpas / esimd-decode)",
    run)


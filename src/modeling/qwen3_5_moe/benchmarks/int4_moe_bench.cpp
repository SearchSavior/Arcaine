// Qwen3.5-MoE AWQ INT4 routed-experts kernel benchmark. Loads one real MoE
// layer from the AWQ checkpoint (router + 256 INT4 routed experts) and
// benchmarks only the routed-expert stage (host top-k already done):
//   onednn:  per-expert oneDNN u4zp matmuls + host scatter
//            (qwen_routed_experts_forward; the numerical reference)
//   dpas:    grouped W4A16 DPAS kernels (qwen_routed_experts_forward_dpas)
//   grouped: oneDNN experimental grouped W4A16 matmuls
//            (qwen_routed_experts_forward_grouped)
// never end-to-end inference. This is the exact region swapped by
// QWEN35_MOE_INT4_IMPL=onednn|dpas|grouped in qwen_moe_forward.
// Registered as `qwen35-awq-int4-moe` in the unified kernel_bench binary.
//
// Run:
//   ./build/kernel_bench qwen35-awq-int4-moe --model \
//       /workspace/models/cyankiwi_Qwen-AgentWorld-35B-A3B-AWQ-INT4 \
//       -p 1,8,64,512 --kernels onednn,dpas

#include "benchmarks/registry.hpp"
#include "benchmarks/util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "runtime/gpu/ops.hpp"
#include "runtime/quantization/quant_loader.hpp"
#include "modeling/qwen3_5_moe/config.hpp"
#include "modeling/qwen3_5_moe/loader_awq.hpp"
#include "modeling/qwen3_5_moe/moe.hpp"
#include "modeling/qwen3_5_moe/weights.hpp"

using arcaine::bench::aggregate;
using arcaine::bench::parse_int_csv;
using arcaine::bench::split_csv;
using arcaine::bench::Stat;

namespace {

void usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s --model <dir> [options]\n"
        "  -p, --p <csv>     token counts to sweep        (default: 1,8,64,512)\n"
        "  -n, --n <N>       operations per timed sample  (default: 1)\n"
        "  -w, --w <N>       warmup runs per cell         (default: 1)\n"
        "  -r, --r <N>       timed runs per cell          (default: 3)\n"
        "  --kernels <csv>   onednn,dpas,grouped           (default: all three)\n"
        "  --layer <N>       decoder layer to load        (default: 0)\n"
        "  --device <N>      visible GPU via ZE_AFFINITY_MASK\n"
        "  --seed <S>        synthetic input seed         (default: 42)\n"
        "  --md              emit a markdown table\n",
        program);
}

// Host softmax + grouped top-k + renorm — identical to qwen_moe_forward's
// router epilogue (norm_topk_prob=1, no per-expert scale).
void host_route(const std::vector<bf16>& scores_h, int S, int E, int top_k,
                std::vector<int>& idx, std::vector<float>& wgt) {
    std::vector<int> order(E);
    std::iota(order.begin(), order.end(), 0);
    for (int t = 0; t < S; ++t) {
        const bf16* row = scores_h.data() + (size_t)t * E;
        float mx = -3.402823466e38f;
        for (int e = 0; e < E; ++e) { float v = bf16_to_float(row[e]); if (v > mx) mx = v; }
        std::vector<float> p(E);
        float sum = 0.0f;
        for (int e = 0; e < E; ++e) { p[e] = std::exp(bf16_to_float(row[e]) - mx); sum += p[e]; }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                          [&](int a, int b) { return p[a] > p[b]; });
        float ssum = 0.0f;
        for (int s = 0; s < top_k; ++s) {
            int e = order[s];
            idx[(size_t)t * top_k + s] = e;
            wgt[(size_t)t * top_k + s] = p[e] * inv;
            ssum += p[e] * inv;
        }
        float sinv = ssum > 0.0f ? 1.0f / ssum : 0.0f;
        for (int s = 0; s < top_k; ++s) wgt[(size_t)t * top_k + s] *= sinv;
    }
}

int run(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model = "/workspace/models/cyankiwi_Qwen-AgentWorld-35B-A3B-AWQ-INT4";
    std::string p_csv = "1,8,64,512";
    std::string kernels_csv = "onednn,dpas,grouped";
    std::string device;
    int warmup = 1;
    int runs = 3;
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
            warmup < 0 || runs <= 0 || layer < 0)
            throw std::runtime_error("invalid benchmark arguments");
        for (int tokens : token_counts)
            if (tokens <= 0) throw std::runtime_error("-p values must be positive");
        for (const std::string& k : kernels)
            if (k != "onednn" && k != "dpas" && k != "grouped")
                throw std::runtime_error("unknown kernel '" + k + "' (use: onednn, dpas, grouped)");
        if (!device.empty()) gpu_device_control::apply_device_index(device);

        QwenConfig cfg = QwenConfig::from_dir(model);
        if (cfg.quant_format != "pack-quantized")
            throw std::runtime_error("expected an AWQ pack-quantized checkpoint");
        const int H     = cfg.hidden_size;
        const int E     = cfg.num_experts;
        const int top_k = cfg.num_experts_per_tok;
        const int inter = cfg.moe_intermediate_size;
        int max_tokens = *std::max_element(token_counts.begin(), token_counts.end());

        auto& context = GpuEngine::get(0);
        auto& queue = context.queue;
        ShardedSafetensors checkpoint(model);
        const std::string mp =
            "model.language_model.layers." + std::to_string(layer) + ".mlp.";

        // Real layer weights through the same AWQ upload path as the loader.
        QwenMoE w;
        w.router_gate = upload(checkpoint.get(mp + "gate.weight"), queue,
                               (mp + "gate.weight").c_str());
        w.grouped = qwen_awq::upload_experts_grouped(checkpoint, mp, E, queue);

        // Synthetic hidden states (post-attention-normed magnitude ~[-1, 1)).
        std::vector<bf16> host_input((size_t)max_tokens * H);
        uint32_t state = seed;
        for (bf16& element : host_input) {
            state = state * 1664525u + 1013904223u;
            int centered = static_cast<int>((state >> 16) % 257u) - 128;
            element = float_to_bf16(static_cast<float>(centered) / 128.0f);
        }
        GpuBuffer<bf16> hidden(host_input.size(), queue);
        hidden.upload(host_input.data(), host_input.size());
        GpuBuffer<bf16> out((size_t)max_tokens * H, queue);

        std::printf("[bench] matrix: kernels={%s} p={%s} | n=%d warmup=%d runs=%d | cells=%zu\n",
                    kernels_csv.c_str(), p_csv.c_str(), iterations, warmup, runs,
                    kernels.size() * token_counts.size());
        std::printf("[bench] layer=%d model=%s H=%d E=%d top_k=%d inter=%d group=%d\n",
                    layer, model.c_str(), H, E, top_k, inter, cfg.quant_group_size);
        if (md)
            std::printf("| kernel | p | runs | mean ms | sd ms | pairs/s | max abs | max rel | cos |\n"
                        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");

        for (int tokens : token_counts) {
            const int pairs = tokens * top_k;

            // Router (untimed, shared by both impls): scores -> host top-k.
            GpuBuffer<bf16> scores((size_t)tokens * E, queue);
            matmul_bf16(hidden.data(), tokens, H, w.router_gate.data(), E,
                        scores.data(), context);
            queue.wait();
            std::vector<bf16> scores_h((size_t)tokens * E);
            queue.memcpy(scores_h.data(), scores.data(),
                         (size_t)tokens * E * sizeof(bf16)).wait();
            std::vector<int>   idx((size_t)pairs);
            std::vector<float> wgt((size_t)pairs);
            host_route(scores_h, tokens, E, top_k, idx, wgt);

            // Device routing buffers: the grouped/dpas production paths
            // consume device idx/wgt (qwen_moe_router_topk output).
            GpuBuffer<int32_t> idx_d((size_t)pairs, queue);
            GpuBuffer<float>   wgt_d((size_t)pairs, queue);
            idx_d.upload(reinterpret_cast<const int32_t*>(idx.data()), pairs);
            wgt_d.upload(wgt.data(), pairs);

            // Reference: production oneDNN per-expert path (host fp32 accum).
            std::vector<float> ref_h((size_t)tokens * H, 0.0f);
            qwen_routed_experts_forward(context, w, hidden.data(), idx, wgt,
                                        ref_h, tokens, cfg);
            std::vector<bf16> ref_b((size_t)tokens * H);
            for (size_t i = 0; i < ref_b.size(); ++i)
                ref_b[i] = float_to_bf16(ref_h[i]);

            for (const std::string& kernel : kernels) {
                auto run = [&] {
                    if (kernel == "dpas") {
                        qwen_routed_experts_forward_dpas(
                            context, w, hidden.data(), idx_d.data(),
                            wgt_d.data(), out.data(), tokens, cfg);
                        queue.wait();
                    } else if (kernel == "grouped") {
                        qwen_routed_experts_forward_grouped(
                            context, w.grouped, hidden.data(), idx_d.data(),
                            wgt_d.data(), out.data(), tokens, top_k);
                        queue.wait();
                    } else {
                        std::vector<float> acc((size_t)tokens * H, 0.0f);
                        qwen_routed_experts_forward(context, w, hidden.data(),
                                                    idx, wgt, acc, tokens, cfg);
                        std::vector<bf16> b(acc.size());
                        for (size_t i = 0; i < b.size(); ++i)
                            b[i] = float_to_bf16(acc[i]);
                        queue.memcpy(out.data(), b.data(),
                                     (size_t)tokens * H * sizeof(bf16)).wait();
                    }
                };

                run();
                std::vector<bf16> actual((size_t)tokens * H);
                out.download(actual.data(), actual.size());
                float max_abs = 0.0f, max_rel = 0.0f;
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    float expected = bf16_to_float(ref_b[i]);
                    float observed = bf16_to_float(actual[i]);
                    float error = std::fabs(observed - expected);
                    max_abs = std::max(max_abs, error);
                    max_rel = std::max(
                        max_rel, error / std::max(1e-3f, std::fabs(expected)));
                    dot += (double)expected * observed;
                    na += (double)expected * expected;
                    nb += (double)observed * observed;
                }
                double cos = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 0.0;

                for (int i = 0; i < warmup; ++i)
                    for (int step = 0; step < iterations; ++step) run();
                std::vector<double> samples;
                samples.reserve(runs);
                for (int i = 0; i < runs; ++i) {
                    auto start = std::chrono::steady_clock::now();
                    for (int step = 0; step < iterations; ++step) run();
                    samples.push_back(
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count() /
                        iterations);
                }
                Stat stat = aggregate(samples);
                double pairs_per_s = pairs * 1000.0 / stat.mean;
                if (md) {
                    std::printf("| %s | %d | %d | %.3f | %.3f | %.1f | %.6f | %.6f | %.6f |\n",
                                kernel.c_str(), tokens, runs, stat.mean, stat.sd,
                                pairs_per_s, max_abs, max_rel, cos);
                } else {
                    std::printf("qwen35_awq_int4_moe kernel=%s layer=%d p=%d runs=%d "
                                "mean_ms=%.3f sd_ms=%.3f pairs_per_s=%.1f "
                                "max_abs=%.6f max_rel=%.6f cos=%.6f\n",
                                kernel.c_str(), layer, tokens, runs, stat.mean,
                                stat.sd, pairs_per_s, max_abs, max_rel, cos);
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

REGISTER_BENCH("qwen35-awq-int4-moe",
    "Qwen3.5-MoE AWQ INT4 routed experts (onednn per-expert u4zp / grouped xe2-dpas / onednn grouped)",
    run)

// Qwen3.5-MoE gated deltanet (linear-attention) core-op benchmark. The op is
// weight-free (q/k/v/beta/g are activations), so no checkpoint is needed;
// synthetic inputs mirror the production layout: q,k l2-normalized per head
// (q pre-scaled by 1/sqrt(d_k)), beta in (0,1), g < 0 (log decay).
//   host:   host_chunk_gated_delta_rule   (download + scalar fp32 + upload)
//   device: qwen_gdn_device_chunk         (SYCL chunked kernel, one WG/head)
//   xmx:    qwen_gdn_device_chunk_xmx     (same, matmuls on XMX via DPAS)
//   hybrid: qwen_gdn_device_chunk_hybrid  (oneDNN batched Gram + parallel
//           T-solve + 2-WG/head persistent fp32-SLM state pass)
// This is the exact region swapped by QWEN35_GDN_IMPL=host|device|xmx|hybrid in
// qwen_linear_attn_forward. Numerics are checked against the host path
// (core output bf16 + final fp32 SSM state). Never end-to-end inference.
// Registered as `qwen35-gdn` in the unified kernel_bench binary.
//
// Run:
//   ./build/arcaine_kbench qwen35-gdn -p 512,1024,2048,4096 --kernels host,device,xmx,hybrid

#include "benchmarks/registry.hpp"
#include "benchmarks/util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "modeling/qwen3_5_moe/gated_deltanet.hpp"

using arcaine::bench::aggregate;
using arcaine::bench::parse_int_csv;
using arcaine::bench::split_csv;
using arcaine::bench::Stat;

namespace {

constexpr int kNV = 32;    // value heads
constexpr int kDK = 128;   // key head dim
constexpr int kDV = 128;   // value head dim
constexpr int kChunk = 64;

void usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -p, --p <csv>     token counts to sweep        (default: 512,1024,2048,4096)\n"
        "  -n, --n <N>       operations per timed sample  (default: 1)\n"
        "  -w, --w <N>       warmup runs per cell         (default: 1)\n"
        "  -r, --r <N>       timed runs per cell          (default: 2)\n"
        "  --kernels <csv>   host,device,xmx,hybrid        (default: all four)\n"
        "  --device <N>      visible GPU via ZE_AFFINITY_MASK\n"
        "  --seed <S>        synthetic input seed         (default: 42)\n"
        "  --md              emit a markdown table\n",
        program);
}

int run(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string p_csv = "512,1024,2048,4096";
    std::string kernels_csv = "host,device,xmx,hybrid";
    std::string device;
    int warmup = 1;
    int runs = 2;
    int iterations = 1;
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
            warmup < 0 || runs <= 0)
            throw std::runtime_error("invalid benchmark arguments");
        for (int tokens : token_counts)
            if (tokens <= 0) throw std::runtime_error("-p values must be positive");
        for (const std::string& k : kernels)
            if (k != "host" && k != "device" && k != "xmx" && k != "hybrid")
                throw std::runtime_error("unknown kernel '" + k + "' (use: host, device, xmx, hybrid)");
        if (!device.empty()) gpu_device_control::apply_device_index(device);

        auto& context = GpuEngine::get(0);
        auto& queue = context.queue;
        int max_tokens = *std::max_element(token_counts.begin(), token_counts.end());

        // Synthetic activations (deterministic LCG), matching production
        // preconditions: q,k l2-normalized per head; q scaled by 1/sqrt(d_k).
        uint32_t rng = seed;
        auto frand = [&](float lo, float hi) {
            rng = rng * 1664525u + 1013904223u;
            return lo + (hi - lo) * (float)((rng >> 8) & 0xFFFFFF) / 16777216.0f;
        };
        const float scale = 1.0f / std::sqrt((float)kDK);
        std::vector<bf16> q_h((size_t)max_tokens * kNV * kDK);
        std::vector<bf16> k_h((size_t)max_tokens * kNV * kDK);
        std::vector<bf16> v_h((size_t)max_tokens * kNV * kDV);
        std::vector<bf16> beta_h((size_t)max_tokens * kNV);
        std::vector<bf16> g_h((size_t)max_tokens * kNV);
        for (size_t row = 0; row < (size_t)max_tokens * kNV; ++row) {
            float nq = 0.0f, nk = 0.0f;
            std::vector<float> tq(kDK), tk(kDK);
            for (int d = 0; d < kDK; ++d) {
                tq[d] = frand(-1.0f, 1.0f); nq += tq[d] * tq[d];
                tk[d] = frand(-1.0f, 1.0f); nk += tk[d] * tk[d];
            }
            nq = 1.0f / std::sqrt(std::max(nq, 1e-12f));
            nk = 1.0f / std::sqrt(std::max(nk, 1e-12f));
            for (int d = 0; d < kDK; ++d) {
                q_h[row * kDK + d] = float_to_bf16(tq[d] * nq * scale);
                k_h[row * kDK + d] = float_to_bf16(tk[d] * nk);
            }
            for (int d = 0; d < kDV; ++d)
                v_h[row * kDV + d] = float_to_bf16(frand(-1.0f, 1.0f));
            beta_h[row] = float_to_bf16(frand(0.05f, 0.95f));
            g_h[row]    = float_to_bf16(-frand(0.0005f, 0.05f));
        }

        GpuBuffer<bf16> qbuf(q_h.size(), queue);
        GpuBuffer<bf16> kbuf(k_h.size(), queue);
        GpuBuffer<bf16> vbuf(v_h.size(), queue);
        GpuBuffer<bf16> bbuf(beta_h.size(), queue);
        GpuBuffer<bf16> gbuf(g_h.size(), queue);
        qbuf.upload(q_h.data(), q_h.size());
        kbuf.upload(k_h.data(), k_h.size());
        vbuf.upload(v_h.data(), v_h.size());
        bbuf.upload(beta_h.data(), beta_h.size());
        gbuf.upload(g_h.data(), g_h.size());

        const size_t state_elems = (size_t)kNV * kDK * kDV;
        std::vector<float> state0(state_elems, 0.0f); // prefill starts cold
        GpuBuffer<float> ssm(state_elems, queue);
        GpuBuffer<bf16> core((size_t)max_tokens * kNV * kDV, queue);

        std::printf("[bench] matrix: kernels={%s} p={%s} | n=%d warmup=%d runs=%d | cells=%zu\n",
                    kernels_csv.c_str(), p_csv.c_str(), iterations, warmup, runs,
                    kernels.size() * token_counts.size());
        std::printf("[bench] n_v=%d d_k=%d d_v=%d chunk=%d (weight-free op, synthetic activations)\n",
                    kNV, kDK, kDV, kChunk);
        if (md)
            std::printf("| kernel | p | runs | mean ms | sd ms | tok/s | core max abs | core cos | state max rel |\n"
                        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");

        for (int tokens : token_counts) {
            const size_t core_elems = (size_t)tokens * kNV * kDV;

            // Reference: host chunked path.
            std::vector<bf16> ref_core(core_elems);
            std::vector<float> ref_state(state_elems);
            {
                ssm.upload(state0.data(), state_elems);
                host_chunk_gated_delta_rule(context, qbuf.data(), kbuf.data(),
                                            vbuf.data(), bbuf.data(), gbuf.data(),
                                            ssm.data(), core.data(),
                                            tokens, kNV, kDK, kDV, kChunk);
                core.download(ref_core.data(), core_elems);
                ssm.download(ref_state.data(), state_elems);
            }

            for (const std::string& kernel : kernels) {
                auto run_once = [&] {
                    ssm.upload(state0.data(), state_elems);
                    if (kernel == "device" || kernel == "xmx" || kernel == "hybrid") {
                        if (kernel == "xmx")
                            qwen_gdn_device_chunk_xmx(queue, qbuf.data(), kbuf.data(),
                                                      vbuf.data(), bbuf.data(), gbuf.data(),
                                                      ssm.data(), core.data(),
                                                      tokens, kNV, kDK, kDV, kChunk);
                        else if (kernel == "hybrid")
                            qwen_gdn_device_chunk_hybrid(context, qbuf.data(), kbuf.data(),
                                                         vbuf.data(), bbuf.data(), gbuf.data(),
                                                         ssm.data(), core.data(),
                                                         tokens, kNV, kDK, kDV, kChunk);
                        else
                            qwen_gdn_device_chunk(queue, qbuf.data(), kbuf.data(),
                                                  vbuf.data(), bbuf.data(), gbuf.data(),
                                                  ssm.data(), core.data(),
                                                  tokens, kNV, kDK, kDV, kChunk);
                        queue.wait();
                    } else {
                        host_chunk_gated_delta_rule(context, qbuf.data(), kbuf.data(),
                                                    vbuf.data(), bbuf.data(), gbuf.data(),
                                                    ssm.data(), core.data(),
                                                    tokens, kNV, kDK, kDV, kChunk);
                    }
                };

                run_once();
                std::vector<bf16> actual(core_elems);
                std::vector<float> actual_state(state_elems);
                core.download(actual.data(), core_elems);
                ssm.download(actual_state.data(), state_elems);

                float max_abs = 0.0f;
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    float expected = bf16_to_float(ref_core[i]);
                    float observed = bf16_to_float(actual[i]);
                    max_abs = std::max(max_abs, std::fabs(observed - expected));
                    dot += (double)expected * observed;
                    na += (double)expected * expected;
                    nb += (double)observed * observed;
                }
                double cos = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 0.0;
                float state_rel = 0.0f;
                for (size_t i = 0; i < state_elems; ++i) {
                    float error = std::fabs(actual_state[i] - ref_state[i]);
                    state_rel = std::max(state_rel,
                                         error / std::max(1e-3f, std::fabs(ref_state[i])));
                }

                // QWEN35_GDN_DUMP=<dir>: dump per-kernel ref/actual state+core
                // (f32 binaries) for offline error localization.
                if (const char* dir = std::getenv("QWEN35_GDN_DUMP")) {
                    auto dump = [&](const char* tag, const float* p, size_t n) {
                        std::string f = std::string(dir) + "/" + tag + ".f32";
                        if (FILE* fp = std::fopen(f.c_str(), "wb")) {
                            std::fwrite(p, sizeof(float), n, fp);
                            std::fclose(fp);
                        }
                    };
                    std::vector<float> rc(core_elems), ac(core_elems);
                    for (size_t i = 0; i < core_elems; ++i) {
                        rc[i] = bf16_to_float(ref_core[i]);
                        ac[i] = bf16_to_float(actual[i]);
                    }
                    char tag[256];
                    std::snprintf(tag, sizeof tag, "state_p%d_%s_ref", tokens, kernel.c_str());
                    dump(tag, ref_state.data(), state_elems);
                    std::snprintf(tag, sizeof tag, "state_p%d_%s_act", tokens, kernel.c_str());
                    dump(tag, actual_state.data(), state_elems);
                    std::snprintf(tag, sizeof tag, "core_p%d_%s_ref", tokens, kernel.c_str());
                    dump(tag, rc.data(), core_elems);
                    std::snprintf(tag, sizeof tag, "core_p%d_%s_act", tokens, kernel.c_str());
                    dump(tag, ac.data(), core_elems);
                }

                for (int i = 0; i < warmup; ++i)
                    for (int step = 0; step < iterations; ++step) run_once();
                std::vector<double> samples;
                samples.reserve(runs);
                for (int i = 0; i < runs; ++i) {
                    auto start = std::chrono::steady_clock::now();
                    for (int step = 0; step < iterations; ++step) run_once();
                    samples.push_back(
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count() /
                        iterations);
                }
                Stat stat = aggregate(samples);
                double tok_s = tokens * 1000.0 / stat.mean;
                if (md) {
                    std::printf("| %s | %d | %d | %.3f | %.3f | %.1f | %.6f | %.6f | %.6f |\n",
                                kernel.c_str(), tokens, runs, stat.mean, stat.sd,
                                tok_s, max_abs, cos, state_rel);
                } else {
                    std::printf("qwen35_gdn kernel=%s p=%d runs=%d mean_ms=%.3f sd_ms=%.3f "
                                "tok_per_s=%.1f core_max_abs=%.6f core_cos=%.6f state_max_rel=%.6f\n",
                                kernel.c_str(), tokens, runs, stat.mean, stat.sd,
                                tok_s, max_abs, cos, state_rel);
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

REGISTER_BENCH("qwen35-gdn",
    "Qwen3.5-MoE gated deltanet core op (host scalar / device chunked SYCL)",
    run)

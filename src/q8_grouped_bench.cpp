// q8_grouped_bench
//
// Standalone micro-benchmark for the MoE expert GEMM grouped kernels, exercising
// the active-only compact-exact row layout (only ~active rows, not all
// num_experts*T experts) WITHOUT loading a model or running end-to-end inference.
//
// Tests two kernels on IDENTICAL synthetic data:
//   - matmul_q8_0_grouped_expert          (custom DPAS, the current default)
//   - matmul_q8_0_grouped_expert_onednn   (oneDNN experimental grouped GEMM,
//                                          gated by DIFF_Q8_GROUPED_EXPERT_BACKEND
//                                          + built w/ ONEDNN_EXPERIMENTAL_GROUPED_MEMORY)
//
// Use a tiny config first to check the oneDNN path completes without a GPU
// fault (UR_RESULT_ERROR_DEVICE_LOST), then scale to the real decode shape for
// perf numbers.  ONEDNN_VERBOSE=2 reveals which oneDNN impl (ref_grouped vs
// grouped_micro) is dispatched.
//
//   ZE_AFFINITY_MASK=0 ./build/q8_grouped_bench [-e E] [-a A] [-m M] [-K K] [-N N]
//                              [-w W] [-r R] [--backend b] [--check] [--md] [--tiny]
//
// Defaults: -e 128 -a 8 -m 256 -K 2816 -N 1408  (expert_mlp gateup: H->2*moe_inter)
//   -e  num_experts            (default 128)
//   -a  active_experts         (default 8, topk)
//   -m  rows_per_active_expert (default 256, rounded up to 8)
//   -K  in_features            (default 2816, must be %32==0)
//   -N  out_features           (default 1408, must be %16==0)
//   -w  warmup                 (default 2)
//   -r  timed runs             (default 3)
//   --backend onednn|custom|both   (default both)
//   --check  correctness: max abs diff of oneDNN vs custom output
//   --md     markdown table
//   --tiny   preset: -e 4 -a 2 -m 16 -K 32 -N 16  (fast hang diagnosis)

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/gpu/q8_0.hpp"

namespace {

struct Args {
    int num_experts = 128;
    int active = 8;
    int rows_per_expert = 256;
    int K = 2816;
    int N = 1408;
    int warmup = 2;
    int runs = 3;
    std::string backend = "both";
    bool check = false;
    bool md = false;
    bool tiny = false;
};

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -e <N>   num_experts              (default 128)\n"
        "  -a <N>   active_experts           (default 8)\n"
        "  -m <N>   rows_per_active_expert   (default 256)\n"
        "  -K <N>   in_features              (default 2816, %%32==0)\n"
        "  -N <N>   out_features             (default 1408, %%16==0)\n"
        "  -w <N>   warmup runs              (default 2)\n"
        "  -r <N>   timed runs               (default 3)\n"
        "  --backend onednn|custom|both      (default both)\n"
        "  --check  correctness vs custom DPAS\n"
        "  --md     markdown table\n"
        "  --tiny   tiny preset for hang diagnosis\n"
        "Backend also gated by DIFF_Q8_GROUPED_EXPERT_BACKEND=onednn.\n", p);
}

uint32_t lcg_state = 0x1234u;
inline uint32_t rng() { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }
inline float frand(float lo, float hi) { return lo + (hi - lo) * (rng() / 4294967296.0f); }

double now_s() {
    using Clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clk::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    return v[v.size() / 2];
}

int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

// Build synthetic Q8BatchedLinear [B, out=N, in=K] with per-32-group f32 scales
// [B, K/32, N].  Same layout as the real expert weights.
Q8BatchedLinear build_synthetic_batched(int B, int K, int N, sycl::queue& q) {
    int G = K / 32;
    std::vector<int8_t> wq((size_t)B * N * K);
    std::vector<float>  sc((size_t)B * G * N);
    for (size_t i = 0; i < wq.size(); ++i) wq[i] = (int8_t)(frand(-127.0f, 127.0f));
    for (size_t i = 0; i < sc.size();  ++i) sc[i]  = frand(0.001f, 0.01f);
    Q8BatchedLinear W;
    W.batch = B; W.in_features = K; W.out_features = N; W.group_size = 32;
    W.weight_qs    = GpuBuffer<int8_t>((size_t)B * N * K, q);
    W.weight_scale = GpuBuffer<float>((size_t)B * G * N, q);
    W.weight_qs.upload(wq.data(), wq.size());
    W.weight_scale.upload(sc.data(), sc.size());
    return W;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(1); } return argv[++i]; };
        if      (s == "-h" || s == "--help") { usage(argv[0]); return 0; }
        else if (s == "-e")      a.num_experts      = std::atoi(next().c_str());
        else if (s == "-a")      a.active           = std::atoi(next().c_str());
        else if (s == "-m")      a.rows_per_expert  = std::atoi(next().c_str());
        else if (s == "-K")      a.K                = std::atoi(next().c_str());
        else if (s == "-N")      a.N                = std::atoi(next().c_str());
        else if (s == "-w")      a.warmup           = std::atoi(next().c_str());
        else if (s == "-r")      a.runs             = std::atoi(next().c_str());
        else if (s == "--backend") a.backend        = next();
        else if (s == "--check") a.check = true;
        else if (s == "--md")    a.md = true;
        else if (s == "--tiny")  a.tiny = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (a.tiny) { a.num_experts = 4; a.active = 2; a.rows_per_expert = 16; a.K = 32; a.N = 16; }

    if (a.K % 32 != 0)  { std::fprintf(stderr, "K=%d must be %%32==0\n", a.K); return 1; }
    if (a.N % 16 != 0)  { std::fprintf(stderr, "N=%d must be %%16==0\n", a.N); return 1; }
    if (a.active > a.num_experts) a.active = a.num_experts;

    int m_round = round_up(a.rows_per_expert, 8);
    int total_rows = a.active * m_round;

    // Compact-exact offsets: expert e < active contributes m_round rows; e >=
    // active contributes 0 (empty).  offsets[i] = cumulative exclusive-end.
    std::vector<int32_t> offsets(a.num_experts, 0);
    {
        int32_t acc = 0;
        for (int e = 0; e < a.num_experts; ++e) {
            if (e < a.active) acc += m_round;
            offsets[e] = acc;
        }
    }

    // Custom DPAS block metadata: one 8-row block per slot, contiguous.
    int blocks = total_rows / 8;
    std::vector<int32_t> block_slot(blocks), block_expert(blocks);
    for (int b = 0; b < blocks; ++b) {
        block_slot[b] = b * 8;
        block_expert[b] = b / (m_round / 8);  // which active expert
    }

    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    std::string dev_name = q.get_device().get_info<sycl::info::device::name>();

    std::printf("[q8_grouped] device: %s\n", dev_name.c_str());
    std::printf("[q8_grouped] E=%d active=%d rows/expert=%d (rounded %d) total_rows=%d K=%d N=%d\n",
               a.num_experts, a.active, a.rows_per_expert, m_round, total_rows, a.K, a.N);
    std::printf("[q8_grouped] backend=%s  warmup=%d runs=%d  check=%d\n",
               a.backend.c_str(), a.warmup, a.runs, (int)a.check);
    std::printf("[q8_grouped] offsets:");
    for (int e = 0; e < a.num_experts && e < 12; ++e) std::printf(" %d", offsets[e]);
    if (a.num_experts > 12) std::printf(" ... %d", offsets.back());
    std::printf("  (last=%d == total_rows? %s)\n\n", offsets.back(),
               offsets.back() == total_rows ? "YES" : "NO");

    // Synthetic weights + activation.
    Q8BatchedLinear W = build_synthetic_batched(a.num_experts, a.K, a.N, q);
    GpuBuffer<bf16> A((size_t)total_rows * a.K, q);
    {
        std::vector<bf16> ha((size_t)total_rows * a.K);
        for (size_t i = 0; i < ha.size(); ++i) ha[i] = float_to_bf16(frand(-1.0f, 1.0f));
        A.upload(ha.data(), ha.size());
    }
    GpuBuffer<int32_t> offsets_dev((size_t)a.num_experts, q);
    offsets_dev.upload(offsets.data(), offsets.size());
    GpuBuffer<int32_t> bslot_dev((size_t)blocks, q);
    GpuBuffer<int32_t> bexp_dev((size_t)blocks, q);
    bslot_dev.upload(block_slot.data(), block_slot.size());
    bexp_dev.upload(block_expert.data(), block_expert.size());

    double flops = 2.0 * (double)total_rows * a.K * a.N;

    auto run_kernel = [&](const std::string& name, auto kernel_fn) {
        GpuBuffer<bf16> C((size_t)total_rows * a.N, q);
        C.zero();
        // warmup
        for (int w = 0; w < a.warmup; ++w) kernel_fn(C.data());
        q.wait();
        std::vector<double> ts;
        for (int r = 0; r < a.runs; ++r) {
            double t0 = now_s();
            kernel_fn(C.data());
            q.wait();
            double t1 = now_s();
            ts.push_back((t1 - t0) * 1e6);
        }
        double us = median(ts);
        double tflops = flops / (us * 1e-6) / 1e12;
        if (a.md)
            std::printf("| %s | %d | %d | %d | %.2f | %.1f |\n",
                       name.c_str(), total_rows, a.K, a.N, us, tflops);
        else
            std::printf("%-12s total_rows=%-5d K=%-5d N=%-5d  %10.2f us  %8.1f TFLOP/s\n",
                       name.c_str(), total_rows, a.K, a.N, us, tflops);
        return C;
    };

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
    bool want_onednn = (a.backend == "onednn" || a.backend == "both");
#else
    bool want_onednn = false;
#endif
    bool want_custom = (a.backend == "custom" || a.backend == "both");

    if (a.md) {
        std::printf("| kernel | total_rows | K | N | us | TFLOP/s |\n");
        std::printf("|---|---:|---:|---:|---:|---:|\n");
    }

    GpuBuffer<bf16> C_custom, C_onednn;

    if (want_custom) {
        auto Cc = run_kernel("custom/dpas", [&](bf16* c) {
            matmul_q8_0_grouped_expert(
                A.data(), a.K, W,
                bslot_dev.data(), bexp_dev.data(), blocks,
                c, ctx);
        });
        if (a.check || want_onednn) C_custom = std::move(Cc);
    }

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
    if (want_onednn) {
        auto Co = run_kernel("onednn", [&](bf16* c) {
            matmul_q8_0_grouped_expert_onednn(
                A.data(), a.K, W,
                a.num_experts, total_rows, offsets_dev.data(),
                c, ctx);
        });
        if (a.check || want_custom) C_onednn = std::move(Co);
    } else if (a.backend == "onednn") {
        std::printf("[q8_grouped] oneDNN grouped kernel not built (ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=OFF)\n");
    }
#else
    if (a.backend == "onednn" || a.backend == "both")
        std::printf("[q8_grouped] oneDNN grouped kernel not built (DNNL_EXPERIMENTAL_GROUPED_MEMORY undefined)\n");
#endif

    // Correctness: max abs diff between custom and oneDNN outputs.
    if (a.check && !C_custom.empty() && !C_onednn.empty()) {
        size_t n = (size_t)total_rows * a.N;
        std::vector<bf16> hc(n), ho(n);
        C_custom.download(hc.data(), n);
        C_onednn.download(ho.data(), n);
        double maxdiff = 0.0, sumdiff = 0.0; size_t ndiff = 0;
        for (size_t i = 0; i < n; ++i) {
            double d = std::fabs(bf16_to_float(hc[i]) - bf16_to_float(ho[i]));
            if (d > maxdiff) maxdiff = d;
            sumdiff += d; ndiff += (d > 1e-3) ? 1 : 0;
        }
        std::printf("\n[check] max abs diff = %.6g  mean = %.6g  entries>1e-3 = %zu/%zu (%.1f%%)\n",
                   maxdiff, sumdiff / n, ndiff, n, 100.0 * ndiff / n);
    }

    return 0;
}

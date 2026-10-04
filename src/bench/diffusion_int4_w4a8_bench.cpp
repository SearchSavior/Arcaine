// Focused DiffusionGemma INT4-AWQ W4A8 prefill benchmark.
//
// Benchmarks the exact kernel sequence a quantized Linear takes in the
// denoiser: per-token bf16->s8 activation quantization followed by a oneDNN
// s8 (src) x s4 (weights) matmul, versus the production W4A16 path (bf16 src,
// s4 weights, oneDNN weight decompression).  No model load, no end-to-end
// inference.
//
// Correctness is checked against a host fp32 reference on a small shape; the
// production shapes are timed only.
//
// Registered as `diffusion-int4-w4a8` in arcaine_kbench.
//
// Run (env-gated; see code):
//   ./build/arcaine_kbench diffusion-int4-w4a8 \
//       --m 64,128,256,512,1024,2048 --k 2816 --n 8192 --iterations 30
//
// Env:
//   DIFF_INT4_W4A8_PREFILL=1   enables the introduced path (required to run)
//   DIFF_INT4_W4A8_MIN_M       dispatch threshold for matmul_int4 (default 128)

#include "common/bench/registry.hpp"
#include "common/bench/util.hpp"

#include "common/gpu/buffer.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/int4.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using arcaine::bench::elapsed_ms;
using arcaine::bench::parse_int_csv;

namespace {

// Deterministic s4 value in [-8, 7] for logical weight element (n, k).
int s4_value(int n, int k) {
    int v = ((n * 131 + k * 17) % 16) - 8;  // [-8, 7]
    return v;
}

// Pack logical (N, K) s4 weights into oneDNN tag::ba byte stream:
// [N, K/2] bytes, low nibble first, two's-complement nibbles.
std::vector<uint8_t> pack_s4(int N, int K) {
    std::vector<uint8_t> packed((size_t)N * K / 2, 0);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int v = s4_value(n, k) & 0xF;
            size_t byte = (size_t)n * (K / 2) + k / 2;
            if ((k & 1) == 0) packed[byte] |= (uint8_t)v;
            else              packed[byte] |= (uint8_t)(v << 4);
        }
    }
    return packed;
}

bf16 pattern_bf16(size_t i) {
    int v = (int)((i * 17 + 13) % 101) - 50;
    return float_to_bf16(0.02f * (float)v);
}

struct Weights {
    Int4Linear w;
    std::vector<uint8_t> packed_h;
    std::vector<bf16> scale_h;
    std::vector<float> dequant;  // host float (N, K), for reference
    int N = 0, K = 0, group = 32;
};

Weights make_weights(int N, int K, int group, sycl::queue& q) {
    Weights out;
    out.N = N; out.K = K; out.group = group;
    out.packed_h = pack_s4(N, K);
    int G = K / group;
    out.scale_h.resize((size_t)G * N);
    out.dequant.resize((size_t)N * K);
    for (int g = 0; g < G; ++g) {
        for (int n = 0; n < N; ++n) {
            // Small per-group scale; keep it bf16-exact for the reference.
            float s = 0.004f + 0.001f * (float)((g + n) % 7);
            bf16 sb = float_to_bf16(s);
            out.scale_h[(size_t)g * N + n] = sb;
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            bf16 sb = out.scale_h[(size_t)(k / group) * N + n];
            out.dequant[(size_t)n * K + k] =
                (float)s4_value(n, k) * bf16_to_float(sb);
        }
    }
    out.w.in_features = K;
    out.w.out_features = N;
    out.w.group_size = group;
    out.w.weight_packed = GpuBuffer<uint8_t>(out.packed_h.size(), q);
    out.w.weight_scale = GpuBuffer<bf16>(out.scale_h.size(), q);
    out.w.weight_packed.upload(out.packed_h.data(), out.packed_h.size());
    out.w.weight_scale.upload(out.scale_h.data(), out.scale_h.size());
    return out;
}

std::vector<bf16> make_activations(int M, int K, sycl::queue& q) {
    std::vector<bf16> h((size_t)M * K);
    for (size_t i = 0; i < h.size(); ++i) h[i] = pattern_bf16(i);
    return h;
}

// C_ref (M, N) fp32 = A (M, K) bf16 @ dequant(W) (N, K)^T.
std::vector<float> host_reference(const std::vector<bf16>& A,
                                  const Weights& W, int M) {
    std::vector<float> C((size_t)M * W.N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < W.N; ++n) {
            float acc = 0.0f;
            const float* wrow = W.dequant.data() + (size_t)n * W.K;
            const bf16* arow = A.data() + (size_t)m * W.K;
            for (int k = 0; k < W.K; ++k)
                acc += bf16_to_float(arow[k]) * wrow[k];
            C[(size_t)m * W.N + n] = acc;
        }
    }
    return C;
}

struct ErrorStats {
    float max_abs = 0.0f;
    double rel_l2 = 0.0;
};

ErrorStats error_vs_ref(const std::vector<bf16>& got,
                        const std::vector<float>& ref) {
    ErrorStats s;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = bf16_to_float(got[i]);
        float d = g - ref[i];
        s.max_abs = std::max(s.max_abs, std::fabs(d));
        num += (double)d * d;
        den += (double)ref[i] * ref[i];
    }
    s.rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    return s;
}

// Small-shape correctness: W4A8 and W4A16 both against host fp32 reference.
void correctness(GpuEngine& ctx) {
    const int M = 64, K = 512, N = 1024, group = 32;
    auto& q = ctx.queue;
    Weights W = make_weights(N, K, group, q);
    std::vector<bf16> A_h = make_activations(M, K, q);

    GpuBuffer<bf16> A((size_t)M * K, q), C8((size_t)M * N, q),
        C16((size_t)M * N, q);
    A.upload(A_h.data(), A_h.size());

    matmul_int4_w4a8(A.data(), M, K, W.w, C8.data(), ctx);
    matmul_int4_w4a16(A.data(), M, K, W.w, C16.data(), ctx);
    q.wait();

    std::vector<bf16> C8_h((size_t)M * N), C16_h((size_t)M * N);
    C8.download(C8_h.data(), C8_h.size());
    C16.download(C16_h.data(), C16_h.size());

    std::vector<float> ref = host_reference(A_h, W, M);
    ErrorStats e8 = error_vs_ref(C8_h, ref);
    ErrorStats e16 = error_vs_ref(C16_h, ref);
    std::printf("[check] M=%d K=%d N=%d group=%d\n", M, K, N, group);
    std::printf("        W4A8 vs fp32 ref: max_abs %.5g  rel_l2 %.5g\n",
                e8.max_abs, e8.rel_l2);
    std::printf("        W4A16 vs fp32 ref: max_abs %.5g  rel_l2 %.5g\n",
                e16.max_abs, e16.rel_l2);
    if (e8.max_abs > 1.0f || e8.rel_l2 > 0.05)
        throw std::runtime_error("W4A8 output diverges from fp32 reference");
}

void perf_shape(GpuEngine& ctx, int M, int K, int N, int group, int iterations) {
    auto& q = ctx.queue;
    Weights W = make_weights(N, K, group, q);
    std::vector<bf16> A_h = make_activations(M, K, q);

    GpuBuffer<bf16> A((size_t)M * K, q), C8((size_t)M * N, q),
        C16((size_t)M * N, q);
    A.upload(A_h.data(), A_h.size());

    auto run16 = [&] { matmul_int4_w4a16(A.data(), M, K, W.w, C16.data(), ctx); };
    auto run8  = [&] { matmul_int4_w4a8(A.data(), M, K, W.w, C8.data(), ctx); };

    // Isolate the introduced per-token activation-quantization pass.
    GpuBuffer<int8_t> Aq((size_t)M * K, q);
    GpuBuffer<float> Ascale((size_t)M, q);
    auto runq = [&] {
        quantize_bf16_to_s8_per_token(q, A.data(), M, K, Aq.data(), Ascale.data());
    };

    for (int i = 0; i < 3; ++i) { run16(); run8(); runq(); }
    q.wait();
    double ms16 = elapsed_ms(q, iterations, run16);
    double ms8  = elapsed_ms(q, iterations, run8);
    double msq  = elapsed_ms(q, iterations, runq);
    double ms8_gemm = ms8 - msq;

    q.wait();
    std::vector<bf16> C8_h((size_t)M * N), C16_h((size_t)M * N);
    C8.download(C8_h.data(), C8_h.size());
    C16.download(C16_h.data(), C16_h.size());
    float max_diff = 0.0f;
    for (size_t i = 0; i < C8_h.size(); ++i)
        max_diff = std::max(max_diff,
            std::fabs(bf16_to_float(C8_h[i]) - bf16_to_float(C16_h[i])));

    double tflops16 = 2.0 * M * N * K / (ms16 * 1e9);
    double tflops8  = 2.0 * M * N * K / (ms8 * 1e9);
    double tflops8g = ms8_gemm > 0 ? 2.0 * M * N * K / (ms8_gemm * 1e9) : 0.0;
    std::printf("[perf]  M=%-5d K=%-6d N=%-6d group=%d | "
                "W4A16 %8.3f ms (%6.1f TF) | W4A8 %8.3f ms (%6.1f TF) | "
                "speedup %.2fx | quant %.3f ms (%.0f%% of w4a8) | "
                "w4a8-gemm %6.1f TF | max_abs %.4g\n",
                M, K, N, group, ms16, tflops16, ms8, tflops8, ms16 / ms8,
                msq, 100.0 * msq / ms8, tflops8g, max_diff);
}

int run(int argc, char** argv) {
    std::vector<int> ms = {64, 128, 256, 512, 1024, 2048};
    int K = 2816, N = 8192, group = 32, iterations = 30;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(arg + " needs a value");
            return argv[++i];
        };
        if (arg == "--m") ms = parse_int_csv(next());
        else if (arg == "--k") K = std::atoi(next().c_str());
        else if (arg == "--n") N = std::atoi(next().c_str());
        else if (arg == "--group") group = std::atoi(next().c_str());
        else if (arg == "--iterations") iterations = std::atoi(next().c_str());
        else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--m CSV] [--k K] [--n N] [--group G] "
                        "[--iterations N]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (iterations <= 0) iterations = 1;
    if (!diff_int4_w4a8_prefill_enabled()) {
        std::fprintf(stderr,
            "Set DIFF_INT4_W4A8_PREFILL=1 to benchmark the introduced path.\n");
        return 2;
    }

    try {
        GpuEngine& ctx = GpuEngine::get(0);
        std::printf("[device] %s | iterations=%d | min_m=%d\n",
                    ctx.queue.get_device()
                        .get_info<sycl::info::device::name>().c_str(),
                    iterations, diff_int4_w4a8_min_m());
        correctness(ctx);
        for (int M : ms) perf_shape(ctx, M, K, N, group, iterations);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

}  // namespace

REGISTER_BENCH("diffusion-int4-w4a8",
    "DiffusionGemma INT4-AWQ W4A8 prefill matmul (s8 acts x s4 weights) vs W4A16",
    run)

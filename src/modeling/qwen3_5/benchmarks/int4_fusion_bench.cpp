// Focused Qwen3.5 dense INT4-AWQ fusion benchmark.
//
// Kernel-path A/B benches (not end-to-end inference) for the fusions ported
// from the diffusion_gemma INT4 work:
//   1) zp-correction fusion: default matmul_int4 (rowsum + corr GEMM +
//      subtract, per-call heap allocs) vs the small-M fused corr+subtract
//      with a workspace rowsum scratch (ARCAINE_QWEN35_INT4_ZP_FUSED).
//   2) add + RMS norm fusion (add_rms_norm): must be bit-exact vs
//      add_inplace + rms_norm.
//   3) q/k norm + MRoPE fusion (qwen35_norm_rope_fused): must be bit-exact vs
//      rms_norm x2 + qwen35_apply_mrope.
//
// Registered as `qwen35-int4-fusion` in arcaine_kbench.
//
// Run:
//   ./build/arcaine_kbench qwen35-int4-fusion [--iterations N]

#include "benchmarks/registry.hpp"
#include "benchmarks/util.hpp"

#include "modeling/qwen3_5/kernels.hpp"
#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "runtime/gpu/ops.hpp"
#include "runtime/kernels/elementwise.hpp"
#include "runtime/kernels/rms_norm.hpp"
#include "runtime/quantization/int4.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using arcaine::bench::elapsed_ms;

namespace {

// Real cyankiwi_Qwen3.6-27B-AWQ-INT4 shapes.
constexpr int kHidden = 5120;
constexpr int kIntermediate = 17408;
constexpr int kGroup = 32;
constexpr float kEps = 1e-6f;

bf16 pattern(size_t i, float scale, float bias = 0.0f) {
    int v = (int)((i * 17 + 13) % 101) - 50;
    return float_to_bf16(bias + scale * (float)v);
}

// Build a synthetic asymmetric-zp Int4Linear (N, K), group=32, on device.
Int4Linear make_int4_linear(int K, int N, sycl::queue& q) {
    Int4Linear W;
    W.in_features = K;
    W.out_features = N;
    W.group_size = kGroup;
    int G = K / kGroup;
    std::vector<uint8_t> packed((size_t)N * K / 2);
    for (size_t i = 0; i < packed.size(); ++i)
        packed[i] = (uint8_t)((i * 7 + 3) % 256);  // arbitrary nibble pairs
    std::vector<bf16> scale((size_t)G * N), zp((size_t)G * N);
    for (size_t i = 0; i < scale.size(); ++i) {
        scale[i] = pattern(i, 0.0005f, 0.01f);
        zp[i] = pattern(i * 3 + 1, 0.002f, 0.005f);  // nonzero -> correction runs
    }
    W.weight_packed = GpuBuffer<uint8_t>(packed.size(), q);
    W.weight_scale = GpuBuffer<bf16>(scale.size(), q);
    W.zp_offset = GpuBuffer<bf16>(zp.size(), q);
    W.weight_packed.upload(packed.data(), packed.size());
    W.weight_scale.upload(scale.data(), scale.size());
    W.zp_offset.upload(zp.data(), zp.size());
    return W;
}

struct ErrorStats {
    float max_abs = 0.0f;
    size_t bit_mismatches = 0;
};

ErrorStats compare(const std::vector<bf16>& a, const std::vector<bf16>& b) {
    if (a.size() != b.size()) throw std::runtime_error("compare size mismatch");
    ErrorStats s;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++s.bit_mismatches;
        s.max_abs = std::max(s.max_abs,
            std::abs(bf16_to_float(a[i]) - bf16_to_float(b[i])));
    }
    return s;
}

void bench_zp_correction(GpuEngine& ctx, int iterations) {
    auto& q = ctx.queue;
    const int M = 1;
    const int K = kHidden;
    const int G = K / kGroup;

    std::vector<bf16> a_h((size_t)M * K);
    for (size_t i = 0; i < a_h.size(); ++i) a_h[i] = pattern(i, 0.002f);
    GpuBuffer<bf16> a(a_h.size(), q);
    a.upload(a_h.data(), a_h.size());

    // Piece-level timing: rowsum alone.
    GpuBuffer<bf16> rowsum_scratch((size_t)M * G, q);
    double t_rs = elapsed_ms(q, iterations, [&] {
        int4_rowsum(a.data(), M, K, kGroup, rowsum_scratch.data(), q);
    });
    std::printf("[zp-pieces] M=%d K=%d | rowsum %.3f ms\n", M, K, t_rs);
    std::fflush(stdout);

    for (int N : {kHidden, 2 * kIntermediate}) {
        Int4Linear W = make_int4_linear(K, N, q);
        GpuBuffer<bf16> c_heap((size_t)M * N, q), c_scratch((size_t)M * N, q);
        GpuBuffer<bf16> rs_scratch((size_t)M * G, q);
        GpuBuffer<bf16> cr_scratch((size_t)M * N, q);

        // Heap (per-call allocation) vs workspace scratch, same GEMM path.
        int4_zp_scratch_set(false);
        matmul_int4(a.data(), M, K, W, c_heap.data(), ctx);
        int4_zp_scratch_set(true);
        matmul_int4(a.data(), M, K, W, c_scratch.data(), ctx,
                    rs_scratch.data(), cr_scratch.data());
        q.wait();
        std::vector<bf16> heap((size_t)M * N), scratch((size_t)M * N);
        c_heap.download(heap.data(), heap.size());
        c_scratch.download(scratch.data(), scratch.size());
        ErrorStats err = compare(heap, scratch);

        int4_zp_scratch_set(false);
        matmul_int4(a.data(), M, K, W, c_heap.data(), ctx);
        int4_zp_scratch_set(true);
        matmul_int4(a.data(), M, K, W, c_scratch.data(), ctx,
                    rs_scratch.data(), cr_scratch.data());
        q.wait();
        double heap_ms = elapsed_ms(q, iterations, [&] {
            int4_zp_scratch_set(false);
            matmul_int4(a.data(), M, K, W, c_heap.data(), ctx);
        });
        double scratch_ms = elapsed_ms(q, iterations, [&] {
            int4_zp_scratch_set(true);
            matmul_int4(a.data(), M, K, W, c_scratch.data(), ctx,
                        rs_scratch.data(), cr_scratch.data());
        });
        std::printf("[zp-corr] M=%d K=%d N=%d | heap %.3f ms | scratch %.3f ms "
                    "| speedup %.3fx | max_abs %.7g\n",
                    M, K, N, heap_ms, scratch_ms, heap_ms / scratch_ms,
                    err.max_abs);
        std::fflush(stdout);
        // Same GEMM path on both sides: must be bit-identical.
        if (err.bit_mismatches > 0)
            throw std::runtime_error("zp scratch path differs bitwise");
    }
    int4_zp_scratch_set(true);
}

void bench_add_norm(GpuEngine& ctx, int iterations) {
    auto& q = ctx.queue;
    const int seqs[] = {1, 256};
    const int H = kHidden;
    std::vector<bf16> x_h((size_t)256 * H), w_h(H);
    for (size_t i = 0; i < x_h.size(); ++i) x_h[i] = pattern(i, 0.01f);
    for (int i = 0; i < H; ++i) w_h[i] = pattern(i, 0.01f, 1.0f);
    GpuBuffer<bf16> x(x_h.size(), q), w(w_h.size(), q);
    x.upload(x_h.data(), x_h.size());
    w.upload(w_h.data(), w_h.size());

    for (int seq : seqs) {
        size_t n = (size_t)seq * H;
        GpuBuffer<bf16> res(n, q), out_ref(n, q), out_fused(n, q);
        std::vector<bf16> res_h(n);
        for (size_t i = 0; i < n; ++i) res_h[i] = pattern(i + 5, 0.01f);
        res.upload(res_h.data(), res_h.size());

        auto baseline = [&] {
            add_inplace(q, res.data(), x.data(), (int)n);
            rms_norm(q, res.data(), w.data(), out_ref.data(), seq, H, kEps);
        };
        auto fused = [&] {
            add_rms_norm(q, x.data(), res.data(), w.data(), out_fused.data(),
                         seq, H, kEps);
        };
        baseline();
        q.wait();
        // Re-upload res so both paths see identical input state.
        res.upload(res_h.data(), res_h.size());
        fused();
        q.wait();

        std::vector<bf16> ref(n), fused_h(n);
        out_ref.download(ref.data(), ref.size());
        out_fused.download(fused_h.data(), fused_h.size());
        ErrorStats err = compare(ref, fused_h);

        res.upload(res_h.data(), res_h.size());
        for (int i = 0; i < 3; ++i) baseline();
        res.upload(res_h.data(), res_h.size());
        for (int i = 0; i < 3; ++i) fused();
        q.wait();
        res.upload(res_h.data(), res_h.size());
        double base_ms = elapsed_ms(q, iterations, [&] { baseline(); });
        res.upload(res_h.data(), res_h.size());
        double fused_ms = elapsed_ms(q, iterations, [&] { fused(); });
        std::printf("[add-norm] seq=%d H=%d | baseline %.3f ms | fused %.3f ms "
                    "| speedup %.3fx | max_abs %.7g | bit_mismatch %zu/%zu\n",
                    seq, H, base_ms, fused_ms, base_ms / fused_ms,
                    err.max_abs, err.bit_mismatches, n);
        if (err.bit_mismatches > 0)
            throw std::runtime_error("add+norm fusion is not bit-exact");
    }
}

void bench_norm_rope(GpuEngine& ctx, int iterations) {
    auto& q = ctx.queue;
    const int seq = 256;
    const int q_heads = 24, k_heads = 4, head_dim = 256;
    const int rotary_dim = 64;
    const float theta = 1e7f;
    const std::vector<int> sections = {11, 11, 10};
    const size_t q_count = (size_t)seq * q_heads * head_dim;
    const size_t k_count = (size_t)seq * k_heads * head_dim;

    std::vector<bf16> q_h(q_count), k_h(k_count), qw_h(head_dim), kw_h(head_dim);
    for (size_t i = 0; i < q_h.size(); ++i) q_h[i] = pattern(i, 0.01f);
    for (size_t i = 0; i < k_h.size(); ++i) k_h[i] = pattern(i + 11, 0.01f);
    for (int i = 0; i < head_dim; ++i) {
        qw_h[i] = pattern(i, 0.01f, 1.0f);
        kw_h[i] = pattern(i + 3, 0.01f, 1.0f);
    }
    std::vector<int32_t> pos_h((size_t)3 * seq);
    for (int t = 0; t < seq; ++t) {
        pos_h[t] = 2 * t + 1;
        pos_h[seq + t] = 2 * t + 2;
        pos_h[2 * seq + t] = 2 * t;
    }
    GpuBuffer<bf16> q_buf(q_count, q), k_buf(k_count, q);
    GpuBuffer<bf16> qw(qw_h.size(), q), kw(kw_h.size(), q);
    GpuBuffer<int32_t> pos(pos_h.size(), q);
    q_buf.upload(q_h.data(), q_h.size());
    k_buf.upload(k_h.data(), k_h.size());
    qw.upload(qw_h.data(), qw_h.size());
    kw.upload(kw_h.data(), kw_h.size());
    pos.upload(pos_h.data(), pos_h.size());

    GpuBuffer<bf16> q_ref(q_count, q), k_ref(k_count, q);
    GpuBuffer<bf16> q_fused(q_count, q), k_fused(k_count, q);
    q_ref.upload(q_h.data(), q_h.size());
    k_ref.upload(k_h.data(), k_h.size());
    q_fused.upload(q_h.data(), q_h.size());
    k_fused.upload(k_h.data(), k_h.size());

    auto baseline = [&] {
        rms_norm(q, q_ref.data(), qw.data(), q_ref.data(), seq * q_heads,
                 head_dim, kEps);
        rms_norm(q, k_ref.data(), kw.data(), k_ref.data(), seq * k_heads,
                 head_dim, kEps);
        qwen35_apply_mrope(q, q_ref.data(), k_ref.data(), pos.data(), seq,
                           q_heads, k_heads, head_dim, rotary_dim, theta,
                           sections);
    };
    auto fused = [&] {
        qwen35_norm_rope_fused(q, q_fused.data(), k_fused.data(), qw.data(),
                               kw.data(), pos.data(), seq, q_heads, k_heads,
                               head_dim, rotary_dim, theta, sections, kEps);
    };
    baseline();
    fused();
    q.wait();

    std::vector<bf16> qr(q_count), kr(k_count), qf(q_count), kf(k_count);
    q_ref.download(qr.data(), qr.size());
    k_ref.download(kr.data(), kr.size());
    q_fused.download(qf.data(), qf.size());
    k_fused.download(kf.data(), kf.size());
    ErrorStats err_q = compare(qr, qf);
    ErrorStats err_k = compare(kr, kf);

    for (int i = 0; i < 3; ++i) baseline();
    for (int i = 0; i < 3; ++i) fused();
    q.wait();
    double base_ms = elapsed_ms(q, iterations, [&] { baseline(); });
    double fused_ms = elapsed_ms(q, iterations, [&] { fused(); });
    std::printf("[norm-rope] seq=%d qh=%d kh=%d hd=%d | baseline %.3f ms | "
                "fused %.3f ms | speedup %.3fx | max_abs q=%.7g k=%.7g | "
                "bit_mismatch q=%zu/%zu k=%zu/%zu\n",
                seq, q_heads, k_heads, head_dim, base_ms, fused_ms,
                base_ms / fused_ms, err_q.max_abs, err_k.max_abs,
                err_q.bit_mismatches, q_count, err_k.bit_mismatches, k_count);
    if (err_q.bit_mismatches > 0 || err_k.bit_mismatches > 0)
        throw std::runtime_error("norm+rope fusion is not bit-exact");
}

void bench_vocab_tail(GpuEngine& ctx, int iterations) {
    auto& q = ctx.queue;
    const int H = 5120;
    const int V = 248320;
    const size_t w_count = (size_t)V * H;  // 2.54 GB bf16

    std::vector<bf16> hidden_h(H);
    for (int i = 0; i < H; ++i) hidden_h[i] = float_to_bf16(std::sin(i * 0.1f));
    GpuBuffer<bf16> hidden(H, q);
    hidden.upload(hidden_h.data(), H);

    GpuBuffer<bf16> w(w_count, q);
    w.zero();
    GpuBuffer<bf16> logits_bf16(V, q);
    GpuBuffer<float> logits_f32(V, q);

    // 1) lm_head GEMM (M=1, K=5120 -> N=248320), the unavoidable full-vocab
    //    computation: reads the 2.54 GB weight row-major once per token.
    double t_gemm = elapsed_ms(q, iterations, [&] {
        matmul_bf16(hidden.data(), 1, H, w.data(), V, logits_bf16.data(), ctx);
    });
    // 2) bf16 -> f32 conversion of the full vocab.
    double t_cvt = elapsed_ms(q, iterations, [&] {
        bf16_to_f32(q, logits_bf16.data(), logits_f32.data(), V);
    });
    // 3) full-vocab D2H with blocking wait (the per-token .wait()).
    std::vector<float> host_logits(V);
    double t_d2h = elapsed_ms(q, iterations, [&] {
        q.memcpy(host_logits.data(), logits_f32.data(), V * sizeof(float)).wait();
    });
    // 4) host top-k via partial_sort over the full vocab (CPU wall time).
    std::vector<int> order(V);
    std::iota(order.begin(), order.end(), 0);
    double t_sort = 0.0;
    {
        auto s0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            std::partial_sort(order.begin(), order.begin() + 64, order.end(),
                [&](int a, int b) { return host_logits[a] > host_logits[b]; });
        auto s1 = std::chrono::steady_clock::now();
        t_sort = std::chrono::duration<double, std::milli>(s1 - s0).count() / iterations;
    }

    std::printf("[vocab-tail] M=1 H=%d V=%d | gemm %.3f ms | bf16->f32 %.3f ms "
                "| d2h+wait %.3f ms | host partial_sort %.3f ms | total %.3f ms\n",
                H, V, t_gemm, t_cvt, t_d2h, t_sort,
                t_gemm + t_cvt + t_d2h + t_sort);
    std::fflush(stdout);
}

int run(int argc, char** argv) {
    int iterations = 20;
    std::string device;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::fprintf(stderr,
                "Usage: %s [--iterations N] [--device N]\n", argv[0]);
            return 0;
        } else if (arg == "--iterations") iterations = std::stoi(next());
        else if (arg == "--device") device = next();
        else throw std::runtime_error("unknown arg: " + arg);
    }
    if (iterations <= 0) throw std::runtime_error("iterations must be > 0");
    if (!device.empty()) gpu_device_control::apply_device_index(device);
    GpuEngine& ctx = GpuEngine::get(0);

    std::printf("[bench] Qwen3.5 dense INT4 fusion A/B (arcaine_kbench)\n");
    bench_vocab_tail(ctx, iterations);
    bench_zp_correction(ctx, iterations);
    bench_add_norm(ctx, iterations);
    bench_norm_rope(ctx, iterations);
    return 0;
}

}  // namespace

REGISTER_BENCH("qwen35-int4-fusion",
    "Qwen3.5 dense INT4-AWQ fusion A/B: zp correction, add+norm, norm+rope",
    run)

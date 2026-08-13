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

// Dequantized-BF16 weight A/B (kbench-only evidence; NOT wired into the dense
// e2e path). Folding the asymmetric-zp correction into a pre-dequantized BF16
// weight runs the GEMM at oneDNN's native BF16 rate (~155 vs ~100 TFLOPS on
// BMG at M=1024) and removes the rowsum + corr GEMM + subtract launches. The
// dequantized copy is N*K*2 bytes = ~48GB for the 27B dense model, which does
// NOT fit the 34GB B70 alongside the packed s4 weights (verified 2026-08-12:
// USM shared allocations spill to system memory -> 10-1000x kernel slowdowns
// e2e). Kept here as an A/B gate for bigger-memory targets / smaller models.
void bench_dequant_bf16(GpuEngine& ctx, int iterations) {
    auto& q = ctx.queue;
    const int M = 512;
    const int K = kHidden;
    const int G = K / kGroup;

    std::vector<bf16> a_h((size_t)M * K);
    for (size_t i = 0; i < a_h.size(); ++i) a_h[i] = pattern(i, 0.002f);
    GpuBuffer<bf16> a(a_h.size(), q);
    a.upload(a_h.data(), a_h.size());

    for (int N : {2 * kIntermediate, kIntermediate}) {
        Int4Linear W = make_int4_linear(K, N, q);
        GpuBuffer<bf16> c_int4((size_t)M * N, q), c_deq((size_t)M * N, q);
        GpuBuffer<bf16> rs_scratch((size_t)M * G, q);
        GpuBuffer<bf16> cr_scratch((size_t)M * N, q);

        // Reference: production int4 + asymmetric-zp correction.
        matmul_int4(a.data(), M, K, W, c_int4.data(), ctx, rs_scratch.data(),
                    cr_scratch.data());
        // Test: dequantized-bf16 weight, zp folded in.
        const bf16* w_deq = W.ensure_dequantized_bf16(ctx);
        matmul_bf16(a.data(), M, K, w_deq, N, c_deq.data(), ctx);
        q.wait();

        std::vector<bf16> ref((size_t)M * N), test((size_t)M * N);
        c_int4.download(ref.data(), ref.size());
        c_deq.download(test.data(), test.size());
        ErrorStats err = compare(ref, test);
        double rms_ref = 0.0, rms_err = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            double r = bf16_to_float(ref[i]), d = bf16_to_float(test[i]) - r;
            rms_ref += r * r;
            rms_err += d * d;
        }
        rms_ref = std::sqrt(rms_ref / ref.size());
        rms_err = std::sqrt(rms_err / ref.size());

        double t_int4 = elapsed_ms(q, iterations, [&] {
            matmul_int4(a.data(), M, K, W, c_int4.data(), ctx,
                        rs_scratch.data(), cr_scratch.data());
        });
        double t_deq = elapsed_ms(q, iterations, [&] {
            matmul_bf16(a.data(), M, K, w_deq, N, c_deq.data(), ctx);
        });
        std::printf("[dequant-bf16] M=%d K=%d N=%d | int4 %.3f ms | "
                    "dequant-bf16 %.3f ms | speedup %.3fx | max_abs %.7g | "
                    "rms_abs %.7g | rms_rel %.7g\n",
                    M, K, N, t_int4, t_deq, t_int4 / t_deq, err.max_abs,
                    rms_err, rms_err / (rms_ref + 1e-12));
        std::fflush(stdout);
    }
}

// Bit-exact gates for the fused-corr consumers (O2) and the GDN micro-fusions
// (O5): each fused path must match the plain (matmul_int4 + plain consumer /
// plain kernel pair) bit-for-bit on synthetic data.
void bench_corr_fusion(GpuEngine& ctx, int iterations) {
    using namespace qwen35_kernels;
    auto& q = ctx.queue;
    const int M = 64;
    const int K = kHidden;
    const int G = K / kGroup;
    const int inter = kIntermediate;
    const int qh = 24, kh = 4, hd = 256;
    const int query_dim = qh * hd;
    const int kv_dim = kh * hd;
    const int row_stride = 2 * query_dim + 2 * kv_dim;

    std::vector<bf16> a_h((size_t)M * K);
    for (size_t i = 0; i < a_h.size(); ++i) a_h[i] = pattern(i, 0.002f);
    GpuBuffer<bf16> a(a_h.size(), q);
    a.upload(a_h.data(), a_h.size());

    Int4Linear Wqkv = make_int4_linear(K, row_stride, q);
    Int4Linear Wgu = make_int4_linear(K, 2 * inter, q);
    Int4Linear Wo = make_int4_linear(query_dim, kHidden, q);
    GpuBuffer<bf16> rs((size_t)M * G, q), corr((size_t)M * (2 * inter), q);

    // --- split_q_gate_kv: plain (matmul_int4 + split) vs fused (zp_corr + split w/ corr)
    {
        GpuBuffer<bf16> c_plain((size_t)M * row_stride, q), c_fused((size_t)M * row_stride, q);
        GpuBuffer<bf16> q1((size_t)M * query_dim, q), g1((size_t)M * query_dim, q);
        GpuBuffer<bf16> k1((size_t)M * kv_dim, q), v1((size_t)M * kv_dim, q);
        GpuBuffer<bf16> q2((size_t)M * query_dim, q), g2((size_t)M * query_dim, q);
        GpuBuffer<bf16> k2((size_t)M * kv_dim, q), v2((size_t)M * kv_dim, q);
        GpuBuffer<bf16> cr_scratch((size_t)M * row_stride, q);
        GpuBuffer<bf16> c_plain2((size_t)M * row_stride, q), c_fused2((size_t)M * row_stride, q);
        (void)c_fused2; (void)c_plain2;
        matmul_int4(a.data(), M, K, Wqkv, c_plain.data(), ctx, rs.data(), cr_scratch.data());
        qwen35_split_q_gate_kv(q, c_plain.data(), q1.data(), g1.data(), k1.data(),
                               v1.data(), M, qh, kh, hd);
        matmul_int4_zp_corr(a.data(), M, K, Wqkv, c_fused.data(), ctx, rs.data(), corr.data());
        qwen35_split_q_gate_kv(q, c_fused.data(), q2.data(), g2.data(), k2.data(),
                               v2.data(), M, qh, kh, hd, nullptr, nullptr, 0,
                               corr.data());
        q.wait();
        std::vector<bf16> ref((size_t)M * query_dim), test((size_t)M * query_dim);
        q1.download(ref.data(), ref.size());
        q2.download(test.data(), test.size());
        ErrorStats e = compare(ref, test);
        std::printf("[corr-fusion] split_q_gate_kv q | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0) throw std::runtime_error("split corr fusion not bit-exact");
        k1.download(ref.data(), ref.size());
        k2.download(test.data(), test.size());
        e = compare(ref, test);
        std::printf("[corr-fusion] split_q_gate_kv k | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0) throw std::runtime_error("split corr fusion k not bit-exact");
    }
    // --- conv-state correction: matmul_int4 (corrected) + update_conv_state
    //     vs matmul_int4_zp_corr (raw) + update_conv_state_corr
    {
        const int cseq = 64, channels = 8192, kernel = 4;
        const int history = kernel - 1;
        const size_t sn = (size_t)history * channels;
        std::vector<bf16> in_h((size_t)cseq * channels), st_h(sn);
        for (size_t i = 0; i < in_h.size(); ++i) in_h[i] = pattern(i, 0.003f);
        for (size_t i = 0; i < sn; ++i) st_h[i] = pattern(i, 0.004f);
        GpuBuffer<bf16> in_buf(in_h.size(), q), corr_buf((size_t)cseq * channels, q);
        GpuBuffer<bf16> c_plain((size_t)cseq * channels, q), st1(sn, q), st2(sn, q);
        in_buf.upload(in_h.data(), in_h.size());
        st1.upload(st_h.data(), sn);
        st2.upload(st_h.data(), sn);
        std::vector<bf16> cr_h((size_t)cseq * channels);
        for (size_t i = 0; i < cr_h.size(); ++i) cr_h[i] = pattern(i + 11, 0.002f);
        corr_buf.upload(cr_h.data(), cr_h.size());
        // baseline: simulate subtract kernel output (C - corr) then update.
        GpuBuffer<bf16> corrected((size_t)cseq * channels, q);
        std::vector<bf16> cc((size_t)cseq * channels);
        for (size_t i = 0; i < cc.size(); ++i)
            cc[i] = float_to_bf16(bf16_to_float(in_buf.data() == nullptr ? 0 : in_h[i]) -
                                  bf16_to_float(cr_h[i]));
        corrected.upload(cc.data(), cc.size());
        qwen35_update_conv_state(q, corrected.data(), st1.data(), cseq, channels,
                                 kernel, true);
        qwen35_update_conv_state_corr(q, in_buf.data(), corr_buf.data(), st2.data(),
                                      cseq, channels, kernel, true);
        q.wait();
        std::vector<bf16> ref(sn), test(sn);
        st1.download(ref.data(), ref.size());
        st2.download(test.data(), test.size());
        ErrorStats e = compare(ref, test);
        std::printf("[corr-fusion] update_conv_state_corr | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0)
            throw std::runtime_error("update_conv_state_corr not bit-exact");
    }
    // --- swiglu: plain vs fused
    {
        GpuBuffer<bf16> c_plain((size_t)M * 2 * inter, q), c_fused((size_t)M * 2 * inter, q);
        GpuBuffer<bf16> o1((size_t)M * inter, q), o2((size_t)M * inter, q);
        GpuBuffer<bf16> cr_scratch((size_t)M * 2 * inter, q);
        matmul_int4(a.data(), M, K, Wgu, c_plain.data(), ctx, rs.data(), cr_scratch.data());
        swiglu_strided(q, c_plain.data(), o1.data(), M, inter);
        matmul_int4_zp_corr(a.data(), M, K, Wgu, c_fused.data(), ctx, rs.data(), corr.data());
        qwen35_swiglu_strided_corr(q, c_fused.data(), corr.data(), o2.data(), M, inter);
        q.wait();
        std::vector<bf16> ref((size_t)M * inter), test((size_t)M * inter);
        o1.download(ref.data(), ref.size());
        o2.download(test.data(), test.size());
        ErrorStats e = compare(ref, test);
        std::printf("[corr-fusion] swiglu | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0) throw std::runtime_error("swiglu corr fusion not bit-exact");
    }
    // --- residual add (o_proj shape): matmul_int4+subtract then add, vs zp_corr then add_corr
    {
        GpuBuffer<bf16> hidden1((size_t)M * kHidden, q), hidden2((size_t)M * kHidden, q);
        GpuBuffer<bf16> c_plain((size_t)M * kHidden, q), c_fused((size_t)M * kHidden, q);
        GpuBuffer<bf16> cr_scratch((size_t)M * kHidden, q);
        std::vector<bf16> hh((size_t)M * kHidden);
        for (size_t i = 0; i < hh.size(); ++i) hh[i] = pattern(i + 3, 0.001f);
        hidden1.upload(hh.data(), hh.size());
        hidden2.upload(hh.data(), hh.size());
        matmul_int4(a.data(), M, query_dim, Wo, c_plain.data(), ctx, rs.data(), cr_scratch.data());
        add_inplace(q, hidden1.data(), c_plain.data(), (int)((size_t)M * kHidden));
        matmul_int4_zp_corr(a.data(), M, query_dim, Wo, c_fused.data(), ctx, rs.data(), corr.data());
        qwen35_add_inplace_corr(q, hidden2.data(), c_fused.data(), corr.data(),
                                (int)((size_t)M * kHidden));
        q.wait();
        std::vector<bf16> ref((size_t)M * kHidden), test((size_t)M * kHidden);
        hidden1.download(ref.data(), ref.size());
        hidden2.download(test.data(), test.size());
        ErrorStats e = compare(ref, test);
        std::printf("[corr-fusion] add_inplace_corr | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0) throw std::runtime_error("add_inplace corr fusion not bit-exact");
    }
}

void bench_gdn_fusions(GpuEngine& ctx, int iterations) {
    using namespace qwen35_kernels;
    auto& q = ctx.queue;
    const int seq = 64;
    const int channels = 8192;  // conv_dim
    const int kernel = 4;
    const int heads = 48, key_dim = 128;
    const float eps = 1e-6f;
    const size_t n = (size_t)seq * channels;

    std::vector<bf16> in_h(n), w_h((size_t)channels * kernel), state_h((size_t)(kernel - 1) * channels);
    for (size_t i = 0; i < n; ++i) in_h[i] = pattern(i, 0.003f);
    for (size_t i = 0; i < w_h.size(); ++i) w_h[i] = pattern(i, 0.01f, 0.5f);
    for (size_t i = 0; i < state_h.size(); ++i) state_h[i] = pattern(i, 0.004f);
    GpuBuffer<bf16> in(n, q), w(w_h.size(), q), old_state(state_h.size(), q);
    in.upload(in_h.data(), n);
    w.upload(w_h.data(), w_h.size());
    old_state.upload(state_h.data(), state_h.size());
    GpuBuffer<bf16> out1(n, q), out2(n, q);
    GpuBuffer<bf16> st1(state_h.size(), q), st2(state_h.size(), q);
    st1.upload(state_h.data(), state_h.size());
    st2.upload(state_h.data(), state_h.size());

    // conv_causal + update_conv_state vs qwen35_conv_causal_state.
    qwen35_conv_causal(q, in.data(), w.data(), old_state.data(), out1.data(),
                       seq, channels, kernel, true);
    qwen35_update_conv_state(q, in.data(), st1.data(), seq, channels, kernel, true);
    qwen35_conv_causal_state(q, in.data(), w.data(), old_state.data(), out2.data(),
                             st2.data(), seq, channels, kernel, true);
    q.wait();
    {
        std::vector<bf16> ref(n), test(n);
        out1.download(ref.data(), ref.size());
        out2.download(test.data(), test.size());
        ErrorStats e = compare(ref, test);
        std::printf("[gdn-fusion] conv_causal_state out | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0) throw std::runtime_error("conv_causal_state out not bit-exact");
        std::vector<bf16> ref_st(state_h.size()), test_st(state_h.size());
        st1.download(ref_st.data(), ref_st.size());
        st2.download(test_st.data(), test_st.size());
        e = compare(ref_st, test_st);
        std::printf("[gdn-fusion] conv_causal_state state | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref_st.size());
        if (e.bit_mismatches > 0) {
            for (size_t i = 0; i < ref_st.size(); ++i)
                if (ref_st[i] != test_st[i]) {
                    std::printf("  first mismatch idx=%zu ref=%.6g test=%.6g\n", i,
                                (double)bf16_to_float(ref_st[i]),
                                (double)bf16_to_float(test_st[i]));
                    break;
                }
            throw std::runtime_error("conv_causal_state state not bit-exact");
        }
    }

    // l2norm x2 + scale vs qwen35_l2norm_scale_k.
    {
        const size_t rn = (size_t)seq * heads * key_dim;
        std::vector<bf16> xh(rn);
        for (size_t i = 0; i < rn; ++i) xh[i] = pattern(i, 0.005f);
        GpuBuffer<bf16> qk(rn, q), kk(rn, q), oq1(rn, q), ok1(rn, q), oq2(rn, q), ok2(rn, q);
        qk.upload(xh.data(), rn);
        kk.upload(xh.data(), rn);
        float scale = 1.0f / std::sqrt((float)key_dim);
        l2norm(q, qk.data(), oq1.data(), seq * heads, key_dim, eps);
        l2norm(q, kk.data(), ok1.data(), seq * heads, key_dim, eps);
        scale_inplace(q, oq1.data(), (int)rn, scale);
        qwen35_l2norm_scale_k(q, qk.data(), kk.data(), oq2.data(), ok2.data(),
                              seq * heads, key_dim, eps, scale);
        q.wait();
        std::vector<bf16> ref(rn), test(rn);
        oq1.download(ref.data(), ref.size());
        oq2.download(test.data(), test.size());
        ErrorStats e = compare(ref, test);
        std::printf("[gdn-fusion] l2norm_scale_k q | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0) throw std::runtime_error("l2norm_scale_k q not bit-exact");
        ok1.download(ref.data(), ref.size());
        ok2.download(test.data(), test.size());
        e = compare(ref, test);
        std::printf("[gdn-fusion] l2norm_scale_k k | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0) throw std::runtime_error("l2norm_scale_k k not bit-exact");
    }

    // sigmoid(beta) + compute_g vs qwen35_sigmoid_beta_compute_g.
    {
        const size_t hn = (size_t)seq * heads;
        std::vector<bf16> bh(hn), ah(hn), Alog(hn), dt(hn);
        for (size_t i = 0; i < hn; ++i) { bh[i] = pattern(i, 0.05f); ah[i] = pattern(i + 7, 0.05f); }
        for (int i = 0; i < heads; ++i) { Alog[i] = pattern(i, 0.02f, -4.0f); dt[i] = pattern(i, 0.03f, -1.0f); }
        for (int s = 1; s < seq; ++s)
            for (int i = 0; i < heads; ++i) {
                Alog[(size_t)s * heads + i] = Alog[i];
                dt[(size_t)s * heads + i] = dt[i];
            }
        GpuBuffer<bf16> beta1(hn, q), g1(hn, q), beta2(hn, q), g2(hn, q);
        GpuBuffer<bf16> a(hn, q), Al(hn, q), Dt(hn, q);
        beta1.upload(bh.data(), hn);
        beta2.upload(bh.data(), hn);
        g1.upload(ah.data(), hn);
        g2.upload(ah.data(), hn);
        a.upload(ah.data(), hn);
        Al.upload(Alog.data(), hn);
        Dt.upload(dt.data(), hn);
        sigmoid_inplace(q, beta1.data(), (int)hn);
        qwen35_compute_g(q, g1.data(), Al.data(), Dt.data(), g1.data(), seq, heads);
        qwen35_sigmoid_beta_compute_g(q, beta2.data(), a.data(), Al.data(), Dt.data(),
                                      g2.data(), seq, heads);
        q.wait();
        std::vector<bf16> ref(hn), test(hn);
        g1.download(ref.data(), ref.size());
        g2.download(test.data(), test.size());
        ErrorStats e = compare(ref, test);
        std::printf("[gdn-fusion] sigmoid_beta_compute_g g | max_abs %.7g bit_mismatch %zu/%zu\n",
                    e.max_abs, e.bit_mismatches, ref.size());
        if (e.bit_mismatches > 0) throw std::runtime_error("sigmoid_beta_compute_g not bit-exact");
    }
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
    std::vector<bf16> w_h(w_count);
    for (size_t i = 0; i < w_h.size(); ++i) w_h[i] = pattern(i, 0.0005f, -0.1f);
    w.upload(w_h.data(), w_h.size());
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

    // f32-dst GEMM: logits keep the full fp32 accumulator (no bf16 rounding).
    GpuBuffer<float> logits_f32_gemm(V, q);
    double t_gemm_f32 = elapsed_ms(q, iterations, [&] {
        matmul_bf16_f32(hidden.data(), 1, H, w.data(), V, logits_f32_gemm.data(), ctx);
    });
    q.wait();
    std::vector<float> logits_bf16_cvt(V), logits_f32_h(V);
    bf16_to_f32(q, logits_bf16.data(), logits_bf16_cvt.data(), V);
    logits_f32_gemm.download(logits_f32_h.data(), V);
    q.wait();
    double max_abs = 0.0;
    for (int i = 0; i < V; ++i)
        max_abs = std::max(max_abs, (double)std::abs(logits_bf16_cvt[i] - logits_f32_h[i]));
    std::printf("[vocab-tail] M=1 H=%d V=%d | gemm %.3f ms | bf16->f32 %.3f ms "
                "| d2h+wait %.3f ms | host partial_sort %.3f ms | total %.3f ms "
                "| f32-gemm %.3f ms | logit bf16-vs-f32 max_abs %.6g\n",
                H, V, t_gemm, t_cvt, t_d2h, t_sort,
                t_gemm + t_cvt + t_d2h + t_sort, t_gemm_f32, max_abs);
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
    bench_dequant_bf16(ctx, iterations);
    bench_corr_fusion(ctx, iterations);
    bench_gdn_fusions(ctx, iterations);
    bench_add_norm(ctx, iterations);
    bench_norm_rope(ctx, iterations);
    return 0;
}

}  // namespace

REGISTER_BENCH("qwen35-int4-fusion",
    "Qwen3.5 dense INT4-AWQ fusion A/B: zp correction, add+norm, norm+rope",
    run)

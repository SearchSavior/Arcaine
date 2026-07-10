// Focused DiffusionGemma INT4-AWQ fusion benchmark.
//
// This deliberately benchmarks the fixed denoiser kernel paths rather than
// end-to-end inference:
//   1) shared dense gate/up projection at M=256, K=2816, N=2112;
//   2) expert weighted-combine -> dual postnorm at seq=256, H=2816, top-k=8.
//
// The fused paths are guarded by the same environment variables as inference.
#include "common/gpu/buffer.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/ops.hpp"
#include "common/kernels/elementwise.hpp"
#include "common/kernels/rms_norm.hpp"
#include "modeling/diffusion_gemma/fusions/int4_awq.hpp"
#include "modeling/diffusion_gemma/fusions/postnorm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSeq = 256;
constexpr int kHidden = 2816;
constexpr int kIntermediate = 2112;
constexpr int kTopK = 8;
constexpr int kExpertRows = 128 * 64;  // current INT4 tail layout at seq=256
constexpr float kEps = 1e-6f;

bf16 pattern(size_t i, float scale, float bias = 0.0f) {
    int v = (int)((i * 17 + 13) % 101) - 50;
    return float_to_bf16(bias + scale * (float)v);
}

template <typename Fn>
double elapsed_ms(sycl::queue& q, int iterations, const Fn& fn) {
    q.wait();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) fn();
    q.wait();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           (double)iterations;
}

void baseline_expert_combine(sycl::queue& q,
                             const bf16* expert_out,
                             const int32_t* slot,
                             const float* route_weight,
                             bf16* moe_out,
                             int seq, int H, int top_k) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(seq, H), [=](sycl::id<2> id) {
            int tok = (int)id[0];
            int d = (int)id[1];
            float acc = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                int a = tok * top_k + k;
                int s = slot[a];
                if (s >= 0)
                    acc += route_weight[a] *
                           bf16_to_float(expert_out[(size_t)s * H + d]);
            }
            moe_out[(size_t)tok * H + d] = float_to_bf16(acc);
        });
    });
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

void bench_dense_gate_up(GpuEngine& ctx, int iterations) {
    auto& q = ctx.queue;
    const size_t x_count = (size_t)kSeq * kHidden;
    const size_t w_count = (size_t)kIntermediate * kHidden;
    const size_t act_count = (size_t)kSeq * kIntermediate;

    std::vector<bf16> x_h(x_count), gate_h(w_count), up_h(w_count);
    for (size_t i = 0; i < x_count; ++i) x_h[i] = pattern(i, 0.002f);
    for (size_t i = 0; i < w_count; ++i) {
        gate_h[i] = pattern(i, 0.0005f);
        up_h[i] = pattern(i + 31, 0.0005f);
    }
    std::vector<bf16> fused_w_h(2 * w_count);
    std::copy(gate_h.begin(), gate_h.end(), fused_w_h.begin());
    std::copy(up_h.begin(), up_h.end(), fused_w_h.begin() + w_count);

    GpuBuffer<bf16> x(x_count, q), gate_w(w_count, q), up_w(w_count, q);
    GpuBuffer<bf16> fused_w(2 * w_count, q);
    GpuBuffer<bf16> gate(act_count, q), up(act_count, q);
    GpuBuffer<bf16> fused_gu(2 * act_count, q), fused_act(act_count, q);
    x.upload(x_h.data(), x_h.size());
    gate_w.upload(gate_h.data(), gate_h.size());
    up_w.upload(up_h.data(), up_h.size());
    fused_w.upload(fused_w_h.data(), fused_w_h.size());

    auto baseline = [&] {
        matmul_bf16(x.data(), kSeq, kHidden, gate_w.data(), kIntermediate,
                    gate.data(), ctx);
        matmul_bf16(x.data(), kSeq, kHidden, up_w.data(), kIntermediate,
                    up.data(), ctx);
        geglu_inplace(q, gate.data(), up.data(), (int)act_count);
    };
    auto fused = [&] {
        matmul_bf16(x.data(), kSeq, kHidden, fused_w.data(),
                    2 * kIntermediate, fused_gu.data(), ctx);
        geglu_strided(q, fused_gu.data(), fused_act.data(), kSeq, kIntermediate);
    };

    baseline();
    fused();
    q.wait();
    std::vector<bf16> baseline_h(act_count), fused_h(act_count);
    gate.download(baseline_h.data(), baseline_h.size());
    fused_act.download(fused_h.data(), fused_h.size());
    ErrorStats err = compare(baseline_h, fused_h);

    // Warm the exact primitive/kernel sequence before timing.
    for (int i = 0; i < 3; ++i) baseline();
    for (int i = 0; i < 3; ++i) fused();
    q.wait();
    double base_ms = elapsed_ms(q, iterations, baseline);
    double fused_ms = elapsed_ms(q, iterations, fused);
    std::printf("[dense-gate-up] shape M=%d K=%d N=%d | baseline %.3f ms | "
                "fused %.3f ms | speedup %.3fx | max_abs %.7g | bit_mismatch %zu/%zu\n",
                kSeq, kHidden, kIntermediate, base_ms, fused_ms,
                base_ms / fused_ms, err.max_abs, err.bit_mismatches, act_count);
    if (err.max_abs > 0.03125f)
        throw std::runtime_error("dense gate/up fusion exceeded BF16 tolerance");
}

void bench_expert_postnorm(GpuEngine& ctx, int iterations) {
    auto& q = ctx.queue;
    const size_t hidden_count = (size_t)kSeq * kHidden;
    const size_t expert_count = (size_t)kExpertRows * kHidden;
    const size_t route_count = (size_t)kSeq * kTopK;

    std::vector<bf16> expert_h(expert_count), mlp_h(hidden_count);
    std::vector<bf16> hidden_h(hidden_count);
    std::vector<bf16> w1_h(kHidden), w2_h(kHidden), w3_h(kHidden);
    std::vector<int32_t> slot_h(route_count);
    std::vector<float> route_h(route_count);
    for (size_t i = 0; i < expert_count; ++i) expert_h[i] = pattern(i, 0.001f);
    for (size_t i = 0; i < hidden_count; ++i) {
        mlp_h[i] = pattern(i + 7, 0.0015f);
        hidden_h[i] = pattern(i + 19, 0.002f);
    }
    for (int d = 0; d < kHidden; ++d) {
        w1_h[d] = pattern(d, 0.00025f, 1.0f);
        w2_h[d] = pattern(d + 5, 0.00025f, 1.0f);
        w3_h[d] = pattern(d + 11, 0.00025f, 1.0f);
    }
    for (size_t a = 0; a < route_count; ++a) {
        slot_h[a] = (int32_t)((a * 37) % kExpertRows);
        route_h[a] = 1.0f / (float)kTopK;
    }

    GpuBuffer<bf16> expert(expert_count, q), mlp(hidden_count, q);
    GpuBuffer<bf16> initial(hidden_count, q), base_hidden(hidden_count, q);
    GpuBuffer<bf16> fused_hidden(hidden_count, q), moe(hidden_count, q);
    GpuBuffer<bf16> w1(kHidden, q), w2(kHidden, q), w3(kHidden, q);
    GpuBuffer<int32_t> slot(route_count, q);
    GpuBuffer<float> route(route_count, q);
    expert.upload(expert_h.data(), expert_h.size());
    mlp.upload(mlp_h.data(), mlp_h.size());
    initial.upload(hidden_h.data(), hidden_h.size());
    w1.upload(w1_h.data(), w1_h.size());
    w2.upload(w2_h.data(), w2_h.size());
    w3.upload(w3_h.data(), w3_h.size());
    slot.upload(slot_h.data(), slot_h.size());
    route.upload(route_h.data(), route_h.size());

    auto reset_outputs = [&] {
        q.memcpy(base_hidden.data(), initial.data(), hidden_count * sizeof(bf16));
        q.memcpy(fused_hidden.data(), initial.data(), hidden_count * sizeof(bf16));
    };
    auto baseline = [&] {
        baseline_expert_combine(q, expert.data(), slot.data(), route.data(),
                                moe.data(), kSeq, kHidden, kTopK);
        fused_dual_postnorm(q, mlp.data(), w1.data(), moe.data(), w2.data(),
                            w3.data(), base_hidden.data(), 1.0f,
                            kSeq, kHidden, kEps);
    };
    auto fused = [&] {
        fused_int4_expert_combine_postnorm(
            q, expert.data(), slot.data(), route.data(), kTopK,
            mlp.data(), w1.data(), w2.data(), w3.data(),
            fused_hidden.data(), 1.0f, kSeq, kHidden, kEps);
    };

    reset_outputs();
    baseline();
    fused();
    q.wait();
    std::vector<bf16> base_h(hidden_count), fused_h(hidden_count);
    base_hidden.download(base_h.data(), base_h.size());
    fused_hidden.download(fused_h.data(), fused_h.size());
    ErrorStats err = compare(base_h, fused_h);

    reset_outputs();
    for (int i = 0; i < 3; ++i) baseline();
    q.wait();
    double base_ms = elapsed_ms(q, iterations, baseline);
    reset_outputs();
    for (int i = 0; i < 3; ++i) fused();
    q.wait();
    double fused_ms = elapsed_ms(q, iterations, fused);
    std::printf("[expert-postnorm] seq=%d H=%d top_k=%d rows=%d | baseline %.3f ms | "
                "fused %.3f ms | speedup %.3fx | max_abs %.7g | bit_mismatch %zu/%zu | "
                "intermediate_saved %.2f MiB\n",
                kSeq, kHidden, kTopK, kExpertRows, base_ms, fused_ms,
                base_ms / fused_ms, err.max_abs, err.bit_mismatches, hidden_count,
                hidden_count * sizeof(bf16) / (1024.0 * 1024.0));
    if (err.bit_mismatches != 0)
        throw std::runtime_error("expert combine/postnorm fusion is not bit-exact");
}

void bench_selfcond_add_norm(GpuEngine& ctx, int iterations) {
    auto& q = ctx.queue;
    const size_t count = (size_t)kSeq * kHidden;
    std::vector<bf16> input_h(count), delta_h(count);
    for (size_t i = 0; i < count; ++i) {
        input_h[i] = pattern(i + 23, 0.002f);
        delta_h[i] = pattern(i + 47, 0.001f);
    }
    GpuBuffer<bf16> initial(count, q), delta(count, q);
    GpuBuffer<bf16> baseline_out(count, q), fused_out(count, q);
    initial.upload(input_h.data(), input_h.size());
    delta.upload(delta_h.data(), delta_h.size());

    auto reset = [&] {
        q.memcpy(baseline_out.data(), initial.data(), count * sizeof(bf16));
        q.memcpy(fused_out.data(), initial.data(), count * sizeof(bf16));
    };
    auto baseline = [&] {
        add_inplace(q, baseline_out.data(), delta.data(), (int)count);
        rms_norm_no_scale(q, baseline_out.data(), baseline_out.data(),
                          kSeq, kHidden, kEps);
    };
    auto fused = [&] {
        fused_int4_selfcond_add_norm(q, fused_out.data(), delta.data(),
                                     kSeq, kHidden, kEps);
    };

    reset();
    baseline();
    fused();
    q.wait();
    std::vector<bf16> base_h(count), fused_h(count);
    baseline_out.download(base_h.data(), base_h.size());
    fused_out.download(fused_h.data(), fused_h.size());
    ErrorStats err = compare(base_h, fused_h);

    reset();
    for (int i = 0; i < 3; ++i) baseline();
    q.wait();
    double base_ms = elapsed_ms(q, iterations, baseline);
    reset();
    for (int i = 0; i < 3; ++i) fused();
    q.wait();
    double fused_ms = elapsed_ms(q, iterations, fused);
    std::printf("[selfcond-add-norm] seq=%d H=%d | baseline %.3f ms | fused %.3f ms | "
                "speedup %.3fx | max_abs %.7g | bit_mismatch %zu/%zu\n",
                kSeq, kHidden, base_ms, fused_ms, base_ms / fused_ms,
                err.max_abs, err.bit_mismatches, count);
    if (err.bit_mismatches != 0)
        throw std::runtime_error("self-conditioning add+norm fusion is not bit-exact");
}

} // namespace

int main(int argc, char** argv) {
    int iterations = 20;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--iterations N]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (iterations <= 0) iterations = 1;
    if (!diff_int4_fuse_dense_gate_up_enabled() ||
        !diff_int4_fuse_expert_postnorm_enabled() ||
        !diff_int4_fuse_selfcond_add_norm_enabled()) {
        std::fprintf(stderr,
            "Set DIFF_INT4_FUSE_DENSE_GATE_UP=1 and "
            "DIFF_INT4_FUSE_EXPERT_POSTNORM=1 and "
            "DIFF_INT4_FUSE_SELFCOND_ADD_NORM=1 to benchmark the introduced paths.\n");
        return 2;
    }

    try {
        GpuEngine& ctx = GpuEngine::get(0);
        std::printf("[device] %s | iterations=%d\n",
                    ctx.queue.get_device().get_info<sycl::info::device::name>().c_str(),
                    iterations);
        bench_dense_gate_up(ctx, iterations);
        bench_expert_postnorm(ctx, iterations);
        bench_selfcond_add_norm(ctx, iterations);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

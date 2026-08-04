// Focused DiffusionGemma INT4-AWQ grouped-MoE benchmark.
//
// This exercises DiffusionGemma's top-8 expert path at a selected context
// length. The baseline is the current per-active-expert oneDNN s4 sequence;
// the candidate is the device-routed grouped DPAS sequence. It deliberately
// does not run end-to-end inference.

#include "common/bench/registry.hpp"
#include "common/bench/util.hpp"
#include "common/gpu/buffer.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/int4.hpp"
#include "common/gpu/int4_grouped_moe.hpp"
#include "common/kernels/elementwise.hpp"
#include "modeling/diffusion_gemma/fusions/int4_awq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using arcaine::bench::elapsed_ms;

namespace {

constexpr int kDefaultSeq = 256;
constexpr int kTopK = 8;
constexpr int kGlobalExperts = 128;
constexpr int kHidden = 2816;
constexpr int kInter = 704;
constexpr int kGroup = 32;
constexpr int kTail = 64;

int round_up(int v, int m) { return (v + m - 1) / m * m; }

bf16 activation_pattern(size_t i) {
    int v = (int)((i * 29 + 17) % 101) - 50;
    return float_to_bf16((float)v * 0.002f);
}

Int4Linear make_weight(GpuEngine& ctx, int K, int N, int seed) {
    Int4Linear w;
    w.in_features = K;
    w.out_features = N;
    w.group_size = kGroup;
    const size_t packed_count = (size_t)N * K / 2;
    const size_t scale_count = (size_t)(K / kGroup) * N;
    std::vector<uint8_t> packed(packed_count);
    std::vector<bf16> scale(scale_count);
    for (size_t i = 0; i < packed_count; ++i) {
        int lo = (int)((i * 13 + seed) % 7) - 3;
        int hi = (int)((i * 19 + seed + 5) % 7) - 3;
        packed[i] = (uint8_t)((lo & 0xf) | ((hi & 0xf) << 4));
    }
    for (size_t i = 0; i < scale_count; ++i)
        scale[i] = float_to_bf16(0.003f + 0.00025f * (float)((i + seed) % 5));
    w.weight_packed = GpuBuffer<uint8_t>(packed_count, ctx.queue);
    w.weight_scale = GpuBuffer<bf16>(scale_count, ctx.queue);
    w.weight_packed.upload(packed.data(), packed.size());
    w.weight_scale.upload(scale.data(), scale.size());
    return w;
}

void combine_baseline(sycl::queue& q, const bf16* expert_out,
                      const int32_t* slot, const float* weight,
                      int seq, int top_k, int H, bf16* out) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)seq, (size_t)H),
                       [=](sycl::id<2> id) {
            int token = (int)id[0], dim = (int)id[1];
            float acc = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                int pair = token * top_k + k;
                int row = slot[pair];
                if (row >= 0)
                    acc += weight[pair] *
                           bf16_to_float(expert_out[(size_t)row * H + dim]);
            }
            out[(size_t)token * H + dim] = float_to_bf16(acc);
        });
    });
}

struct ErrorStats {
    float max_abs = 0.0f;
    double rms = 0.0;
    double cosine = 0.0;
    size_t bit_mismatches = 0;
};

ErrorStats compare(const std::vector<bf16>& a, const std::vector<bf16>& b) {
    if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
    ErrorStats s;
    double err2 = 0.0, aa = 0.0, bb = 0.0, ab = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float av = bf16_to_float(a[i]);
        float bv = bf16_to_float(b[i]);
        float d = av - bv;
        if (a[i] != b[i]) ++s.bit_mismatches;
        s.max_abs = std::max(s.max_abs, std::abs(d));
        err2 += (double)d * d;
        aa += (double)av * av;
        bb += (double)bv * bv;
        ab += (double)av * bv;
    }
    s.rms = std::sqrt(err2 / (double)a.size());
    s.cosine = ab / std::sqrt(std::max(aa * bb, 1e-30));
    return s;
}

int run(int argc, char** argv) {
    int local_experts = 44;
    int first_expert = 0;
    int iterations = 10;
    int seq = kDefaultSeq;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--experts" && i + 1 < argc) local_experts = std::atoi(argv[++i]);
        else if (arg == "--first" && i + 1 < argc) first_expert = std::atoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (arg == "--seq" && i + 1 < argc) seq = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--experts N] [--first N] "
                        "[--seq N] [--iterations N]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (!diff_int4_grouped_dpas_moe_enabled()) {
        std::fprintf(stderr, "Set DIFF_INT4_GROUPED_DPAS_MOE=1 to benchmark the candidate path.\n");
        return 2;
    }
    if (local_experts <= 0 || first_expert < 0 ||
        first_expert + local_experts > kGlobalExperts || iterations <= 0 || seq <= 0)
        throw std::runtime_error("invalid expert range, context length, or iteration count");

    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    const int pairs = seq * kTopK;
    const int max_active = std::min(local_experts, pairs);
    std::printf("[device] %s | range=[%d,%d) | iterations=%d | slice=%d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                first_expert, first_expert + local_experts, iterations,
                diff_int4_grouped_dpas_expert_slice());
    if (!diff_int4_grouped_dpas_device_supported(q.get_device())) {
        std::printf("[unsupported] grouped BF16 DPAS requires Xe2/Battlemage; "
                    "this device uses the oneDNN INT4 path.\n");
        return 0;
    }

    std::vector<bf16> input_h((size_t)seq * kHidden);
    std::vector<int32_t> idx_h(pairs), slot_h(pairs, -1);
    std::vector<float> weight_h(pairs, 1.0f / (float)kTopK);
    for (size_t i = 0; i < input_h.size(); ++i) input_h[i] = activation_pattern(i);
    for (int t = 0; t < seq; ++t)
        for (int k = 0; k < kTopK; ++k)
            idx_h[t * kTopK + k] = (t * 13 + k * 17) % kGlobalExperts;

    std::vector<int> counts(local_experts, 0), cursor(local_experts, 0);
    for (int pair = 0; pair < pairs; ++pair) {
        int e = idx_h[pair] - first_expert;
        if (e >= 0 && e < local_experts) ++counts[e];
    }
    const int tail = std::min(kTail, round_up(seq, 8));
    std::vector<int> base(local_experts), bucket_cap(local_experts, tail);
    int total_rows = local_experts * tail;
    for (int e = 0; e < local_experts; ++e) {
        if (counts[e] > tail) {
            base[e] = total_rows;
            bucket_cap[e] = round_up(counts[e], 32);
            total_rows += bucket_cap[e];
        } else {
            base[e] = e * tail;
        }
    }
    for (int pair = 0; pair < pairs; ++pair) {
        int e = idx_h[pair] - first_expert;
        if (e >= 0 && e < local_experts) slot_h[pair] = base[e] + cursor[e]++;
    }
    GpuBuffer<bf16> input(input_h.size(), q);
    GpuBuffer<int32_t> idx(pairs, q), slot(pairs, q);
    GpuBuffer<float> weight(pairs, q);
    input.upload(input_h.data(), input_h.size());
    idx.upload(idx_h.data(), idx_h.size());
    slot.upload(slot_h.data(), slot_h.size());
    weight.upload(weight_h.data(), weight_h.size());

    std::vector<Int4Linear> gate(local_experts), down(local_experts);
    std::vector<const uint8_t*> gate_w_h(local_experts), down_w_h(local_experts);
    std::vector<const bf16*> gate_s_h(local_experts), down_s_h(local_experts);
    for (int e = 0; e < local_experts; ++e) {
        gate[e] = make_weight(ctx, kHidden, 2 * kInter, 31 + e);
        down[e] = make_weight(ctx, kInter, kHidden, 197 + e);
        gate_w_h[e] = gate[e].weight_packed.data();
        gate_s_h[e] = gate[e].weight_scale.data();
        down_w_h[e] = down[e].weight_packed.data();
        down_s_h[e] = down[e].weight_scale.data();
    }
    GpuBuffer<const uint8_t*> gate_w(local_experts, q), down_w(local_experts, q);
    GpuBuffer<const bf16*> gate_s(local_experts, q), down_s(local_experts, q);
    gate_w.upload(gate_w_h.data(), gate_w_h.size());
    gate_s.upload(gate_s_h.data(), gate_s_h.size());
    down_w.upload(down_w_h.data(), down_w_h.size());
    down_s.upload(down_s_h.data(), down_s_h.size());

    GpuBuffer<bf16> xe((size_t)total_rows * kHidden, q);
    GpuBuffer<bf16> gu((size_t)total_rows * 2 * kInter, q);
    GpuBuffer<bf16> act((size_t)total_rows * kInter, q);
    GpuBuffer<bf16> ye((size_t)total_rows * kHidden, q);
    GpuBuffer<bf16> baseline_out((size_t)seq * kHidden, q);

    GpuBuffer<int32_t> offsets((size_t)local_experts + 1, q), tokens(pairs, q);
    GpuBuffer<int32_t> active_experts(max_active, q), active_count(1, q);
    GpuBuffer<bf16> grouped_act((size_t)pairs * kInter, q);
    GpuBuffer<bf16> pair_out((size_t)pairs * kHidden, q);
    GpuBuffer<bf16> grouped_out((size_t)seq * kHidden, q);

    auto baseline = [&] {
        q.memset(xe.data(), 0, xe.count() * sizeof(bf16));
        const int32_t* slot_ptr = slot.data();
        const bf16* input_ptr = input.data();
        bf16* xe_ptr = xe.data();
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>((size_t)pairs, (size_t)kHidden),
                           [=](sycl::id<2> id) {
                int pair = (int)id[0], dim = (int)id[1];
                int row = slot_ptr[pair];
                if (row >= 0)
                    xe_ptr[(size_t)row * kHidden + dim] =
                        input_ptr[(size_t)(pair / kTopK) * kHidden + dim];
            });
        });
        for (int e = 0; e < local_experts; ++e) {
            if (counts[e] == 0) continue;
            int m = std::min(bucket_cap[e], round_up(counts[e], 32));
            matmul_int4(xe.data() + (size_t)base[e] * kHidden,
                        m, kHidden, gate[e],
                        gu.data() + (size_t)base[e] * 2 * kInter, ctx);
        }
        geglu_strided(q, gu.data(), act.data(), total_rows, kInter);
        for (int e = 0; e < local_experts; ++e) {
            if (counts[e] == 0) continue;
            int m = std::min(bucket_cap[e], round_up(counts[e], 32));
            matmul_int4(act.data() + (size_t)base[e] * kInter,
                        m, kInter, down[e],
                        ye.data() + (size_t)base[e] * kHidden, ctx);
        }
        combine_baseline(q, ye.data(), slot.data(), weight.data(),
                         seq, kTopK, kHidden, baseline_out.data());
    };

    auto grouped = [&] {
        int4_grouped_moe_build_routes(
            q, idx.data(), first_expert, local_experts, pairs,
            offsets.data(), tokens.data(), active_experts.data(), active_count.data());
        matmul_int4_grouped_dpas_gateup_geglu(
            q, input.data(), kHidden, gate_w.data(), gate_s.data(),
            offsets.data(), tokens.data(), local_experts, pairs, kTopK,
            kInter, kGroup, grouped_act.data(), active_experts.data(),
            active_count.data(), max_active);
        matmul_int4_grouped_dpas_down(
            q, grouped_act.data(), kInter, down_w.data(), down_s.data(),
            offsets.data(), tokens.data(), local_experts, pairs, kHidden,
            kGroup, pair_out.data(), active_experts.data(),
            active_count.data(), max_active);
        int4_grouped_moe_combine(
            q, pair_out.data(), idx.data(), weight.data(), first_expert,
            local_experts, seq, kTopK, kHidden, grouped_out.data());
    };

    baseline();
    grouped();
    q.wait();
    std::vector<bf16> baseline_h(baseline_out.count()), grouped_h(grouped_out.count());
    baseline_out.download(baseline_h.data(), baseline_h.size());
    grouped_out.download(grouped_h.data(), grouped_h.size());
    ErrorStats error = compare(baseline_h, grouped_h);

    for (int i = 0; i < 2; ++i) baseline();
    for (int i = 0; i < 2; ++i) grouped();
    q.wait();
    double baseline_ms = elapsed_ms(q, iterations, baseline);
    double grouped_ms = elapsed_ms(q, iterations, grouped);
    auto grouped_routes = [&] {
        int4_grouped_moe_build_routes(
            q, idx.data(), first_expert, local_experts, pairs,
            offsets.data(), tokens.data(), active_experts.data(), active_count.data());
    };
    auto grouped_gateup = [&] {
        matmul_int4_grouped_dpas_gateup_geglu(
            q, input.data(), kHidden, gate_w.data(), gate_s.data(),
            offsets.data(), tokens.data(), local_experts, pairs, kTopK,
            kInter, kGroup, grouped_act.data(), active_experts.data(),
            active_count.data(), max_active);
    };
    auto grouped_down = [&] {
        matmul_int4_grouped_dpas_down(
            q, grouped_act.data(), kInter, down_w.data(), down_s.data(),
            offsets.data(), tokens.data(), local_experts, pairs, kHidden,
            kGroup, pair_out.data(), active_experts.data(),
            active_count.data(), max_active);
    };
    auto grouped_combine = [&] {
        int4_grouped_moe_combine(
            q, pair_out.data(), idx.data(), weight.data(), first_expert,
            local_experts, seq, kTopK, kHidden, grouped_out.data());
    };
    double routes_ms = elapsed_ms(q, iterations, grouped_routes);
    double gateup_ms = elapsed_ms(q, iterations, grouped_gateup);
    double down_ms = elapsed_ms(q, iterations, grouped_down);
    double combine_ms = elapsed_ms(q, iterations, grouped_combine);
    std::printf("[moe] seq=%d topk=%d H=%d I=%d localE=%d rows=%d | "
                "baseline %.3f ms | grouped %.3f ms | speedup %.3fx\n",
                seq, kTopK, kHidden, kInter, local_experts, total_rows,
                baseline_ms, grouped_ms, baseline_ms / grouped_ms);
    std::printf("[grouped stages] routes %.3f ms | gateup %.3f ms | "
                "down %.3f ms | combine %.3f ms\n",
                routes_ms, gateup_ms, down_ms, combine_ms);
    std::printf("[correctness] max_abs=%.7g rms=%.7g cosine=%.9f "
                "bit_mismatch=%zu/%zu\n",
                error.max_abs, error.rms, error.cosine,
                error.bit_mismatches, baseline_h.size());
    if (!std::isfinite(error.cosine) || error.cosine < 0.999)
        throw std::runtime_error("grouped INT4 DPAS output failed cosine tolerance");
    return 0;
}

} // namespace

REGISTER_BENCH("diffusion-int4-grouped-moe",
    "DiffusionGemma INT4-AWQ per-expert oneDNN vs device-routed grouped DPAS MoE",
    run)

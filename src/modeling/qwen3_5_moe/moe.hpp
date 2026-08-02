#pragma once
//
// Qwen3.5-MoE block forward (Qwen3_5MoeSparseMoeBlock). Reference:
//   reference/transformers/.../modeling_qwen3_5_moe.py lines 798-817.
//
// Flow (per token, hidden already post-attention-normed):
//   1. Router:  scores = hidden @ router_gate.T            -> [S, 256]
//               probs  = softmax(scores, fp32)             -> [S, 256]
//               top8   = topk(probs); renorm /= sum(top8)   -> idx[S,8], wgt[S,8]
//               (NO per-expert scale — unlike the block-diffusion MoE.)
//   2. Routed experts (expert-major, reference lines 752-776):
//               for each active expert e: gather its tokens,
//                 h_e = silu(gate_up_e[:inter]) * gate_up_e[inter:]   (SwiGLU)
//                 y_e = down_e(h_e); y_e *= wgt[t, slot]
//                 scatter-add y_e into out[t].
//   3. Shared expert (always-on, DeepSeekMoE):
//               sh = down_shared(silu(gate_up_shared[:inter]) * gate_up_shared[inter:])
//               sh *= sigmoid(hidden @ shared_expert_gate.T)          (per-token scalar)
//   4. out = routed + shared.
//
// Routing is fully device-resident: the router matmul feeds
// qwen_moe_router_topk (softmax + top-k + renorm in one kernel) which writes
// idx/wgt straight into device scratch the routed paths consume — no host
// round trip, no per-layer pipeline sync. The routed experts then run either
// the oneDNN grouped W4A16 path (primary) or the custom grouped DPAS kernels
// (tiny-M / decode fallback), both consuming device idx/wgt; a host-
// orchestrated per-expert reference path remains for impl=onednn (AB test),
// selected via QWEN35_MOE_INT4_IMPL. The swap point is the single function
// below.
//
// Expert projections dispatch through qwen_matmul_proj (NVFP4 / AWQ INT4 / dense).
// router_gate and shared_expert_gate are BF16.
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#include "../../runtime/gpu/buffer.hpp"
#include "../../runtime/gpu/engine.hpp"
#include "../../runtime/quantization/nvfp4.hpp"               // matmul_nvfp4, Nvfp4Linear
#include "../../runtime/gpu/ops.hpp"                // matmul_bf16
#include "kernels/elementwise.hpp"    // add_inplace
#include "kernels/int4_grouped_moe.hpp"   // grouped DPAS INT4 routed path (tiny-M fallback)
#include "kernels/int4_grouped_onednn.hpp" // oneDNN grouped W4A16 routed path (primary)
#include "kernels/router_topk.hpp"    // device softmax+top-k router, active-expert compaction

#include "config.hpp"
#include "weights.hpp"   // QwenMoE
#include "kernels.hpp"    // swiglu_strided, scale_rows_by_sigmoid

using namespace qwen35moe_kernels;

// Routed-expert INT4 implementation selector (AB test).
//   grouped: oneDNN experimental grouped W4A16 matmuls (primary prefill path;
//            falls back to dpas below the size thresholds)
//   dpas:    custom grouped W4A16 DPAS kernels (device routing, no host round
//            trips; the tiny-M / decode path)
//   onednn:  per-expert oneDNN u4zp matmul + host scatter (reference)
enum QwenMoeInt4Impl { kQwenMoeImplOnednn = 0, kQwenMoeImplDpas = 1,
                       kQwenMoeImplGrouped = 2 };
inline QwenMoeInt4Impl qwen_moe_int4_impl()
{
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("QWEN35_MOE_INT4_IMPL");
        if (v && std::strcmp(v, "onednn") == 0) cached = kQwenMoeImplOnednn;
        else if (v && std::strcmp(v, "dpas") == 0) cached = kQwenMoeImplDpas;
        else cached = kQwenMoeImplGrouped;
    }
    return (QwenMoeInt4Impl)cached;
}

// Dispatch policy for impl=grouped: run the oneDNN grouped path only when the
// batch is large enough (env-overridable). Measured on B70 (int4_moe_bench,
// layer 0): the grouped path has a flat ~0.62 ms floor for pairs <= E (it
// pads total to E for the oneDNN micro kernel) and beats DPAS from ~48 pairs
// (p=6, top_k=8) upward; below that DPAS wins regardless of routing skew, so
// a max-per-expert escape has no winning regime and was removed in the M3
// sweep.
inline int qwen_moe_grouped_min_pairs()
{
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("QWEN35_MOE_GROUPED_MIN_PAIRS");
        cached = v ? std::atoi(v) : 48;
    }
    return cached;
}

// Host-orchestrated routed-experts path (correctness / decode). See file header.
// idx/wgt are [S*top_k] row-major (token-major, slot within token).
inline void qwen_routed_experts_forward(
    GpuEngine& ctx, const QwenMoE& w,
    const bf16* hidden,                // device [S, H]
    const std::vector<int>& idx,        // host [S*top_k]  expert ids
    const std::vector<float>& wgt,      // host [S*top_k]  renormed weights
    std::vector<float>& out_h,          // host [S*H]      fp32 accumulator (in/out)
    int S, const QwenConfig& cfg)
{
    auto& q = ctx.queue;
    int H     = cfg.hidden_size;              // 2048
    int E     = cfg.num_experts;              // 256
    int top_k = cfg.num_experts_per_tok;      // 8
    int inter = cfg.moe_intermediate_size;    // 512

    // Download hidden once (host gather source).
    std::vector<bf16> h_h((size_t)S * H);
    q.memcpy(h_h.data(), hidden, (size_t)S * H * sizeof(bf16)).wait();

    // Group (token, slot) assignments by expert.
    std::vector<std::vector<std::pair<int,int>>> by_expert(E);
    for (int t = 0; t < S; ++t)
        for (int s = 0; s < top_k; ++s)
            by_expert[idx[(size_t)t * top_k + s]].push_back({t, s});

    for (int e = 0; e < E; ++e) {
        const auto& lst = by_expert[e];
        int M = (int)lst.size();
        if (M == 0) continue;

        // Host gather: hidden[t] -> sub[i] for each assigned token.
        std::vector<bf16> sub((size_t)M * H);
        for (int i = 0; i < M; ++i) {
            int t = lst[i].first;
            std::memcpy(sub.data() + (size_t)i * H,
                        h_h.data() + (size_t)t * H, (size_t)H * sizeof(bf16));
        }
        GpuBuffer<bf16> d_sub((size_t)M * H, q);
        d_sub.upload(sub.data(), (size_t)M * H);

        GpuBuffer<bf16> d_dn((size_t)M * H, q);
        if (w.grouped.ready()) {
            // AWQ INT4: per-expert oneDNN u4zp matmuls over expert slices of
            // the canonical contiguous storage (separate gate/up + SwiGLU).
            const auto& g = w.grouped;
            const int gs = g.group_size;
            const int gH = H / gs, gI = inter / gs;
            GpuBuffer<bf16> d_gate((size_t)M * inter, q);
            GpuBuffer<bf16> d_up((size_t)M * inter, q);
            qwen_int4_matmul_u4zp(
                ctx, d_sub.data(), M, H, inter, gs,
                g.gate_q.data() + (size_t)e * inter * (H / 2),
                g.gate_s.data() + (size_t)e * gH * inter,
                g.gate_has_zp() ? g.gate_zp.data() + (size_t)e * gH * (inter / 2)
                                : nullptr,
                d_gate.data());
            qwen_int4_matmul_u4zp(
                ctx, d_sub.data(), M, H, inter, gs,
                g.up_q.data() + (size_t)e * inter * (H / 2),
                g.up_s.data() + (size_t)e * gH * inter,
                g.up_has_zp() ? g.up_zp.data() + (size_t)e * gH * (inter / 2)
                              : nullptr,
                d_up.data());
            swiglu_inplace(q, d_gate.data(), d_up.data(), M * inter);
            qwen_int4_matmul_u4zp(
                ctx, d_gate.data(), M, inter, H, gs,
                g.down_q.data() + (size_t)e * H * (inter / 2),
                g.down_s.data() + (size_t)e * gI * H,
                g.down_has_zp() ? g.down_zp.data() + (size_t)e * gI * (H / 2)
                                : nullptr,
                d_dn.data());
        } else {
            // gate_up: [M, 2*inter]  (gate in [0,inter), up in [inter,2*inter))
            GpuBuffer<bf16> d_gu((size_t)M * 2 * inter, q);
            qwen_matmul_proj(d_sub.data(), M, H, w.experts_gate_up[e], d_gu.data(), ctx);
            // SwiGLU -> [M, inter]
            GpuBuffer<bf16> d_act((size_t)M * inter, q);
            swiglu_strided(q, d_gu.data(), d_act.data(), M, inter);
            // down -> [M, H]
            qwen_matmul_proj(d_act.data(), M, inter, w.experts_down[e], d_dn.data(), ctx);
        }

        std::vector<bf16> dn_h((size_t)M * H);
        d_dn.download(dn_h.data(), (size_t)M * H);

        // Host fp32 scatter-add, scaled by the (renormed) routing weight.
        for (int i = 0; i < M; ++i) {
            int t = lst[i].first;
            int s = lst[i].second;
            float wv = wgt[(size_t)t * top_k + s];
            const bf16* src = dn_h.data() + (size_t)i * H;
            float* dst = out_h.data() + (size_t)t * H;
            for (int d = 0; d < H; ++d)
                dst[d] += bf16_to_float(src[d]) * wv;
        }
    }
}

// Device-router scratch (scores + idx/wgt) and shared-expert scratch,
// cached grow-only: the MoE runs once per layer per batch, so per-call USM
// allocation (and the q.wait() needed to free it safely) would serialize the
// pipeline. A mutex serializes host-side reuse across sessions; device-side
// reuse is ordered by the in-order queue.
namespace qwen_moe_router_detail {
struct Scratch {
    GpuBuffer<bf16>    scores;  // [S, E]
    GpuBuffer<int32_t> idx;     // [S*top_k]
    GpuBuffer<float>   wgt;     // [S*top_k]
    GpuBuffer<bf16>    sgu;     // [S, 2*inter]  shared gate/up
    GpuBuffer<bf16>    sact;    // [S, inter]
    GpuBuffer<bf16>    sdn;     // [S, H]
    GpuBuffer<bf16>    glogit;  // [S]
    size_t scores_cap = 0;
    size_t pairs_cap = 0;
    size_t shared_cap = 0;      // rows (S)
};
inline Scratch& scratch()
{
    static Scratch s;
    return s;
}
inline std::mutex& scratch_mutex()
{
    static std::mutex m;
    return m;
}
} // namespace qwen_moe_router_detail

// Grouped DPAS INT4 routed-experts path (AWQ checkpoints, w.grouped.ready()).
// Everything runs on device: router top-k (caller), route build, fused
// gate/up+SwiGLU (routing weight folded in), down, and the top-k combine
// into `out`. idx/wgt are DEVICE [S*top_k] buffers from qwen_moe_router_topk.
// Scratch buffers are cached across calls (grow-only) — the MoE runs once
// per layer per token, so per-call USM allocation would dominate launch
// overhead; a mutex serializes host-side reuse across sessions.
namespace qwen_moe_dpas_detail {
struct Scratch {
    GpuBuffer<int32_t> offsets;
    GpuBuffer<int32_t> tokens;
    GpuBuffer<int32_t> active;
    GpuBuffer<int32_t> count;
    GpuBuffer<bf16>    inter;
    GpuBuffer<bf16>    pair_out;
    size_t pairs_cap = 0;
    int    experts_cap = 0;
};
inline Scratch& scratch()
{
    static Scratch s;
    return s;
}
inline std::mutex& scratch_mutex()
{
    static std::mutex m;
    return m;
}
} // namespace qwen_moe_dpas_detail

inline void qwen_routed_experts_forward_dpas(
    GpuEngine& ctx, const QwenMoE& w,
    const bf16* hidden,                 // device [S, H]
    const int32_t* idx,                 // device [S*top_k]  expert ids
    const float* wgt,                   // device [S*top_k]  renormed weights
    bf16* out,                          // device [S, H]  (output)
    int S, const QwenConfig& cfg)
{
    auto& q = ctx.queue;
    int H     = cfg.hidden_size;
    int E     = cfg.num_experts;
    int top_k = cfg.num_experts_per_tok;
    int inter = cfg.moe_intermediate_size;
    int pairs = S * top_k;

    std::lock_guard<std::mutex> lock(qwen_moe_dpas_detail::scratch_mutex());
    auto& sc = qwen_moe_dpas_detail::scratch();
    if (sc.pairs_cap < (size_t)pairs) {
        sc.tokens   = GpuBuffer<int32_t>((size_t)pairs, q);
        sc.inter    = GpuBuffer<bf16>((size_t)pairs * inter, q);
        sc.pair_out = GpuBuffer<bf16>((size_t)pairs * H, q);
        sc.pairs_cap = (size_t)pairs;
    }
    if (sc.experts_cap < E) {
        sc.offsets = GpuBuffer<int32_t>((size_t)E + 1, q);
        sc.active  = GpuBuffer<int32_t>((size_t)E, q);
        sc.count   = GpuBuffer<int32_t>(1, q);
        sc.experts_cap = E;
    }

    qwen_int4_grouped_build_routes(q, idx, E, pairs,
                                   sc.offsets.data(), sc.tokens.data());

    // Compact the active-expert list on device and launch the GEMMs over it,
    // sliced for the DPAS wedge: the grid is bounded by min(E, pairs) and
    // work-groups past *count exit immediately — no idle-expert work-groups,
    // no host readback. At decode sizes this is the difference between ~30
    // mostly-idle submissions and ~5 total.
    qwen_moe_compact_active(q, idx, pairs, E, sc.active.data(), sc.count.data());
    const int max_active = std::min(E, pairs);

    const auto& g = w.grouped;
    qwen_int4_grouped_dpas_gateup_swiglu(
        q, hidden, H, g.gate_q.data(), g.up_q.data(), g.gate_s.data(),
        g.up_s.data(), g.gate_has_zp() ? g.gate_zp.data() : nullptr,
        g.up_has_zp() ? g.up_zp.data() : nullptr, sc.offsets.data(),
        sc.tokens.data(), wgt, E, pairs, top_k, inter,
        cfg.quant_group_size, sc.inter.data(), sc.active.data(),
        sc.count.data(), max_active);

    qwen_int4_grouped_dpas_down(
        q, sc.inter.data(), inter, g.down_q.data(), g.down_s.data(),
        g.down_has_zp() ? g.down_zp.data() : nullptr, sc.offsets.data(),
        sc.tokens.data(), E, pairs, H, cfg.quant_group_size,
        sc.pair_out.data(), sc.active.data(), sc.count.data(), max_active);

    // Top-k combine, weights pre-folded (fp32 accumulate, one bf16 rounding).
    qwen_int4_grouped_combine(q, sc.pair_out.data(), S, top_k, H, out);
}

// Full MoE block forward. `out` is the caller-allocated device [S, H] buffer;
// written with routed + shared.
inline void qwen_moe_forward(
    GpuEngine& ctx, const QwenMoE& w,
    const bf16* hidden,                // device [S, H]  (post-attention-normed)
    bf16* out,                         // device [S, H]  (output)
    int S, const QwenConfig& cfg)
{
    auto& q = ctx.queue;
    int H     = cfg.hidden_size;              // 2048
    int E     = cfg.num_experts;              // 256
    int top_k = cfg.num_experts_per_tok;      // 8
    int inter = cfg.moe_intermediate_size;    // 512

    // ---- 1. Router: hidden @ router_gate.T -> [S, E] (device) ----
    const int pairs = S * top_k;
    std::lock_guard<std::mutex> lock(qwen_moe_router_detail::scratch_mutex());
    auto& rsc = qwen_moe_router_detail::scratch();
    if (rsc.scores_cap < (size_t)S * E) {
        rsc.scores = GpuBuffer<bf16>((size_t)S * E, q);
        rsc.scores_cap = (size_t)S * E;
    }
    if (rsc.pairs_cap < (size_t)pairs) {
        rsc.idx = GpuBuffer<int32_t>((size_t)pairs, q);
        rsc.wgt = GpuBuffer<float>((size_t)pairs, q);
        rsc.pairs_cap = (size_t)pairs;
    }
    matmul_bf16(hidden, S, H, w.router_gate.data(), E, rsc.scores.data(), ctx);

    // ---- 2. Routed experts -> `out` ----
    const QwenMoeInt4Impl impl = qwen_moe_int4_impl();
    const bool use_grouped = impl == kQwenMoeImplGrouped && w.grouped.ready() &&
                             pairs >= qwen_moe_grouped_min_pairs();
    const bool use_dpas = !use_grouped && impl != kQwenMoeImplOnednn &&
                          w.grouped.ready();
    if (use_grouped || use_dpas) {
        // Device routing: softmax + top-k + renorm in one kernel, idx/wgt
        // stay on device (no download, no pipeline sync).
        qwen_moe_router_topk(q, rsc.scores.data(), S, E, top_k,
                             rsc.idx.data(), rsc.wgt.data());
        if (use_grouped)
            qwen_routed_experts_forward_grouped(ctx, w.grouped, hidden,
                                                rsc.idx.data(), rsc.wgt.data(),
                                                out, S, top_k);
        else
            qwen_routed_experts_forward_dpas(ctx, w, hidden,
                                             rsc.idx.data(), rsc.wgt.data(),
                                             out, S, cfg);
    } else {
        // Host-orchestrated per-expert reference path (oneDNN / dense /
        // NVFP4): host softmax + top-k + renorm on the downloaded scores.
        q.wait();
        std::vector<bf16> scores_h((size_t)S * E);
        q.memcpy(scores_h.data(), rsc.scores.data(), (size_t)S * E * sizeof(bf16)).wait();

        std::vector<int>   idx((size_t)pairs);
        std::vector<float> wgt((size_t)pairs);
        std::vector<int>   order(E);
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

        std::vector<float> out_h((size_t)S * H, 0.0f);
        qwen_routed_experts_forward(ctx, w, hidden, idx, wgt, out_h, S, cfg);

        // Upload routed (bf16) into `out`.
        std::vector<bf16> out_b((size_t)S * H);
        for (size_t i = 0; i < (size_t)S * H; ++i) out_b[i] = float_to_bf16(out_h[i]);
        q.memcpy(out, out_b.data(), (size_t)S * H * sizeof(bf16));
    }

    // ---- 3. Shared expert (device, cached scratch — no per-layer free) ----
    if (rsc.shared_cap < (size_t)S) {
        rsc.sgu    = GpuBuffer<bf16>((size_t)S * 2 * inter, q);
        rsc.sact   = GpuBuffer<bf16>((size_t)S * inter, q);
        rsc.sdn    = GpuBuffer<bf16>((size_t)S * H, q);
        rsc.glogit = GpuBuffer<bf16>((size_t)S, q);
        rsc.shared_cap = (size_t)S;
    }
    qwen_matmul_proj(hidden, S, H, w.shared_gate_up, rsc.sgu.data(), ctx);
    swiglu_strided(q, rsc.sgu.data(), rsc.sact.data(), S, inter);
    qwen_matmul_proj(rsc.sact.data(), S, inter, w.shared_down, rsc.sdn.data(), ctx);

    // shared_expert_gate: hidden @ gate.T -> [S,1]; scale shared_out by sigmoid.
    matmul_bf16(hidden, S, H, w.shared_expert_gate.data(), 1, rsc.glogit.data(), ctx);
    scale_rows_by_sigmoid(q, rsc.sdn.data(), rsc.glogit.data(), S, H);

    // ---- 4. Combine: out (routed) += shared ----
    add_inplace(q, out, rsc.sdn.data(), (size_t)S * H);
}

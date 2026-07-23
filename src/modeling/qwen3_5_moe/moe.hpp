#pragma once
//
// Qwen3.5-MoE block forward (Qwen3_5MoeSparseMoeBlock). Reference:
//   reference/transformers/.../modeling_qwen3_5_moe.py lines 798-817.
//
// Flow (per token, hidden already post-attention-normed):
//   1. Router:  scores = hidden @ router_gate.T            -> [S, 256]
//               probs  = softmax(scores, fp32)             -> [S, 256]
//               top8   = topk(probs); renorm /= sum(top8)   -> idx[S,8], wgt[S,8]
//               (NO per-expert scale — unlike diffusion_gemma.)
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
// Routing + the routed-expert gather/scatter run host-orchestrated (download
// hidden + scores, host softmax/topk/renorm, host gather, device NVFP4 GEMM per
// active expert, host fp32 scatter-add, re-upload). This is the correctness /
// decode path: for decode (S==1) it is ~8 tiny GEMMs and trivially cheap, and it
// needs no new device gather/scatter/atomic kernels. The shared-expert path and
// the final combine run fully on-device reusing existing primitives. A
// prefill-optimised device grouped-GEMM routed path can later replace
// qwen_routed_experts_forward behind an env var (AB test); the swap point is the
// single function below.
//
// NVFP4 expert projections reuse matmul_nvfp4 (handles its own input packing).
// All routed+shared expert weights are NVFP4; router_gate and shared_expert_gate
// are BF16. Numerical validation vs HF is Phase 6 (deferred).
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/nvfp4.hpp"               // matmul_nvfp4, Nvfp4Linear
#include "../../common/gpu/ops.hpp"                // matmul_bf16
#include "../../common/kernels/elementwise.hpp"    // add_inplace

#include "config.hpp"
#include "weights.hpp"   // QwenMoE
#include "kernels.hpp"    // swiglu_strided, scale_rows_by_sigmoid

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

        // gate_up: [M, 2*inter]  (gate in [0,inter), up in [inter,2*inter))
        GpuBuffer<bf16> d_gu((size_t)M * 2 * inter, q);
        matmul_nvfp4(d_sub.data(), M, H, w.experts_gate_up[e], d_gu.data(), ctx);
        // SwiGLU -> [M, inter]
        GpuBuffer<bf16> d_act((size_t)M * inter, q);
        swiglu_strided(q, d_gu.data(), d_act.data(), M, inter);
        // down -> [M, H]
        GpuBuffer<bf16> d_dn((size_t)M * H, q);
        matmul_nvfp4(d_act.data(), M, inter, w.experts_down[e], d_dn.data(), ctx);

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

    // ---- 1. Router: hidden @ router_gate.T -> [S, E]; host softmax+topk+renorm ----
    GpuBuffer<bf16> scores((size_t)S * E, q);
    matmul_bf16(hidden, S, H, w.router_gate.data(), E, scores.data(), ctx);
    q.wait();
    std::vector<bf16> scores_h((size_t)S * E);
    q.memcpy(scores_h.data(), scores.data(), (size_t)S * E * sizeof(bf16)).wait();

    std::vector<int>   idx((size_t)S * top_k);
    std::vector<float> wgt((size_t)S * top_k);
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

    // ---- 2. Routed experts (host-orchestrated) -> fp32 accumulator ----
    std::vector<float> out_h((size_t)S * H, 0.0f);
    qwen_routed_experts_forward(ctx, w, hidden, idx, wgt, out_h, S, cfg);

    // Upload routed (bf16) into `out`.
    std::vector<bf16> out_b((size_t)S * H);
    for (size_t i = 0; i < (size_t)S * H; ++i) out_b[i] = float_to_bf16(out_h[i]);
    q.memcpy(out, out_b.data(), (size_t)S * H * sizeof(bf16));

    // ---- 3. Shared expert (device) ----
    GpuBuffer<bf16> sgu((size_t)S * 2 * inter, q);
    matmul_nvfp4(hidden, S, H, w.shared_gate_up, sgu.data(), ctx);
    GpuBuffer<bf16> sact((size_t)S * inter, q);
    swiglu_strided(q, sgu.data(), sact.data(), S, inter);
    GpuBuffer<bf16> sdn((size_t)S * H, q);
    matmul_nvfp4(sact.data(), S, inter, w.shared_down, sdn.data(), ctx);

    // shared_expert_gate: hidden @ gate.T -> [S,1]; scale shared_out by sigmoid.
    GpuBuffer<bf16> glogit((size_t)S, q);
    matmul_bf16(hidden, S, H, w.shared_expert_gate.data(), 1, glogit.data(), ctx);
    scale_rows_by_sigmoid(q, sdn.data(), glogit.data(), S, H);

    // ---- 4. Combine: out (routed) += shared ----
    add_inplace(q, out, sdn.data(), (size_t)S * H);
    q.wait();
}

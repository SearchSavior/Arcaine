#pragma once
// Device-resident MoE router for Qwen3.5-MoE (Qwen3_5MoeSparseMoeBlock
// routing, reference modeling_qwen3_5_moe.py lines 800-806):
//   probs = softmax(hidden @ router_gate.T)        (fp32, max-subtracted)
//   top_k = argtopk(probs); wgt = probs[top_k]; wgt /= sum(wgt)
// Previously this ran on the host per layer per batch (scores download ->
// host softmax/top-k/renorm -> idx/wgt upload), a full pipeline sync per MoE
// layer. These kernels keep routing entirely on device: qwen_moe_router_topk
// consumes the router-matmul scores in place and writes idx/wgt straight into
// the buffers the routed-expert paths (grouped oneDNN / DPAS) consume, and
// qwen_moe_compact_active builds the dense active-expert list + count the
// DPAS GEMMs slice over (read back on device, never on host).
//
// Numerics match the host reference: bf16->fp32 logits, max-subtracted exp,
// fp32 softmax, renorm over the selected top-k weights. Ties (exactly equal
// fp32 probs) break toward the lower expert index; the host reference's
// partial_sort tie order is unspecified, and exact ties do not occur with
// real router weights.
#include <cstdint>
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "runtime/gpu/buffer.hpp"

// Model-local namespace: these kernels are per-model COPIES (see
// AGENTS.md model isolation). Global-scope inline functions with
// identical names in other models would ODR-merge at link time;
// divergent bodies (e.g. tiled attention) then crash at runtime.
namespace qwen35moe_kernels {

// Per-token softmax + top-k + renorm. One 256-lane work-group per token,
// probs staged in SLM. scores: device [S, E] bf16 router logits.
// idx_out: device [S*top_k] int32 expert ids; wgt_out: device [S*top_k] fp32
// renormed weights (top-k weights sum to 1 per token).
inline void qwen_moe_router_topk(
    sycl::queue& q, const bf16* scores, int S, int E, int top_k,
    int32_t* idx_out, float* wgt_out)
{
    if (S <= 0 || E <= 0 || top_k <= 0) return;
    constexpr int WG = 256;
    if (top_k > WG)
        throw std::runtime_error("qwen_moe_router_topk: top_k exceeds work-group size");
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> probs((size_t)E, h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)S * WG, (size_t)WG),
            [=](sycl::nd_item<1> it) {
                const int t   = (int)it.get_group(0);
                const int lid = (int)it.get_local_id(0);
                auto g = it.get_group();
                const bf16* row = scores + (size_t)t * E;

                // fp32 softmax into SLM.
                float v = -3.402823466e38f;
                for (int e = lid; e < E; e += WG)
                    v = sycl::fmax(v, bf16_to_float(row[e]));
                const float mx =
                    sycl::reduce_over_group(g, v, sycl::maximum<float>());
                float psum = 0.0f;
                for (int e = lid; e < E; e += WG) {
                    float p = sycl::exp(bf16_to_float(row[e]) - mx);
                    probs[e] = p;
                    psum += p;
                }
                const float sum =
                    sycl::reduce_over_group(g, psum, sycl::plus<float>());
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int e = lid; e < E; e += WG) probs[e] *= inv;
                it.barrier();

                // Iterative argmax x top_k. Slot e is only ever read/written
                // by lane e % WG, so the -inf knock-out needs no barrier.
                int   sel_i = -1;
                float sel_v = 0.0f;
                float ssum  = 0.0f;
                for (int s = 0; s < top_k; ++s) {
                    float bv = -3.402823466e38f;
                    int   bi = 0x7fffffff;
                    for (int e = lid; e < E; e += WG) {
                        float p = probs[e];
                        if (p > bv) { bv = p; bi = e; }
                    }
                    const float wv0 =
                        sycl::reduce_over_group(g, bv, sycl::maximum<float>());
                    const int cand = (bv == wv0) ? bi : 0x7fffffff;
                    const int wi0 =
                        sycl::reduce_over_group(g, cand, sycl::minimum<int>());
                    // Degenerate row (e.g. all-NaN probs from inf/NaN logits):
                    // wi0 stays INT_MAX. Clamp to expert 0 with weight 0 —
                    // never index SLM or idx_out out of bounds.
                    const int   wi = wi0 < E ? wi0 : 0;
                    const float wv = wi0 < E ? wv0 : 0.0f;
                    if (lid == (wi & (WG - 1))) probs[wi] = -3.402823466e38f;
                    ssum += wv;
                    if (lid == s) { sel_i = wi; sel_v = wv; }
                }
                const float sinv = ssum > 0.0f ? 1.0f / ssum : 0.0f;
                if (lid < top_k) {
                    idx_out[(size_t)t * top_k + lid] = sel_i;
                    wgt_out[(size_t)t * top_k + lid] = sel_v * sinv;
                }
            });
    });
}

// Dense list of experts referenced by idx (device [pairs]), plus a
// device-side count — feeds the DPAS GEMMs' expert slicing so they launch
// min(E, pairs) expert tiles worst-case and early-exit past *count, with no
// host readback. `active` must hold E entries; order is unspecified.
inline void qwen_moe_compact_active(
    sycl::queue& q, const int32_t* idx, int pairs, int E,
    int32_t* active, int32_t* count)
{
    if (pairs <= 0 || E <= 0) {
        q.memset(count, 0, sizeof(int32_t));
        return;
    }
    constexpr int WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t> cnt(1, h);
        h.parallel_for(
            sycl::nd_range<1>(WG, WG),
            [=](sycl::nd_item<1> it) {
                const int lid = (int)it.get_local_id(0);
                if (lid == 0) cnt[0] = 0;
                it.barrier();
                for (int e = lid; e < E; e += WG) {
                    bool found = false;
                    for (int p = 0; p < pairs; ++p) {
                        if (idx[p] == e) { found = true; break; }
                    }
                    if (found) {
                        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            c(cnt[0]);
                        active[c.fetch_add(1)] = e;
                    }
                }
                it.barrier();
                if (lid == 0) count[0] = cnt[0];
            });
    });
}

} // namespace qwen35moe_kernels

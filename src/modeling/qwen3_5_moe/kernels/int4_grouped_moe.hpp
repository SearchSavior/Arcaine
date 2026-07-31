#pragma once
// Native Xe2/BMG grouped W4A16 MoE kernels for the Qwen3.5-MoE AWQ INT4
// checkpoint (compressed-tensors pack-quantized, group_size=32, asymmetric
// zero-points). Port of modeling/diffusion_gemma/kernels/int4_grouped_moe.hpp
// (itself after reference/llm-scaler .../moe_batch/int4_nmajor_gemm.h) with
// model-specific changes and a redesign informed by the llm-scaler survey
// (reports/llm-scaler-grouped-moe.md):
//
//   * SwiGLU epilogue (silu(gate) * up) instead of DiffusionGemma's GeGLU;
//   * exact per-element asymmetric zero-point handling. The packed weights
//     are two's-complement s4 (q_u - 8, GPU XOR 0x88 at load) but true
//     dequant is scale * (q_u - zp_u) = scale * q_s4 - zp_offset with
//     zp_offset[g,n] = scale[g,n] * (zp_u[g,n] - 8) precomputed by the
//     loader. The DPAS B operand is bf16(scale * q_s4 - zp_offset) directly
//     (zp_offset is constant across a group's 32 elements, so it costs one
//     load + one subtract per element), matching the oneDNN path's math
//     without a rowsum + correction pass;
//   * Qwen top-k pair indexing: pair p = token * top_k + slot;
//   * 64-lane work-groups (4 subgroups of 16), one 16-wide output tile per
//     subgroup — 4x fewer work-groups than one-subgroup WGs (better XeCore
//     occupancy on BMG-G31's 32 XeCores). No SLM staging of GEMM operands:
//     llm-scaler's in-tree evidence is that register/GRF dataflow straight
//     from L2 beats SLM staging for these tile sizes;
//   * single-launch device route builder (SLM histogram + serial prefix +
//     SLM-cursor scatter in one 256-lane work-group, after
//     moe_prefill_int4.sycl's gather kernel) instead of memset + 3 submits;
//   * routing weight folded into the gate/up epilogue
//     (intermediate = rw * silu(g) * u), after llm-scaler's decode kernels
//     which fold rw into the activation before the MAC — the top-k combine
//     is then a plain sum with no weight array;
//   * min()-clamped M tails (duplicate last row into the DPAS tile, masked
//     on store) and hoisted per-tile pair/token loads (no integer division
//     in the K loop);
//   * packed weight rows read as one aligned uint64 per K-tile instead of
//     eight byte loads.
//
// Layout (identical to Int4Linear): packed s4 rows [N, K/2] low-nibble first
// along K, scales BF16 [G, N], zp_offset BF16 [G, N] (nullptr per expert when
// that expert is symmetric, i.e. zp_u == 8 everywhere). Names are
// qwen_-prefixed: kernel_bench links several models into one binary and the
// DiffusionGemma originals are inline with external linkage.
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "runtime/quantization/int4.hpp"
#include "runtime/quantization/q8_0.hpp" // DPAS builtin declaration / vector operand types.

static constexpr int kQwenInt4GroupedDpasBF16 = 0x3000;

// Expert-slice size for the GEMM launches below. The SPIR-V
// SubgroupMatrixMultiplyAccumulateINTEL path wedges the device (work-items
// never retire) once more than ~16K concurrent work-items execute DPAS —
// reproduced with synthetic weights, independent of WG size (16/64), load
// vectorization, and hoisting; the identical kernel with scalar MACs runs
// clean at all sizes. BMG-G31 has no joint_matrix aspect, so this builtin is
// a thinly-tested IGC path (llm-scaler ships ESIMD xmx::dpas instead).
// Launching the expert dimension in slices (in-order queue serializes them)
// caps concurrent DPAS work far below the wedge threshold at negligible
// cost. 0 disables slicing. The GEMMs launch over the compacted active-
// expert list (`active_experts`, device buffer, host-computed from the
// top-k idx the caller already holds), sliced in chunks of `estep` — no
// idle-expert work-groups and the minimum number of submissions.
inline int qwen_int4_dpas_expert_slice()
{
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("QWEN35_MOE_INT4_DPAS_SLICE");
        cached = v ? std::atoi(v) : 16;
        if (cached < 0) cached = 16;
    }
    return cached;
}

// Compact expert-major route list built entirely on device in a single
// launch. One 256-lane work-group: SLM histogram, serial prefix on lane 0,
// SLM-cursor scatter. offsets has local_experts+1 entries (global); tokens
// holds the original top-k pair index. No host count/download boundary.
inline void qwen_int4_grouped_build_routes(
    sycl::queue& q, const int* expert_idx,
    int local_experts, int pairs, int32_t* offsets, int32_t* tokens)
{
    if (pairs <= 0) {
        q.memset(offsets, 0, ((size_t)local_experts + 1) * sizeof(int32_t));
        return;
    }
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t> slm_counts(
            sycl::range<1>((size_t)local_experts), h);
        sycl::local_accessor<int32_t> slm_off(
            sycl::range<1>((size_t)local_experts + 1), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) {
                int wi = (int)it.get_local_id(0);
                int nw = (int)it.get_local_range(0);
                for (int e = wi; e < local_experts; e += nw) slm_counts[e] = 0;
                it.barrier();
                for (int p = wi; p < pairs; p += nw) {
                    int e = expert_idx[p];
                    if (e < 0 || e >= local_experts) continue;
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        count(slm_counts[e]);
                    count.fetch_add(1);
                }
                it.barrier();
                if (wi == 0) {
                    int32_t running = 0;
                    for (int e = 0; e < local_experts; ++e) {
                        int32_t c = slm_counts[e];
                        slm_off[e] = running;
                        slm_counts[e] = running; // reuse as scatter cursors
                        running += c;
                    }
                    slm_off[local_experts] = running;
                }
                it.barrier();
                for (int e = wi; e <= local_experts; e += nw)
                    offsets[e] = slm_off[e];
                for (int p = wi; p < pairs; p += nw) {
                    int e = expert_idx[p];
                    if (e < 0 || e >= local_experts) continue;
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        cursor(slm_counts[e]);
                    tokens[cursor.fetch_add(1)] = p;
                }
            });
    });
}

// Fused gate/up W4A16 grouped GEMM + SwiGLU + routing-weight fold.
// One 64-lane work-group (4 subgroups) covers one local expert and 64
// intermediate channels (16 per subgroup), looping over its compacted pair
// list in eight-row DPAS tiles. `gate_w` points at raw packed
// [2*inter, hidden/2] rows (gate rows [0,inter), up rows [inter,2*inter));
// `gate_s`/`gate_zp` at [hidden/group, 2*inter]. Writes
// intermediate[pair * inter + n] = pair_wgt[pair] * silu(gate) * up.
inline void qwen_int4_grouped_dpas_gateup_swiglu(
    sycl::queue& q,
    const bf16* input, int hidden,
    const uint8_t* const* gate_w, const bf16* const* gate_s,
    const bf16* const* gate_zp,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    const float* pair_wgt,
    int local_experts, int pairs, int top_k, int inter, int group_size,
    bf16* intermediate, const int32_t* active_experts, int num_active)
{
    if (hidden % 16 || inter % 64 || hidden % group_size || group_size != 32)
        throw std::runtime_error("qwen grouped INT4 DPAS requires H % 16, I % 64 and AWQ group_size=32");
    if (pairs <= 0 || num_active <= 0) return;
    const int slice = qwen_int4_dpas_expert_slice();
    const int estep = slice > 0 ? slice : num_active;
    for (int a0 = 0; a0 < num_active; a0 += estep) {
        const int ec = std::min(estep, num_active - a0);
        q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)ec, (size_t)inter),
                              sycl::range<2>(1, 64)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = active_experts[a0 + (int)it.get_group(0)];
                int lane = (int)(it.get_local_id(1) & 15);
                int n = (int)it.get_global_id(1);
                int t0 = expert_offsets[e];
                int t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;

                const uint8_t* w = gate_w[e];
                const bf16* s = gate_s[e];
                const bf16* zp = gate_zp[e];
                const int packed_k = hidden / 2;
                const int out_features = 2 * inter;

                for (int m0 = t0; m0 < t1; m0 += 8) {
                    // Hoisted per-tile pair/token ids (min-clamped tail:
                    // duplicate the last row, masked on store below).
                    int pair[8], tok[8];
                    for (int m = 0; m < 8; ++m) {
                        int pos = m0 + m < t1 ? m0 + m : t1 - 1;
                        pair[m] = expert_tokens[pos];
                        tok[m] = pair[m] / top_k;
                    }
                    diff_dpas_v8f cg = {0,0,0,0,0,0,0,0};
                    diff_dpas_v8f cu = {0,0,0,0,0,0,0,0};
                    for (int k0 = 0; k0 < hidden; k0 += 16) {
                        int kg = k0 / group_size;
                        float sg = bf16_to_float(s[(size_t)kg * out_features + n]);
                        float su = bf16_to_float(s[(size_t)kg * out_features + inter + n]);
                        float zg = zp ? bf16_to_float(zp[(size_t)kg * out_features + n]) : 0.0f;
                        float zu = zp ? bf16_to_float(zp[(size_t)kg * out_features + inter + n]) : 0.0f;
                        uint64_t gw, uw;
                        std::memcpy(&gw, w + (size_t)n * packed_k + k0 / 2, 8);
                        std::memcpy(&uw, w + (size_t)(inter + n) * packed_k + k0 / 2, 8);
                        diff_dpas_v8i bg, bu;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t gbyte = (uint8_t)(gw >> (8 * j));
                            uint8_t ubyte = (uint8_t)(uw >> (8 * j));
                            int g0 = (int)(gbyte & 0x0f); if (g0 >= 8) g0 -= 16;
                            int g1 = (int)(gbyte >> 4);   if (g1 >= 8) g1 -= 16;
                            int u0 = (int)(ubyte & 0x0f); if (u0 >= 8) u0 -= 16;
                            int u1 = (int)(ubyte >> 4);   if (u1 >= 8) u1 -= 16;
                            uint16_t g0b = float_to_bf16((float)g0 * sg - zg);
                            uint16_t g1b = float_to_bf16((float)g1 * sg - zg);
                            uint16_t u0b = float_to_bf16((float)u0 * su - zu);
                            uint16_t u1b = float_to_bf16((float)u1 * su - zu);
                            bg[j] = (int)((uint32_t)g0b | ((uint32_t)g1b << 16));
                            bu[j] = (int)((uint32_t)u0b | ((uint32_t)u1b << 16));
                        }
                        diff_dpas_v8s av;
                        int kk = k0 + lane;
                        for (int m = 0; m < 8; ++m)
                            av[m] = (short)input[(size_t)tok[m] * hidden + kk];
                        cg = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bg, cg, kQwenInt4GroupedDpasBF16);
                        cu = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bu, cu, kQwenInt4GroupedDpasBF16);
                    }
                    for (int m = 0; m < 8 && m0 + m < t1; ++m) {
                        float g = cg[m], u = cu[m];
                        float rw = pair_wgt[pair[m]];
                        intermediate[(size_t)pair[m] * inter + n] = float_to_bf16(
                            rw * (g / (1.0f + sycl::exp(-g))) * u);
                    }
                }
            });
        });
    }
}

// Grouped W4A16 down projection with per-element zp folding. Rows keep the
// original pair index; routing weights are already folded into the
// intermediate, so the top-k combine is a plain sum.
inline void qwen_int4_grouped_dpas_down(
    sycl::queue& q,
    const bf16* intermediate, int inter,
    const uint8_t* const* down_w, const bf16* const* down_s,
    const bf16* const* down_zp,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    int local_experts, int pairs, int hidden, int group_size,
    bf16* output, const int32_t* active_experts, int num_active)
{
    if (hidden % 64 || inter % 16 || inter % group_size || group_size != 32)
        throw std::runtime_error("qwen grouped INT4 DPAS requires H % 64, I % 16 and AWQ group_size=32");
    if (pairs <= 0 || num_active <= 0) return;
    const int slice = qwen_int4_dpas_expert_slice();
    const int estep = slice > 0 ? slice : num_active;
    for (int a0 = 0; a0 < num_active; a0 += estep) {
        const int ec = std::min(estep, num_active - a0);
        q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)ec, (size_t)hidden),
                              sycl::range<2>(1, 64)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = active_experts[a0 + (int)it.get_group(0)];
                int lane = (int)(it.get_local_id(1) & 15);
                int n = (int)it.get_global_id(1);
                int t0 = expert_offsets[e];
                int t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;

                const uint8_t* w = down_w[e];
                const bf16* s = down_s[e];
                const bf16* zp = down_zp[e];
                const int packed_k = inter / 2;
                for (int m0 = t0; m0 < t1; m0 += 8) {
                    int pair[8];
                    for (int m = 0; m < 8; ++m) {
                        int pos = m0 + m < t1 ? m0 + m : t1 - 1;
                        pair[m] = expert_tokens[pos];
                    }
                    diff_dpas_v8f c = {0,0,0,0,0,0,0,0};
                    for (int k0 = 0; k0 < inter; k0 += 16) {
                        int kg = k0 / group_size;
                        float scale = bf16_to_float(s[(size_t)kg * hidden + n]);
                        float z = zp ? bf16_to_float(zp[(size_t)kg * hidden + n]) : 0.0f;
                        uint64_t wv;
                        std::memcpy(&wv, w + (size_t)n * packed_k + k0 / 2, 8);
                        diff_dpas_v8i bv;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t byte = (uint8_t)(wv >> (8 * j));
                            int v0 = (int)(byte & 0x0f); if (v0 >= 8) v0 -= 16;
                            int v1 = (int)(byte >> 4);   if (v1 >= 8) v1 -= 16;
                            uint16_t b0 = float_to_bf16((float)v0 * scale - z);
                            uint16_t b1 = float_to_bf16((float)v1 * scale - z);
                            bv[j] = (int)((uint32_t)b0 | ((uint32_t)b1 << 16));
                        }
                        diff_dpas_v8s av;
                        int kk = k0 + lane;
                        for (int m = 0; m < 8; ++m)
                            av[m] = (short)intermediate[(size_t)pair[m] * inter + kk];
                        c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bv, c, kQwenInt4GroupedDpasBF16);
                    }
                    for (int m = 0; m < 8 && m0 + m < t1; ++m)
                        output[(size_t)pair[m] * hidden + n] = float_to_bf16(c[m]);
                }
            });
        });
    }
}

// Top-k combine over pair-indexed, routing-weight-pre-folded expert outputs:
//   out[t, d] = sum_s pair_out[(t*top_k + s), d]
// fp32 accumulation, single bf16 rounding — matches the host reference path.
inline void qwen_int4_grouped_combine(
    sycl::queue& q, const bf16* pair_out,
    int seq, int top_k, int hidden, bf16* out)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)seq, (size_t)hidden),
                       [=](sycl::id<2> id) {
            int t = (int)id[0], d = (int)id[1];
            float acc = 0.0f;
            for (int k = 0; k < top_k; ++k)
                acc += bf16_to_float(pair_out[(size_t)(t * top_k + k) * hidden + d]);
            out[(size_t)t * hidden + d] = float_to_bf16(acc);
        });
    });
}

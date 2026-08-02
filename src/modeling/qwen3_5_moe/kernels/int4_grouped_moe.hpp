#pragma once
// Native Xe2/BMG grouped W4A16 MoE kernels for the Qwen3.5-MoE AWQ INT4
// checkpoint (compressed-tensors pack-quantized, group_size=32, asymmetric
// zero-points). Port of modeling/diffusion_gemma/kernels/int4_grouped_moe.hpp
// (itself after reference/llm-scaler .../moe_batch/int4_nmajor_gemm.h) with
// model-specific changes and a redesign informed by the llm-scaler survey
// (reports/llm-scaler-grouped-moe.md):
//
//   * SwiGLU epilogue (silu(gate) * up) instead of DiffusionGemma's GeGLU;
//   * exact per-element asymmetric zero-point handling from the canonical
//     raw storage: weights stay unsigned nibbles q_u and zero points are the
//     checkpoint's original unsigned nibbles zp_u (packed along N). The DPAS
//     B operand is bf16((q_u - zp_u) * scale) directly — one exact integer
//     subtract and one bf16 rounding per element, matching the oneDNN
//     grouped path's math without a rowsum + correction pass;
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
// Layout (canonical QwenInt4ExpertsGrouped): contiguous per-projection expert
// tensors, packed u4 rows [E][N][K/2] low-nibble first along K, scales BF16
// [E][K/gs][N], zero points u4 [E][K/gs][N/2] packed along N (nullptr when
// the projection is symmetric, i.e. zp_u == 8 everywhere). Kernels take base
// pointers + compile-time-constant per-expert strides. Names are
// qwen_-prefixed: kernel_bench links several models into one binary and the
// DiffusionGemma originals are inline with external linkage.
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "runtime/gpu/buffer.hpp"
#include "runtime/quantization/q8_0.hpp" // DPAS builtin declaration / vector operand types.

// Model-local namespace: these kernels are per-model COPIES (see
// AGENTS.md model isolation). Global-scope inline functions with
// identical names in other models would ODR-merge at link time;
// divergent bodies (e.g. tiled attention) then crash at runtime.
namespace qwen35moe_kernels {

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
// expert list (`active_experts` + device `active_count`, built on device by
// qwen_moe_compact_active from the router's idx), sliced in chunks of
// `estep`. The grid is sized by the host-side upper bound
// max_active = min(E, pairs) and work-groups past *active_count exit
// immediately — no host readback, no idle-expert work beyond the bound.
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
// list in eight-row DPAS tiles. `gate_q`/`up_q` are the canonical contiguous
// packed u4 expert tensors ([E][inter][hidden/2] each), `gate_s`/`up_s` the
// [E][hidden/gs][inter] BF16 scales and `gate_zp`/`up_zp` the raw u4 zero
// points [E][hidden/gs][inter/2] (nullptr = symmetric, zp_u == 8). Writes
// intermediate[pair * inter + n] = pair_wgt[pair] * silu(gate) * up.
inline void qwen_int4_grouped_dpas_gateup_swiglu(
    sycl::queue& q,
    const bf16* input, int hidden,
    const uint8_t* gate_q, const uint8_t* up_q,
    const bf16* gate_s, const bf16* up_s,
    const uint8_t* gate_zp, const uint8_t* up_zp,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    const float* pair_wgt,
    int local_experts, int pairs, int top_k, int inter, int group_size,
    bf16* intermediate, const int32_t* active_experts,
    const int32_t* active_count, int max_active)
{
    if (hidden % 16 || inter % 64 || hidden % group_size || group_size != 32)
        throw std::runtime_error("qwen grouped INT4 DPAS requires H % 16, I % 64 and AWQ group_size=32");
    if (pairs <= 0 || max_active <= 0) return;
    const int slice = qwen_int4_dpas_expert_slice();
    const int estep = slice > 0 ? slice : max_active;
    for (int a0 = 0; a0 < max_active; a0 += estep) {
        const int ec = std::min(estep, max_active - a0);
        q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)ec, (size_t)inter),
                              sycl::range<2>(1, 64)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int a = a0 + (int)it.get_group(0);
                if (a >= *active_count) return;
                int e = active_experts[a];
                int lane = (int)(it.get_local_id(1) & 15);
                int n = (int)it.get_global_id(1);
                int t0 = expert_offsets[e];
                int t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;

                const int packed_k = hidden / 2;
                const int kg_n = hidden / group_size;   // groups along K
                const uint8_t* w_g = gate_q + (size_t)e * inter * packed_k;
                const uint8_t* w_u = up_q + (size_t)e * inter * packed_k;
                const bf16* s_g = gate_s + (size_t)e * kg_n * inter;
                const bf16* s_u = up_s + (size_t)e * kg_n * inter;
                const uint8_t* z_g = gate_zp ? gate_zp + (size_t)e * kg_n * (inter / 2) : nullptr;
                const uint8_t* z_u = up_zp ? up_zp + (size_t)e * kg_n * (inter / 2) : nullptr;

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
                        float sg = bf16_to_float(s_g[(size_t)kg * inter + n]);
                        float su = bf16_to_float(s_u[(size_t)kg * inter + n]);
                        int zg = 8, zu = 8;
                        if (z_g) {
                            uint8_t b = z_g[(size_t)kg * (inter / 2) + n / 2];
                            zg = (n & 1) ? (b >> 4) : (b & 0xF);
                        }
                        if (z_u) {
                            uint8_t b = z_u[(size_t)kg * (inter / 2) + n / 2];
                            zu = (n & 1) ? (b >> 4) : (b & 0xF);
                        }
                        uint64_t gw, uw;
                        std::memcpy(&gw, w_g + (size_t)n * packed_k + k0 / 2, 8);
                        std::memcpy(&uw, w_u + (size_t)n * packed_k + k0 / 2, 8);
                        diff_dpas_v8i bg, bu;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t gbyte = (uint8_t)(gw >> (8 * j));
                            uint8_t ubyte = (uint8_t)(uw >> (8 * j));
                            int g0 = (int)(gbyte & 0x0f) - zg;
                            int g1 = (int)(gbyte >> 4) - zg;
                            int u0 = (int)(ubyte & 0x0f) - zu;
                            int u1 = (int)(ubyte >> 4) - zu;
                            uint16_t g0b = float_to_bf16((float)g0 * sg);
                            uint16_t g1b = float_to_bf16((float)g1 * sg);
                            uint16_t u0b = float_to_bf16((float)u0 * su);
                            uint16_t u1b = float_to_bf16((float)u1 * su);
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

// Grouped W4A16 down projection with per-element native zero points. Rows
// keep the original pair index; routing weights are already folded into the
// intermediate, so the top-k combine is a plain sum. `down_q` is [E][hidden]
// [inter/2] u4, `down_s` [E][inter/gs][hidden] BF16, `down_zp`
// [E][inter/gs][hidden/2] u4 (nullptr = symmetric).
inline void qwen_int4_grouped_dpas_down(
    sycl::queue& q,
    const bf16* intermediate, int inter,
    const uint8_t* down_q, const bf16* down_s, const uint8_t* down_zp,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    int local_experts, int pairs, int hidden, int group_size,
    bf16* output, const int32_t* active_experts,
    const int32_t* active_count, int max_active)
{
    if (hidden % 64 || inter % 16 || inter % group_size || group_size != 32)
        throw std::runtime_error("qwen grouped INT4 DPAS requires H % 64, I % 16 and AWQ group_size=32");
    if (pairs <= 0 || max_active <= 0) return;
    const int slice = qwen_int4_dpas_expert_slice();
    const int estep = slice > 0 ? slice : max_active;
    for (int a0 = 0; a0 < max_active; a0 += estep) {
        const int ec = std::min(estep, max_active - a0);
        q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)ec, (size_t)hidden),
                              sycl::range<2>(1, 64)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int a = a0 + (int)it.get_group(0);
                if (a >= *active_count) return;
                int e = active_experts[a];
                int lane = (int)(it.get_local_id(1) & 15);
                int n = (int)it.get_global_id(1);
                int t0 = expert_offsets[e];
                int t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;

                const int packed_k = inter / 2;
                const int kg_n = inter / group_size;
                const uint8_t* w = down_q + (size_t)e * hidden * packed_k;
                const bf16* s = down_s + (size_t)e * kg_n * hidden;
                const uint8_t* zp = down_zp ? down_zp + (size_t)e * kg_n * (hidden / 2) : nullptr;
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
                        int z = 8;
                        if (zp) {
                            uint8_t b = zp[(size_t)kg * (hidden / 2) + n / 2];
                            z = (n & 1) ? (b >> 4) : (b & 0xF);
                        }
                        uint64_t wv;
                        std::memcpy(&wv, w + (size_t)n * packed_k + k0 / 2, 8);
                        diff_dpas_v8i bv;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t byte = (uint8_t)(wv >> (8 * j));
                            int v0 = (int)(byte & 0x0f) - z;
                            int v1 = (int)(byte >> 4) - z;
                            uint16_t b0 = float_to_bf16((float)v0 * scale);
                            uint16_t b1 = float_to_bf16((float)v1 * scale);
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

} // namespace qwen35moe_kernels

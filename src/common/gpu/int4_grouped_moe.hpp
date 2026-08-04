#pragma once
// Native Xe2/BMG grouped W4A16 MoE kernels for DiffusionGemma INT4-AWQ.
//
// The checkpoint's raw Int4Linear layout is already N-major: packed s4 rows
// are [N, K/2] (low nibble first) and scales are [K/group, N].  That lets a
// subgroup keep one output channel in a lane, construct a BF16 DPAS B tile
// directly from a packed row, and iterate its device-grouped routes.  No
// oneDNN primitive or padded per-expert activation bucket is involved.
//
// This follows the N-major DPAS structure in
// reference/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/
// int4_nmajor_gemm.h, but uses the project-wide SPIR-V DPAS builtin.  The
// latter is already used by q8_0.hpp and permits direct BF16-bit operands.
#include "int4.hpp"
#include "q8_0.hpp" // DPAS builtin declaration / vector operand types.
#include <algorithm>
#include <cstdlib>
#include <cstring>

static constexpr int kInt4GroupedDpasBF16 = 0x3000;

// This SPIR-V BF16 operand form is accepted by Xe2/Battlemage.  ACM's IGC
// requires an int32 A operand for K=16 and rejects the kernel at JIT time, so
// callers must keep it out of the A770 path.  SYCL reports Intel's GPU
// architecture version as 20.x for Xe2 and 12.x for ACM.
inline bool diff_int4_grouped_dpas_device_supported(const sycl::device& device) {
    const std::string version =
        device.get_info<sycl::info::device::version>();
    return std::atoi(version.c_str()) >= 20;
}

// Limit each DPAS submission to a bounded active-expert slice.  The raw
// SPV_INTEL subgroup-matrix path has been observed to wedge above roughly
// 16K work-items, while slicing is effectively free on an in-order queue.
// DiffusionGemma's 2816-wide down projection launches 2816 work-items per
// expert, so five experts (14080 work-items) is the largest slice below that
// threshold. Keep this independently tunable for BMG A/B work; 0 disables
// slicing.
inline int diff_int4_grouped_dpas_expert_slice() {
    static int value = [] {
        const char* env = std::getenv("DIFF_INT4_GROUPED_DPAS_SLICE");
        int v = env ? std::atoi(env) : 5;
        return v < 0 ? 5 : v;
    }();
    return value;
}

// Make a compact expert-major route list entirely on device.  offsets has
// local_experts+1 entries, and tokens contains the original top-k pair index.
// A token may appear once for each selected expert; pair/top_k gets its source
// activation row.  This deliberately has no host count/download boundary.
inline void int4_grouped_moe_build_routes(
    sycl::queue& q, const int* expert_idx, int first_expert,
    int local_experts, int pairs, int32_t* offsets, int32_t* tokens,
    int32_t* active_experts, int32_t* active_count)
{
    if (pairs <= 0) {
        q.memset(offsets, 0, ((size_t)local_experts + 1) * sizeof(int32_t));
        q.memset(active_count, 0, sizeof(int32_t));
        return;
    }
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> counts((size_t)local_experts, h);
        sycl::local_accessor<int32_t, 1> local_offsets((size_t)local_experts + 1, h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) {
                int wi = (int)it.get_local_id(0);
                int nw = (int)it.get_local_range(0);
                for (int e = wi; e < local_experts; e += nw) counts[e] = 0;
                it.barrier();

                for (int pair = wi; pair < pairs; pair += nw) {
                    int e = expert_idx[pair] - first_expert;
                    if (e < 0 || e >= local_experts) continue;
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        count(counts[e]);
                    count.fetch_add(1);
                }
                it.barrier();

                if (wi == 0) {
                    int32_t running = 0;
                    int32_t active = 0;
                    for (int e = 0; e < local_experts; ++e) {
                        int32_t count = counts[e];
                        local_offsets[e] = running;
                        counts[e] = running; // reuse as scatter cursors
                        if (count > 0) active_experts[active++] = e;
                        running += count;
                    }
                    local_offsets[local_experts] = running;
                    *active_count = active;
                }
                it.barrier();

                for (int e = wi; e <= local_experts; e += nw)
                    offsets[e] = local_offsets[e];
                for (int pair = wi; pair < pairs; pair += nw) {
                    int e = expert_idx[pair] - first_expert;
                    if (e < 0 || e >= local_experts) continue;
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        cursor(counts[e]);
                    tokens[cursor.fetch_add(1)] = pair;
                }
            }
        );
    });
}

// Fused gate/up W4A16 GEMM + DiffusionGemma's GeGLU.  One 64-lane work-group
// handles one active local expert and 64 intermediate channels (four
// 16-wide DPAS subgroups), looping over its compacted pair list in eight-row
// tiles. `gate_w` points at raw packed
// [2*inter, hidden/2] rows and `gate_s` points at [hidden/group, 2*inter].
inline void matmul_int4_grouped_dpas_gateup_geglu(
    sycl::queue& q,
    const bf16* input, int hidden,
    const uint8_t* const* gate_w, const bf16* const* gate_s,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    int local_experts, int pairs, int top_k, int inter, int group_size,
    bf16* intermediate, const int32_t* active_experts,
    const int32_t* active_count, int max_active)
{
    if (hidden % 16 || inter % 64 || hidden % group_size || group_size != 32)
        throw std::runtime_error("grouped INT4 DPAS requires H % 16, I % 64 and AWQ group_size=32");
    if (pairs <= 0 || max_active <= 0) return;
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEF = 0.044715f;
    int slice = diff_int4_grouped_dpas_expert_slice();
    int estep = slice > 0 ? slice : max_active;
    for (int a0 = 0; a0 < max_active; a0 += estep) {
        int ec = std::min(estep, max_active - a0);
        q.submit([&](sycl::handler& h) {
          h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)ec, (size_t)inter),
                              sycl::range<2>(1, 64)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int a = a0 + (int)it.get_group(0);
                if (a >= *active_count) return;
                int e = active_experts[a];
                int lane = (int)it.get_local_id(1) & 15;
                int n = (int)it.get_global_id(1);
                int t0 = expert_offsets[e];
                int t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;

                const uint8_t* w = gate_w[e];
                const bf16* s = gate_s[e];
                const int packed_k = hidden / 2;
                const int out_features = 2 * inter;

                for (int m0 = t0; m0 < t1; m0 += 8) {
                    int pair[8], token[8];
                    for (int m = 0; m < 8; ++m) {
                        int pos = m0 + m < t1 ? m0 + m : t1 - 1;
                        pair[m] = expert_tokens[pos];
                        token[m] = pair[m] / top_k;
                    }
                    diff_dpas_v8f cg = {0,0,0,0,0,0,0,0};
                    diff_dpas_v8f cu = {0,0,0,0,0,0,0,0};
                    for (int k0 = 0; k0 < hidden; k0 += 16) {
                        int kg = k0 / group_size;
                        float sg = bf16_to_float(s[(size_t)kg * out_features + n]);
                        float su = bf16_to_float(s[(size_t)kg * out_features + inter + n]);
                        uint64_t gate_word, up_word;
                        std::memcpy(&gate_word, w + (size_t)n * packed_k + k0 / 2, 8);
                        std::memcpy(&up_word, w + (size_t)(inter + n) * packed_k + k0 / 2, 8);
                        diff_dpas_v8i bg, bu;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t gbyte = (uint8_t)(gate_word >> (8 * j));
                            uint8_t ubyte = (uint8_t)(up_word >> (8 * j));
                            int g0 = (int)(gbyte & 0x0f); if (g0 >= 8) g0 -= 16;
                            int g1 = (int)(gbyte >> 4);   if (g1 >= 8) g1 -= 16;
                            int u0 = (int)(ubyte & 0x0f); if (u0 >= 8) u0 -= 16;
                            int u1 = (int)(ubyte >> 4);   if (u1 >= 8) u1 -= 16;
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
                            av[m] = (short)input[(size_t)token[m] * hidden + kk];
                        cg = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bg, cg, kInt4GroupedDpasBF16);
                        cu = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bu, cu, kInt4GroupedDpasBF16);
                    }
                    for (int m = 0; m < 8 && m0 + m < t1; ++m) {
                        // Match the original oneDNN GEMM -> BF16 buffer ->
                        // GeGLU boundary before fusing the epilogue.
                        float g = bf16_to_float(float_to_bf16(cg[m]));
                        float u = bf16_to_float(float_to_bf16(cu[m]));
                        float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
                        intermediate[(size_t)pair[m] * inter + n] = float_to_bf16(
                            0.5f * g * (1.0f + sycl::tanh(inner)) * u);
                    }
                }
            });
        });
    }
}

// Grouped W4A16 down projection.  Input/output rows use the original pair
// index, so the existing route-weighted combine (and postnorm fusion) can use
// an identity slot map without a scatter/reorder kernel.
inline void matmul_int4_grouped_dpas_down(
    sycl::queue& q,
    const bf16* intermediate, int inter,
    const uint8_t* const* down_w, const bf16* const* down_s,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    int local_experts, int pairs, int hidden, int group_size,
    bf16* output, const int32_t* active_experts,
    const int32_t* active_count, int max_active)
{
    if (hidden % 64 || inter % 16 || inter % group_size || group_size != 32)
        throw std::runtime_error("grouped INT4 DPAS requires H % 64, I % 16 and AWQ group_size=32");
    if (pairs <= 0 || max_active <= 0) return;
    int slice = diff_int4_grouped_dpas_expert_slice();
    int estep = slice > 0 ? slice : max_active;
    for (int a0 = 0; a0 < max_active; a0 += estep) {
        int ec = std::min(estep, max_active - a0);
        q.submit([&](sycl::handler& h) {
          h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)ec, (size_t)hidden),
                              sycl::range<2>(1, 64)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int a = a0 + (int)it.get_group(0);
                if (a >= *active_count) return;
                int e = active_experts[a];
                int lane = (int)it.get_local_id(1) & 15;
                int n = (int)it.get_global_id(1);
                int t0 = expert_offsets[e];
                int t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;

                const uint8_t* w = down_w[e];
                const bf16* s = down_s[e];
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
                        uint64_t weight_word;
                        std::memcpy(&weight_word, w + (size_t)n * packed_k + k0 / 2, 8);
                        diff_dpas_v8i bv;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t byte = (uint8_t)(weight_word >> (8 * j));
                            int v0 = (int)(byte & 0x0f); if (v0 >= 8) v0 -= 16;
                            int v1 = (int)(byte >> 4);   if (v1 >= 8) v1 -= 16;
                            uint16_t b0 = float_to_bf16((float)v0 * scale);
                            uint16_t b1 = float_to_bf16((float)v1 * scale);
                            bv[j] = (int)((uint32_t)b0 | ((uint32_t)b1 << 16));
                        }
                        diff_dpas_v8s av;
                        int kk = k0 + lane;
                        for (int m = 0; m < 8; ++m)
                            av[m] = (short)intermediate[(size_t)pair[m] * inter + kk];
                        c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bv, c, kInt4GroupedDpasBF16);
                    }
                    for (int m = 0; m < 8 && m0 + m < t1; ++m)
                        output[(size_t)pair[m] * hidden + n] = float_to_bf16(c[m]);
                }
            });
        });
    }
}

// Combine one expert shard's pair-indexed output. Routing weights remain here
// (after the down projection), preserving DiffusionGemma's original rounding
// and weighting order. Pairs routed to another shard are skipped.
inline void int4_grouped_moe_combine(
    sycl::queue& q, const bf16* pair_out,
    const int* expert_idx, const float* pair_weight,
    int first_expert, int local_experts,
    int seq, int top_k, int hidden, bf16* out)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)seq, (size_t)hidden),
                       [=](sycl::id<2> id) {
            int token = (int)id[0];
            int dim = (int)id[1];
            float acc = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                int pair = token * top_k + k;
                int e = expert_idx[pair] - first_expert;
                if (e >= 0 && e < local_experts)
                    acc += pair_weight[pair] *
                           bf16_to_float(pair_out[(size_t)pair * hidden + dim]);
            }
            out[(size_t)token * hidden + dim] = float_to_bf16(acc);
        });
    });
}

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

static constexpr int kInt4GroupedDpasBF16 = 0x3000;

// Make a compact expert-major route list entirely on device.  offsets has
// local_experts+1 entries, and tokens contains the original top-k pair index.
// A token may appear once for each selected expert; pair/top_k gets its source
// activation row.  This deliberately has no host count/download boundary.
inline void int4_grouped_moe_build_routes(
    sycl::queue& q, const int* expert_idx, int first_expert,
    int local_experts, int pairs, int32_t* counts, int32_t* offsets,
    int32_t* cursors, int32_t* tokens)
{
    q.memset(counts, 0, (size_t)local_experts * sizeof(int32_t));
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)pairs), [=](sycl::id<1> id) {
            int pair = (int)id[0];
            int e = expert_idx[pair] - first_expert;
            if (e < 0 || e >= local_experts) return;
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                count(counts[e]);
            count.fetch_add(1);
        });
    });
    // E=128, so a serial device prefix is a few dozen instructions and avoids
    // a host round trip or a general scan temporary.  This is ordered before
    // the following scatter by the in-order model queue.
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            int32_t running = 0;
            for (int e = 0; e < local_experts; ++e) {
                offsets[e] = running;
                cursors[e] = running;
                running += counts[e];
            }
            offsets[local_experts] = running;
        });
    });
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)pairs), [=](sycl::id<1> id) {
            int pair = (int)id[0];
            int e = expert_idx[pair] - first_expert;
            if (e < 0 || e >= local_experts) return;
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                cursor(cursors[e]);
            tokens[cursor.fetch_add(1)] = pair;
        });
    });
}

// Fused gate/up W4A16 GEMM + DiffusionGemma's GeGLU.  One 16-wide subgroup
// handles one local expert and 16 intermediate channels, looping over its
// compacted pair list in eight-row DPAS tiles.  `gate_w` points at raw packed
// [2*inter, hidden/2] rows and `gate_s` points at [hidden/group, 2*inter].
inline void matmul_int4_grouped_dpas_gateup_geglu(
    sycl::queue& q,
    const bf16* input, int hidden,
    const uint8_t* const* gate_w, const bf16* const* gate_s,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    int local_experts, int pairs, int top_k, int inter, int group_size,
    bf16* intermediate)
{
    if (hidden % 16 || inter % 16 || hidden % group_size || group_size != 32)
        throw std::runtime_error("grouped INT4 DPAS requires H/I % 16 and AWQ group_size=32");
    if (pairs <= 0) return;
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEF = 0.044715f;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)local_experts, (size_t)inter),
                              sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                int t0 = expert_offsets[e];
                int t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;

                const uint8_t* w = gate_w[e];
                const bf16* s = gate_s[e];
                const int packed_k = hidden / 2;
                const int out_features = 2 * inter;

                for (int m0 = t0; m0 < t1; m0 += 8) {
                    diff_dpas_v8f cg = {0,0,0,0,0,0,0,0};
                    diff_dpas_v8f cu = {0,0,0,0,0,0,0,0};
                    for (int k0 = 0; k0 < hidden; k0 += 16) {
                        int kg = k0 / group_size;
                        float sg = bf16_to_float(s[(size_t)kg * out_features + n]);
                        float su = bf16_to_float(s[(size_t)kg * out_features + inter + n]);
                        const uint8_t* wg = w + (size_t)n * packed_k + k0 / 2;
                        const uint8_t* wu = w + (size_t)(inter + n) * packed_k + k0 / 2;
                        diff_dpas_v8i bg, bu;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t gbyte = wg[j], ubyte = wu[j];
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
                        for (int m = 0; m < 8; ++m) {
                            int pos = m0 + m < t1 ? m0 + m : t1 - 1;
                            int pair = expert_tokens[pos];
                            av[m] = (short)input[(size_t)(pair / top_k) * hidden + kk];
                        }
                        cg = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bg, cg, kInt4GroupedDpasBF16);
                        cu = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bu, cu, kInt4GroupedDpasBF16);
                    }
                    for (int m = 0; m < 8 && m0 + m < t1; ++m) {
                        int pair = expert_tokens[m0 + m];
                        float g = cg[m], u = cu[m];
                        float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
                        intermediate[(size_t)pair * inter + n] = float_to_bf16(
                            0.5f * g * (1.0f + sycl::tanh(inner)) * u);
                    }
                }
            });
    });
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
    bf16* output)
{
    if (hidden % 16 || inter % 16 || inter % group_size || group_size != 32)
        throw std::runtime_error("grouped INT4 DPAS requires H/I % 16 and AWQ group_size=32");
    if (pairs <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)local_experts, (size_t)hidden),
                              sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                int t0 = expert_offsets[e];
                int t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;

                const uint8_t* w = down_w[e];
                const bf16* s = down_s[e];
                const int packed_k = inter / 2;
                for (int m0 = t0; m0 < t1; m0 += 8) {
                    diff_dpas_v8f c = {0,0,0,0,0,0,0,0};
                    for (int k0 = 0; k0 < inter; k0 += 16) {
                        int kg = k0 / group_size;
                        float scale = bf16_to_float(s[(size_t)kg * hidden + n]);
                        const uint8_t* wr = w + (size_t)n * packed_k + k0 / 2;
                        diff_dpas_v8i bv;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t byte = wr[j];
                            int v0 = (int)(byte & 0x0f); if (v0 >= 8) v0 -= 16;
                            int v1 = (int)(byte >> 4);   if (v1 >= 8) v1 -= 16;
                            uint16_t b0 = float_to_bf16((float)v0 * scale);
                            uint16_t b1 = float_to_bf16((float)v1 * scale);
                            bv[j] = (int)((uint32_t)b0 | ((uint32_t)b1 << 16));
                        }
                        diff_dpas_v8s av;
                        int kk = k0 + lane;
                        for (int m = 0; m < 8; ++m) {
                            int pos = m0 + m < t1 ? m0 + m : t1 - 1;
                            int pair = expert_tokens[pos];
                            av[m] = (short)intermediate[(size_t)pair * inter + kk];
                        }
                        c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, av, bv, c, kInt4GroupedDpasBF16);
                    }
                    for (int m = 0; m < 8 && m0 + m < t1; ++m) {
                        int pair = expert_tokens[m0 + m];
                        output[(size_t)pair * hidden + n] = float_to_bf16(c[m]);
                    }
                }
            });
    });
}

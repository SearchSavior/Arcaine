#pragma once
// Native Xe2/BMG grouped W4A16 MoE kernels for DiffusionGemma INT4-AWQ, built
// on the project-wide SPIR-V DPAS intrinsic
// (__spirv_SubgroupMatrixMultiplyAccumulateINTEL, operand 0x3000 = packed bf16).
//
// Why not the ESIMD xmx::dpas path: the raw intrinsic is ~6.6x faster on this
// device (18.4 vs 2.8 G dpas/s, measured in the kbench), and the intrinsic
// cannot express mixed s8 x s4 anyway.  ESIMD's s8xs4 is therefore a dead end
// for throughput here, so weights stay 4-bit in DRAM and are dequantized to
// bf16 in registers, with the per-group weight scale folded into the dequant.
// That is W4A16 compute over 4-bit weight traffic, on the fast dpas path.
//
// Operand semantics (K-tile = 16, M = 8 rows, N = 16 channels, lane = channel):
//   A (v8s): a[m] = bf16 bits of activation row m at K index (k0 + lane)
//   B (v8i): b[j] = bf16(w[n][k0+2j] * scale) | bf16(w[n][k0+2j+1] * scale) << 16
//   C (v8f): c[m] accumulates row m, channel n
//
// The checkpoint's raw s4 weights are [N, K/2] (low nibble = even K), strided
// by K/2 between channels, so lanes would read uncoalesced.  Weights are
// therefore repacked once into [k_tile(16)][n_tile(16)][16 channels x 8 bytes]
// (128 B/block) so a lane's 8 s4 bytes for its channel are contiguous and the
// 16 lanes of a tile cover a single coalesced 128-byte block.
//
// Routing is built entirely on device (no host count / download boundary).
#include <sycl/sycl.hpp>
#include <cstdint>
#include <stdexcept>

#include "buffer.hpp"
#include "int4.hpp"
#include "q8_0.hpp"  // diff_dpas_* types + the DPAS intrinsic declaration

namespace igb16 {
constexpr int kKT = 16;   // DPAS K tile (bf16 systolic depth * 2)
constexpr int kRC = 8;    // M rows per subgroup per dpas
constexpr int kES = 16;   // N channels per dpas (= sub-group size)
constexpr int kBLK = kKT / 2;     // s4 bytes per (channel, k_tile) = 8
constexpr int kNBLK = 16 * kBLK;  // repacked bytes per (k_tile, n_tile) = 128
// N register-blocking: n-tiles (16 channels) computed per subgroup.
constexpr int kNBG = 2;  // gate/up
constexpr int kNBD = 4;  // down
}  // namespace igb16

// ---------------------------------------------------------------------------
// Device route builder: compact expert-major list, no host round trip.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// One-time repack of [N, K/2] s4 weights into [k_tile][n_tile][16 x 8 B] so a
// lane's per-channel s4 bytes are contiguous and a tile load is coalesced.
// ---------------------------------------------------------------------------
inline void repack_s4_bf16_dpas(
    sycl::queue& q, const uint8_t* src, int N, int K, uint8_t* dst)
{
    using namespace igb16;
    if (N % kES || K % kKT)
        throw std::runtime_error("bf16 DPAS repack requires N % 16 == 0, K % 16 == 0");
    const int ntiles = N / kES;
    const size_t total = (size_t)N * K / 2;
    const int kbytes = K / 2;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
            size_t o = (size_t)id[0];
            int tile = (int)(o / kNBLK);
            int within = (int)(o % kNBLK);
            int ntile = tile % ntiles;
            int ktile = tile / ntiles;
            int nl = within / kBLK;
            int b = within % kBLK;
            int n = ntile * kES + nl;
            dst[o] = src[(size_t)n * kbytes + ktile * kBLK + b];
        });
    });
}

// ---------------------------------------------------------------------------
// Fused grouped gate/up W4A16 GEMM + GeGLU.  gate channels [0,inter), up
// channels [inter,2*inter).  Output is pair-indexed (identity slot map).
// ---------------------------------------------------------------------------
inline void matmul_int4_grouped_bf16_gateup_geglu(
    sycl::queue& q, const bf16* input, int hidden,
    const uint8_t* const* gate_w_repacked, const bf16* const* gate_s,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    int local_experts, int pairs, int top_k, int inter, bf16* intermediate)
{
    using namespace igb16;
    if (hidden % kKT || inter % (kES * kNBG) || pairs <= 0)
        throw std::runtime_error(
            "grouped bf16 gateup requires hidden % 16 == 0, inter % 32 == 0");
    const int out_features = 2 * inter;
    const int ntiles = out_features / kES;
    const int nblk = inter / (kES * kNBG);

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)local_experts * nblk * kES, kES),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kES)]] {
                int sg = (int)it.get_group(0);
                int lane = (int)it.get_local_id(0);
                int e = sg / nblk;
                int n0 = (sg % nblk) * (kES * kNBG);
                int t0 = expert_offsets[e], t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;
                const uint8_t* w = gate_w_repacked[e];
                const bf16* s = gate_s[e];
                const int gate_tile0 = n0 / kES;
                const int up_tile0 = (inter + n0) / kES;

                for (int m0 = t0; m0 < t1; m0 += kRC) {
                    int tok[kRC];
#pragma unroll
                    for (int r = 0; r < kRC; ++r) {
                        int pos = m0 + r < t1 ? m0 + r : t1 - 1;
                        tok[r] = expert_tokens[pos] / top_k;
                    }
                    diff_dpas_v8f fg[kNBG], fu[kNBG];
                    float scg_arr[kNBG], scu_arr[kNBG];
#pragma unroll
                    for (int i = 0; i < kNBG; ++i) {
                        fg[i] = diff_dpas_v8f{0,0,0,0,0,0,0,0};
                        fu[i] = diff_dpas_v8f{0,0,0,0,0,0,0,0};
                        scg_arr[i] = 0.0f; scu_arr[i] = 0.0f;
                    }
                    for (int k0 = 0, kt = 0; k0 < hidden; k0 += kKT, ++kt) {
                        diff_dpas_v8s a;
#pragma unroll
                        for (int m = 0; m < kRC; ++m)
                            a[m] = (short)input[(size_t)tok[m] * hidden + k0 + lane];

                        if ((k0 % 32) == 0) {
#pragma unroll
                            for (int i = 0; i < kNBG; ++i) {
                                scg_arr[i] = bf16_to_float(
                                    s[(size_t)(k0 / 32) * out_features + n0 + i * kES + lane]);
                                scu_arr[i] = bf16_to_float(
                                    s[(size_t)(k0 / 32) * out_features + inter + n0 + i * kES + lane]);
                            }
                        }
#pragma unroll
                        for (int i = 0; i < kNBG; ++i) {
                            float sg_ = scg_arr[i];
                            float su_ = scu_arr[i];
                            const uint8_t* wg =
                                w + ((size_t)kt * ntiles + gate_tile0 + i) * kNBLK
                                  + (size_t)lane * kBLK;
                            const uint8_t* wu =
                                w + ((size_t)kt * ntiles + up_tile0 + i) * kNBLK
                                  + (size_t)lane * kBLK;
                            diff_dpas_v8i bg, bu;
#pragma unroll
                            for (int j = 0; j < kBLK; ++j) {
                                uint8_t yg = wg[j], yu = wu[j];
                                int g0 = yg & 0xf; if (g0 >= 8) g0 -= 16;
                                int g1 = yg >> 4;   if (g1 >= 8) g1 -= 16;
                                int u0 = yu & 0xf; if (u0 >= 8) u0 -= 16;
                                int u1 = yu >> 4;   if (u1 >= 8) u1 -= 16;
                                uint16_t g0b = float_to_bf16((float)g0 * sg_);
                                uint16_t g1b = float_to_bf16((float)g1 * sg_);
                                uint16_t u0b = float_to_bf16((float)u0 * su_);
                                uint16_t u1b = float_to_bf16((float)u1 * su_);
                                bg[j] = (int)((uint32_t)g0b | ((uint32_t)g1b << 16));
                                bu[j] = (int)((uint32_t)u0b | ((uint32_t)u1b << 16));
                            }
                            fg[i] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                                kKT, a, bg, fg[i], 0x3000);
                            fu[i] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                                kKT, a, bu, fu[i], 0x3000);
                        }
                    }
                    constexpr float S2 = 0.7978845608028654f;
                    constexpr float CO = 0.044715f;
#pragma unroll
                    for (int i = 0; i < kNBG; ++i) {
#pragma unroll
                        for (int m = 0; m < kRC; ++m) {
                            if (m0 + m >= t1) continue;
                            float g = fg[i][m], u = fu[i][m];
                            float inner = S2 * (g + CO * g * g * g);
                            float t = sycl::tanh(inner);
                            intermediate[(size_t)expert_tokens[m0 + m] * inter
                                         + n0 + i * kES + lane] =
                                float_to_bf16(0.5f * g * (1.0f + t) * u);
                        }
                    }
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Grouped down-projection W4A16 GEMM.  Input/output rows are the pair index.
// ---------------------------------------------------------------------------
inline void matmul_int4_grouped_bf16_down(
    sycl::queue& q, const bf16* intermediate, int inter,
    const uint8_t* const* down_w_repacked, const bf16* const* down_s,
    const int32_t* expert_offsets, const int32_t* expert_tokens,
    int local_experts, int pairs, int hidden, bf16* output)
{
    using namespace igb16;
    if (inter % kKT || hidden % (kES * kNBD) || pairs <= 0)
        throw std::runtime_error(
            "grouped bf16 down requires inter % 16 == 0, hidden % 64 == 0");
    const int ntiles = hidden / kES;
    const int nblk = hidden / (kES * kNBD);

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)local_experts * nblk * kES, kES),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kES)]] {
                int sg = (int)it.get_group(0);
                int lane = (int)it.get_local_id(0);
                int e = sg / nblk;
                int n0 = (sg % nblk) * (kES * kNBD);
                int t0 = expert_offsets[e], t1 = expert_offsets[e + 1];
                if (t0 >= t1) return;
                const uint8_t* w = down_w_repacked[e];
                const bf16* s = down_s[e];
                const int dtile0 = n0 / kES;

                for (int m0 = t0; m0 < t1; m0 += kRC) {
                    int pr[kRC];
#pragma unroll
                    for (int r = 0; r < kRC; ++r) {
                        int pos = m0 + r < t1 ? m0 + r : t1 - 1;
                        pr[r] = expert_tokens[pos];
                    }
                    diff_dpas_v8f f[kNBD];
                    float sc_arr[kNBD];
#pragma unroll
                    for (int i = 0; i < kNBD; ++i) {
                        f[i] = diff_dpas_v8f{0,0,0,0,0,0,0,0};
                        sc_arr[i] = 0.0f;
                    }
                    for (int k0 = 0, kt = 0; k0 < inter; k0 += kKT, ++kt) {
                        diff_dpas_v8s a;
#pragma unroll
                        for (int m = 0; m < kRC; ++m)
                            a[m] = (short)intermediate[(size_t)pr[m] * inter + k0 + lane];

                        if ((k0 % 32) == 0) {
#pragma unroll
                            for (int i = 0; i < kNBD; ++i)
                                sc_arr[i] = bf16_to_float(
                                    s[(size_t)(k0 / 32) * hidden + n0 + i * kES + lane]);
                        }
#pragma unroll
                        for (int i = 0; i < kNBD; ++i) {
                            float sc_ = sc_arr[i];
                            const uint8_t* wb =
                                w + ((size_t)kt * ntiles + dtile0 + i) * kNBLK
                                  + (size_t)lane * kBLK;
                            diff_dpas_v8i b;
#pragma unroll
                            for (int j = 0; j < kBLK; ++j) {
                                uint8_t y = wb[j];
                                int v0 = y & 0xf; if (v0 >= 8) v0 -= 16;
                                int v1 = y >> 4;   if (v1 >= 8) v1 -= 16;
                                uint16_t b0 = float_to_bf16((float)v0 * sc_);
                                uint16_t b1 = float_to_bf16((float)v1 * sc_);
                                b[j] = (int)((uint32_t)b0 | ((uint32_t)b1 << 16));
                            }
                            f[i] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                                kKT, a, b, f[i], 0x3000);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < kNBD; ++i) {
#pragma unroll
                        for (int m = 0; m < kRC; ++m) {
                            if (m0 + m >= t1) continue;
                            output[(size_t)expert_tokens[m0 + m] * hidden
                                   + n0 + i * kES + lane] = float_to_bf16(f[i][m]);
                        }
                    }
                }
            });
    });
}

#pragma once
// XMX (DPAS) variant of the device chunked gated delta rule — same math as
// qwen_gdn_device_chunk (gated_delta_chunk.hpp) but every large matmul runs
// on the XMX cores via __spirv_SubgroupMatrixMultiplyAccumulateINTEL with
// fp16 operands and fp32 accumulate (M=8, N=16/SG16, K=16 per instruction,
// same idiom as int4_grouped_moe.hpp). Operands that only exist in fp32
// (T, kcd, v_new, S) are staged as fp16 in SLM per chunk; the SSM state
// master copy stays fp32 in global memory, so rounding enters per-chunk
// increments but never compounds in storage. fp16 carries 10 mantissa bits
// (vs bf16's 7): bf16 Q/K/V inputs convert exactly, so K@K^T and Q@K^T
// products stay bit-identical to the scalar fp32 path, and the per-chunk
// staging roundings are ~8x smaller than with bf16 operands.
// A bf16-operand A/B variant (QWEN35_GDN_XMX_FP16=0) was measured and
// REMOVED: bf16 staging destabilizes the multi-chunk recurrence (core cos
// 0.82 @ p=96, 0.59 @ p=128) while fp16 holds cos >= 0.99999 through p=512.
// The T=(I-A)^-1 forward substitution (64x64, ~C^3/6 MACs) stays scalar —
// it is latency-bound on the sequential row recurrence, not flops.
//
// Per chunk (C=64, DK=DV=128), one 256-lane WG per head (16 SGs):
//   1. sA  = mask(-beta*decay) * (K @ K^T)         32 tiles  (8 k-steps)
//      [c==0: stage sSb = f16(S) from global fp32]
//   2. T = (I-A)^-1 scalar solve (fp32 SLM)
//   3. stage sTb = f16(T+I), sW = f16[beta*V | sw*K]   ([64,256])
//   4. v_tmp: sW[:,0:128] -= (sW[:,128:256] @ sSb) 64 tiles (8 k-steps)
//      (runs on the untransformed sW: disjoint columns, no aliasing)
//   5. sVn = (T+I) @ v_tmp -> M2                   64 tiles  (4 k-steps)
//      (out of place: an in-place (T+I)@X would read rows earlier tiles
//       already overwrote; v_new = (T+I)@(betaV - swK@S) by associativity)
//   6. sQK = mask(decay) * (Q @ K^T)               32 tiles  (8 k-steps)
//   7. stage sQeg = f16(Q * eg) (aliases dead sW)
//   8. core = sQeg @ sSb + sQK @ sVn               64 tiles (12 k-steps)
//   9. stage sKT = f16(K^T * dec) ([128,64])
//  10. S = S*eg_last + sKT @ sVn; sSb = f16(S)    128 tiles (4 k-steps)
//
// SLM arena (115,968 B < 128KB), aliasing by phase lifetime:
//   0      sSb  32K f16 [128][128]   (persists across chunks)
//   32768  small 1.25K fp32[64] x5    (beta, gc, eg, sw, dec)
//   34048  sK   16K f16 [64][128]    (until sKT built)
//   50432  M1   16K union: sA fp32 / sTb / sQK / sKT
//   66816  M2   16K union: sT fp32 / sVn f16 [64][128]
//   83200  sW   32K f16 [64][256]    (dead after phase 5; sQeg aliases it)
//
// Dispatch: QWEN35_GDN_IMPL=xmx (see gated_deltanet.hpp).

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "runtime/gpu/buffer.hpp"
#include "runtime/quantization/q8_0.hpp" // DPAS builtin declaration / vector operand types.

inline uint16_t float_to_f16_bits(float f) {
    return sycl::bit_cast<uint16_t>((sycl::half)f);
}

inline void qwen_gdn_device_chunk_xmx(
    sycl::queue& q,
    const bf16* qbuf, const bf16* kbuf, const bf16* vbuf,
    const bf16* beta, const bf16* g,
    float* ssm_state, bf16* core,
    int S, int n_v, int d_k, int d_v, int chunk)
{
    if (chunk != 64 || d_k != 128 || d_v != 128)
        throw std::runtime_error("qwen_gdn_device_chunk_xmx requires chunk=64, d_k=d_v=128");
    if (S <= 0) return;
    constexpr int C = 64, DK = 128, DV = 128, WG = 256;
    const int nch = (S + C - 1) / C;

    constexpr size_t OFF_SB = 0;                   // bf16 [128][128]
    constexpr size_t OFF_SM = OFF_SB + 32768;      // fp32 [64] x5
    constexpr size_t OFF_K  = OFF_SM + 1280;       // bf16 [64][128]
    constexpr size_t OFF_M1 = OFF_K + 16384;       // union 16K
    constexpr size_t OFF_M2 = OFF_M1 + 16384;      // union 16K
    constexpr size_t OFF_W  = OFF_M2 + 16384;      // bf16 [64][256]
    constexpr size_t ARENA  = OFF_W + 32768;       // 115,968

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<char> arena(sycl::range<1>(ARENA), h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_v, WG),
                              sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                char*  base_ = &arena[0];
                auto dpas = [&](diff_dpas_v8s av, diff_dpas_v8i bv, diff_dpas_v8f acc) -> diff_dpas_v8f {
                    return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, av, bv, acc, kQ8DpasFP16);
                };
                auto to_op = [](float x) -> uint16_t { return float_to_f16_bits(x); };
                auto op_to_float = [](uint16_t b) -> float {
                    return (float)sycl::bit_cast<sycl::half>(b);
                };
                bf16*  sSb   = (bf16*)(base_ + OFF_SB);
                float* sBeta = (float*)(base_ + OFF_SM);
                float* sGc   = sBeta + C;
                float* sEg   = sGc + C;
                float* sSW   = sEg + C;
                float* sDec  = sSW + C;
                bf16*  sK    = (bf16*)(base_ + OFF_K);
                float* sA    = (float*)(base_ + OFF_M1);
                float* sT    = (float*)(base_ + OFF_M2);
                bf16*  sW    = (bf16*)(base_ + OFF_W);

                const int hd   = (int)it.get_group(0);
                const int lane = (int)it.get_local_id(1);
                const int sg   = lane >> 4;
                const int sl   = lane & 15;
                float* ssm     = ssm_state + (size_t)hd * DK * DV;

                for (int c = 0; c < nch; ++c) {
                    const int cbase = c * C;
                    const int rows  = std::min(C, S - cbase);

                    // -- stage K (zero-padded), beta/g, cumsum gc -----------
                    for (int i = lane; i < C * DK; i += WG) {
                        int t = i >> 7;
                        sK[i] = (t < rows)
                            ? to_op(bf16_to_float(kbuf[((size_t)(cbase + t) * n_v + hd) * DK + (i & 127)]))
                            : (bf16)0;
                    }
                    for (int t = lane; t < C; t += WG) {
                        sBeta[t] = (t < rows)
                            ? bf16_to_float(beta[(size_t)(cbase + t) * n_v + hd])
                            : 0.0f;
                        sGc[t] = (t < rows)
                            ? bf16_to_float(g[(size_t)(cbase + t) * n_v + hd])
                            : 0.0f;
                    }
                    if (c == 0) {
                        for (int i = lane; i < DK * DV; i += WG)
                            sSb[i] = to_op(ssm[i]);
                    }
                    it.barrier();
                    if (lane == 0) {
                        float acc = 0.0f;
                        for (int t = 0; t < C; ++t) { acc += sGc[t]; sGc[t] = acc; }
                    }
                    it.barrier();
                    for (int t = lane; t < C; t += WG) {
                        float eg = sycl::exp(sGc[t]);
                        sEg[t]  = eg;
                        sSW[t]  = eg * sBeta[t];
                        sDec[t] = sycl::exp(sGc[C - 1] - sGc[t]);
                    }
                    it.barrier();

                    // -- 1. sA = -beta*decay * (K @ K^T), strictly lower -----
                    for (int t = sg; t < 32; t += 16) {
                        const int mt = (t >> 2) * 8, nt = (t & 3) * 16;
                        const bf16* kcol = sK + (nt + sl) * DK;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < DK; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)sK[(mt + m) * DK + k0 + sl];
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)kcol[k0 + 2 * j] |
                                              ((uint32_t)kcol[k0 + 2 * j + 1] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        const int bcol = nt + sl;
                        for (int m = 0; m < 8; ++m) {
                            const int a = mt + m;
                            sA[a * C + bcol] = (bcol < a)
                                ? -sBeta[a] * acc[m] * sycl::exp(sGc[a] - sGc[bcol])
                                : 0.0f;
                        }
                    }
                    it.barrier();

                    // -- 2. T = (I - A)^-1 (scalar, strictly lower) ----------
                    for (int j = lane; j < C; j += WG) sT[j] = 0.0f;
                    it.barrier();
                    for (int i = 1; i < C; ++i) {
                        for (int j = lane; j < i; j += WG) {
                            float acc = sA[i * C + j];
                            for (int r = j + 1; r < i; ++r)
                                acc += sA[i * C + r] * sT[r * C + j];
                            sT[i * C + j] = acc;
                        }
                        it.barrier();
                    }

                    // -- 3. stage sTb = bf16(T+I), sW = bf16[beta*V | sw*K] --
                    bf16* sTb = (bf16*)(base_ + OFF_M1); // sA dead
                    for (int idx = lane; idx < C * C; idx += WG) {
                        int i = idx >> 6, j = idx & 63;
                        sTb[idx] = to_op(sT[idx] + (i == j ? 1.0f : 0.0f));
                    }
                    for (int idx = lane; idx < C * 256; idx += WG) {
                        int a = idx >> 8, col = idx & 255;
                        int ar = a < rows ? a : rows - 1;
                        float xv;
                        if (col < 128)
                            xv = sBeta[a] * bf16_to_float(vbuf[
                                ((size_t)(cbase + ar) * n_v + hd) * DV + col]);
                        else
                            xv = sSW[a] * op_to_float(sK[a * DK + (col - 128)]);
                        sW[idx] = to_op(xv);
                    }
                    it.barrier();

                    // -- 4. v_tmp: sW[:,0:128] -= (sW[:,128:256] @ sSb) ------
                    for (int t = sg; t < 64; t += 16) {
                        const int mt = (t >> 3) * 8, nt = (t & 7) * 16;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < DK; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)sW[(mt + m) * 256 + 128 + k0 + sl];
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)sSb[(k0 + 2 * j) * DV + nt + sl] |
                                              ((uint32_t)sSb[(k0 + 2 * j + 1) * DV + nt + sl] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int m = 0; m < 8; ++m) {
                            const int a = mt + m;
                            float vi = op_to_float(sW[a * 256 + nt + sl]);
                            sW[a * 256 + nt + sl] = to_op(vi - acc[m]);
                        }
                    }
                    it.barrier();

                    // -- 5. sVn = (T+I) @ sW[:,0:128] (out of place -> M2) ---
                    bf16* sVn = (bf16*)(base_ + OFF_M2); // sT dead
                    for (int t = sg; t < 64; t += 16) {
                        const int mt = (t >> 3) * 8, nt = (t & 7) * 16;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < C; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)sTb[(mt + m) * C + k0 + sl];
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)sW[(k0 + 2 * j) * 256 + nt + sl] |
                                              ((uint32_t)sW[(k0 + 2 * j + 1) * 256 + nt + sl] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int m = 0; m < 8; ++m)
                            sVn[(mt + m) * DV + nt + sl] = to_op(acc[m]);
                    }
                    it.barrier();

                    // -- 6+7. sQK = decay*(Q @ K^T) (b<=a), sQeg = Q*eg ------
                    bf16* sQK  = (bf16*)(base_ + OFF_M1); // sTb dead
                    bf16* sQeg = (bf16*)(base_ + OFF_W);  // sW dead
                    for (int t = sg; t < 32; t += 16) {
                        const int mt = (t >> 2) * 8, nt = (t & 3) * 16;
                        const bf16* kcol = sK + (nt + sl) * DK;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < DK; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m) {
                                int a  = mt + m;
                                int ar = a < rows ? a : rows - 1;
                                av[m] = (short)to_op(bf16_to_float(qbuf[
                                    ((size_t)(cbase + ar) * n_v + hd) * DK + k0 + sl]));
                            }
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)kcol[k0 + 2 * j] |
                                              ((uint32_t)kcol[k0 + 2 * j + 1] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        const int bcol = nt + sl;
                        for (int m = 0; m < 8; ++m) {
                            const int a = mt + m;
                            sQK[a * C + bcol] = (bcol <= a && a < rows)
                                ? to_op(acc[m] * sycl::exp(sGc[a] - sGc[bcol]))
                                : (bf16)0;
                        }
                    }
                    for (int idx = lane; idx < C * DK; idx += WG) {
                        int a = idx >> 7, dk = idx & 127;
                        int ar = a < rows ? a : rows - 1;
                        sQeg[idx] = (a < rows)
                            ? to_op(bf16_to_float(qbuf[
                                  ((size_t)(cbase + ar) * n_v + hd) * DK + dk]) * sEg[a])
                            : (bf16)0;
                    }
                    it.barrier();

                    // -- 8. core = Qeg @ S + QK @ v_new ----------------------
                    for (int t = sg; t < 64; t += 16) {
                        const int mt = (t >> 3) * 8, nt = (t & 7) * 16;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < DK; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)sQeg[(mt + m) * DK + k0 + sl];
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)sSb[(k0 + 2 * j) * DV + nt + sl] |
                                              ((uint32_t)sSb[(k0 + 2 * j + 1) * DV + nt + sl] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int k0 = 0; k0 < C; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)sQK[(mt + m) * C + k0 + sl];
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)sVn[(k0 + 2 * j) * DV + nt + sl] |
                                              ((uint32_t)sVn[(k0 + 2 * j + 1) * DV + nt + sl] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int m = 0; m < 8; ++m) {
                            const int a = mt + m;
                            if (a < rows)
                                core[((size_t)(cbase + a) * n_v + hd) * DV + nt + sl] =
                                    float_to_bf16(acc[m]);
                        }
                    }
                    it.barrier();

                    // -- 9. stage sKT = bf16(K^T * dec) -----------------------
                    bf16* sKT = (bf16*)(base_ + OFF_M1); // sQK dead
                    for (int idx = lane; idx < DK * C; idx += WG) {
                        int dk = idx >> 6, b = idx & 63;
                        sKT[idx] = to_op(
                            op_to_float(sK[b * DK + dk]) * sDec[b]);
                    }
                    it.barrier();

                    // -- 10. S = S*eg_last + sKT @ v_new; refresh sSb --------
                    // 128 tiles = 16 M-tiles (dk) x 8 N-tiles (dv).
                    const float eg_last = sEg[C - 1];
                    for (int t = sg; t < 128; t += 16) {
                        const int mt = (t >> 3) * 8, nt = (t & 7) * 16;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < C; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)sKT[(mt + m) * C + k0 + sl];
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)sVn[(k0 + 2 * j) * DV + nt + sl] |
                                              ((uint32_t)sVn[(k0 + 2 * j + 1) * DV + nt + sl] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int m = 0; m < 8; ++m) {
                            const int dk = mt + m, dv = nt + sl;
                            float sn = ssm[dk * DV + dv] * eg_last + acc[m];
                            ssm[dk * DV + dv] = sn;
                            sSb[dk * DV + dv] = to_op(sn);
                        }
                    }
                    it.barrier();
                }
            });
    });
}

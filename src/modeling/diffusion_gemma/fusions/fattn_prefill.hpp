#pragma once
// ===========================================================================
// SKETCH — Fused DPAS flash-attention for the DiffusionGemma *prefill* path.
//
// Gated by DIFF_FATTN_PREFILL (A/B against the banded batched-GEMM + fused
// softmax path in gqa_attention).  This is the model-specialized analogue of
// llama.cpp's `ggml_sycl_flash_attn_ext_tile` (ggml/src/ggml-sycl/fattn-tile.*):
// same streaming online-softmax algorithm, but the two matmuls are mapped onto
// Intel DPAS via __spirv_SubgroupMatrixMultiplyAccumulateINTEL (the same
// systolic intrinsic Arcaine already drives in common/gpu/nvfp4.hpp) instead of
// llama.cpp's portable half2-FMA vector dots.
//
// Correctness contract: this is a *drop-in* for the {score_mm, fused_masked_
// softmax, value_mm} sequence in gqa_attention() for the causal (encoder /
// prefill) case.  Q arrives already scaled (the query pre-attn scalar is baked
// upstream in project_qkv, exactly as the banded path assumes), so this kernel
// applies NO extra softmax scale.  Output is written time-major (seq, nq, hd)
// so it needs neither transpose_q_into on the way in nor scatter_ctx on the way
// out — it reads time-major Q and writes time-major O directly.
//
// STATUS: unverified on hardware (no Intel GPU / icpx in this tree; repo has no
// tests per AGENTS.md).  Block sizes and the V read pattern are the first tuning
// knobs.  Marked TODO where a real bring-up must confirm layout/register cost.
// ===========================================================================
#include <cstdint>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "../../../common/gpu/buffer.hpp"
#include "../../../common/gpu/engine.hpp"

// DPAS intrinsic — shared decl with common/gpu/nvfp4.hpp (same include guard so
// the two headers never redefine it).
#ifndef DIFF_DPAS_INTRINSIC_DECL
#define DIFF_DPAS_INTRINSIC_DECL
using diff_dpas_v8s = short __attribute__((ext_vector_type(8)));
using diff_dpas_v8i = int   __attribute__((ext_vector_type(8)));
using diff_dpas_v8f = float __attribute__((ext_vector_type(8)));

SYCL_EXTERNAL inline diff_dpas_v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, diff_dpas_v8s A, diff_dpas_v8i B, diff_dpas_v8f C, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return diff_dpas_v8f{}; }
#endif
#endif

namespace fattn_prefill {

// Operands selector: bf16 A/B inputs, f32 accumulate (matches nvfp4.hpp).
static constexpr int kDpasBF16 = 0x3000;
static constexpr int MQ = 8;    // query rows per DPAS M-tile (v8 vectors)
static constexpr int SG = 16;   // subgroup size == DPAS N/K tile == VNNI depth
static constexpr int BK = SG;   // KV positions per streamed block

// One work-group == one subgroup of 16 lanes, owning MQ=8 query rows of one
// query head.  It streams the KV band in blocks of BK, keeping the running
// online-softmax state (m_run, l_run) and the O accumulator in registers.
//
// Layouts (all bf16 == uint16 raw bits):
//   Q  : (seq, nq, hd) time-major   Q[qi*nq*hd + h*hd + d]
//   Kc : (kv_len, nkv, hd)          Kc[kj*nkv*hd + b*hd + d]   (b = h / gqa)
//   Vc : (kv_len, nkv, hd)          Vc[kj*nkv*hd + b*hd + d]
//   O  : (seq, nq, hd) time-major   written in place, no scatter
//
// DPAS mapping:
//   S = Q Kᵀ  -> M=query(8) N=key(lane) K=hd(step 16).  K's hd is contiguous,
//                so the B operand VNNI-packs straight from Kc with NO transpose.
//   O += P V  -> M=query(8) N=hd(lane)  K=key(step 16). V's contraction (key)
//                is the strided dim; packed by a per-lane strided read (TODO:
//                stage V block into SLM for coalescing on real hardware).
inline void launch(
    GpuEngine& gpu,
    const bf16* Q, bf16* O,
    int seq, int nq, int nkv, int hd,
    const bf16* Kc, const bf16* Vc, int kv_len,
    int past_offset, int sliding_window, bool causal)
{
    // hd must tile the DPAS K/N step; caller falls back to the banded path
    // otherwise (see gqa_attention dispatch).
    const int hdt = hd / SG;             // hd tiles (both matmuls)
    const int gqa = nq / nkv;
    const int ntq = (seq + MQ - 1) / MQ; // query tiles per head
    const int kv0 = past_offset - (kv_len - seq);  // absolute pos of Kc[0]

    auto& q = gpu.queue;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)nq * ntq * SG, SG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg   = it.get_sub_group();
                int  grp  = (int)it.get_group(0);
                int  lane = (int)it.get_local_id(0);
                int  head = grp / ntq;
                int  qtile= grp % ntq;
                int  qbase= qtile * MQ;
                int  bkv  = head / gqa;   // shared kv head for this query head

                // Per-query online-softmax state (replicated across lanes).
                float m_run[MQ], l_run[MQ];
                for (int m = 0; m < MQ; ++m) { m_run[m] = -INFINITY; l_run[m] = 0.0f; }

                // O accumulator: per lane holds hd column `nt*16+lane` for each
                // of MQ queries, one v8f per hd tile.  (MQ*hdt f32 per lane.)
                diff_dpas_v8f acc[/*hdt max*/ 32];
                for (int nt = 0; nt < hdt; ++nt) acc[nt] = diff_dpas_v8f{};

                // Causal / sliding band for this query tile (same math as the
                // banded gqa_attention path — absolute coords).
                int q_lo = past_offset + qbase;
                int q_hi = past_offset + std::min(qbase + MQ - 1, seq - 1);
                int ke = kv_len, ks = 0;
                if (causal) {
                    ke = std::min(kv_len, (q_hi - kv0) + 1);
                    ks = (sliding_window == INT_MAX)
                       ? 0 : std::max(0, (q_lo - sliding_window + 1) - kv0);
                }
                int ks_aligned = (ks / BK) * BK;

                for (int kj = ks_aligned; kj < ke; kj += BK) {
                    int key = kj + lane;              // this lane's key column
                    bool key_ok = key < kv_len;

                    // ---- 1. S = Q·Kᵀ for this block: c[m] = S[query m][key=lane]
                    diff_dpas_v8f c = diff_dpas_v8f{};
                    for (int kt = 0; kt < hdt; ++kt) {
                        int d0 = kt * SG;
                        // A: a[m] = Q[query m][d0+lane]  (lane over hd == DPAS K)
                        diff_dpas_v8s a;
                        for (int m = 0; m < MQ; ++m) {
                            int qi = qbase + m;
                            uint16_t qv = (qi < seq)
                                ? Q[(size_t)qi * nq * hd + head * hd + d0 + lane] : 0;
                            a[m] = (short)qv;
                        }
                        // B: b[i] VNNI-packs K[key][bkv][d0+2i .. +2i+1] (hd
                        // contiguous -> no transpose).  lane == key == DPAS N.
                        diff_dpas_v8i b;
                        const bf16* krow = key_ok
                            ? Kc + (size_t)key * nkv * hd + (size_t)bkv * hd + d0
                            : nullptr;
                        for (int i = 0; i < 8; ++i) {
                            uint16_t lo = krow ? krow[2 * i]     : 0;
                            uint16_t hi = krow ? krow[2 * i + 1] : 0;
                            b[i] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
                        }
                        c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(SG, a, b, c, kDpasBF16);
                    }

                    // ---- 2. mask + online softmax; produce P[m] for this lane's key
                    float P[MQ];
                    for (int m = 0; m < MQ; ++m) {
                        int qi = qbase + m;
                        float s = c[m];
                        bool masked = !key_ok || qi >= seq;
                        if (causal && !masked) {
                            int qg = past_offset + qi, kg = kv0 + key;
                            masked = kg > qg ||
                                (sliding_window != INT_MAX && kg < qg - sliding_window + 1);
                        }
                        if (masked) s = -INFINITY;

                        float bmax = sycl::reduce_over_group(sg, s, sycl::maximum<float>());
                        if (bmax == -INFINITY) { P[m] = 0.0f; continue; }  // block fully masked for q m
                        float m_new = sycl::fmax(m_run[m], bmax);
                        float corr  = sycl::exp(m_run[m] == -INFINITY ? 0.0f : m_run[m] - m_new);
                        float p     = (s == -INFINITY) ? 0.0f : sycl::exp(s - m_new);
                        float bsum  = sycl::reduce_over_group(sg, p, sycl::plus<float>());

                        l_run[m] = l_run[m] * corr + bsum;
                        m_run[m] = m_new;
                        // rescale the O accumulator rows for query m by corr
                        for (int nt = 0; nt < hdt; ++nt) acc[nt][m] *= corr;
                        P[m] = p;   // this lane's contribution (key = kj+lane)
                    }

                    // ---- 3. O += P·V for this block.  M=query N=hd(lane) K=key
                    diff_dpas_v8s pa;                     // A: pa[m] = P[query m][key=lane]
                    for (int m = 0; m < MQ; ++m) pa[m] = (short)float_to_bf16(P[m]);
                    for (int nt = 0; nt < hdt; ++nt) {
                        int d0 = nt * SG;
                        // B: b[i] VNNI-packs V[kj+2i .. +2i+1][bkv][d0+lane]
                        // (contraction=key is strided by nkv*hd — TODO SLM stage).
                        diff_dpas_v8i b;
                        for (int i = 0; i < 8; ++i) {
                            int k0 = kj + 2 * i, k1 = kj + 2 * i + 1;
                            uint16_t lo = (k0 < kv_len)
                                ? Vc[(size_t)k0 * nkv * hd + (size_t)bkv * hd + d0 + lane] : 0;
                            uint16_t hi = (k1 < kv_len)
                                ? Vc[(size_t)k1 * nkv * hd + (size_t)bkv * hd + d0 + lane] : 0;
                            b[i] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
                        }
                        acc[nt] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(SG, pa, b, acc[nt], kDpasBF16);
                    }
                }

                // ---- finalize: O[query][head][hd] = acc / l_run  (time-major)
                for (int nt = 0; nt < hdt; ++nt) {
                    int d = nt * SG + lane;
                    for (int m = 0; m < MQ; ++m) {
                        int qi = qbase + m;
                        if (qi >= seq) continue;
                        float inv = l_run[m] > 0.0f ? 1.0f / l_run[m] : 0.0f;
                        O[(size_t)qi * nq * hd + head * hd + d] = float_to_bf16(acc[nt][m] * inv);
                    }
                }
            });
    });
}

// hd tile bound for the fixed-size acc[] register array above.
inline bool supported(int hd) { return hd % SG == 0 && hd / SG <= 32; }

}  // namespace fattn_prefill

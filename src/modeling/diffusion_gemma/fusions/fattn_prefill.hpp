#pragma once
// ===========================================================================
// Fused DPAS flash-attention for the DiffusionGemma *prefill* path.
//
// Gated by DIFF_FATTN_PREFILL (A/B against the banded batched-GEMM + fused
// softmax path in gqa_attention).  Model-specialized analogue of llama.cpp's
// ggml_sycl_flash_attn_ext_tile: same streaming online-softmax algorithm, but
// both matmuls are mapped onto Intel DPAS via
// __spirv_SubgroupMatrixMultiplyAccumulateINTEL (the intrinsic Arcaine already
// drives in common/gpu/nvfp4.hpp) instead of llama.cpp's half2-FMA vector dots.
//
// Correctness contract: drop-in for {score_mm, fused_masked_softmax, value_mm}
// in gqa_attention() for the causal (encoder/prefill) case.  Q arrives already
// scaled (query pre-attn scalar baked upstream in project_qkv), so NO extra
// softmax scale.  Output is written time-major (seq, nq, hd) — no transpose_q_
// into on the way in, no scatter_ctx on the way out.
//
// DPAS mapping (subgroup 16, 8x16 bf16 tiles, f32 accumulate):
//   S = Q Kᵀ  -> M=query(8) N=key(lane) K=hd(step 16).  K's hd axis is the
//                cache's contiguous axis AND QKᵀ's contraction axis, so the B
//                operand VNNI-packs straight from Kc with NO transpose.
//   O += P V  -> M=query(8) N=hd(lane)  K=key(step 16). V's contraction (key)
//                is the strided axis, so each KV block of V is staged into SLM
//                transposed to (hd, kv); the B operand then reads its
//                contraction contiguously from SLM.
//
// A/B tuning knobs (read once at launch; see dispatch at bottom):
//   DIFF_FATTN_QTILES=1|2|4   query M-tiles per subgroup, sharing one V stage
//                             (V-load amortization / occupancy). default 1.
//   DIFF_FATTN_ACC=reg|slm    O accumulator in registers vs SLM
//                             (register-pressure ceiling, matters at hd=512).
//                             default reg.
//   DIFF_FATTN_BK=16|32|48    KV positions staged per outer step; multiple of
//                             16 (fewer barriers / larger V stage). default 16.
//
// STATUS: unverified on hardware (no Intel GPU / icpx in this tree; repo has no
// tests per AGENTS.md).  Logic is complete and stub-checked; bring-up on
// Battlemage owns the numeric cross-check vs the banded path and picking the
// winning knob combo.  SLM budget (Vt + optional acc) is checked at dispatch;
// configs that overflow kSlmBudget fall back to the register accumulator.
// ===========================================================================
#include <cstdint>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
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

static constexpr int kDpasBF16 = 0x3000;  // bf16 A/B, f32 accumulate
static constexpr int MQ  = 8;   // query rows per DPAS M-tile (v8 vectors)
static constexpr int SG  = 16;  // subgroup size == DPAS N/K tile == VNNI depth
static constexpr int MAXHDT = 32;                 // hd/16 ceiling (global_head_dim=512)
static constexpr size_t kSlmBudget = 128 * 1024;  // per-work-group SLM budget (bytes)

// ---------------------------------------------------------------------------
// Templated kernel.  QT = query M-tiles per subgroup (each MQ rows, all sharing
// the staged V block); ACC_SLM = keep the O accumulator in SLM instead of
// registers.  bk (runtime, multiple of SG) = KV positions staged per step.
//
// One subgroup owns QT*MQ query rows of one head.  Layouts (bf16 == uint16):
//   Q  : (seq, nq, hd) time-major   Q[qi*nq*hd + h*hd + d]
//   Kc : (kv_len, nkv, hd)          Kc[kj*nkv*hd + b*hd + d]   (b = h / gqa)
//   Vc : (kv_len, nkv, hd)          Vc[kj*nkv*hd + b*hd + d]
//   O  : (seq, nq, hd) time-major   written in place, no scatter
// ---------------------------------------------------------------------------
template <int QT, bool ACC_SLM>
inline void flash(
    GpuEngine& gpu,
    const bf16* Q, bf16* O,
    int seq, int nq, int nkv, int hd,
    const bf16* Kc, const bf16* Vc, int kv_len,
    int past_offset, int sliding_window, bool causal, int bk)
{
    const int hdt = hd / SG;              // hd tiles (both matmuls)
    const int gqa = nq / nkv;
    const int qpt = MQ * QT;              // query rows per subgroup
    const int ntq = (seq + qpt - 1) / qpt;
    const int kv0 = past_offset - (kv_len - seq);
    const int nsub = bk / SG;             // 16-key DPAS sub-blocks per stage
    const int MR = MQ * QT;               // total query rows tracked per subgroup

    auto& q = gpu.queue;
    q.submit([&](sycl::handler& h) {
        // Vt: one bk-key block of V transposed to (hd, kv).
        sycl::local_accessor<bf16, 1> Vt((size_t)hd * bk, h);
        // acc_slm (ACC_SLM only): [qt][nt][m][lane] O accumulator.
        sycl::local_accessor<float, 1> AccS(
            ACC_SLM ? (size_t)QT * hdt * MQ * SG : 1, h);

        h.parallel_for(
            sycl::nd_range<1>((size_t)nq * ntq * SG, SG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg   = it.get_sub_group();
                int  grp  = (int)it.get_group(0);
                int  lane = (int)it.get_local_id(0);
                int  head = grp / ntq;
                int  qbase= (grp % ntq) * qpt;
                int  bkv  = head / gqa;

                // Per-query online-softmax state (replicated across lanes).
                float m_run[MR], l_run[MR];
                for (int r = 0; r < MR; ++r) { m_run[r] = -INFINITY; l_run[r] = 0.0f; }

                // O accumulator: registers, or SLM.  accr sized to the hd=512
                // ceiling; only [*][0..hdt) touched.
                diff_dpas_v8f accr[QT][MAXHDT];
                if constexpr (!ACC_SLM)
                    for (int t = 0; t < QT; ++t)
                        for (int nt = 0; nt < hdt; ++nt) accr[t][nt] = diff_dpas_v8f{};
                auto acc_idx = [&](int t, int nt, int m) {
                    return (size_t)((t * hdt + nt) * MQ + m) * SG + lane;
                };
                if constexpr (ACC_SLM) {
                    for (int t = 0; t < QT; ++t)
                        for (int nt = 0; nt < hdt; ++nt)
                            for (int m = 0; m < MQ; ++m) AccS[acc_idx(t, nt, m)] = 0.0f;
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // Causal / sliding band spanning all QT tiles of this subgroup.
                int q_lo = past_offset + qbase;
                int q_hi = past_offset + std::min(qbase + qpt - 1, seq - 1);
                int ke = kv_len, ks = 0;
                if (causal) {
                    ke = std::min(kv_len, (q_hi - kv0) + 1);
                    ks = (sliding_window == INT_MAX)
                       ? 0 : std::max(0, (q_lo - sliding_window + 1) - kv0);
                }
                int ks_aligned = (ks / bk) * bk;

                for (int kj = ks_aligned; kj < ke; kj += bk) {
                    // ---- stage this bk-key block of V into SLM, transposed to
                    // (hd, kv).  Global read coalesced (consecutive lanes hit
                    // consecutive hd == stride 1); transpose lands in the SLM
                    // write.  OOB keys store 0 (their P is 0 anyway).
                    it.barrier(sycl::access::fence_space::local_space);  // prev block done reading Vt
                    for (int idx = lane; idx < bk * hd; idx += SG) {
                        int kvv = idx / hd, d = idx % hd, k = kj + kvv;
                        Vt[(size_t)d * bk + kvv] =
                            (k < kv_len) ? Vc[(size_t)k * nkv * hd + (size_t)bkv * hd + d] : 0;
                    }
                    it.barrier(sycl::access::fence_space::local_space);  // Vt filled

                    // For each query M-tile and each 16-key sub-block: QKᵀ,
                    // online-softmax step, P·V accumulate.  All QT tiles reuse
                    // the one staged Vt.
                    for (int t = 0; t < QT; ++t) {
                        for (int sb = 0; sb < nsub; ++sb) {
                            int keysub = kj + sb * SG;
                            int key = keysub + lane;
                            bool key_ok = key < kv_len;

                            // 1. S = Q·Kᵀ : c[m] = S[query m][key=lane]
                            diff_dpas_v8f c = diff_dpas_v8f{};
                            for (int kt = 0; kt < hdt; ++kt) {
                                int d0 = kt * SG;
                                diff_dpas_v8s a;
                                for (int m = 0; m < MQ; ++m) {
                                    int qi = qbase + t * MQ + m;
                                    uint16_t qv = (qi < seq)
                                        ? Q[(size_t)qi * nq * hd + head * hd + d0 + lane] : 0;
                                    a[m] = (short)qv;
                                }
                                const bf16* krow = key_ok
                                    ? Kc + (size_t)key * nkv * hd + (size_t)bkv * hd + d0
                                    : nullptr;
                                diff_dpas_v8i b;
                                for (int i = 0; i < 8; ++i) {
                                    uint16_t lo = krow ? krow[2 * i]     : 0;
                                    uint16_t hi = krow ? krow[2 * i + 1] : 0;
                                    b[i] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
                                }
                                c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(SG, a, b, c, kDpasBF16);
                            }

                            // 2. mask + online-softmax step -> P[m] for key=lane
                            float P[MQ];
                            for (int m = 0; m < MQ; ++m) {
                                int r = t * MQ + m;
                                int qi = qbase + r;
                                float s = c[m];
                                bool masked = !key_ok || qi >= seq;
                                if (causal && !masked) {
                                    int qg = past_offset + qi, kg = kv0 + key;
                                    masked = kg > qg ||
                                        (sliding_window != INT_MAX && kg < qg - sliding_window + 1);
                                }
                                if (masked) s = -INFINITY;

                                float bmax = sycl::reduce_over_group(sg, s, sycl::maximum<float>());
                                if (bmax == -INFINITY) { P[m] = 0.0f; continue; }
                                float m_new = sycl::fmax(m_run[r], bmax);
                                float corr  = sycl::exp(m_run[r] == -INFINITY ? 0.0f : m_run[r] - m_new);
                                float p     = (s == -INFINITY) ? 0.0f : sycl::exp(s - m_new);
                                float bsum  = sycl::reduce_over_group(sg, p, sycl::plus<float>());

                                l_run[r] = l_run[r] * corr + bsum;
                                m_run[r] = m_new;
                                // rescale the O accumulator rows for this query by corr
                                if (corr != 1.0f) {
                                    if constexpr (ACC_SLM)
                                        for (int nt = 0; nt < hdt; ++nt) AccS[acc_idx(t, nt, m)] *= corr;
                                    else
                                        for (int nt = 0; nt < hdt; ++nt) accr[t][nt][m] *= corr;
                                }
                                P[m] = p;
                            }

                            // 3. O += P·V.  A: pa[m]=P[query m][key=lane].  B:
                            // Vt row for hd=nt*16+lane, contiguous over kv.
                            diff_dpas_v8s pa;
                            for (int m = 0; m < MQ; ++m) pa[m] = (short)float_to_bf16(P[m]);
                            for (int nt = 0; nt < hdt; ++nt) {
                                size_t vrow = (size_t)(nt * SG + lane) * bk + sb * SG;
                                diff_dpas_v8i b;
                                for (int i = 0; i < 8; ++i)
                                    b[i] = (int)((uint32_t)Vt[vrow + 2 * i] |
                                                 ((uint32_t)Vt[vrow + 2 * i + 1] << 16));
                                if constexpr (ACC_SLM) {
                                    diff_dpas_v8f cc;
                                    for (int m = 0; m < MQ; ++m) cc[m] = AccS[acc_idx(t, nt, m)];
                                    cc = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(SG, pa, b, cc, kDpasBF16);
                                    for (int m = 0; m < MQ; ++m) AccS[acc_idx(t, nt, m)] = cc[m];
                                } else {
                                    accr[t][nt] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                                        SG, pa, b, accr[t][nt], kDpasBF16);
                                }
                            }
                        }
                    }
                }

                // ---- finalize: O[query][head][hd] = acc / l_run (time-major)
                for (int t = 0; t < QT; ++t)
                    for (int nt = 0; nt < hdt; ++nt) {
                        int d = nt * SG + lane;
                        for (int m = 0; m < MQ; ++m) {
                            int r = t * MQ + m, qi = qbase + r;
                            if (qi >= seq) continue;
                            float inv = l_run[r] > 0.0f ? 1.0f / l_run[r] : 0.0f;
                            float a;
                            if constexpr (ACC_SLM) a = AccS[acc_idx(t, nt, m)];
                            else                   a = accr[t][nt][m];
                            O[(size_t)qi * nq * hd + head * hd + d] = float_to_bf16(a * inv);
                        }
                    }
            });
    });
}

inline bool supported(int hd) { return hd % SG == 0 && hd / SG <= MAXHDT; }

// ---------------------------------------------------------------------------
// Dispatch: read the A/B knobs once, clamp to the supported set, budget-check
// the SLM footprint, and call the matching template instantiation.
// ---------------------------------------------------------------------------
inline void launch(
    GpuEngine& gpu,
    const bf16* Q, bf16* O,
    int seq, int nq, int nkv, int hd,
    const bf16* Kc, const bf16* Vc, int kv_len,
    int past_offset, int sliding_window, bool causal)
{
    static const int envQT = [] {
        const char* e = std::getenv("DIFF_FATTN_QTILES");
        int v = e ? std::atoi(e) : 1;
        return (v >= 4) ? 4 : (v >= 2) ? 2 : 1;   // {1,2,4}
    }();
    static const bool envAccSlm = [] {
        const char* e = std::getenv("DIFF_FATTN_ACC");
        return e && std::strcmp(e, "slm") == 0;
    }();
    static const int envBK = [] {
        const char* e = std::getenv("DIFF_FATTN_BK");
        int v = e ? std::atoi(e) : SG;
        v = (v / SG) * SG;                        // floor to multiple of 16
        return v < SG ? SG : v;
    }();

    int qt = envQT, bk = envBK;
    bool acc_slm = envAccSlm;
    const int hdt = hd / SG;

    // SLM budget: Vt (hd*bk bf16) + optional acc (QT*hdt*MQ*SG f32).  Shrink the
    // knobs (bk, then acc->reg, then QT) until it fits, so a too-aggressive combo
    // degrades instead of failing to launch.
    auto slm_bytes = [&](int q_, bool s_, int b_) {
        size_t v = (size_t)hd * b_ * sizeof(bf16);
        size_t a = s_ ? (size_t)q_ * hdt * MQ * SG * sizeof(float) : 0;
        return v + a;
    };
    while (bk > SG && slm_bytes(qt, acc_slm, bk) > kSlmBudget) bk -= SG;
    if (acc_slm && slm_bytes(qt, acc_slm, bk) > kSlmBudget) acc_slm = false;
    while (qt > 1 && slm_bytes(qt, acc_slm, bk) > kSlmBudget) qt >>= 1;

    // (QT, ACC_SLM) -> template instantiation; BK stays runtime.
    auto call = [&](auto QTc, auto ACCc) {
        flash<decltype(QTc)::value, decltype(ACCc)::value>(
            gpu, Q, O, seq, nq, nkv, hd, Kc, Vc, kv_len,
            past_offset, sliding_window, causal, bk);
    };
    #define FATTN_QT(N) \
        if (qt == N) { \
            if (acc_slm) call(std::integral_constant<int,N>{}, std::true_type{}); \
            else         call(std::integral_constant<int,N>{}, std::false_type{}); \
            return; \
        }
    FATTN_QT(4) FATTN_QT(2) FATTN_QT(1)
    #undef FATTN_QT
}

}  // namespace fattn_prefill

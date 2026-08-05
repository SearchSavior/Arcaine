#pragma once
// Fused DECODE attention (seq==1) for Qwen3.5-MoE full-attention layers:
// flash-decoding (KV-split) + XMX/DPAS with GQA reuse, one KV pass total.
//
// The baseline decode path through batched_attention runs ~10 launches with
// 4+ amplified KV passes per layer per token (expand_kv x2 to (nq,kv,hd),
// GEMV scores over k_exp, softmax round-trip, GEMV ctx over v_exp): at 32K
// that is ~1.1 GB of traffic per layer and dominates long-context decode.
//
// Design (sycl-tla flash-epilogue + dense qwen3_5 xmx kernels as references):
//   * One work-group per (kv_head, kv_split); the 8 query heads of the GQA
//     group are the M rows of the 8x16 DPAS tile, so K/V stream exactly once
//     per kv head (no GQA amplification, no expanded copies).
//   * K/V tiles are staged through SLM by WG-cooperative 32B-per-lane
//     vectorized loads (the sycl-tla staged-mainloop pattern): the DPAS
//     fragment layouts are row- and column-scattered in gmem, so direct
//     fragment fills would issue hundreds of 2B gathers per block; SLM
//     absorbs the scatter after one coalesced pass.
//   * 16-key blocks: SG0 computes S = Q[8,256] @ K[16,256]^T via DPAS +
//     online softmax (running m/l per row in registers, probabilities to
//     SLM); all 16 SGs then accumulate O[8, 16-slice] += P[8,16] @ V[16,256]
//     via DPAS. fp32 accumulation end to end; the only bf16 round-trip is P
//     (which the baseline GEMV chain also performs).
//   * KV splits (flash-decoding) give nkv*nsplit work-groups (32 on BMG-G31
//     at kv>=16K -> one per XeCore); a tiny combine kernel merges the
//     (m, l, O) partials across splits.
//
// Guards (dispatch falls back to batched_attention otherwise): seq==1,
// head_dim==256, integral GQA ratio <= 8, skip_mask (mask provably all-zero).
//
// Numerics vs baseline: same bf16 products with fp32 accumulation; the
// baseline's scores bf16 round-trip is skipped and the single-pass softmax
// becomes online, so outputs match to bf16 rounding (validated in
// attention_bench decode mode against the split path).
//
// A/B: QWEN35_ATTN_DECODE_FUSED=1 selects this path (default: split chain).

#include <algorithm>
#include <cstdlib>

#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/engine.hpp"
#include "runtime/quantization/nvfp4.hpp" // diff_dpas_v8*, kNvfp4DpasBF16

namespace qwen35moe_kernels {

inline bool qwen_attn_decode_fused_enabled()
{
    static const bool cached = [] {
        const char* e = std::getenv("QWEN35_ATTN_DECODE_FUSED");
        return e && e[0] == '1';
    }();
    return cached;
}

constexpr int kQwenAttnDecodeMaxSplit = 16;
constexpr int kQwenAttnDecodeRows     = 8;   // GQA group per DPAS M tile
constexpr int kQwenAttnDecodeHd       = 256;

// Scratch elements (fp32) for the decode partials: O per (kv head, split,
// row, dim) plus (m, l) per (kv head, split, row).
inline size_t qwen_attn_decode_fused_scratch_elems(int nkv)
{
    return (size_t)nkv * kQwenAttnDecodeMaxSplit * kQwenAttnDecodeRows *
           (kQwenAttnDecodeHd + 2);
}

// Split count / chunk (multiple of the 16-key DPAS block) for a given
// depth. Target ~1024 tokens/split so nkv*nsplit reaches 32 WGs (one per
// BMG-G31 XeCore) from kv=16K on.
inline int qwen_attn_decode_nsplit(int kv_len)
{
    return std::min((size_t)kQwenAttnDecodeMaxSplit,
                    std::max((size_t)1, ((size_t)kv_len + 1023) / 1024));
}

// Partial pass: one WG (256 lanes = 16 SGs) per (kv_head, split).
// Q: (nq, hd) flat (seq==1); K/V: (kv_len, nkv, hd) cache rows.
// o_part: [(kvh*nsplit+split)*8*hd + row*hd + dim]
// ml_part: [(kvh*nsplit+split)*8*2 + row*2 + {0,1}]
inline void qwen_attn_decode_fused_partial(
    sycl::queue& q,
    const bf16* Q, const bf16* K, const bf16* V,
    float* o_part, float* ml_part,
    int kv_len, int nkv, int nsplit, int chunk, int gqa_ratio, float scale)
{
    constexpr int SG   = 16;
    constexpr int ROWS = kQwenAttnDecodeRows;
    constexpr int HD   = kQwenAttnDecodeHd;
    // bf16 row pad: keeps SLM row starts 16B-aligned (uint4 tile stores)
    // and spreads SG0's row-strided K-fragment reads over 8 banks (2-way
    // conflict, the minimum reachable with 16B-aligned rows).
    constexpr int PAD  = 8;
    constexpr int LDP  = HD + PAD;
    q.submit([&](sycl::handler& h) {
        // q_slm [8][LDP] | k_slm [16][LDP] | v_slm [16][LDP] (bf16), then
        // probs [ROWS*SG] | alphas [ROWS] (f32).
        sycl::local_accessor<bf16, 1> tiles((ROWS + 2 * SG) * LDP, h);
        sycl::local_accessor<float, 1> shared(ROWS * SG + ROWS, h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)nkv * nsplit * HD, HD),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int group = it.get_group(0);
                const int kvh   = group / nsplit;
                const int split = group % nsplit;
                const int kv0   = split * chunk;
                const int kv1   = sycl::min(kv_len, kv0 + chunk);
                const int lid   = it.get_local_id(0);
                auto sg = it.get_sub_group();
                const int tile = (int)sg.get_group_linear_id();
                const int lane = (int)sg.get_local_linear_id();
                const int qh0  = kvh * gqa_ratio;

                bf16* q_slm = &tiles[0];
                bf16* k_slm = q_slm + ROWS * LDP;
                bf16* v_slm = k_slm + SG * LDP;

                // Stage this WG's Q rows (the full GQA group) into SLM once;
                // padding rows (gqa_ratio < 8) are zeroed.
                for (int i = lid; i < ROWS * HD; i += HD) {
                    const int r = i / HD, d = i % HD;
                    q_slm[r * LDP + d] = r < gqa_ratio
                        ? Q[(size_t)(qh0 + r) * HD + d] : float_to_bf16(0.0f);
                }

                diff_dpas_v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
                float m_run[ROWS], l_run[ROWS];
                for (int r = 0; r < ROWS; ++r) {
                    m_run[r] = -INFINITY;
                    l_run[r] = 0.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const size_t row_bytes = (size_t)HD * sizeof(bf16);
                // Cooperative vectorized tile stage: 256 lanes x 32B. Lane i
                // covers row i/16, dims [(i%16)*16, +16) of both the K and V
                // 16-row tiles; rows past kv1 are zeroed. The gmem->register
                // load for block k0+16 is issued before the current block's
                // compute so the latency hides behind the DPAS phases (the
                // sycl-tla staged-mainloop pipeline), then lands in SLM once
                // all SGs are done reading the previous tile.
                sycl::uint4 klo, khi, vlo, vhi;
                auto load_tile = [&](int kb) {
                    const int r  = lid >> 4;         // 0..15
                    const int d0 = (lid & 15) << 4;  // 0..240 step 16
                    klo = khi = vlo = vhi = sycl::uint4(0, 0, 0, 0);
                    if (kb + r < kv1) {
                        const size_t base =
                            ((size_t)(kb + r) * nkv + kvh) * row_bytes + d0 * 2;
                        const char* kp = (const char*)K + base;
                        const char* vp = (const char*)V + base;
                        klo = *(const sycl::uint4*)(kp);
                        khi = *(const sycl::uint4*)(kp + 16);
                        vlo = *(const sycl::uint4*)(vp);
                        vhi = *(const sycl::uint4*)(vp + 16);
                    }
                };
                auto store_tile = [&] {
                    const int r  = lid >> 4;
                    const int d0 = (lid & 15) << 4;
                    bf16* kd = k_slm + r * LDP + d0;
                    bf16* vd = v_slm + r * LDP + d0;
                    *(sycl::uint4*)(kd)      = klo;
                    *(sycl::uint4*)(kd + 8)  = khi;
                    *(sycl::uint4*)(vd)      = vlo;
                    *(sycl::uint4*)(vd + 8)  = vhi;
                };

                load_tile(kv0);
                store_tile();
                it.barrier(sycl::access::fence_space::local_space);

                for (int k0 = kv0; k0 < kv1; k0 += SG) {
                    const int knext = k0 + SG;
                    if (knext < kv1) load_tile(knext);   // prefetch next tile

                    if (tile == 0) {
                        // S = Q[8,256] @ K[16,256]^T for this 16-key block.
                        diff_dpas_v8f scores = {0, 0, 0, 0, 0, 0, 0, 0};
                        for (int kt = 0; kt < HD / SG; ++kt) {
                            const int dim = kt * SG + lane;
                            diff_dpas_v8s qf;
                            for (int r = 0; r < ROWS; ++r)
                                qf[r] = (short)q_slm[r * LDP + dim];
                            diff_dpas_v8i kf;
                            const bf16* krow = k_slm + lane * LDP;
                            for (int p = 0; p < 8; ++p) {
                                uint32_t packed =
                                    (uint32_t)krow[kt * 16 + p * 2] |
                                    ((uint32_t)krow[kt * 16 + p * 2 + 1] << 16);
                                kf[p] = (int)packed;
                            }
                            scores = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                                16, qf, kf, scores, kNvfp4DpasBF16);
                        }
                        // Online softmax per row over the block.
                        for (int r = 0; r < ROWS; ++r) {
                            const bool vis = r < gqa_ratio && (k0 + lane) < kv1;
                            const float s = vis ? scores[r] * scale : -INFINITY;
                            const float bm = sycl::reduce_over_group(
                                sg, s, sycl::maximum<float>());
                            const float mn = sycl::fmax(m_run[r], bm);
                            const float alpha = sycl::exp(m_run[r] - mn);
                            const float p = vis ? sycl::exp(s - mn) : 0.0f;
                            const float bs = sycl::reduce_over_group(
                                sg, p, sycl::plus<float>());
                            l_run[r] = l_run[r] * alpha + bs;
                            m_run[r] = mn;
                            shared[r * SG + lane] = p;
                            if (lane == 0) shared[ROWS * SG + r] = alpha;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // O[8, tile-slice] = O*alpha + P[8,16] @ V[16, tile-slice].
                    for (int r = 0; r < ROWS; ++r) acc[r] *= shared[ROWS * SG + r];
                    diff_dpas_v8s pf;
                    for (int r = 0; r < ROWS; ++r)
                        pf[r] = (short)float_to_bf16(shared[r * SG + lane]);
                    diff_dpas_v8i vf;
                    const int odim = tile * SG + lane;
                    for (int p = 0; p < 8; ++p) {
                        uint32_t packed =
                            (uint32_t)v_slm[(p * 2) * LDP + odim] |
                            ((uint32_t)v_slm[(p * 2 + 1) * LDP + odim] << 16);
                        vf[p] = (int)packed;
                    }
                    acc = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, pf, vf, acc, kNvfp4DpasBF16);
                    it.barrier(sycl::access::fence_space::local_space);
                    if (knext < kv1) {
                        store_tile();   // prefetched tile -> SLM
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                }

                const size_t pbase = ((size_t)kvh * nsplit + split) * ROWS * HD;
                for (int r = 0; r < ROWS; ++r)
                    o_part[pbase + (size_t)r * HD + tile * SG + lane] = acc[r];
                if (tile == 0 && lane < ROWS) {
                    const size_t mbase = ((size_t)kvh * nsplit + split) * ROWS * 2;
                    ml_part[mbase + lane * 2 + 0] = m_run[lane];
                    ml_part[mbase + lane * 2 + 1] = l_run[lane];
                }
            });
    });
}

// Combine pass: one WG (256 lanes = hd) per query head; merges the split
// partials with the standard flash epilogue and normalizes.
inline void qwen_attn_decode_fused_combine(
    sycl::queue& q,
    const float* o_part, const float* ml_part, bf16* out,
    int nq, int nkv, int nsplit, int gqa_ratio)
{
    constexpr int ROWS = kQwenAttnDecodeRows;
    constexpr int HD   = kQwenAttnDecodeHd;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)nq * HD, HD),
            [=](sycl::nd_item<1> it) {
                const int qh  = it.get_group(0);
                const int d   = it.get_local_id(0);
                const int kvh = qh / gqa_ratio;
                const int row = qh % gqa_ratio;
                float M = -INFINITY;
                for (int s = 0; s < nsplit; ++s) {
                    const size_t mbase = ((size_t)kvh * nsplit + s) * ROWS * 2;
                    M = sycl::fmax(M, ml_part[mbase + row * 2]);
                }
                float acc = 0.0f, L = 0.0f;
                for (int s = 0; s < nsplit; ++s) {
                    const size_t base = ((size_t)kvh * nsplit + s);
                    const float m = ml_part[base * ROWS * 2 + row * 2];
                    const float l = ml_part[base * ROWS * 2 + row * 2 + 1];
                    const float w = sycl::exp(m - M);   // 0 for an empty split
                    acc += w * o_part[base * ROWS * HD + (size_t)row * HD + d];
                    L   += w * l;
                }
                out[(size_t)qh * HD + d] = float_to_bf16(acc / L);
            });
    });
}

} // namespace qwen35moe_kernels

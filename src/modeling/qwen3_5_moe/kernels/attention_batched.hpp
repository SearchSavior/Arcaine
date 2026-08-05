#pragma once

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>

#include "runtime/gpu/ops.hpp"
#include "attention_mask.hpp"
#include "attention_layout.hpp"
#include "attention_decode_fused.hpp"

// Model-local namespace: these kernels are per-model COPIES (see
// AGENTS.md model isolation). Global-scope inline functions with
// identical names in other models would ODR-merge at link time;
// divergent bodies (e.g. tiled attention) then crash at runtime.
namespace qwen35moe_kernels {

// ---------------------------------------------------------------------------
// Reusable GQA-aware batched dot-product attention building blocks.
//
// These pieces compose the low-level layout helpers into a full causal SDPA
// pipeline over time-major Q/K/V tensors.
//
//   Q:  (seq, nq_heads, q_head_dim)        — time-major (as projected)
//   K:  (kv_len, nkv_heads, kv_head_dim)   — from cache
//   V:  (kv_len, nkv_heads, kv_head_dim)   — from cache
//
// `scale` multiplies the raw QK^T scores before masking/softmax. Gemma4 passes
// 1.0 (Q and K are RMSNorm-ed per head); architectures using the conventional
// 1/sqrt(head_dim) factor pass that instead.
// ---------------------------------------------------------------------------

// Apply scale + score mask and FP32 conversion in one kernel.
// Use size_t for total/index: nq*seq*kv_len can exceed INT_MAX at long contexts.
inline void apply_mask_f32(
    sycl::queue& q,
    const bf16* scores_bf16, float* scores_f32,
    const float* mask,
    int nq, int seq, int kv_len,
    float scale
) {
    size_t total = (size_t)nq * seq * kv_len;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            size_t i  = gid[0];
            int    sq = (int)((i / (size_t)kv_len) % (size_t)seq);
            int    kv = (int)(i % (size_t)kv_len);
            scores_f32[i] = bf16_to_float(scores_bf16[i]) * scale + mask[sq * kv_len + kv];
        });
    });
}

// Scale + BF16->FP32 without mask (mask is all-zeros).
inline void scores_bf16_to_f32(
    sycl::queue& q,
    const bf16* src, float* dst, size_t n,
    float scale = 1.0f
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            dst[gid[0]] = bf16_to_float(src[gid[0]]) * scale;
        });
    });
}

// Convert FP32 softmax output back to BF16.
inline void f32_to_bf16_buf(sycl::queue& q, const float* src, bf16* dst, size_t n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            size_t i = id[0];
            dst[i] = float_to_bf16(src[i]);
        });
    });
}

// Grow-only scratch for batched_attention. The attention path runs once per
// full-attention layer per forward call, so per-call USM alloc/free would
// dominate launch overhead. The CALLER must hold scratch_mutex() across the
// batched_attention call and any consumption of the returned pointer (a
// resize by a concurrent session would free the buffer behind it). Safe on
// the in-order queue: kernels from the previous layer complete before the
// next layer's kernels reuse the same buffers.
namespace qwen_attn_batched_detail {
struct Scratch {
    GpuBuffer<float> mask;        // [seq*kv]
    GpuBuffer<bf16>  q_hm;        // [nq*seq*q_hd]
    GpuBuffer<bf16>  k_exp;       // [nq*kv*kv_hd]
    GpuBuffer<bf16>  v_exp;       // [nq*kv*kv_hd]
    GpuBuffer<bf16>  scores_bf16; // [nq*seq*kv]
    GpuBuffer<float> scores_f32;  // [nq*seq*kv]
    GpuBuffer<bf16>  ctx_hm;      // [nq*seq*q_hd] (tiled: o_tile)
    GpuBuffer<bf16>  ctx_tm;      // [seq*nq*q_hd] (returned)
    // Tiled (flash-style) path only:
    GpuBuffer<bf16>  kt_exp;      // [nq*tile*kv_hd]
    GpuBuffer<bf16>  vt_exp;      // [nq*tile*kv_hd]
    GpuBuffer<bf16>  p_tile;      // [nq*seq*tile] scores/P
    GpuBuffer<float> o_acc;       // [nq*seq*q_hd] f32 running output
    GpuBuffer<float> ml;          // [4, nq*seq]: m_run | l_run | m_t | l_t
    // Decode-fused (flash-decoding) path only:
    GpuBuffer<float> o_part;      // [nkv*MAXSPLIT*8*hd] split partials
    GpuBuffer<float> ml_part;     // [nkv*MAXSPLIT*8*2]: m | l per partial
    size_t mask_cap = 0;
    size_t q_cap = 0;             // covers q_hm / ctx_hm / ctx_tm (nq*seq*q_hd)
    size_t kv_cap = 0;            // covers k_exp / v_exp (nq*kv*kv_hd)
    size_t scores_cap = 0;
    size_t tile_kv_cap = 0;       // covers kt_exp / vt_exp
    size_t p_cap = 0;             // covers p_tile
    size_t acc_cap = 0;           // covers o_acc (rows*q_hd) and ml (rows*4)
    size_t part_cap = 0;          // covers o_part / ml_part (decode fused)
};
inline Scratch& scratch()
{
    static Scratch s;
    return s;
}
inline std::mutex& scratch_mutex()
{
    static std::mutex m;
    return m;
}
} // namespace qwen_attn_batched_detail

// Read per call (no static caching) so tests can flip them between calls.
// QWEN35_ATTN_SCORES_BUDGET_MB: single-shot path is used while
// nq*seq*kv*6B (bf16 + f32 scores) fits the budget; above it the KV-tiled
// online-softmax path runs instead (default 2048 MB -> pp<=4096 single-shot,
// pp>=8192 tiled).
inline size_t qwen_attn_scores_budget_bytes()
{
    const char* v = std::getenv("QWEN35_ATTN_SCORES_BUDGET_MB");
    long mb = v ? std::atol(v) : 2048;
    if (mb < 0) mb = 0;
    return (size_t)mb << 20;
}
// QWEN35_ATTN_TILE: KV tile length for the tiled path (0 = auto).
inline int qwen_attn_tile_override()
{
    const char* v = std::getenv("QWEN35_ATTN_TILE");
    return v ? std::atoi(v) : 0;
}

// Worst-case device bytes batched_attention will ensure in grow-only scratch
// for one call with (seq, kv). Mirrors the dispatch rule and alloc blocks
// below exactly; used by QwenModel's max_seq preflight. Keep in sync.
inline size_t qwen_attn_scratch_bytes(int seq, int kv, int nq, int nkv,
                                      int q_hd, int kv_hd)
{
    const size_t rows      = (size_t)nq * seq;
    const size_t q_sz      = rows * q_hd;
    const size_t scores_sz = rows * kv;
    size_t total = 3 * q_sz * sizeof(bf16);               // q_hm / ctx_hm / ctx_tm
    if (qwen_attn_decode_fused_enabled())
        total += qwen_attn_decode_fused_scratch_elems(nkv) * sizeof(float);
    if (scores_sz * 6 <= qwen_attn_scores_budget_bytes()) {
        // single-shot
        total += 2 * (size_t)nq * kv * kv_hd * sizeof(bf16); // k_exp / v_exp
        total += scores_sz * (sizeof(bf16) + sizeof(float)); // scores bf16 + f32
        total += (size_t)seq * kv * sizeof(float);           // mask
    } else {
        // tiled (mirror of the tile computation in batched_attention_tiled)
        int tile = qwen_attn_tile_override();
        if (tile <= 0) {
            size_t t = ((size_t)1 << 28) / rows;
            tile = (int)std::min<size_t>(std::max<size_t>(t & ~(size_t)255, 256), 4096);
        }
        tile = std::min(tile, kv);
        total += 2 * (size_t)nq * tile * kv_hd * sizeof(bf16); // kt_exp / vt_exp
        total += rows * tile * sizeof(bf16);                   // p_tile
        total += q_sz * sizeof(float);                         // o_acc
        total += rows * 4 * sizeof(float);                     // ml
    }
    return total;
}

// ---------------------------------------------------------------------------
// KV-tiled flash-style attention (online softmax), for prefills whose full
// (nq, seq, kv) scores would exceed the memory budget. Composed from the
// batched GEMMs + fused elementwise kernels; the mask is evaluated inline
// (never materialized) and scores live only per (nq, seq, tile) slice.
//
//   per tile: S_t = Q_hm @ K_t^T;  m_t/l_t = row max/sum of exp(S_t*scale+mask)
//             O   = O*exp(m-m') + (P_t @ V_t)*exp(m_t-m');  l, m updated
//   final:    ctx = O / l
// ---------------------------------------------------------------------------

// Per-row (nq*seq rows, one WG per row) partial softmax over one KV tile:
// applies scale + inline causal/sliding mask, computes the row max m_t and
// the exp sum l_t, and rewrites the scores buffer with P = exp(x - m_t) bf16.
// A fully masked tile row yields m_t=-inf, P=0, l_t=0 (no NaN).
inline void flash_partial_softmax(
    sycl::queue& q, bf16* scores,  // (rows, tlen) in-place -> P
    float* m_t, float* l_t,
    size_t rows, int seq, int tlen,
    int kv0_global,           // absolute position of the tile's first kv
    int past_offset, int sliding_window,
    float scale, bool skip_mask
) {
    constexpr int WG = 256;
    static const float NEG_INF = -std::numeric_limits<float>::infinity();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
                const size_t row = it.get_group(0);
                const int    lid = it.get_local_id(0);
                const int    s   = (int)(row % (size_t)seq);
                const int    q_pos = past_offset + s;
                const bf16*  srow = scores + row * (size_t)tlen;
                bf16*        prow = scores + row * (size_t)tlen;

                auto masked_val = [&](int j) -> float {
                    float x = bf16_to_float(srow[j]) * scale;
                    if (!skip_mask) {
                        int kvg = kv0_global + j;
                        bool m = kvg > q_pos ||
                            (sliding_window != INT_MAX &&
                             kvg < q_pos - sliding_window + 1);
                        if (m) x = NEG_INF;
                    }
                    return x;
                };

                float mx = NEG_INF;
                for (int j = lid; j < tlen; j += WG)
                    mx = sycl::fmax(mx, masked_val(j));
                mx = sycl::reduce_over_group(it.get_group(), mx,
                                             sycl::maximum<float>());

                float sum = 0.0f;
                for (int j = lid; j < tlen; j += WG) {
                    // mx == -inf (fully masked tile row): P = 0, l_t = 0.
                    float p = (mx == NEG_INF) ? 0.0f
                                              : sycl::exp(masked_val(j) - mx);
                    prow[j] = float_to_bf16(p);
                    sum += p;
                }
                sum = sycl::reduce_over_group(it.get_group(), sum,
                                              sycl::plus<float>());
                if (lid == 0) { m_t[row] = mx; l_t[row] = sum; }
            });
    });
}

// Online-softmax combine: O = O*exp(m-m') + O_t*exp(m_t-m'), with
// m' = max(m, m_t), l = l*exp(m-m') + l_t*exp(m_t-m'). On the first tile
// (first=true) O/m/l are initialized from the tile directly (avoids having to
// zero O_acc; m is always finite after tile 0 since kv 0 attends to all
// queries). O_t is the bf16 P@V product for the current tile.
inline void flash_combine(
    sycl::queue& q,
    float* o_acc, const bf16* o_tile,
    float* m_run, float* l_run, const float* m_t, const float* l_t,
    size_t rows, int hd, bool first
) {
    const size_t total = rows * (size_t)hd;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            size_t i = gid[0];
            size_t r = i / (size_t)hd;
            if (first) {
                o_acc[i] = bf16_to_float(o_tile[i]);
                if (i % (size_t)hd == 0) { m_run[r] = m_t[r]; l_run[r] = l_t[r]; }
                return;
            }
            float mo = m_run[r];
            float mn = sycl::fmax(mo, m_t[r]);
            float a  = sycl::exp(mo - mn);
            float b  = sycl::exp(m_t[r] - mn);   // 0 for a fully masked tile
            o_acc[i] = o_acc[i] * a + bf16_to_float(o_tile[i]) * b;
            if (i % (size_t)hd == 0) {
                m_run[r] = mn;
                l_run[r] = l_run[r] * a + l_t[r] * b;
            }
        });
    });
}

// Final normalize: ctx_hm = O_acc / l (bf16 out). l > 0 always (tile 0 has an
// unmasked entry for every query).
inline void flash_finalize(
    sycl::queue& q, const float* o_acc, const float* l_run,
    bf16* ctx_hm, size_t rows, int hd
) {
    const size_t total = rows * (size_t)hd;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            size_t i = gid[0];
            ctx_hm[i] = float_to_bf16(o_acc[i] / l_run[i / (size_t)hd]);
        });
    });
}

// KV-tiled batched attention. Same signature/semantics as batched_attention
// minus block_ids (the tiled path evaluates the mask inline and does not
// support vision blocks); returns the same scratch-owned ctx_tm pointer.
// The caller holds qwen_attn_batched_detail::scratch_mutex().
inline bf16* batched_attention_tiled(
    GpuEngine& ctx,
    const bf16* Q_dev, int seq_len, int nq_heads, int q_head_dim,
    const bf16* K_dev, const bf16* V_dev, int kv_len, int nkv_heads, int kv_head_dim,
    int past_offset, int sliding_window,
    float scale, bool skip_mask
) {
    auto& q  = ctx.queue;
    auto& sc = qwen_attn_batched_detail::scratch();

    const size_t rows = (size_t)nq_heads * seq_len;
    const size_t q_sz = rows * q_head_dim;

    // Tile length: cap p_tile at ~2^28 elements (512 MB bf16), multiple of
    // 256, clamped to [256, 4096]; env override for tests/tuning.
    int tile = qwen_attn_tile_override();
    if (tile <= 0) {
        size_t t = ((size_t)1 << 28) / rows;
        tile = (int)std::min<size_t>(std::max<size_t>(t & ~(size_t)255, 256), 4096);
    }
    tile = std::min(tile, kv_len);

    const size_t tile_kv_sz = (size_t)nq_heads * tile * kv_head_dim;
    const size_t p_sz       = rows * tile;

    if (sc.q_cap < q_sz) {
        sc.q_hm   = GpuBuffer<bf16>(q_sz, q);
        sc.ctx_hm = GpuBuffer<bf16>(q_sz, q);
        sc.ctx_tm = GpuBuffer<bf16>(q_sz, q);
        sc.q_cap  = q_sz;
    }
    if (sc.tile_kv_cap < tile_kv_sz) {
        sc.kt_exp      = GpuBuffer<bf16>(tile_kv_sz, q);
        sc.vt_exp      = GpuBuffer<bf16>(tile_kv_sz, q);
        sc.tile_kv_cap = tile_kv_sz;
    }
    if (sc.p_cap < p_sz) {
        sc.p_tile = GpuBuffer<bf16>(p_sz, q);
        sc.p_cap  = p_sz;
    }
    if (sc.acc_cap < q_sz) {
        sc.o_acc   = GpuBuffer<float>(q_sz, q);
        sc.ml      = GpuBuffer<float>(rows * 4, q);
        sc.acc_cap = q_sz;
    }
    float* m_run = sc.ml.data();
    float* l_run = sc.ml.data() + rows;
    float* m_t   = sc.ml.data() + rows * 2;
    float* l_t   = sc.ml.data() + rows * 3;

    // Q: time-major (seq, nq, hd) -> head-major (nq, seq, hd), once.
    transpose_q_into(q, sc.q_hm.data(), Q_dev, seq_len, nq_heads, q_head_dim);

    // Absolute position of kv[0] (mirrors fill_causal_mask).
    const int kv_base = past_offset - (kv_len - seq_len);

    bool first = true;
    for (int kv0 = 0; kv0 < kv_len; kv0 += tile) {
        const int tlen = std::min(tile, kv_len - kv0);
        const bf16* K_slice = K_dev + (size_t)kv0 * nkv_heads * kv_head_dim;
        const bf16* V_slice = V_dev + (size_t)kv0 * nkv_heads * kv_head_dim;

        expand_kv_into(q, sc.kt_exp.data(), K_slice, tlen, nkv_heads, nq_heads, kv_head_dim);
        expand_kv_into(q, sc.vt_exp.data(), V_slice, tlen, nkv_heads, nq_heads, kv_head_dim);

        // S_t = Q_hm @ K_t^T -> (nq, seq, tlen)
        matmul_bf16_batched(sc.q_hm.data(), nq_heads, seq_len, q_head_dim,
                            sc.kt_exp.data(), tlen, /*transpose_W=*/true,
                            sc.p_tile.data(), ctx);
        flash_partial_softmax(q, sc.p_tile.data(), m_t, l_t,
                              rows, seq_len, tlen, kv_base + kv0,
                              past_offset, sliding_window, scale, skip_mask);
        // O_t = P_t @ V_t -> (nq, seq, q_hd), into ctx_hm (o_tile role)
        matmul_bf16_batched(sc.p_tile.data(), nq_heads, seq_len, tlen,
                            sc.vt_exp.data(), q_head_dim, /*transpose_W=*/false,
                            sc.ctx_hm.data(), ctx);
        flash_combine(q, sc.o_acc.data(), sc.ctx_hm.data(),
                      m_run, l_run, m_t, l_t, rows, q_head_dim, first);
        first = false;
    }

    flash_finalize(q, sc.o_acc.data(), l_run, sc.ctx_hm.data(), rows, q_head_dim);
    scatter_ctx(q, sc.ctx_hm.data(), sc.ctx_tm.data(), nq_heads, seq_len, q_head_dim);
    return sc.ctx_tm.data();
}

// Fused decode attention (seq==1): flash-decoding + DPAS with GQA reuse,
// one KV pass, 2 launches. Returns the same scratch-owned ctx_tm pointer as
// batched_attention ([nq*hd] flat == (1, nq, hd) time-major). The caller
// holds qwen_attn_batched_detail::scratch_mutex().
inline bf16* qwen_attn_decode_fused(
    GpuEngine& ctx,
    const bf16* Q_dev, int nq_heads,
    const bf16* K_dev, const bf16* V_dev, int kv_len, int nkv_heads,
    float scale
) {
    auto& q  = ctx.queue;
    auto& sc = qwen_attn_batched_detail::scratch();
    constexpr int HD = kQwenAttnDecodeHd;
    const int ratio  = nq_heads / nkv_heads;
    const int nsplit = qwen_attn_decode_nsplit(kv_len);
    const int chunk  = (((kv_len + nsplit - 1) / nsplit) + 15) & ~15;

    const size_t q_sz = (size_t)nq_heads * HD;
    if (sc.q_cap < q_sz) {
        sc.q_hm   = GpuBuffer<bf16>(q_sz, q);
        sc.ctx_hm = GpuBuffer<bf16>(q_sz, q);
        sc.ctx_tm = GpuBuffer<bf16>(q_sz, q);
        sc.q_cap  = q_sz;
    }
    const size_t o_elems = (size_t)nkv_heads * kQwenAttnDecodeMaxSplit *
                           kQwenAttnDecodeRows * HD;
    if (sc.part_cap < o_elems) {
        sc.o_part   = GpuBuffer<float>(o_elems, q);
        sc.ml_part  = GpuBuffer<float>(
            (size_t)nkv_heads * kQwenAttnDecodeMaxSplit * kQwenAttnDecodeRows * 2, q);
        sc.part_cap = o_elems;
    }

    qwen_attn_decode_fused_partial(q, Q_dev, K_dev, V_dev,
                                   sc.o_part.data(), sc.ml_part.data(),
                                   kv_len, nkv_heads, nsplit, chunk, ratio, scale);
    qwen_attn_decode_fused_combine(q, sc.o_part.data(), sc.ml_part.data(),
                                   sc.ctx_tm.data(),
                                   nq_heads, nkv_heads, nsplit, ratio);
    return sc.ctx_tm.data();
}

// Full GQA batched attention: scores = scale * QK^T, mask, softmax, then @ V.
// Returns a pointer to the context in (seq, nq, q_head_dim) time-major layout,
// ready for o_proj. The pointer is into grow-only scratch; the caller must
// hold qwen_attn_batched_detail::scratch_mutex() until all kernels consuming
// it have been enqueued.
// Baseline batched attention (single-shot / KV-tiled chain). Exposed for
// A/B benchmarks that must run the split decode path regardless of the
// QWEN35_ATTN_DECODE_FUSED env toggle; production callers use
// batched_attention.
inline bf16* batched_attention_baseline(
    GpuEngine& ctx,
    const bf16* Q_dev, int seq_len, int nq_heads, int q_head_dim,
    const bf16* K_dev, const bf16* V_dev, int kv_len, int nkv_heads, int kv_head_dim,
    int past_offset,
    int sliding_window,
    float scale = 1.0f,
    bool skip_mask = false,
    const int32_t* block_ids = nullptr
) {
    auto& q = ctx.queue;
    auto& sc = qwen_attn_batched_detail::scratch();

    const size_t q_sz      = (size_t)nq_heads * seq_len * q_head_dim;
    const size_t kv_sz     = (size_t)nq_heads * kv_len * kv_head_dim;
    const size_t scores_sz = (size_t)nq_heads * seq_len * kv_len;
    const size_t mask_sz   = (size_t)seq_len * kv_len;

    // Long-context dispatch: if the materialized bf16+f32 scores would exceed
    // the budget, run the KV-tiled online-softmax path instead (no block_ids
    // support there; this model never passes them).
    if (!block_ids && scores_sz * 6 > qwen_attn_scores_budget_bytes()) {
        return batched_attention_tiled(ctx,
            Q_dev, seq_len, nq_heads, q_head_dim,
            K_dev, V_dev, kv_len, nkv_heads, kv_head_dim,
            past_offset, sliding_window, scale, skip_mask);
    }

    if (sc.q_cap < q_sz) {
        sc.q_hm   = GpuBuffer<bf16>(q_sz, q);
        sc.ctx_hm = GpuBuffer<bf16>(q_sz, q);
        sc.ctx_tm = GpuBuffer<bf16>(q_sz, q);
        sc.q_cap  = q_sz;
    }
    if (sc.kv_cap < kv_sz) {
        sc.k_exp  = GpuBuffer<bf16>(kv_sz, q);
        sc.v_exp  = GpuBuffer<bf16>(kv_sz, q);
        sc.kv_cap = kv_sz;
    }
    if (sc.scores_cap < scores_sz) {
        sc.scores_bf16 = GpuBuffer<bf16>(scores_sz, q);
        sc.scores_f32  = GpuBuffer<float>(scores_sz, q);
        sc.scores_cap  = scores_sz;
    }

    // 1. Causal mask: (seq, kv_len)
    const float* mask = nullptr;
    if (!skip_mask) {
        if (sc.mask_cap < mask_sz) {
            sc.mask     = GpuBuffer<float>(mask_sz, q);
            sc.mask_cap = mask_sz;
        }
        if (block_ids) {
            fill_causal_mask_with_block_ids(q, sc.mask.data(), block_ids,
                                            seq_len, kv_len, past_offset, sliding_window);
        } else {
            fill_causal_mask(q, sc.mask.data(), seq_len, kv_len, past_offset, sliding_window);
        }
        mask = sc.mask.data();
    }

    // 2. Q: time-major (seq, nq, hd) -> head-major (nq, seq, hd)
    transpose_q_into(q, sc.q_hm.data(), Q_dev, seq_len, nq_heads, q_head_dim);

    // 3. K and V: time-major (kv_len, nkv, hd) -> head-major (nq, kv_len, hd) + GQA expansion
    expand_kv_into(q, sc.k_exp.data(), K_dev, kv_len, nkv_heads, nq_heads, kv_head_dim);
    expand_kv_into(q, sc.v_exp.data(), V_dev, kv_len, nkv_heads, nq_heads, kv_head_dim);

    // 4. Scores: (nq, seq, kv_len) = Q_hm @ K_exp^T
    matmul_bf16_batched(sc.q_hm.data(), nq_heads, seq_len, q_head_dim,
                        sc.k_exp.data(), kv_len, /*transpose_W=*/true,
                        sc.scores_bf16.data(), ctx);

    // 5. Scale + mask + BF16->FP32
    if (skip_mask) {
        scores_bf16_to_f32(q, sc.scores_bf16.data(), sc.scores_f32.data(),
                           scores_sz, scale);
    } else {
        apply_mask_f32(q, sc.scores_bf16.data(), sc.scores_f32.data(), mask,
                       nq_heads, seq_len, kv_len, scale);
    }

    // 6. Softmax over kv_len: treat as (nq*seq, kv_len)
    softmax_f32(sc.scores_f32.data(), nq_heads * seq_len, kv_len, ctx);

    // 7. FP32->BF16 (reuse scores_bf16 buffer)
    f32_to_bf16_buf(q, sc.scores_f32.data(), sc.scores_bf16.data(), scores_sz);

    // 8. Context: (nq, seq, q_hd) = scores @ V_exp
    matmul_bf16_batched(sc.scores_bf16.data(), nq_heads, seq_len, kv_len,
                        sc.v_exp.data(), q_head_dim, /*transpose_W=*/false,
                        sc.ctx_hm.data(), ctx);

    // 9. Scatter back to (seq, nq, q_hd) time-major for o_proj
    scatter_ctx(q, sc.ctx_hm.data(), sc.ctx_tm.data(), nq_heads, seq_len, q_head_dim);

    return sc.ctx_tm.data();
}

// Full GQA batched attention dispatcher: routes decode (seq==1) to the fused
// flash-decoding DPAS path when enabled and shape-compatible; otherwise the
// baseline single-shot / KV-tiled chain.
inline bf16* batched_attention(
    GpuEngine& ctx,
    const bf16* Q_dev, int seq_len, int nq_heads, int q_head_dim,
    const bf16* K_dev, const bf16* V_dev, int kv_len, int nkv_heads, int kv_head_dim,
    int past_offset,
    int sliding_window,
    float scale = 1.0f,
    bool skip_mask = false,  // true when mask is provably all-zeros (decode after KV truncation)
    const int32_t* block_ids = nullptr
) {
    // Decode (seq==1) fused path: flash-decoding + DPAS, one KV pass. The
    // mask is provably all-zero (skip_mask) and this model uses no sliding
    // window, so all cached tokens attend. Below kv=2048 the fixed split
    // overhead outweighs the traffic win (see attention_bench decode mode).
    if (seq_len == 1 && skip_mask && !block_ids && kv_len >= 2048 &&
        qwen_attn_decode_fused_enabled() &&
        q_head_dim == kQwenAttnDecodeHd && kv_head_dim == kQwenAttnDecodeHd &&
        nq_heads % nkv_heads == 0 &&
        nq_heads / nkv_heads <= kQwenAttnDecodeRows) {
        return qwen_attn_decode_fused(ctx, Q_dev, nq_heads,
                                      K_dev, V_dev, kv_len, nkv_heads, scale);
    }
    return batched_attention_baseline(ctx,
        Q_dev, seq_len, nq_heads, q_head_dim,
        K_dev, V_dev, kv_len, nkv_heads, kv_head_dim,
        past_offset, sliding_window, scale, skip_mask, block_ids);
}

} // namespace qwen35moe_kernels

#pragma once
// Leaf attention kernels (RoPE-norm + masked softmax) for DiffusionGemma.
//
// Header-only + inline so they can be exercised directly by the
// diffusion_attn_selfcheck binary (links only DNNL::dnnl, like the qwen
// self-checks) without pulling the full attention.cpp / oneDNN-matmul deps.
// The higher-level orchestration (project_qkv, gqa_attention, ...) lives in
// attention.cpp and #includes this header.
//
// Two AB-test env knobs (cached once per process, like onednn_sdpa_mode):
//   DIFF_FUSED_NORM_ROPE = pow|table|fused   (default: table)
//     pow   = original per-head pow/cos/sin recomputation (baseline)
//     table = precompute one (seq, n_active_pairs) cos/sin table, share heads
//     fused = single kernel: one work-group per token, cos/sin in local memory,
//             loop over heads reusing it (no separate table fill launch)
//   DIFF_SOFTMAX_SPAN = 0|1                   (default: 1, contiguous span)
//     0 = original per-column masked() branch over the full row
//     1 = compute contiguous [lo,hi) valid span, iterate only it (no exp on
//         masked columns)
#include <sycl/sycl.hpp>
#include "arena.hpp"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Env knobs
// ---------------------------------------------------------------------------
enum class RopeKernelMode { Pow, Table, Fused };

inline RopeKernelMode diff_fused_norm_rope_mode() {
    static RopeKernelMode mode = [] {
        const char* e = std::getenv("DIFF_FUSED_NORM_ROPE");
        if (e) {
            if (std::strcmp(e, "pow") == 0 || std::strcmp(e, "0") == 0 ||
                std::strcmp(e, "off") == 0)
                return RopeKernelMode::Pow;
            if (std::strcmp(e, "fused") == 0 || std::strcmp(e, "1") == 0)
                return RopeKernelMode::Fused;
        }
        return RopeKernelMode::Table;  // default: shared cos/sin table
    }();
    return mode;
}

inline bool softmax_span_mode() {
    static bool span = [] {
        const char* e = std::getenv("DIFF_SOFTMAX_SPAN");
        if (!e) return true;  // default: contiguous-span (shipped)
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 ||
                 std::strcmp(e, "false") == 0 || std::strcmp(e, "no") == 0);
    }();
    return span;
}

// ---------------------------------------------------------------------------
// Shared RoPE cos/sin table (mode == Table / Fused-for-KV).
// ---------------------------------------------------------------------------
struct RopeTable {
    diffarena::Alloc<float> cos, sin;
    int n_pairs = 0;
    bool empty() const { return n_pairs == 0; }
    const float* cos_data() const { return cos.data(); }
    const float* sin_data() const { return sin.data(); }
};

inline RopeTable make_rope_table(GpuEngine& ctx, int seq, int offset,
                                 int hd, float theta, float partial) {
    RopeTable r;
    r.n_pairs = static_cast<int>(partial * hd / 2.0f);
    if (r.n_pairs <= 0 || seq <= 0) { r.n_pairs = 0; return r; }
    auto& ar = diffarena::arena(ctx.index);
    r.cos = ar.alloc<float>((size_t)seq * r.n_pairs);
    r.sin = ar.alloc<float>((size_t)seq * r.n_pairs);
    float freq_denom = static_cast<float>(hd);
    float* cos = r.cos.data();
    float* sin = r.sin.data();
    int n_pairs = r.n_pairs;
    ctx.queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)seq * n_pairs), [=](sycl::id<1> gid) {
            int idx = (int)gid[0];
            int p = idx % n_pairs;
            int t = idx / n_pairs;
            float inv_freq = 1.0f / sycl::pow(theta, 2.0f * p / freq_denom);
            float angle = (float)(offset + t) * inv_freq;
            cos[idx] = sycl::cos(angle);
            sin[idx] = sycl::sin(angle);
        });
    });
    return r;
}

// ---------------------------------------------------------------------------
// Fused RMSNorm + RoPE, one work-group per (token, head).
// rope_cos/rope_sin == nullptr -> compute pow/cos/sin inline (Pow baseline).
// otherwise -> read the shared (seq, n_active_pairs) table (Table mode).
// ---------------------------------------------------------------------------
inline void fused_norm_rope_inplace(
    sycl::queue& q, bf16* x, const bf16* weight,
    int seq, int nheads, int hd, int offset,
    float theta, float partial, float eps,
    const float* rope_cos, const float* rope_sin) {
    int pair_offset = hd / 2;
    int n_active_pairs = static_cast<int>(partial * hd / 2.0f);
    float freq_denom = static_cast<float>(hd);
    size_t local = std::min(256, hd);
    while (local & (local - 1)) local--;
    int rows = seq * nheads;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)rows * local, local),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                int tok = row / nheads;
                int head = row - tok * nheads;
                bf16* xrow = x + ((size_t)tok * nheads + head) * hd;

                float ss = 0.0f;
                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(xrow[d]);
                    ss += v * v;
                }
                float ss_sum = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                float rms_inv = sycl::rsqrt(ss_sum / float(hd) + eps);

                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(xrow[d]) * rms_inv * bf16_to_float(weight[d]);
                    xrow[d] = float_to_bf16(v);
                }

                it.barrier(sycl::access::fence_space::local_space);
                for (int pair_i = lid; pair_i < n_active_pairs; pair_i += lsz) {
                    float c, s;
                    if (rope_cos) {
                        int ti = tok * n_active_pairs + pair_i;
                        c = rope_cos[ti];
                        s = rope_sin[ti];
                    } else {
                        float inv_freq = 1.0f / sycl::pow(theta, 2.0f * pair_i / freq_denom);
                        float angle = (float)(offset + tok) * inv_freq;
                        c = sycl::cos(angle);
                        s = sycl::sin(angle);
                    }
                    float x0 = bf16_to_float(xrow[pair_i]);
                    float x1 = bf16_to_float(xrow[pair_i + pair_offset]);
                    xrow[pair_i] = float_to_bf16(x0 * c - x1 * s);
                    xrow[pair_i + pair_offset] = float_to_bf16(x0 * s + x1 * c);
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Fused RMSNorm + RoPE, single-kernel per-token fusion (Fused mode).
//
// One work-group per token. cos/sin for that token are computed once into
// local memory, then every head in the group reuses them — removing the
// per-head pow/cos/sin recomputation without a separate table-fill launch or
// a transient arena allocation. Numerically identical to the table path (cos/sin
// in float, same norm/rope math); only the work mapping differs.
// ---------------------------------------------------------------------------
inline void fused_norm_rope_inplace_fused(
    sycl::queue& q, bf16* x, const bf16* weight,
    int seq, int nheads, int hd, int offset,
    float theta, float partial, float eps) {
    int pair_offset = hd / 2;
    int n_active_pairs = static_cast<int>(partial * hd / 2.0f);
    float freq_denom = static_cast<float>(hd);
    size_t local = std::min(256, hd);
    while (local & (local - 1)) local--;
    int nap = n_active_pairs;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> cos_l(std::max(1, nap), h);
        sycl::local_accessor<float, 1> sin_l(std::max(1, nap), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * local, local),
            [=](sycl::nd_item<1> it) {
                int tok = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);

                if (nap > 0) {
                    for (int p = lid; p < nap; p += lsz) {
                        float inv_freq = 1.0f / sycl::pow(theta, 2.0f * p / freq_denom);
                        float angle = (float)(offset + tok) * inv_freq;
                        cos_l[p] = sycl::cos(angle);
                        sin_l[p] = sycl::sin(angle);
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                for (int head = 0; head < nheads; ++head) {
                    bf16* xrow = x + ((size_t)tok * nheads + head) * hd;

                    float ss = 0.0f;
                    for (int d = lid; d < hd; d += lsz) {
                        float v = bf16_to_float(xrow[d]);
                        ss += v * v;
                    }
                    float ss_sum = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                    float rms_inv = sycl::rsqrt(ss_sum / float(hd) + eps);

                    for (int d = lid; d < hd; d += lsz) {
                        float v = bf16_to_float(xrow[d]) * rms_inv * bf16_to_float(weight[d]);
                        xrow[d] = float_to_bf16(v);
                    }

                    if (nap > 0) {
                        it.barrier(sycl::access::fence_space::local_space);
                        for (int p = lid; p < nap; p += lsz) {
                            float c = cos_l[p], s = sin_l[p];
                            float x0 = bf16_to_float(xrow[p]);
                            float x1 = bf16_to_float(xrow[p + pair_offset]);
                            xrow[p] = float_to_bf16(x0 * c - x1 * s);
                            xrow[p + pair_offset] = float_to_bf16(x0 * s + x1 * c);
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Fused masked softmax (contiguous-span variant, DIFF_SOFTMAX_SPAN=1 default).
// Valid KV columns form a single [lo,hi) span; max + z passes iterate only it.
// Write pass zeroes columns outside [lo,hi) so the ValueMM reads 0.
// ---------------------------------------------------------------------------
inline void fused_masked_softmax(
    sycl::queue& q, bf16* scores,
    int nq, int seq, int kv_len,
    int q_pos0, int kv_pos0, int sliding_window, bool causal) {
    constexpr int WG = 256;
    constexpr float NEG_INF = -3.4028235e38f;
    size_t rows = (size_t)nq * seq;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sf(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>(rows * WG, WG),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                int sq  = row % seq;
                bf16* x = scores + (size_t)row * kv_len;
                int q_global = q_pos0 + sq;

                int lo = 0, hi = kv_len;
                if (causal) {
                    int h = q_global - kv_pos0 + 1;
                    hi = h < kv_len ? h : kv_len;
                    if (sliding_window != INT_MAX) {
                        int l = q_global - kv_pos0 - sliding_window + 1;
                        lo = l > 0 ? l : 0;
                    }
                }

                float m = NEG_INF;
                for (int c = lo + lid; c < hi; c += WG)
                    m = sycl::fmax(m, bf16_to_float(x[c]));
                sf[lid] = m;
                it.barrier(sycl::access::fence_space::local_space);
                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) sf[lid] = sycl::fmax(sf[lid], sf[lid + o]);
                    it.barrier(sycl::access::fence_space::local_space);
                }
                m = sf[0];
                it.barrier(sycl::access::fence_space::local_space);

                float z = 0.0f;
                for (int c = lo + lid; c < hi; c += WG)
                    z += sycl::exp(bf16_to_float(x[c]) - m);
                sf[lid] = z;
                it.barrier(sycl::access::fence_space::local_space);
                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) sf[lid] += sf[lid + o];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float inv_z = 1.0f / sf[0];

                for (int c = lid; c < kv_len; c += WG) {
                    float p = (c >= lo && c < hi)
                            ? sycl::exp(bf16_to_float(x[c]) - m) * inv_z
                            : 0.0f;
                    x[c] = float_to_bf16(p);
                }
            });
    });
}

// Original per-column masked() branch over the full row (DIFF_SOFTMAX_SPAN=0).
// AB baseline for fused_masked_softmax.
inline void fused_masked_softmax_branchy(
    sycl::queue& q, bf16* scores,
    int nq, int seq, int kv_len,
    int q_pos0, int kv_pos0, int sliding_window, bool causal) {
    constexpr int WG = 256;
    constexpr float NEG_INF = -3.4028235e38f;
    size_t rows = (size_t)nq * seq;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sf(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>(rows * WG, WG),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                int sq  = row % seq;
                bf16* x = scores + (size_t)row * kv_len;
                int q_global = q_pos0 + sq;

                auto masked = [=](int c) {
                    if (!causal) return false;
                    int kv_global = kv_pos0 + c;
                    if (kv_global > q_global) return true;
                    return sliding_window != INT_MAX &&
                           kv_global < q_global - sliding_window + 1;
                };

                float m = NEG_INF;
                for (int c = lid; c < kv_len; c += WG)
                    if (!masked(c)) m = sycl::fmax(m, bf16_to_float(x[c]));
                sf[lid] = m;
                it.barrier(sycl::access::fence_space::local_space);
                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) sf[lid] = sycl::fmax(sf[lid], sf[lid + o]);
                    it.barrier(sycl::access::fence_space::local_space);
                }
                m = sf[0];
                it.barrier(sycl::access::fence_space::local_space);

                float z = 0.0f;
                for (int c = lid; c < kv_len; c += WG)
                    if (!masked(c)) z += sycl::exp(bf16_to_float(x[c]) - m);
                sf[lid] = z;
                it.barrier(sycl::access::fence_space::local_space);
                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) sf[lid] += sf[lid + o];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float inv_z = 1.0f / sf[0];

                for (int c = lid; c < kv_len; c += WG) {
                    float p = masked(c) ? 0.0f
                            : sycl::exp(bf16_to_float(x[c]) - m) * inv_z;
                    x[c] = float_to_bf16(p);
                }
            });
    });
}

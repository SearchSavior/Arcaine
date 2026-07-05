#pragma once
//
// Qwen3.5-MoE small elementwise kernels + Qwen HF-convention RoPE.
//
// All ops are BF16 in/out (compute in FP32, cast at the end), header-only and
// `inline` to match src/common/kernels/{elementwise,rms_norm,rope}.hpp. Math is
// transcribed from reference/transformers/.../modeling_qwen3_5_moe.py (line refs
// below). Runtime numerical validation is deferred to Phase 6.
//
//   silu/sigmoid            : F.silu / F.sigmoid
//   swiglu                  : act_fn(gate)*up with hidden_act="silu" (line 735/771)
//   mul_sigmoid_inplace     : attn_output * sigmoid(gate)         (line 717)
//   scale_rows_by_sigmoid   : sigmoid(shared_expert_gate(x)) * out (line 813, per-row scalar)
//   apply_qwen_rope         : rotate_half + partial rotary          (lines 563-606, 133-144)
//   gated_rmsnorm           : (w * rmsnorm(x)) * silu(z), w ones-init (lines 192-201)
//   l2norm                  : x * rsqrt(sum(x^2)+eps)              (lines 239-242)
//

#include <sycl/sycl.hpp>
#include "../../common/gpu/buffer.hpp"
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Elementwise activations
// ---------------------------------------------------------------------------

// x[i] = silu(x[i]) = x / (1 + exp(-x)).
inline void silu_inplace(sycl::queue& q, bf16* x, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float v = bf16_to_float(x[id[0]]);
            x[id[0]] = float_to_bf16(v / (1.0f + sycl::exp(-v)));
        });
    });
}

// x[i] = sigmoid(x[i]) = 1 / (1 + exp(-x)).
inline void sigmoid_inplace(sycl::queue& q, bf16* x, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float v = bf16_to_float(x[id[0]]);
            x[id[0]] = float_to_bf16(1.0f / (1.0f + sycl::exp(-v)));
        });
    });
}

// SwiGLU (separate gate/up): gate[i] = silu(gate[i]) * up[i].
inline void swiglu_inplace(sycl::queue& q, bf16* gate, const bf16* up, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float g = bf16_to_float(gate[id[0]]);
            float u = bf16_to_float(up[id[0]]);
            gate[id[0]] = float_to_bf16((g / (1.0f + sycl::exp(-g))) * u);
        });
    });
}

// SwiGLU from stacked (seq, 2*inter) layout into compact (seq, inter) output.
//   gate_up[tok, dim]       = gate proj  (dim < inter)
//   gate_up[tok, inter+dim] = up   proj  (dim < inter)
//   out[tok, dim] = silu(gate_up[tok, dim]) * gate_up[tok, inter + dim]
inline void swiglu_strided(sycl::queue& q, const bf16* gate_up, bf16* out,
                           int seq, int inter) {
    int total = seq * inter;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int tok = gid[0] / inter;
            int dim = gid[0] % inter;
            float g = bf16_to_float(gate_up[tok * 2 * inter + dim]);
            float u = bf16_to_float(gate_up[tok * 2 * inter + inter + dim]);
            out[tok * inter + dim] = float_to_bf16((g / (1.0f + sycl::exp(-g))) * u);
        });
    });
}

// ---------------------------------------------------------------------------
// Full-attention output gate: a[i] *= sigmoid(gate[i]).  (modeling line 717)
// ---------------------------------------------------------------------------
inline void mul_sigmoid_inplace(sycl::queue& q, bf16* a, const bf16* gate, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float g = bf16_to_float(gate[id[0]]);
            float sig = 1.0f / (1.0f + sycl::exp(-g));
            a[id[0]] = float_to_bf16(bf16_to_float(a[id[0]]) * sig);
        });
    });
}

// ---------------------------------------------------------------------------
// Shared-expert gate: out[row, :] *= sigmoid(gate_vec[row]).  (line 813)
// shared_expert_gate is Linear(H,1) -> one scalar per row, broadcast over D.
// ---------------------------------------------------------------------------
inline void scale_rows_by_sigmoid(sycl::queue& q, bf16* out, const bf16* gate_vec,
                                  int n_rows, int row_dim) {
    int total = n_rows * row_dim;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int r = gid[0] / row_dim;
            int d = gid[0] % row_dim;
            float g = bf16_to_float(gate_vec[r]);
            float sig = 1.0f / (1.0f + sycl::exp(-g));
            out[r * row_dim + d] =
                float_to_bf16(bf16_to_float(out[r * row_dim + d]) * sig);
        });
    });
}

// ---------------------------------------------------------------------------
// Qwen3.5 RoPE (HF rotate_half, partial rotary).  (lines 563-606, 133-144)
//
//   rotary_dim = head_dim * partial_rotary_factor   (= 256 * 0.25 = 64)
//   inv_freq[i] = 1 / theta^(2i / rotary_dim)        i in [0, rotary_dim/2)
//   half = rotary_dim / 2                            (= 32)
//   emb = cat(freqs, freqs)  -> cos/sin shape [..., rotary_dim], cos[i]==cos[i+half]
//   rotate_half(x)[:half]      = -x[half:rotary_dim]
//   rotate_half(x)[half:rotary_dim] = x[:half]
//
// Per (tok, head), for i in [0, half):
//   row[i]      = x[i]*c - x[i+half]*s
//   row[i+half] = x[i+half]*c + x[i]*s
// Channels [rotary_dim, head_dim) are left untouched (pass-through identity).
// Applied to Q (nq_heads) and K (nkv_heads). pos = offset + tok.
// ---------------------------------------------------------------------------
inline void apply_qwen_rope(
    sycl::queue& q,
    bf16* q_ptr, bf16* k_ptr,
    int seq_len, int offset,
    int nq_heads, int nkv_heads,
    int head_dim, int rotary_dim, float rope_theta
) {
    int half = rotary_dim / 2;                    // active pairs per head
    float freq_denom = static_cast<float>(rotary_dim);
    auto apply = [&](bf16* tensor, int nheads) {
        int total = seq_len * nheads * half;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
                int idx    = gid[0];
                int pair_i = idx % half;
                int head   = (idx / half) % nheads;
                int tok    = idx / (half * nheads);
                int pos    = offset + tok;
                float inv_freq = 1.0f / sycl::pow(rope_theta, 2.0f * pair_i / freq_denom);
                float angle = static_cast<float>(pos) * inv_freq;
                float c = sycl::cos(angle);
                float s = sycl::sin(angle);
                bf16* row = tensor + (tok * nheads + head) * head_dim;
                float x0 = bf16_to_float(row[pair_i]);
                float x1 = bf16_to_float(row[pair_i + half]);
                row[pair_i]       = float_to_bf16(x0 * c - x1 * s);
                row[pair_i + half] = float_to_bf16(x0 * s + x1 * c);
            });
        });
    };
    apply(q_ptr, nq_heads);
    apply(k_ptr, nkv_heads);
}

// ---------------------------------------------------------------------------
// Gated RMSNorm (Qwen3_5MoeRMSNormGated, linear-attn output norm).  (lines 192-201)
//   out[n, d] = (weight[d] * rmsnorm_D(x[n, :])) * silu(z[n, d])
// weight is ONES-init, PLAIN scale (no +1). Reduction is the MEAN of squares
// (matches Qwen3_5MoeRMSNorm). x, z, weight, out are BF16; compute FP32.
// ---------------------------------------------------------------------------
inline void gated_rmsnorm(
    sycl::queue& q,
    const bf16* x, const bf16* z, const bf16* weight,
    bf16* out, int N, int D, float eps
) {
    size_t local_size = static_cast<size_t>(std::min(256, D));
    while (local_size & (local_size - 1)) local_size--;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local_size, h);
        h.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
            [=](sycl::nd_item<1> it) {
                int n   = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                const bf16* xrow = x + static_cast<size_t>(n) * D;
                const bf16* zrow = z + static_cast<size_t>(n) * D;
                bf16*       orow = out + static_cast<size_t>(n) * D;
                float ss = 0.0f;
                for (int d = lid; d < D; d += lsz) { float v = bf16_to_float(xrow[d]); ss += v * v; }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz >> 1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid + s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float rms_inv = sycl::rsqrt(lmem[0] / static_cast<float>(D) + eps);
                for (int d = lid; d < D; d += lsz) {
                    float v = bf16_to_float(xrow[d]) * rms_inv * bf16_to_float(weight[d]);
                    float g = bf16_to_float(zrow[d]);
                    orow[d] = float_to_bf16(v * (g / (1.0f + sycl::exp(-g))));
                }
            });
    });
}

// ---------------------------------------------------------------------------
// L2 normalization (Gated DeltaNet q/k, FLA-style).  (lines 239-242)
//   out[n, d] = x[n, d] * rsqrt(SUM_d(x^2) + eps)   (SUM, not mean; no weight)
// ---------------------------------------------------------------------------
inline void l2norm(sycl::queue& q, const bf16* x, bf16* out, int N, int D, float eps) {
    size_t local_size = static_cast<size_t>(std::min(256, D));
    while (local_size & (local_size - 1)) local_size--;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local_size, h);
        h.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
            [=](sycl::nd_item<1> it) {
                int n   = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                const bf16* xrow = x + static_cast<size_t>(n) * D;
                bf16*       orow = out + static_cast<size_t>(n) * D;
                float ss = 0.0f;
                for (int d = lid; d < D; d += lsz) { float v = bf16_to_float(xrow[d]); ss += v * v; }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz >> 1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid + s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float inv_norm = sycl::rsqrt(lmem[0] + eps);
                for (int d = lid; d < D; d += lsz)
                    orow[d] = float_to_bf16(bf16_to_float(xrow[d]) * inv_norm);
            });
    });
}

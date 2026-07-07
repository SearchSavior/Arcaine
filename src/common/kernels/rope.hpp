#pragma once
#include <sycl/sycl.hpp>
#include "../gpu/buffer.hpp"
#include <cmath>

// Apply RoPE in-place to Q and K using the rotate_half (non-interleaved) convention.
//
// Matches Gemma4Unified: pairs (x[i], x[i + head_dim/2]) for i < n_active_pairs.
//
// Parameters:
//   head_dim:    full head dimension (256 for sliding, 512 for full attention)
//   rope_theta:  base frequency (10000 for sliding, 1e6 for full)
//   partial_rotary_factor: fraction determining active pairs
//     - sliding: 1.0  → n_active_pairs = 128, pair_offset = 128
//     - full:    0.25 → n_active_pairs = 64,  pair_offset = 256
//
// Frequency for pair_i: 1 / (theta ^ (2*pair_i / head_dim))
// Rotation: row[pair_i]           = x0*cos - x1*sin
//           row[pair_i+pair_offset] = x0*sin + x1*cos
// where x0 = row[pair_i], x1 = row[pair_i + pair_offset].
// Dimensions [n_active_pairs, pair_offset) and [pair_offset+n_active_pairs, head_dim)
// are unchanged (implicit zero-frequency identity).

inline void apply_rope(
    sycl::queue& q,
    bf16* q_ptr, bf16* k_ptr,
    int seq_len, int offset,
    int nq_heads, int nkv_heads, int head_dim,
    float rope_theta, float partial_rotary_factor,
    const float* rope_cos = nullptr, const float* rope_sin = nullptr
) {
    int pair_offset   = head_dim / 2;
    int n_active_pairs = static_cast<int>(partial_rotary_factor * head_dim / 2.0f);
    // freq denom = head_dim (not rotary_dims)
    float freq_denom = static_cast<float>(head_dim);

    auto apply = [&](bf16* tensor, int nheads) {
        int total = seq_len * nheads * n_active_pairs;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
                int idx    = gid[0];
                int pair_i = idx % n_active_pairs;
                int head   = (idx / n_active_pairs) % nheads;
                int tok    = idx / (n_active_pairs * nheads);

                bf16* row = tensor + (tok * nheads + head) * head_dim;
                float x0 = bf16_to_float(row[pair_i]);
                float x1 = bf16_to_float(row[pair_i + pair_offset]);
                float c, s;
                if (rope_cos) {
                    int ti = tok * n_active_pairs + pair_i;
                    c = rope_cos[ti];
                    s = rope_sin[ti];
                } else {
                    int pos = offset + tok;
                    float inv_freq = 1.0f / sycl::pow(rope_theta, 2.0f * pair_i / freq_denom);
                    float angle = (float)pos * inv_freq;
                    c = sycl::cos(angle);
                    s = sycl::sin(angle);
                }
                row[pair_i]              = float_to_bf16(x0 * c - x1 * s);
                row[pair_i + pair_offset] = float_to_bf16(x0 * s + x1 * c);
            });
        });
    };

    apply(q_ptr, nq_heads);
    apply(k_ptr, nkv_heads);
}

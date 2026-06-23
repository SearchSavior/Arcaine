#pragma once
#include <sycl/sycl.hpp>
#include "../gpu/buffer.hpp"
#include <cmath>

// GELU with tanh approximation. Applied in-place.
inline void gelu_tanh_inplace(sycl::queue& q, bf16* x, int n) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEF = 0.044715f;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float v = bf16_to_float(x[id[0]]);
            float inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
            float g = 0.5f * v * (1.0f + sycl::tanh(inner));
            x[id[0]] = float_to_bf16(g);
        });
    });
}

// Element-wise multiply in-place: a[i] *= b[i]
inline void mul_inplace(sycl::queue& q, bf16* a, const bf16* b, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float va = bf16_to_float(a[id[0]]);
            float vb = bf16_to_float(b[id[0]]);
            a[id[0]] = float_to_bf16(va * vb);
        });
    });
}

// Fused GeGLU: gate[i] = gelu(gate[i]) * up[i]
// gate and up are separate contiguous buffers (used when projections are separate).
inline void geglu_inplace(sycl::queue& q, bf16* gate, const bf16* up, int n) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEF = 0.044715f;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            float g = bf16_to_float(gate[id[0]]);
            float u = bf16_to_float(up[id[0]]);
            float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
            gate[id[0]] = float_to_bf16(0.5f * g * (1.0f + sycl::tanh(inner)) * u);
        });
    });
}

// GeGLU from stacked (seq, 2*inter) layout into compact (seq, inter) output.
// gate_up[tok, dim]       = gate projection output  (dim < inter)
// gate_up[tok, inter+dim] = up   projection output  (dim < inter)
// Produces gate_act[tok, dim] = gelu(gate[tok,dim]) * up[tok,dim]
inline void geglu_strided(sycl::queue& q,
                           const bf16* gate_up, bf16* gate_act,
                           int seq, int inter) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEF = 0.044715f;
    int total = seq * inter;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int i   = gid[0];
            int tok = i / inter;
            int dim = i % inter;
            float g = bf16_to_float(gate_up[tok * 2 * inter + dim]);
            float u = bf16_to_float(gate_up[tok * 2 * inter + inter + dim]);
            float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
            gate_act[tok * inter + dim] = float_to_bf16(0.5f * g * (1.0f + sycl::tanh(inner)) * u);
        });
    });
}

// Scale in-place: x[i] *= s
inline void scale_inplace(sycl::queue& q, bf16* x, int n, float s) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            x[id[0]] = float_to_bf16(bf16_to_float(x[id[0]]) * s);
        });
    });
}

// Logit softcapping (Gemma): x = tanh(x / cap) * cap
// Applied in-place on float buffer.
inline void softcap_inplace(sycl::queue& q, float* x, int n, float cap) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            x[id[0]] = sycl::tanh(x[id[0]] / cap) * cap;
        });
    });
}

// Convert BF16 buffer to FP32 buffer (for logits download)
inline void bf16_to_f32(sycl::queue& q, const bf16* src, float* dst, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            dst[id[0]] = bf16_to_float(src[id[0]]);
        });
    });
}

// Add two BF16 buffers in-place: a[i] += b[i]
inline void add_inplace(sycl::queue& q, bf16* a, const bf16* b, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            a[id[0]] = float_to_bf16(bf16_to_float(a[id[0]]) + bf16_to_float(b[id[0]]));
        });
    });
}

#pragma once

#include "../gpu/buffer.hpp"
#include "../gpu/engine.hpp"

// Attention tensor layout helpers shared by multiple model families.
// These stay narrowly focused on reshaping and scattering buffers.

// Transpose Q from (seq, nq, hd) time-major to (nq, seq, hd) head-major, into a
// caller-provided buffer (lets callers reuse arena/planned scratch instead of
// allocating per call).
inline void transpose_q_into(
    sycl::queue& q,
    bf16* dst, const bf16* src, int seq, int nq, int hd
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>(seq, nq, hd), [=](sycl::id<3> id) {
            int t = id[0], qh = id[1], d = id[2];
            dst[(qh * seq + t) * hd + d] = src[(t * nq + qh) * hd + d];
        });
    });
}

// Transpose Q from (seq, nq, hd) time-major to (nq, seq, hd) head-major.
inline GpuBuffer<bf16> transpose_q(
    sycl::queue& q,
    const bf16* src, int seq, int nq, int hd,
    sycl::queue& alloc_q
) {
    GpuBuffer<bf16> out((size_t)nq * seq * hd, alloc_q);
    transpose_q_into(q, out.data(), src, seq, nq, hd);
    return out;
}

// Expand K or V from (kv_len, nkv, hd) time-major to (nq, kv_len, hd) head-major
// with GQA repetition: expanded[h, t, d] = src[t, h/gqa_ratio, d].
inline GpuBuffer<bf16> expand_kv(
    sycl::queue& q,
    const bf16* src, int kv_len, int nkv, int nq, int hd,
    sycl::queue& alloc_q
) {
    int gqa_ratio = nq / nkv;
    GpuBuffer<bf16> out((size_t)nq * kv_len * hd, alloc_q);
    q.submit([&](sycl::handler& h) {
        bf16* dst = out.data();
        h.parallel_for(sycl::range<3>(nq, kv_len, hd), [=](sycl::id<3> id) {
            int qh = id[0], t = id[1], d = id[2];
            int kvh = qh / gqa_ratio;
            dst[(qh * kv_len + t) * hd + d] = src[(t * nkv + kvh) * hd + d];
        });
    });
    return out;
}

// Scatter ctx from (nq, seq, q_hd) head-major back to (seq, nq, q_hd) time-major.
inline void scatter_ctx(
    sycl::queue& q,
    const bf16* ctx, bf16* out,
    int nq, int seq, int q_hd
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<3>(nq, seq, q_hd), [=](sycl::id<3> id) {
            int qh = id[0], t = id[1], d = id[2];
            out[(t * nq + qh) * q_hd + d] = ctx[(qh * seq + t) * q_hd + d];
        });
    });
}

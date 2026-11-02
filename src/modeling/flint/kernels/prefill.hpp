#pragma once

#include <sycl/sycl.hpp>

#include "../config.hpp"
#include "../weights.hpp"

// Batched prefill kernels (M tokens). Two projection paths:
//   bf16 : activations stay bf16, dot accumulates fp32.
//   int8 : activations quantized per 64-group, integer code dot, dequantized.
// Both use the same 8-token weight-reuse tile and optional row scatter/gather.
namespace flint {
namespace prefill {

inline constexpr int kSG = 32;
inline constexpr int kThreads = 256;
inline constexpr int kRows = 8;

inline void embed_batch(sycl::queue& q, const QTensor& t, const int* tokens,
                        bf16* out, int M) {
    const int H = t.k;
    if (t.format == WeightFormat::BF16) {
        q.parallel_for(sycl::range<1>((size_t)M * H), [=](sycl::id<1> id) {
            const size_t i = id[0];
            out[i] = t.dense[(size_t)tokens[i / H] * H + i % H];
        });
        return;
    }
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(M * 256),
                                         sycl::range<1>(256)),
                       [=](sycl::nd_item<1> it) {
                           const int m = (int)it.get_group(0);
                           const int tok = tokens[m];
                           const uint8_t* w = t.w + (size_t)tok * H;
                           const bf16* s = t.s + (size_t)tok * (H / 64);
                           const bf16* b = t.b + (size_t)tok * (H / 64);
                           for (int i = (int)it.get_local_id(0); i < H; i += 256)
                               out[(size_t)m * H + i] =
                                   bf16(float(s[i >> 6]) * float(w[i]) +
                                        float(b[i >> 6]));
                       });
    });
}

inline void rmsnorm_batch(sycl::queue& q, const bf16* x, const bf16* w,
                          bf16* out, int M, int H, float eps) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(M * 256),
                                         sycl::range<1>(256)),
                       [=](sycl::nd_item<1> it) {
                           const int m = (int)it.get_group(0);
                           const int tid = (int)it.get_local_id(0);
                           const bf16* src = x + (size_t)m * H;
                           bf16* dst = out + (size_t)m * H;
                           const int step = H / 256;
                           float sum = 0.f;
                           for (int j = 0; j < step; ++j) {
                               const float v = float(src[tid * step + j]);
                               sum += v * v;
                           }
                           red[tid] = sum;
                           sycl::group_barrier(it.get_group());
                           for (int st = 128; st > 0; st >>= 1) {
                               if (tid < st) red[tid] += red[tid + st];
                               sycl::group_barrier(it.get_group());
                           }
                           const float scale = sycl::rsqrt(red[0] / H + eps);
                           for (int j = 0; j < step; ++j) {
                               const int i = tid * step + j;
                               dst[i] = bf16(float(w[i]) *
                                             float(bf16(float(src[i]) * scale)));
                           }
                       });
    });
}

inline void head_norm_rope_batch(sycl::queue& q, bf16* x, const bf16* w,
                                 int M, int heads, bool rope, int pos_base,
                                 float log2base, float eps) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)M * heads * kSG),
                                         sycl::range<1>(kSG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int g = (int)it.get_group(0);
                           const int m = g / heads;
                           const int head = g % heads;
                           const auto sg = it.get_sub_group();
                           const int lane = (int)sg.get_local_id()[0];
                           bf16* p = x + ((size_t)m * heads + head) * 128;
                           float ss = 0.f;
                           for (int d = 0; d < 4; ++d) {
                               const float v = float(p[lane * 4 + d]);
                               ss += v * v;
                           }
                           ss = sycl::reduce_over_group(sg, ss, sycl::plus<>());
                           const float inv = sycl::rsqrt(ss / 128.f + eps);
                           if (rope) {
                               const int pos = pos_base + m;
                               for (int j = 0; j < 2; ++j) {
                                   const int pair = lane * 2 + j;
                                   const float a = float(w[pair]) *
                                       float(bf16(float(p[pair]) * inv));
                                   const float bb = float(w[pair + 64]) *
                                       float(bf16(float(p[pair + 64]) * inv));
                                   const float th = float(pos) * sycl::exp2(
                                       -float(pair) / 64.f * log2base);
                                   const float c = sycl::cos(th);
                                   const float s = sycl::sin(th);
                                   p[pair] = bf16(a * c - bb * s);
                                   p[pair + 64] = bf16(a * s + bb * c);
                               }
                           } else {
                               for (int d = 0; d < 4; ++d) {
                                   const int i = lane * 4 + d;
                                   p[i] = bf16(float(w[i]) *
                                               float(bf16(float(p[i]) * inv)));
                               }
                           }
                       });
    });
}

inline void kv_finish_batch(sycl::queue& q, bf16* k, const bf16* v,
                            const bf16* k_norm, bf16* kcache, bf16* vcache,
                            int M, int kv_heads, bool global, int pos_base,
                            int cap, bool rope, float log2base, float eps) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)M * 2 * kv_heads * kSG),
                              sycl::range<1>(kSG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int g = (int)it.get_group(0);
                const int m = g / (2 * kv_heads);
                const int work = g % (2 * kv_heads);
                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_id()[0];
                const int pos = pos_base + m;
                const int slot = global ? pos : (pos % cap);
                if (work < kv_heads) {
                    bf16* p = k + ((size_t)m * kv_heads + work) * 128;
                    float ss = 0.f;
                    for (int d = 0; d < 4; ++d) {
                        const float x = float(p[lane * 4 + d]);
                        ss += x * x;
                    }
                    ss = sycl::reduce_over_group(sg, ss, sycl::plus<>());
                    const float inv = sycl::rsqrt(ss / 128.f + eps);
                    bf16* dst = kcache + ((size_t)work * cap + slot) * 128;
                    if (rope) {
                        for (int j = 0; j < 2; ++j) {
                            const int pair = lane * 2 + j;
                            const float a = float(k_norm[pair]) *
                                float(bf16(float(p[pair]) * inv));
                            const float bb = float(k_norm[pair + 64]) *
                                float(bf16(float(p[pair + 64]) * inv));
                            const float th = float(pos) * sycl::exp2(
                                -float(pair) / 64.f * log2base);
                            const float c = sycl::cos(th);
                            const float s = sycl::sin(th);
                            dst[pair] = bf16(a * c - bb * s);
                            dst[pair + 64] = bf16(a * s + bb * c);
                        }
                    } else {
                        for (int d = 0; d < 4; ++d) {
                            const int i = lane * 4 + d;
                            dst[i] = bf16(float(k_norm[i]) *
                                          float(bf16(float(p[i]) * inv)));
                        }
                    }
                } else {
                    const int kvh = work - kv_heads;
                    const bf16* src = v + ((size_t)m * kv_heads + kvh) * 128;
                    bf16* dst = vcache + ((size_t)kvh * cap + slot) * 128;
                    for (int d = 0; d < 4; ++d)
                        dst[lane * 4 + d] = src[lane * 4 + d];
                }
            });
    });
}

// C[t, row] = dot(A[row_index ? row_index[t] : t], W[row]) (+ residual).
inline void gemm_awq(sycl::queue& q, const QTensor& wm, const bf16* A,
                     const int* row_index, bf16* C, int M, int N,
                     const bf16* residual = nullptr, const int* out_index = nullptr,
                     int out_stride = 0, int expert = 0) {
    const int K = wm.k;
    const int tiles = (M + kRows - 1) / kRows;
    const uint8_t* weights = wm.w + (size_t)expert * wm.stride_w;
    const sycl::half* scales = wm.awq_s + (size_t)expert * wm.stride_s;
    const uint8_t* zeros = wm.awq_z + (size_t)expert * wm.stride_b;
    if (out_stride == 0) out_stride = N;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<bf16, 1> lw(sycl::range<1>(K), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)tiles * N * kThreads),
                              sycl::range<1>(kThreads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int gid = (int)it.get_group(0);
                const int row = gid % N;
                const int tid = (int)it.get_local_id(0);
                const uint8_t* packed = weights + (size_t)row * (K / 2);
                for (int i = tid; i < K; i += kThreads) {
                    const int code = (packed[i >> 1] >> ((i & 1) * 4)) & 15;
                    const size_t param = (size_t)(i >> 6) * wm.n + row;
                    lw[i] = bf16(float(code - int(zeros[param])) * float(scales[param]));
                }
                sycl::group_barrier(it.get_group());
                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_id()[0];
                const int t = (gid / N) * kRows + tid / kSG;
                if (t >= M) return;
                const int source = row_index ? row_index[t] : t;
                const bf16* input = A + (size_t)source * K;
                float acc = 0.f;
                for (int i = lane; i < K; i += kSG)
                    acc += float(lw[i]) * float(input[i]);
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                if (lane == 0) {
                    const int destination = out_index ? out_index[t] : t;
                    const size_t at = (size_t)destination * out_stride + row;
                    const bf16 value = bf16(acc);
                    C[at] = residual ? bf16(float(residual[at]) + float(value)) : value;
                }
            });
    });
}

template <bool INT8>
inline void gemm_q4(sycl::queue& q, const QTensor& wm, const bf16* A,
                    const int8_t* Aq, const float* xs, const int* row_index,
                    bf16* C, int M, int N, const bf16* residual = nullptr,
                    const int* out_index = nullptr, int out_stride = 0,
                    int expert = 0) {
    const int K = wm.k;
    const size_t ewoff = (size_t)expert * wm.stride_w;
    const size_t esoff = (size_t)expert * wm.stride_s;
    const int tiles = (M + kRows - 1) / kRows;
    if (out_stride == 0) out_stride = N;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lw(sycl::range<1>(4096), h);
        sycl::local_accessor<bf16, 1> ls(sycl::range<1>(128), h);
        sycl::local_accessor<bf16, 1> lb(sycl::range<1>(128), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)tiles * N * kThreads),
                              sycl::range<1>(kThreads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int gid = (int)it.get_group(0);
                const int tile = gid / N;
                const int row = gid % N;
                const int tid = (int)it.get_local_id(0);
                const uint8_t* wrow = wm.w + ewoff + (size_t)row * (K / 2);
                const bf16* srow = wm.s + esoff + (size_t)row * (K / 64);
                const bf16* brow = wm.b + esoff + (size_t)row * (K / 64);
                for (int i = tid; i < K / 2; i += kThreads) lw[i] = wrow[i];
                for (int g = tid; g < K / 64; g += kThreads) {
                    ls[g] = srow[g];
                    lb[g] = brow[g];
                }
                sycl::group_barrier(it.get_group());

                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_id()[0];
                const int sgid = tid / kSG;
                const int t = tile * kRows + sgid;
                if (t >= M) return;
                const int arow = row_index ? row_index[t] : t;
                float acc = 0.f;
                if constexpr (INT8) {
                    const int8_t* x = Aq + (size_t)arow * K;
                    const float* sx = xs + (size_t)arow * (K / 64);
                    for (int g = 0; g < K / 64; ++g) {
                        int ai = 0, xi = 0;
                        for (int i = g * 64 + lane; i < g * 64 + 64; i += kSG) {
                            const uint8_t byte = lw[i >> 1];
                            const int code = (i & 1) ? ((byte >> 4) & 15) : (byte & 15);
                            ai += code * (int)x[i];
                            xi += (int)x[i];
                        }
                        ai = sycl::reduce_over_group(sg, ai, sycl::plus<int>());
                        xi = sycl::reduce_over_group(sg, xi, sycl::plus<int>());
                        const float s = float(ls[g]) * sx[g];
                        acc += s * float(ai) + float(lb[g]) * sx[g] * float(xi);
                    }
                } else {
                    const bf16* x = A + (size_t)arow * K;
                    for (int i = lane; i < K; i += kSG) {
                        const uint8_t byte = lw[i >> 1];
                        const int code = (i & 1) ? ((byte >> 4) & 15) : (byte & 15);
                        const int g = i >> 6;
                        acc += (float(ls[g]) * code + float(lb[g])) * float(x[i]);
                    }
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                }
                if (lane == 0) {
                    const int orow = out_index ? out_index[t] : t;
                    const bf16 v = bf16(acc);
                    C[(size_t)orow * out_stride + row] =
                        residual ? bf16(float(residual[(size_t)orow * out_stride + row]) +
                                        float(v))
                                 : v;
                }
            });
    });
}

inline void quantize_int8(sycl::queue& q, const bf16* x, int8_t* xq, float* xs,
                          int M, int K) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)M * (K / 64) * kSG),
                                         sycl::range<1>(kSG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int g = (int)it.get_group(0);
                           const int row = g / (K / 64);
                           const int grp = g % (K / 64);
                           const auto sg = it.get_sub_group();
                           const int lane = (int)sg.get_local_id()[0];
                           float mx = 0.f;
                           for (int i = grp * 64 + lane; i < grp * 64 + 64; i += kSG)
                               mx = sycl::fmax(mx, sycl::fabs(float(x[(size_t)row * K + i])));
                           mx = sycl::reduce_over_group(sg, mx, sycl::maximum<>());
                           const float scale = mx > 0.f ? mx / 127.f : 1.f;
                           for (int i = grp * 64 + lane; i < grp * 64 + 64; i += kSG)
                               xq[(size_t)row * K + i] =
                                   (int8_t)sycl::round(float(x[(size_t)row * K + i]) / scale);
                           if (lane == 0) xs[(size_t)row * (K / 64) + grp] = scale;
                       });
    });
}

inline void silu_mul_batch(sycl::queue& q, bf16* g, const bf16* u, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const int i = (int)id[0];
            const float gf = float(g[i]);
            const float tail = 1.f / (1.f + sycl::exp(sycl::fabs(gf)));
            const float sig = gf < 0.f ? tail : 1.f - tail;
            g[i] = bf16(float(bf16(gf * sig)) * float(u[i]));
        });
    });
}

inline void router_batch(sycl::queue& q, const bf16* w, const bf16* x,
                          int M, int experts, int H, int top_k, int* ids,
                          float* weights) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scores(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)M * 256),
                                         sycl::range<1>(256)),
                       [=](sycl::nd_item<1> it) {
                           const int t = (int)it.get_group(0);
                           const int tid = (int)it.get_local_id(0);
                           const auto sg = it.get_sub_group();
                           const int lane = (int)sg.get_local_id()[0];
                           const int sgid = tid / kSG;
                           const int rows = (experts + 7) / 8;
                           const bf16* xr = x + (size_t)t * H;
                           for (int r = 0; r < rows; ++r) {
                               const int row = sgid * rows + r;
                               if (row >= experts) break;
                               const bf16* wr = w + (size_t)row * H;
                               float a = 0.f;
                               for (int i = lane; i < H; i += kSG)
                                   a += float(wr[i]) * float(xr[i]);
                               a = sycl::reduce_over_group(sg, a, sycl::plus<>());
                               if (lane == 0) {
                                   a = float(bf16(a));
                                   scores[row] = 1.f / (1.f + sycl::exp(-a));
                               }
                           }
                           sycl::group_barrier(it.get_group());
                           if (tid == 0) {
                               float chosen[16];
                               int chosen_id[16];
                               for (int r = 0; r < top_k; ++r) {
                                   float best = -3.402823466e+38f;
                                   int idx = 0;
                                   for (int i = 0; i < experts; ++i) {
                                       const float v = scores[i];
                                       if (v > best || (v == best && i > idx)) {
                                           best = v;
                                           idx = i;
                                       }
                                   }
                                   chosen[r] = best;
                                   chosen_id[r] = idx;
                                   scores[idx] = -3.402823466e+38f;
                               }
                               float sum = 1.0e-20f;
                               for (int r = 0; r < top_k; ++r) sum += chosen[r];
                               for (int r = 0; r < top_k; ++r) {
                                   ids[t * top_k + r] = chosen_id[r];
                                   weights[t * top_k + r] = chosen[r] / sum;
                               }
                           }
                       });
    });
}

inline void combine_batch(sycl::queue& q, bf16* h, const bf16* routed,
                         const bf16* shared, const float* weights, int M,
                         int H, int top_k) {
    q.submit([&](sycl::handler& hd) {
        hd.parallel_for(sycl::range<1>((size_t)M * H), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int t = (int)(i / H);
            const int j = (int)(i % H);
            float acc = 0.f;
            for (int r = 0; r < top_k; ++r) {
                const float term = float(bf16(
                    float(routed[((size_t)t * top_k + r) * H + j]) *
                    weights[t * top_k + r]));
                acc = r == 0 ? term : float(bf16(term + acc));
            }
            const float mlp =
                float(bf16(acc + float(shared[(size_t)t * H + j])));
            h[i] = bf16(float(h[i]) + mlp);
        });
    });
}

}  // namespace prefill
}  // namespace flint

#pragma once

#include <sycl/sycl.hpp>

#include "config.hpp"
#include "weights.hpp"

// All Flint decode kernels (M = 1). No oneDNN, no joint_matrix.
// Weights are MLX affine int4: w = s*q + b over groups of 64.

namespace flint {
namespace kernels {

inline constexpr int kSG = 32;
inline constexpr int kRowsPerGroup = 8;  // 8 sub-groups of 32

// ---------------------------------------------------------------- primitives

inline float dot_q4(const uint8_t* w, const bf16* s, const bf16* b, int k,
                    const bf16* x, int lane, sycl::sub_group sg) {
    float acc = 0.f;
    int gi = -1;
    float sv = 0.f, bv = 0.f;
    for (int i = lane; i < k; i += kSG) {
        const int g = i >> 6;
        if (g != gi) {
            gi = g;
            sv = float(s[g]);
            bv = float(b[g]);
        }
        const uint8_t byte = w[i >> 1];
        const int code = (i & 1) ? ((byte >> 4) & 15) : (byte & 15);
        acc += (sv * code + bv) * float(x[i]);
    }
    return sycl::reduce_over_group(sg, acc, sycl::plus<>());
}

inline void dot_q4_pair(const uint8_t* w0, const bf16* s0, const bf16* b0,
                        const uint8_t* w1, const bf16* s1, const bf16* b1, int k,
                        const bf16* x, int lane, sycl::sub_group sg, float& a0,
                        float& a1) {
    float y0 = 0.f, y1 = 0.f;
    int gi = -1;
    float sv0 = 0.f, bv0 = 0.f, sv1 = 0.f, bv1 = 0.f;
    for (int i = lane; i < k; i += kSG) {
        const int g = i >> 6;
        if (g != gi) {
            gi = g;
            sv0 = float(s0[g]);
            bv0 = float(b0[g]);
            sv1 = float(s1[g]);
            bv1 = float(b1[g]);
        }
        const int byte = i >> 1;
        const int shift = (i & 1) ? 4 : 0;
        const int c0 = (w0[byte] >> shift) & 15;
        const int c1 = (w1[byte] >> shift) & 15;
        const float xv = float(x[i]);
        y0 += (sv0 * c0 + bv0) * xv;
        y1 += (sv1 * c1 + bv1) * xv;
    }
    a0 = sycl::reduce_over_group(sg, y0, sycl::plus<>());
    a1 = sycl::reduce_over_group(sg, y1, sycl::plus<>());
}

inline bf16 silu_mul(bf16 g, bf16 u) {
    const float gf = float(g);
    const float tail = 1.f / (1.f + sycl::exp(sycl::fabs(gf)));
    const float sig = gf < 0.f ? tail : 1.f - tail;
    const bf16 prod = bf16(gf * sig);
    return bf16(float(prod) * float(u));
}

// ---------------------------------------------------------------- embedding

inline void embed_i8(sycl::queue& q, const QTensor& t, int token, bf16* out) {
    const int H = t.k;
    const uint8_t* w = t.w + (size_t)token * H;
    const bf16* s = t.s + (size_t)token * (H / 64);
    const bf16* b = t.b + (size_t)token * (H / 64);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(256), [=](sycl::id<1> id) {
            for (int i = (int)id[0]; i < H; i += 256) {
                const int g = i >> 6;
                out[i] = bf16(float(s[g]) * float(w[i]) + float(b[g]));
            }
        });
    });
}

// ---------------------------------------------------------------- rmsnorm

inline void rmsnorm(sycl::queue& q, const bf16* x, const bf16* w, bf16* out,
                    int H, float eps) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                       [=](sycl::nd_item<1> it) {
                           const int tid = (int)it.get_local_id(0);
                           const int step = H / 256;
                           float sum = 0.f;
                           for (int j = 0; j < step; ++j) {
                               const float v = float(x[tid * step + j]);
                               sum += v * v;
                           }
                           red[tid] = sum;
                           sycl::group_barrier(it.get_group());
                           for (int stride = 128; stride > 0; stride >>= 1) {
                               if (tid < stride) red[tid] += red[tid + stride];
                               sycl::group_barrier(it.get_group());
                           }
                           const float scale =
                               sycl::rsqrt(red[0] / float(H) + eps);
                           for (int j = 0; j < step; ++j) {
                               const int i = tid * step + j;
                               out[i] = bf16(float(w[i]) *
                                             float(bf16(float(x[i]) * scale)));
                           }
                       });
    });
}

// ---------------------------------------------------------------- q4 GEMV

// y[row] = dot(w[row], x) (+ residual[row] when add != 0).
inline void q4_gemv(sycl::queue& q, const QTensor& t, const bf16* x, bf16* y,
                    int expert = 0, const bf16* residual = nullptr) {
    const int n = t.n, k = t.k;
    const uint8_t* w = t.w + (size_t)expert * t.stride_w;
    const bf16* s = t.s + (size_t)expert * t.stride_s;
    const bf16* b = t.b + (size_t)expert * t.stride_b;
    const int groups = n / kRowsPerGroup;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int sgid = (int)it.get_local_id(0) / kSG;
                const int row = (int)it.get_group(0) * kRowsPerGroup + sgid;
                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_id()[0];
                const float v = dot_q4(w + (size_t)row * (k / 2), s + (size_t)row * (k / 64),
                                       b + (size_t)row * (k / 64), k, x, lane, sg);
                if (lane == 0) {
                    const bf16 r = bf16(v);
                    y[row] = residual ? bf16(float(residual[row]) + float(r)) : r;
                }
            });
    });
}

// lm_head: bf16 dot, then float logits.
inline void lm_head(sycl::queue& q, const QTensor& t, const bf16* x, float* y) {
    const int n = t.n, k = t.k;
    const int groups = n / kRowsPerGroup;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int sgid = (int)it.get_local_id(0) / kSG;
                const int row = (int)it.get_group(0) * kRowsPerGroup + sgid;
                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_id()[0];
                const float v = dot_q4(t.w + (size_t)row * (k / 2),
                                       t.s + (size_t)row * (k / 64),
                                       t.b + (size_t)row * (k / 64), k, x, lane, sg);
                if (lane == 0) y[row] = float(bf16(v));
            });
    });
}

// ---------------------------------------------------------------- qk norm + rope

inline void head_norm_rope(sycl::queue& q, bf16* x, const bf16* w, int heads,
                           bool rope, int pos, float log2base, float eps) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(heads * kSG),
                                         sycl::range<1>(kSG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int head = (int)it.get_group(0);
                           const auto sg = it.get_sub_group();
                           const int lane = (int)sg.get_local_id()[0];
                           bf16* p = x + (size_t)head * 128;
                           float ss = 0.f;
                           for (int d = 0; d < 4; ++d) {
                               const float v = float(p[lane * 4 + d]);
                               ss += v * v;
                           }
                           ss = sycl::reduce_over_group(sg, ss, sycl::plus<>());
                           const float inv = sycl::rsqrt(ss / 128.f + eps);
                           if (rope) {
                               // Rotate straight from the raw values; no in-place
                               // dependency between lanes.
                               for (int j = 0; j < 2; ++j) {
                                   const int pair = lane * 2 + j;
                                   const float a =
                                       float(w[pair]) *
                                       float(bf16(float(p[pair]) * inv));
                                   const float bb =
                                       float(w[pair + 64]) *
                                       float(bf16(float(p[pair + 64]) * inv));
                                   const float theta =
                                       float(pos) * sycl::exp2(-float(pair) / 64.f * log2base);
                                   const float c = sycl::cos(theta);
                                   const float s = sycl::sin(theta);
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

// Normalize+rope K heads and store K/V for this position.
inline void kv_finish(sycl::queue& q, bf16* kbuf, const bf16* vbuf,
                      const bf16* k_norm, bf16* kcache, bf16* vcache,
                      int kv_heads, bool global, int pos, int cap,
                      bool rope, float log2base, float eps) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(2 * kv_heads * kSG),
                                         sycl::range<1>(kSG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int work = (int)it.get_group(0);
                           const auto sg = it.get_sub_group();
                           const int lane = (int)sg.get_local_id()[0];
                           const int slot = global ? pos : (pos % cap);
                           if (work < kv_heads) {
                               bf16* p = kbuf + (size_t)work * 128;
                               float ss = 0.f;
                               for (int d = 0; d < 4; ++d) {
                                   const float v = float(p[lane * 4 + d]);
                                   ss += v * v;
                               }
                               ss = sycl::reduce_over_group(sg, ss, sycl::plus<>());
                               const float inv = sycl::rsqrt(ss / 128.f + eps);
                               bf16* dst =
                                   kcache + ((size_t)work * cap + slot) * 128;
                               if (rope) {
                                   for (int j = 0; j < 2; ++j) {
                                       const int pair = lane * 2 + j;
                                       const float a =
                                           float(k_norm[pair]) *
                                           float(bf16(float(p[pair]) * inv));
                                       const float bb =
                                           float(k_norm[pair + 64]) *
                                           float(bf16(float(p[pair + 64]) * inv));
                                       const float theta = float(pos) *
                                           sycl::exp2(-float(pair) / 64.f * log2base);
                                       const float c = sycl::cos(theta);
                                       const float s = sycl::sin(theta);
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
                               const bf16* src = vbuf + (size_t)kvh * 128;
                               bf16* dst =
                                   vcache + ((size_t)kvh * cap + slot) * 128;
                               for (int d = 0; d < 4; ++d)
                                   dst[lane * 4 + d] = src[lane * 4 + d];
                           }
                       });
    });
}

// ---------------------------------------------------------------- attention

inline void attention(sycl::queue& q, const bf16* qbuf, const bf16* kcache,
                      const bf16* vcache, const bf16* sink, bf16* out,
                      int heads, int kv_heads, bool global, int pos, int cap) {
    const float scale = 0.08838834764831845f;  // 1/sqrt(128)
    const int repeat = heads / kv_heads;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(heads * kSG),
                                         sycl::range<1>(kSG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           const int head = (int)it.get_group(0);
                           const auto sg = it.get_sub_group();
                           const int lane = (int)sg.get_local_id()[0];
                           const int kvh = head / repeat;
                           const int count = global ? pos + 1 : sycl::min(pos + 1, cap);
                           const int start = pos + 1 - count;

                           float qv[4];
                           for (int d = 0; d < 4; ++d)
                               qv[d] = float(qbuf[(size_t)head * 128 + lane * 4 + d]) * scale;
                           float acc[4] = {0.f, 0.f, 0.f, 0.f};
                           float hi = float(sink[head]);
                           float z = 1.f;
                           for (int t = start; t <= pos; ++t) {
                               const int slot = global ? t : (t % cap);
                               const bf16* kp =
                                   kcache + ((size_t)kvh * cap + slot) * 128 + lane * 4;
                               const bf16* vp =
                                   vcache + ((size_t)kvh * cap + slot) * 128 + lane * 4;
                               float s = 0.f;
                               for (int d = 0; d < 4; ++d) s += qv[d] * float(kp[d]);
                               s = sycl::reduce_over_group(sg, s, sycl::plus<>());
                               const float m = sycl::fmax(hi, s);
                               const float old_w = sycl::exp(hi - m);
                               const float new_w = sycl::exp(s - m);
                               z = z * old_w + new_w;
                               for (int d = 0; d < 4; ++d)
                                   acc[d] = acc[d] * old_w + new_w * float(vp[d]);
                               hi = m;
                           }
                           const float inv = z == 0.f ? 0.f : 1.f / z;
                           for (int d = 0; d < 4; ++d)
                               out[(size_t)head * 128 + lane * 4 + d] =
                                   bf16(acc[d] * inv);
                       });
    });
}

// ---------------------------------------------------------------- router

inline void router_topk(sycl::queue& q, const bf16* w, const bf16* x, int experts,
                        int H, int top_k, int* ids, float* weights) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scores(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                       [=](sycl::nd_item<1> it) {
                           const int tid = (int)it.get_local_id(0);
                           if (tid < experts) {
                               float a = 0.f;
                               const bf16* row = w + (size_t)tid * H;
                               for (int i = 0; i < H; ++i)
                                   a += float(row[i]) * float(x[i]);
                               a = float(bf16(a));
                               scores[tid] = 1.f / (1.f + sycl::exp(-a));
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
                                   ids[r] = chosen_id[r];
                                   weights[r] = chosen[r] / sum;
                               }
                           }
                       });
    });
}

// ---------------------------------------------------------------- MoE

// Routed roles 0..top_k-1 use the 3-D expert tensors; role top_k is shared.
inline void expert_gateup(sycl::queue& q, const FlintLayer& L, int hidden, int moe_n,
                          int shared_n, int top_k, const bf16* x, const int* ids,
                          bf16* act) {
    const int width_r = moe_n;
    const int rows = top_k * width_r + shared_n;
    const int groups = rows / kRowsPerGroup;
    const uint8_t* gw = L.egate.w;
    const bf16* gs = L.egate.s;
    const bf16* gb = L.egate.b;
    const uint8_t* uw = L.eup.w;
    const bf16* us = L.eup.s;
    const bf16* ub = L.eup.b;
    const uint8_t* sgw = L.sgate.w;
    const bf16* sgs = L.sgate.s;
    const bf16* sgb = L.sgate.b;
    const uint8_t* suw = L.sup.w;
    const bf16* sus = L.sup.s;
    const bf16* sub = L.sup.b;
    const size_t ew = L.egate.stride_w, es = L.egate.stride_s;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int sgid = (int)it.get_local_id(0) / kSG;
                const int row = (int)it.get_group(0) * kRowsPerGroup + sgid;
                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_id()[0];
                int role, local, expert, width;
                if (row < top_k * width_r) {
                    role = row / width_r;
                    local = row - role * width_r;
                    expert = ids[role];
                    width = width_r;
                } else {
                    role = top_k;
                    local = row - top_k * width_r;
                    expert = 0;
                    width = shared_n;
                }
                const uint8_t *gw_row, *uw_row;
                const bf16 *gs_row, *gb_row, *us_row, *ub_row;
                if (role < top_k) {
                    gw_row = gw + (size_t)expert * ew + (size_t)local * (hidden / 2);
                    gs_row = gs + (size_t)expert * es + (size_t)local * (hidden / 64);
                    gb_row = gb + (size_t)expert * es + (size_t)local * (hidden / 64);
                    uw_row = uw + (size_t)expert * ew + (size_t)local * (hidden / 2);
                    us_row = us + (size_t)expert * es + (size_t)local * (hidden / 64);
                    ub_row = ub + (size_t)expert * es + (size_t)local * (hidden / 64);
                } else {
                    gw_row = sgw + (size_t)local * (hidden / 2);
                    gs_row = sgs + (size_t)local * (hidden / 64);
                    gb_row = sgb + (size_t)local * (hidden / 64);
                    uw_row = suw + (size_t)local * (hidden / 2);
                    us_row = sus + (size_t)local * (hidden / 64);
                    ub_row = sub + (size_t)local * (hidden / 64);
                }
                float ga = 0.f, ua = 0.f;
                dot_q4_pair(gw_row, gs_row, gb_row, uw_row, us_row, ub_row, hidden,
                            x, lane, sg, ga, ua);
                if (lane == 0)
                    act[(size_t)role * shared_n + local] =
                        silu_mul(bf16(ga), bf16(ua));
                (void)width;
            });
    });
}

inline void expert_down(sycl::queue& q, const FlintLayer& L, int hidden, int moe_n,
                        int shared_n, int top_k, const bf16* act, const int* ids,
                        bf16* down) {
    const int rows = (top_k + 1) * hidden;
    const int groups = rows / kRowsPerGroup;
    const uint8_t* dw = L.edown.w;
    const bf16* ds = L.edown.s;
    const bf16* db = L.edown.b;
    const uint8_t* sdw = L.sdown.w;
    const bf16* sds = L.sdown.s;
    const bf16* sdb = L.sdown.b;
    const size_t ew = L.edown.stride_w, es = L.edown.stride_s;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int sgid = (int)it.get_local_id(0) / kSG;
                const int row = (int)it.get_group(0) * kRowsPerGroup + sgid;
                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_id()[0];
                const int role = row / hidden;
                const int local = row - role * hidden;
                const int k = role < top_k ? moe_n : shared_n;
                const bf16* input = act + (size_t)role * shared_n;
                const uint8_t* w_row;
                const bf16 *s_row, *b_row;
                if (role < top_k) {
                    const int expert = ids[role];
                    w_row = dw + (size_t)expert * ew + (size_t)local * (k / 2);
                    s_row = ds + (size_t)expert * es + (size_t)local * (k / 64);
                    b_row = db + (size_t)expert * es + (size_t)local * (k / 64);
                } else {
                    w_row = sdw + (size_t)local * (k / 2);
                    s_row = sds + (size_t)local * (k / 64);
                    b_row = sdb + (size_t)local * (k / 64);
                }
                const float v = dot_q4(w_row, s_row, b_row, k, input, lane, sg);
                if (lane == 0) down[(size_t)role * hidden + local] = bf16(v);
            });
    });
}

inline void moe_combine(sycl::queue& q, bf16* h, const bf16* down,
                        const float* weights, int hidden, int top_k) {
    q.submit([&](sycl::handler& hdl) {
        hdl.parallel_for(sycl::range<1>(hidden), [=](sycl::id<1> id) {
            const int j = (int)id[0];
            float acc = 0.f;
            for (int r = 0; r < top_k; ++r) {
                const float term = float(bf16(float(down[(size_t)r * hidden + j]) *
                                              weights[r]));
                acc = r == 0 ? term : float(bf16(term + acc));
            }
            const float mlp =
                float(bf16(acc + float(down[(size_t)top_k * hidden + j])));
            h[j] = bf16(float(h[j]) + mlp);
        });
    });
}

// ---------------------------------------------------------------- dense MLP

inline void dense_gateup(sycl::queue& q, const QTensor& g, const QTensor& u,
                         const bf16* x, bf16* gout, bf16* uout) {
    const int n = g.n, k = g.k;
    const int groups = n / kRowsPerGroup;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int sgid = (int)it.get_local_id(0) / kSG;
                const int row = (int)it.get_group(0) * kRowsPerGroup + sgid;
                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_id()[0];
                float ga = 0.f, ua = 0.f;
                dot_q4_pair(g.w + (size_t)row * (k / 2), g.s + (size_t)row * (k / 64),
                            g.b + (size_t)row * (k / 64), u.w + (size_t)row * (k / 2),
                            u.s + (size_t)row * (k / 64),
                            u.b + (size_t)row * (k / 64), k, x, lane, sg, ga, ua);
                if (lane == 0) {
                    gout[row] = bf16(ga);
                    uout[row] = bf16(ua);
                }
            });
    });
}

inline void silu_mul_vec(sycl::queue& q, bf16* g, const bf16* u, int n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const int i = (int)id[0];
            g[i] = silu_mul(g[i], u[i]);
        });
    });
}

}  // namespace kernels
}  // namespace flint

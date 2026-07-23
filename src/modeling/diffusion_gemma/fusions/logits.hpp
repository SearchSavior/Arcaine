#pragma once
// Fused logits/sampling kernels for the DiffusionGemma hot path.
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "../../../common/gpu/buffer.hpp"

// Counter-based device RNG — must match diffsamp::rng_u32 in device_sampler.hpp.
// Duplicated here (rather than #include'ing device_sampler.hpp) to keep the
// logits head free of a sampler-module dependency.
namespace difflogits {
inline uint32_t rng_u32(uint64_t seed, uint32_t a, uint32_t b, uint32_t c) {
    uint64_t x = seed + 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t)a * 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t)b * 0xbf58476d1ce4e5b9ULL;
    x ^= (uint64_t)c * 0x94d049bb1335ebULL;
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb1335ebULL;
    x ^= x >> 31;
    return (uint32_t)(x >> 32);
}
}

// ---------------------------------------------------------------------------
// F1a — fused logits head.  Per canvas row (vocab-wide WG reduction):
//   processed = tanh(logits/softcap) * softcap / temp
//   probs     = softmax(processed)           -> bf16 (soft-cond matmul)
//   argmax, entropy(nats), multinomial sample -> per row
// Replaces: bf16_to_f32 + softcap_inplace + scale_f32 + oneDNN softmax +
//           f32_to_bf16 + sample_rows/multinomial_rows.
// ---------------------------------------------------------------------------
inline void fused_logits_head(
    sycl::queue& q,
    const bf16* logits,    // (seq, V)
    float softcap, float inv_temp,
    const float* sample_u,   // (seq,) uniform [0,1)
    bf16*    probs_bf16,   // (seq, V), optional nullptr to skip materialization
    int32_t* argmax,       // (seq,)
    float*   entropy,      // (seq,)
    int32_t* sample,       // (seq,)
    int seq, int V)
{
    constexpr int WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sf(sycl::range<1>(WG + 1), h);
        sycl::local_accessor<int, 1>   si(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * WG, WG),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                const bf16* x = logits + (size_t)row * V;

                auto proc = [=](int c) {
                    float v = bf16_to_float(x[c]);
                    return sycl::tanh(v / softcap) * softcap * inv_temp;
                };

                // Pass 1: max + argmax.
                float m = -3.4028235e38f; int am = 0;
                for (int c = lid; c < V; c += WG) {
                    float p = proc(c);
                    if (p > m) { m = p; am = c; }
                }
                sf[lid] = m; si[lid] = am;
                it.barrier(sycl::access::fence_space::local_space);
                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) {
                        // tie-break to the lower index, matching a serial scan
                        if (sf[lid + o] > sf[lid] ||
                            (sf[lid + o] == sf[lid] && si[lid + o] < si[lid])) {
                            sf[lid] = sf[lid + o]; si[lid] = si[lid + o];
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                m = sf[0];
                if (lid == 0) argmax[row] = si[0];
                it.barrier(sycl::access::fence_space::local_space);

                // Pass 2: Z = sum e^t, S1 = sum t*e^t  (t = processed - max).
                float z = 0.0f, s1 = 0.0f;
                for (int c = lid; c < V; c += WG) {
                    float t = proc(c) - m;
                    float e = sycl::exp(t);
                    z += e; s1 += t * e;
                }
                sf[lid] = z;
                it.barrier(sycl::access::fence_space::local_space);
                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) sf[lid] += sf[lid + o];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                z = sf[0];
                it.barrier(sycl::access::fence_space::local_space);
                sf[lid] = s1;
                it.barrier(sycl::access::fence_space::local_space);
                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) sf[lid] += sf[lid + o];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                s1 = sf[0];
                if (lid == 0) entropy[row] = sycl::log(z) - s1 / z;

                // Pass 3: write bf16 probabilities and sample from the same row.
                float inv_z = 1.0f / z;
                bf16* pb = probs_bf16 ? probs_bf16 + (size_t)row * V : nullptr;
                int chunk = (V + WG - 1) / WG;
                int c0 = lid * chunk;
                int c1 = sycl::min(c0 + chunk, V);

                float part = 0.0f;
                for (int c = c0; c < c1; ++c) {
                    float p = sycl::exp(proc(c) - m) * inv_z;
                    if (pb) pb[c] = float_to_bf16(p);
                    part += p;
                }
                sf[lid] = part;
                if (lid == 0) si[0] = V - 1;
                it.barrier(sycl::access::fence_space::local_space);

                if (lid == 0) {
                    float acc = 0.0f;
                    for (int k = 0; k < WG; ++k) {
                        float v = sf[k];
                        sf[k] = acc;
                        acc += v;
                    }
                    sf[WG] = acc;
                }
                it.barrier(sycl::access::fence_space::local_space);

                float target = sample_u[row];
                float off = sf[lid];
                if (off < target && target <= off + part) {
                    float cum = off;
                    for (int c = c0; c < c1; ++c) {
                        cum += sycl::exp(proc(c) - m) * inv_z;
                        if (cum >= target) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group>
                                a(si[0]);
                            a.fetch_min(c);
                            break;
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) sample[row] = si[0];
            });
    });
}

// ---------------------------------------------------------------------------
// F2a+F2b — online (Flash) softmax.  Merges the 3-pass kernel's Pass 1
// (max+argmax) and Pass 2 (Z, S1 for entropy) into ONE strided scan over V:
// each thread carries a running (m, am, z, s1) and a WG merge-tree rescales
// losers by exp(m_local - m_global).  Halves the 128 MiB logits read and the
// tanh recomputation (3 passes -> 2; Pass 3 unchanged).  argmax/entropy are
// mathematically identical but the FP-accumulation order differs -> NOT
// bit-exact with the 3-pass kernel; selfcheck before trusting.  Sampling
// stays inverse-CDF (Pass 3 is the original).
// Enable: DIFF_ONLINE_SOFTMAX (default off -> original 3-pass kernel).
// ---------------------------------------------------------------------------
inline void fused_logits_head_online(
    sycl::queue& q,
    const bf16* logits, float softcap, float inv_temp,
    const float* sample_u,
    bf16* probs_bf16,
    int32_t* argmax, float* entropy, int32_t* sample,
    int seq, int V)
{
    constexpr int WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sm(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1> sz(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1> ss(sycl::range<1>(WG), h);
        sycl::local_accessor<int, 1>   si(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1> sf(sycl::range<1>(WG + 1), h);
        sycl::local_accessor<int, 1>   scand(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * WG, WG),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                const bf16* x = logits + (size_t)row * V;
                auto proc = [=](int c) {
                    float v = bf16_to_float(x[c]);
                    return sycl::tanh(v / softcap) * softcap * inv_temp;
                };

                // Online pass: running (m, am, z, s1).
                float m = -3.4028235e38f; int am = 0;
                float z = 0.0f, s1 = 0.0f;
                for (int c = lid; c < V; c += WG) {
                    float p = proc(c);
                    if (p > m) {
                        float r = sycl::exp(m - p);   // rescale old accumulators
                        z = z * r + 1.0f;             // new peak contributes exp(0)=1
                        s1 = s1 * r;                  // (p - m_new) = 0
                        m = p; am = c;
                    } else {
                        float t = p - m;
                        float e = sycl::exp(t);
                        z += e; s1 += t * e;
                    }
                }
                sm[lid] = m; sz[lid] = z; ss[lid] = s1; si[lid] = am;
                it.barrier(sycl::access::fence_space::local_space);

                // WG merge tree: combine (m, z, s1, am) pairs.
                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) {
                        float ma = sm[lid], mb = sm[lid + o];
                        float za = sz[lid], zb = sz[lid + o];
                        float sa = ss[lid], sb = ss[lid + o];
                        int   ia = si[lid], ib = si[lid + o];
                        if (ma >= mb) {
                            float r = sycl::exp(mb - ma);
                            sm[lid] = ma; sz[lid] = za + zb * r; ss[lid] = sa + sb * r;
                            si[lid] = (ma == mb && ib < ia) ? ib : ia;  // tie-break lower idx
                        } else {
                            float r = sycl::exp(ma - mb);
                            sm[lid] = mb; sz[lid] = zb + za * r; ss[lid] = sb + sa * r;
                            si[lid] = ib;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float gm = sm[0];
                float Z  = sz[0];
                if (lid == 0) {
                    argmax[row]  = si[0];
                    entropy[row] = sycl::log(Z) - ss[0] / Z;
                }

                // Pass 3 (unchanged from the 3-pass kernel): probs + inverse-CDF sample.
                float inv_z = 1.0f / Z;
                bf16* pb = probs_bf16 ? probs_bf16 + (size_t)row * V : nullptr;
                int chunk = (V + WG - 1) / WG;
                int c0 = lid * chunk;
                int c1 = sycl::min(c0 + chunk, V);
                float part = 0.0f;
                for (int c = c0; c < c1; ++c) {
                    float p = sycl::exp(proc(c) - gm) * inv_z;
                    if (pb) pb[c] = float_to_bf16(p);
                    part += p;
                }
                sf[lid] = part;
                if (lid == 0) scand[0] = V - 1;
                it.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) {
                    float acc = 0.0f;
                    for (int k = 0; k < WG; ++k) { float v = sf[k]; sf[k] = acc; acc += v; }
                    sf[WG] = acc;
                }
                it.barrier(sycl::access::fence_space::local_space);
                float target = sample_u[row];
                float off = sf[lid];
                if (off < target && target <= off + part) {
                    float cum = off;
                    for (int c = c0; c < c1; ++c) {
                        cum += sycl::exp(proc(c) - gm) * inv_z;
                        if (cum >= target) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group>
                                a(scand[0]);
                            a.fetch_min(c);
                            break;
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) sample[row] = scand[0];
            });
    });
}

// ---------------------------------------------------------------------------
// F2c (+F4) — Gumbel-max sampling on the online-softmax pass.  The multinomial
// draw folds into the argmax reduction: sample = argmax_c (proc(c) + gumbel(c)),
// gumbel(c) = -log(-log(u_c)), u_c drawn INLINE from the counter-based RNG (no
// fill_uniform launch, no u_dev buffer).  Eliminates Pass 3's inverse-CDF
// machinery (chunked partial sums, serial scan, atomic).  For Hard/TopK
// soft_next (no probs) the whole head is ONE pass over V; for Exact (probs
// needed) it's two (online pass + a probs-write pass).  Built on the online
// pass so it also gets the F2a/F2b read/tanh win.
// NOT bit-exact with inverse-CDF sampling (different sample path), but
// device_sampler.hpp already disclaims HF RNG bit-exactness for the device
// sampler -> policy call.  Enable: DIFF_GUMBEL_MAX (implies online softmax).
// Default off.
// ---------------------------------------------------------------------------
inline void fused_logits_head_gumbel(
    sycl::queue& q,
    const bf16* logits, float softcap, float inv_temp,
    uint64_t seed, uint32_t blk, uint32_t step,
    bf16* probs_bf16,
    int32_t* argmax, float* entropy, int32_t* sample,
    int seq, int V)
{
    constexpr int WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sm(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1> sz(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1> ss(sycl::range<1>(WG), h);
        sycl::local_accessor<int, 1>   si(sycl::range<1>(WG), h);
        sycl::local_accessor<float, 1> sg(sycl::range<1>(WG), h);
        sycl::local_accessor<int, 1>   sgi(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * WG, WG),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                const bf16* x = logits + (size_t)row * V;
                auto proc = [=](int c) {
                    float v = bf16_to_float(x[c]);
                    return sycl::tanh(v / softcap) * softcap * inv_temp;
                };
                auto gumbel = [=](int c) {
                    uint32_t r = difflogits::rng_u32(seed, blk, step,
                                (uint32_t)((size_t)row * (size_t)V + (size_t)c));
                    float u = (float)((r >> 8) + 1u) * (1.0f / 16777217.0f);  // (0,1)
                    return -sycl::log(-sycl::log(u));
                };

                float m = -3.4028235e38f; int am = 0; float z = 0.0f, s1 = 0.0f;
                float gm = -3.4028235e38f; int gam = 0;
                for (int c = lid; c < V; c += WG) {
                    float p = proc(c);
                    if (p > m) {
                        float r = sycl::exp(m - p);
                        z = z * r + 1.0f; s1 = s1 * r; m = p; am = c;
                    } else {
                        float t = p - m; float e = sycl::exp(t);
                        z += e; s1 += t * e;
                    }
                    float g = p + gumbel(c);
                    if (g > gm) { gm = g; gam = c; }
                }
                sm[lid] = m; sz[lid] = z; ss[lid] = s1; si[lid] = am;
                sg[lid] = gm; sgi[lid] = gam;
                it.barrier(sycl::access::fence_space::local_space);

                for (int o = WG / 2; o > 0; o >>= 1) {
                    if (lid < o) {
                        // softmax merge
                        float ma = sm[lid], mb = sm[lid + o];
                        float za = sz[lid], zb = sz[lid + o];
                        float sa = ss[lid], sb = ss[lid + o];
                        int   ia = si[lid], ib = si[lid + o];
                        if (ma >= mb) {
                            float r = sycl::exp(mb - ma);
                            sm[lid] = ma; sz[lid] = za + zb * r; ss[lid] = sa + sb * r;
                            si[lid] = (ma == mb && ib < ia) ? ib : ia;
                        } else {
                            float r = sycl::exp(ma - mb);
                            sm[lid] = mb; sz[lid] = zb + za * r; ss[lid] = sb + sa * r;
                            si[lid] = ib;
                        }
                        // gumbel merge (max + lower-index tie-break)
                        float ga = sg[lid], gb = sg[lid + o];
                        int   gia = sgi[lid], gib = sgi[lid + o];
                        if (ga >= gb) {
                            sg[lid] = ga;
                            sgi[lid] = (ga == gb && gib < gia) ? gib : gia;
                        } else {
                            sg[lid] = gb; sgi[lid] = gib;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                if (lid == 0) {
                    argmax[row]  = si[0];
                    entropy[row] = sycl::log(sz[0]) - ss[0] / sz[0];
                    sample[row]  = sgi[0];
                }

                // Pass 3 (probs only, no CDF — gumbel already drew the sample).
                if (probs_bf16) {
                    float gmax = sm[0];
                    float inv_z = 1.0f / sz[0];
                    bf16* pb = probs_bf16 + (size_t)row * V;
                    for (int c = lid; c < V; c += WG)
                        pb[c] = float_to_bf16(sycl::exp(proc(c) - gmax) * inv_z);
                }
            });
    });
}

// ---------------------------------------------------------------------------
// F1c — top-k soft-conditioning select.  Per canvas row, find the top-k tokens
// of the softcapped/tempered distribution and their renormalized softmax
// weights, WITHOUT materializing the (seq,V) probs buffer.  Feeds a top-k
// weighted embedding gather (see weighted_embed_gather*) so the next-step
// self-conditioning signal costs k*H per row instead of the full V*H GEMM —
// the accuracy-preserving middle ground between exact probs@embed and the
// argmax-only "hard" path.
//
// Z cancels under top-k renormalization, so only the global max is needed for
// numerical stability: weight_i = exp(proc_i - m) / sum_{j in topk} exp(proc_j - m).
// Selection is k parallel argmax passes (k*V work, still << V*H), masking
// already-picked columns — same construction as the MoE router top-k.
// ---------------------------------------------------------------------------
inline void topk_soft_select(
    sycl::queue& q,
    const bf16* logits,    // (seq, V)
    float softcap, float inv_temp,
    int k,
    int32_t* topk_idx,     // (seq, k)
    float*   topk_w,       // (seq, k) renormalized softmax weights
    int seq, int V)
{
    constexpr int WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sf(sycl::range<1>(WG), h);
        sycl::local_accessor<int, 1>   si(sycl::range<1>(WG), h);
        sycl::local_accessor<int, 1>   sel(sycl::range<1>(k), h);
        sycl::local_accessor<float, 1> selv(sycl::range<1>(k), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * WG, WG),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                const bf16* x = logits + (size_t)row * V;

                auto proc = [=](int c) {
                    float v = bf16_to_float(x[c]);
                    return sycl::tanh(v / softcap) * softcap * inv_temp;
                };

                for (int s = 0; s < k; ++s) {
                    // Parallel argmax over columns not already selected.
                    float m = -3.4028235e38f; int am = -1;
                    for (int c = lid; c < V; c += WG) {
                        bool taken = false;
                        for (int j = 0; j < s; ++j) if (sel[j] == c) { taken = true; break; }
                        if (taken) continue;
                        float p = proc(c);
                        if (p > m || (p == m && (am < 0 || c < am))) { m = p; am = c; }
                    }
                    sf[lid] = m; si[lid] = am;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int o = WG / 2; o > 0; o >>= 1) {
                        if (lid < o) {
                            if (sf[lid + o] > sf[lid] ||
                                (sf[lid + o] == sf[lid] && si[lid + o] >= 0 &&
                                 (si[lid] < 0 || si[lid + o] < si[lid]))) {
                                sf[lid] = sf[lid + o]; si[lid] = si[lid + o];
                            }
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0) { sel[s] = si[0]; selv[s] = sf[0]; }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // Renormalized softmax over the k selected (selv[0] == global max).
                if (lid == 0) {
                    float m0 = selv[0];
                    float z = 0.0f;
                    for (int s = 0; s < k; ++s) { float e = sycl::exp(selv[s] - m0); selv[s] = e; z += e; }
                    float inv_z = z > 0.0f ? 1.0f / z : 0.0f;
                    for (int s = 0; s < k; ++s) {
                        topk_idx[(size_t)row * k + s] = sel[s];
                        topk_w[(size_t)row * k + s]   = selv[s] * inv_z;
                    }
                }
            });
    });
}

// out[t, :] = scale * sum_{s<k} w[t,s] * table[idx[t,s], :]   (BF16 table)
inline void weighted_embed_gather(
    sycl::queue& q,
    const bf16* table,     // (vocab, H)
    const int32_t* idx,    // (seq, k)
    const float* w,        // (seq, k)
    bf16* out,             // (seq, H)
    int seq, int k, int H, float scale)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(seq, H), [=](sycl::id<2> id) {
            int t = (int)id[0], d = (int)id[1];
            float acc = 0.0f;
            for (int s = 0; s < k; ++s) {
                int tok = idx[(size_t)t * k + s];
                acc += w[(size_t)t * k + s] * bf16_to_float(table[(size_t)tok * H + d]);
            }
            out[(size_t)t * H + d] = float_to_bf16(acc * scale);
        });
    });
}

// ---------------------------------------------------------------------------
// F1b — parallel multinomial over probs rows.  Thread k owns a contiguous
// column chunk; partial sums + local exclusive scan find the chunk containing
// the target, which is then rescanned serially.
// ---------------------------------------------------------------------------
inline void multinomial_rows(
    sycl::queue& q,
    const float* probs,    // (seq, V)
    const float* u,        // (seq,) uniform [0,1)
    int32_t* sample,       // (seq,)
    int seq, int V)
{
    constexpr int WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sf(sycl::range<1>(WG + 1), h);
        sycl::local_accessor<int, 1>   cand(sycl::range<1>(1), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * WG, WG),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                const float* p = probs + (size_t)row * V;
                int chunk = (V + WG - 1) / WG;
                int c0 = lid * chunk, c1 = sycl::min(c0 + chunk, V);

                float part = 0.0f;
                for (int c = c0; c < c1; ++c) part += p[c];
                sf[lid] = part;
                if (lid == 0) cand[0] = V - 1;   // fallback: last column
                it.barrier(sycl::access::fence_space::local_space);

                // Serial exclusive scan by thread 0 (WG=256 entries — cheap).
                if (lid == 0) {
                    float acc = 0.0f;
                    for (int k = 0; k < WG; ++k) { float v = sf[k]; sf[k] = acc; acc += v; }
                    sf[WG] = acc;
                }
                it.barrier(sycl::access::fence_space::local_space);

                float target = u[row];
                float off = sf[lid];
                if (off < target && target <= off + part) {
                    float cum = off;
                    for (int c = c0; c < c1; ++c) {
                        cum += p[c];
                        if (cum >= target) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group>
                                a(cand[0]);
                            a.fetch_min(c);
                            break;
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) sample[row] = cand[0];
            });
    });
}

#pragma once
// Fused logits/sampling kernels for the DiffusionGemma hot path.
#include <cstdint>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "../../../common/gpu/buffer.hpp"

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

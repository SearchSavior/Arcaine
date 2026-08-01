#pragma once
// Device chunked gated delta rule (prefill) for Qwen3.5-MoE linear-attention
// layers. SYCL port of host_chunk_gated_delta_rule (gated_deltanet.hpp),
// which mirrors torch_chunk_gated_delta_rule in the reference. The host path
// downloads q/k/v/beta/g, runs scalar single-threaded FP32 loops per head and
// re-uploads — it dominates prefill (~14 ms/tok flat across pp 512..4096).
//
// Structure: one 256-lane work-group per value head (n_v=32 -> 32 XeCores on
// BMG-G31), looping over 64-token chunks sequentially (SSM state carry). All
// math fp32 like the host reference; q/v stream from global (L2), k is staged
// in SLM as bf16. Per chunk:
//   1. gc = inclusive cumsum(g); A[a,b] = -beta[a]*(K[a].K[b])*exp(gc[a]-gc[b])
//      (strictly lower), staged in SLM.
//   2. T = (I-A)^-1 by forward substitution, row-parallel over columns:
//      T[i,j] = A[i,j] + sum_{r=j+1}^{i-1} A[i,r]*T[r,j]. Derived from the
//      host's in-place sequential-j loop: terms with r<=j vanish (T strictly
//      lower, zero diagonal during the solve) and r>j reads see the original
//      A row, so lanes over j are race-free against a separate A buffer.
//   3. kcd = (T+I) @ (K*beta*exp(gc))            [C, d_k] fp32 SLM
//   4. per d_v half (S columns are disjoint, so the half-0 state update may
//      precede half-1 reads):
//      v_intra = (T+I) @ (V*beta)                [C, 64]
//      v_new   = v_intra - kcd @ S_old           (in place)
//      core    = (Q*exp(gc)) @ S_old
//                + ((Q.K^T)*exp(gc[a]-gc[b]), b<=a) @ v_new
//      S       = S*exp(gc[last]) + (K*exp(gc[last]-gc))^T @ v_new
// The QK^T term maps lane = a*4 + w (w<4): each lane holds <=16 dot results
// in registers and the 4-lane partial sums are combined with xor shuffles
// (4-lane groups never cross an SG16 boundary, any sub-group width >= 4).
//
// Requires chunk=64, d_k=d_v=128 (checked by the caller, which falls back to
// the host path otherwise). Dispatch: QWEN35_GDN_IMPL=host|device.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "runtime/gpu/buffer.hpp" // bf16, bf16_to_float, float_to_bf16

// 0 = host scalar reference, 1 = device scalar SYCL, 2 = device XMX (DPAS).
inline int qwen_gdn_impl()
{
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("QWEN35_GDN_IMPL");
        if (v && std::strcmp(v, "host") == 0)       cached = 0;
        else if (v && std::strcmp(v, "device") == 0) cached = 1;
        else                                        cached = 2; // "xmx" / unset
    }
    return cached;
}

inline constexpr int kQwenGdnChunk = 64;
inline constexpr int kQwenGdnWG    = 256;

// q,k,v: device bf16 [S, n_v, d] (q,k already l2norm'd, q pre-scaled).
// beta,g: device bf16 [S, n_v]. ssm_state: device fp32 [n_v, d_k, d_v] (in/out).
// core: device bf16 [S, n_v, d_v] (out). Mirrors host_chunk_gated_delta_rule.
inline void qwen_gdn_device_chunk(
    sycl::queue& q,
    const bf16* qbuf, const bf16* kbuf, const bf16* vbuf,
    const bf16* beta, const bf16* g,
    float* ssm_state, bf16* core,
    int S, int n_v, int d_k, int d_v, int chunk)
{
    if (chunk != kQwenGdnChunk || d_k != 128 || d_v != 128)
        throw std::runtime_error("qwen_gdn_device_chunk requires chunk=64, d_k=d_v=128");
    if (S <= 0) return;
    const int C = kQwenGdnChunk, DK = 128, DV = 128, DVH = 64, WG = kQwenGdnWG;
    const int nch = (S + C - 1) / C;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<bf16>  sK(sycl::range<1>(C * DK), h);
        sycl::local_accessor<float> sA(sycl::range<1>(C * C), h);
        sycl::local_accessor<float> sT(sycl::range<1>(C * C), h);
        sycl::local_accessor<float> sKCD(sycl::range<1>(C * DK), h);
        sycl::local_accessor<float> sVH(sycl::range<1>(C * DVH), h);
        sycl::local_accessor<float> sBeta(sycl::range<1>(C), h);
        sycl::local_accessor<float> sGc(sycl::range<1>(C), h);
        sycl::local_accessor<float> sEg(sycl::range<1>(C), h);
        sycl::local_accessor<float> sSW(sycl::range<1>(C), h);   // eg*beta
        sycl::local_accessor<float> sDec(sycl::range<1>(C), h);  // exp(gc_last - gc)
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_v, WG),
                              sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) {
                const int h    = (int)it.get_group(0);
                const int lane = (int)it.get_local_id(1);
                float* ssm = ssm_state + (size_t)h * DK * DV;

                for (int c = 0; c < nch; ++c) {
                    const int base = c * C;
                    const int rows = std::min(C, S - base);

                    // -- stage K chunk (zero-padded), beta/g, cumsum gc ------
                    for (int i = lane; i < C * DK; i += WG) {
                        int t = i >> 7;
                        sK[i] = (t < rows)
                            ? kbuf[((size_t)(base + t) * n_v + h) * DK + (i & 127)]
                            : float_to_bf16(0.0f);
                    }
                    for (int t = lane; t < C; t += WG) {
                        sBeta[t] = (t < rows)
                            ? bf16_to_float(beta[(size_t)(base + t) * n_v + h])
                            : 0.0f;
                        sGc[t] = (t < rows)
                            ? bf16_to_float(g[(size_t)(base + t) * n_v + h])
                            : 0.0f;
                    }
                    it.barrier();
                    if (lane == 0) {
                        float acc = 0.0f;
                        for (int t = 0; t < C; ++t) { acc += sGc[t]; sGc[t] = acc; }
                    }
                    it.barrier();
                    for (int t = lane; t < C; t += WG) {
                        float eg = sycl::exp(sGc[t]);
                        sEg[t]  = eg;
                        sSW[t]  = eg * sBeta[t];
                        sDec[t] = sycl::exp(sGc[C - 1] - sGc[t]);
                    }
                    it.barrier();

                    // -- A (strictly lower) ----------------------------------
                    for (int idx = lane; idx < C * C; idx += WG) {
                        int a = idx >> 6, b = idx & 63;
                        float v = 0.0f;
                        if (b < a) {
                            float dot = 0.0f;
                            for (int dk = 0; dk < DK; ++dk)
                                dot += bf16_to_float(sK[a * DK + dk]) *
                                       bf16_to_float(sK[b * DK + dk]);
                            v = -sBeta[a] * dot * sycl::exp(sGc[a] - sGc[b]);
                        }
                        sA[idx] = v;
                    }
                    it.barrier();

                    // -- T = (I - A)^-1 (strictly lower part, diagonal 0) ----
                    for (int j = lane; j < C; j += WG) sT[j] = 0.0f;
                    it.barrier();
                    for (int i = 1; i < C; ++i) {
                        for (int j = lane; j < i; j += WG) {
                            float acc = sA[i * C + j];
                            for (int r = j + 1; r < i; ++r)
                                acc += sA[i * C + r] * sT[r * C + j];
                            sT[i * C + j] = acc;
                        }
                        it.barrier();
                    }

                    // -- kcd = (T + I) @ (K * beta * eg) ----------------------
                    for (int idx = lane; idx < C * DK; idx += WG) {
                        int a = idx >> 7, dk = idx & 127;
                        float acc = sSW[a] * bf16_to_float(sK[a * DK + dk]);
                        for (int b = 0; b < a; ++b)
                            acc += sT[a * C + b] * sSW[b] *
                                   bf16_to_float(sK[b * DK + dk]);
                        sKCD[idx] = acc;
                    }
                    it.barrier();

                    const float eg_last = sEg[C - 1];
                    for (int half = 0; half < 2; ++half) {
                        const int dv0 = half * DVH;

                        // -- v_intra = (T + I) @ (V * beta) ------------------
                        for (int idx = lane; idx < C * DVH; idx += WG) {
                            int a = idx >> 6, dv = idx & 63;
                            float acc = 0.0f;
                            if (a < rows)
                                acc = sBeta[a] * bf16_to_float(vbuf[
                                    ((size_t)(base + a) * n_v + h) * DV + dv0 + dv]);
                            const int bmax = std::min(a, rows); // sBeta[b>=rows]==0
                            for (int b = 0; b < bmax; ++b)
                                acc += sT[a * C + b] * sBeta[b] * bf16_to_float(
                                    vbuf[((size_t)(base + b) * n_v + h) * DV + dv0 + dv]);
                            sVH[idx] = acc;
                        }
                        it.barrier();

                        // -- v_new = v_intra - kcd @ S_old (in place) --------
                        for (int idx = lane; idx < C * DVH; idx += WG) {
                            int a = idx >> 6, dv = idx & 63;
                            float acc = sVH[idx];
                            for (int dk = 0; dk < DK; ++dk)
                                acc -= sKCD[a * DK + dk] * ssm[dk * DV + dv0 + dv];
                            sVH[idx] = acc;
                        }
                        it.barrier();

                        // -- core = (Q*eg) @ S_old + (QK decay) @ v_new ------
                        {
                            const int a = lane >> 2, w = lane & 3;
                            const int ar = a < rows ? a : rows - 1; // clamp: padded lanes compute garbage but never write
                            const bf16* qrow =
                                qbuf + ((size_t)(base + ar) * n_v + h) * DK;
                            const float eg_a = sEg[a];
                            float dots[C / 4];
                            int nb = 0;
                            for (int b = w; b <= a; b += 4) {
                                float dot = 0.0f;
                                for (int dk = 0; dk < DK; ++dk)
                                    dot += bf16_to_float(qrow[dk]) *
                                           bf16_to_float(sK[b * DK + dk]);
                                dots[nb++] = dot * sycl::exp(sGc[a] - sGc[b]);
                            }
                            sycl::sub_group sg = it.get_sub_group();
                            for (int dv = 0; dv < DVH; ++dv) {
                                float part = 0.0f;
                                for (int i2 = 0, b = w; b <= a; b += 4, ++i2)
                                    part += dots[i2] * sVH[b * DVH + dv];
                                for (int dk = w * 32; dk < w * 32 + 32; ++dk)
                                    part += eg_a * bf16_to_float(qrow[dk]) *
                                            ssm[dk * DV + dv0 + dv];
                                part += sycl::permute_group_by_xor(sg, part, 1);
                                part += sycl::permute_group_by_xor(sg, part, 2);
                                if (w == 0 && a < rows)
                                    core[((size_t)(base + a) * n_v + h) * DV + dv0 + dv] =
                                        float_to_bf16(part);
                            }
                        }
                        it.barrier();

                        // -- S = S*eg_last + (K*dec)^T @ v_new (this half) ---
                        for (int idx = lane; idx < DK * DVH; idx += WG) {
                            int dk = idx >> 6, dv = idx & 63;
                            float acc = ssm[dk * DV + dv0 + dv] * eg_last;
                            for (int a = 0; a < C; ++a)
                                acc += bf16_to_float(sK[a * DK + dk]) * sDec[a] *
                                       sVH[a * DVH + dv];
                            ssm[dk * DV + dv0 + dv] = acc;
                        }
                        it.barrier();
                    }
                }
            });
    });
}

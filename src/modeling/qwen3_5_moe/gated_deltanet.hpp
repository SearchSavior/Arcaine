#pragma once
//
// Qwen3.5-MoE Gated DeltaNet (Qwen3_5MoeGatedDeltaNet) forward + per-layer cache.
//
// Reference: reference/transformers/.../modeling_qwen3_5_moe.py lines 221-560.
//
// Geometry (per config): hidden H=2048; num_k_heads=16 (head_k_dim=128) ->
// key_dim=2048; num_v_heads=32 (head_v_dim=128) -> value_dim=4096; conv_dim =
// 2*key_dim + value_dim = 8192; conv kernel k=4 (silu). q/k are repeated x2
// (16 -> 32) before the core op. SSM state S[32,128,128] is fp32.
//
// Forward flow (hidden already input_layernorm-ed by the caller):
//   mixed_qkv = hidden @ in_proj_qkv.T                 -> [S, 8192]   (BF16 GEMM)
//   conv1d    = silu(causal depthwise conv, k=4)       -> [S, 8192]   (device kernel)
//   q,k,v     = split(mixed) ; q,k repeat x2           -> [S, 32, 128] (device kernel)
//   q,k       = l2norm(q), l2norm(k); q *= 1/sqrt(128)
//   z         = hidden @ in_proj_z.T                  -> [S, 4096]   (BF16 GEMM, gate)
//   b,a       = hidden @ in_proj_{b,a}.T               -> [S, 32]
//   beta      = sigmoid(b);  g = -exp(A_log)*softplus(a+dt_bias)  -> [S, 32]
//   core      = gated_delta_rule(q,k,v,g,beta, S0)    -> [S, 32, 128]  (see below)
//   core      = (norm * rmsnorm_D(core)) * silu(z)     -> gated_rmsnorm
//   out       = core @ out_proj.T                     -> [S, H]      (NVFP4)
//
// Core op:
//   * DECODE (S==1, cached): recurrent per-token update of S (device kernel,
//     lines 326-367). Each thread owns one (head, d_v) column of S; fully
//     parallel, no cross-thread deps:
//       S  *= exp(g);  kv_mem = S^T k;  delta = (v - kv_mem)*beta;
//       S  += k (x) delta;  out = S^T q
//   * PREFILL (S>1): chunked gated delta rule (chunk=64), host-orchestrated for
//     correctness (lines 245-323). q/k are already l2norm'd and q scaled on
//     device before download. Computes final S and uploads it to the cache.
//     A prefill-optimised device path can later replace the host function
//     behind an env var; the swap point is the single call below.
//
// All in_proj params are BF16 (unquantized); only out_proj is NVFP4. Numerical
// validation vs HF is Phase 6 (deferred); the self-check instead verifies the
// chunked-vs-recurrent equivalence (prefill(N) == prefill(N-1)+decode).
//

#include <cmath>
#include <cstring>
#include <vector>

#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/nvfp4.hpp"               // matmul_nvfp4, Nvfp4Linear
#include "../../common/gpu/ops.hpp"                // matmul_bf16
#include "../../common/kernels/elementwise.hpp"    // scale_inplace, add_inplace

#include "config.hpp"
#include "weights.hpp"   // QwenLinearAttn
#include "kernels.hpp"    // l2norm, gated_rmsnorm, sigmoid_inplace

// ---------------------------------------------------------------------------
// Per-layer linear-attention cache. conv_state holds the last k-1 pre-conv
// inputs (BF16, [conv_dim, k-1]); ssm_state is the recurrent S (fp32,
// [n_v, d_k, d_v]). Allocated only for linear-attn layers; zero-initialized.
// ---------------------------------------------------------------------------
struct QwenLinearAttnCache {
    GpuBuffer<bf16>  conv_state;   // [conv_dim, k-1]
    GpuBuffer<float> ssm_state;    // [n_v, d_k, d_v]  fp32
    bool has_state = false;
};

struct QwenLinearAttnCaches {
    std::vector<QwenLinearAttnCache> layers;   // size == num_hidden_layers
    void init(const QwenConfig& cfg, sycl::queue& q) {
        int n_k = cfg.linear_num_key_heads;
        int n_v = cfg.linear_num_value_heads;
        int d_k = cfg.linear_key_head_dim;
        int d_v = cfg.linear_value_head_dim;
        int k   = cfg.linear_conv_kernel_dim;
        int key_dim   = n_k * d_k;                       // 2048
        int value_dim = n_v * d_v;                       // 4096
        int conv_dim  = 2 * key_dim + value_dim;          // 8192
        layers.resize(cfg.num_hidden_layers);
        for (int l = 0; l < cfg.num_hidden_layers; ++l) {
            if (cfg.is_full_attn(l)) continue;
            auto& c = layers[l];
            c.conv_state = GpuBuffer<bf16>((size_t)conv_dim * (k - 1), q);
            c.ssm_state  = GpuBuffer<float>((size_t)n_v * d_k * d_v, q);
            q.memset(c.conv_state.data(), 0, (size_t)conv_dim * (k - 1) * sizeof(bf16));
            q.memset(c.ssm_state.data(),  0, (size_t)n_v * d_k * d_v * sizeof(float));
            c.has_state = false;
        }
        q.wait();
    }
    void reset(sycl::queue& q) {
        for (auto& c : layers) {
            if (c.conv_state.count()) {
                q.memset(c.conv_state.data(), 0, c.conv_state.count() * sizeof(bf16));
                q.memset(c.ssm_state.data(),  0, c.ssm_state.count() * sizeof(float));
            }
            c.has_state = false;
        }
        q.wait();
    }
};

// ===========================================================================
// Device kernels
// ===========================================================================

// Causal depthwise conv1d (k=4) + silu for prefill (zero left-pad).
//   mixed: [S, conv_dim] BF16 in.  conv_w: [conv_dim, k] BF16.
//   out[t, c] = silu( sum_j conv_w[c, j] * mixed[t-(k-1)+j, c] ), mixed[neg]=0.
inline void conv1d_causal_prefill(sycl::queue& q, const bf16* mixed, const bf16* conv_w,
                                  bf16* out, int S, int conv_dim, int k) {
    int km1 = k - 1;
    int total = S * conv_dim;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int c = (int)gid[0] % conv_dim;
            int t = (int)gid[0] / conv_dim;
            float acc = 0.0f;
            for (int j = 0; j < k; ++j) {
                int idx = t - km1 + j;
                if (idx >= 0 && idx < S)
                    acc += bf16_to_float(conv_w[c * k + j]) *
                           bf16_to_float(mixed[(size_t)idx * conv_dim + c]);
            }
            out[(size_t)t * conv_dim + c] = float_to_bf16(acc / (1.0f + sycl::exp(-acc)));
        });
    });
}

// Causal depthwise conv1d for single-token decode. Shifts conv_state in place.
//   mixed: [conv_dim] (new token). conv_state: [conv_dim, k-1] (in/out).
//   x_cat[j] = conv_state[c, j] (j<k-1) | mixed[c] (j=k-1).
//   out[c] = silu( sum_j conv_w[c,j] * x_cat[j] ); conv_state shifts in mixed[c].
inline void conv1d_causal_decode(sycl::queue& q, const bf16* mixed, const bf16* conv_w,
                                 bf16* conv_state, bf16* out, int conv_dim, int k) {
    int km1 = k - 1;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(conv_dim), [=](sycl::id<1> gid) {
            int c = (int)gid[0];
            float acc = 0.0f;
            for (int j = 0; j < km1; ++j)
                acc += bf16_to_float(conv_w[c * k + j]) * bf16_to_float(conv_state[c * km1 + j]);
            acc += bf16_to_float(conv_w[c * k + km1]) * bf16_to_float(mixed[c]);
            out[c] = float_to_bf16(acc / (1.0f + sycl::exp(-acc)));
            for (int j = 0; j < km1 - 1; ++j)
                conv_state[c * km1 + j] = conv_state[c * km1 + j + 1];
            conv_state[c * km1 + km1 - 1] = mixed[c];
        });
    });
}

// Save conv_state = last k-1 pre-conv inputs after prefill (causal zero-pad
// for the unavailable leading positions when S < k-1).
inline void save_conv_state(sycl::queue& q, const bf16* mixed, bf16* conv_state,
                            int S, int conv_dim, int k) {
    int km1 = k - 1;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)conv_dim * km1), [=](sycl::id<1> gid) {
            int j = (int)gid[0] % km1;
            int c = (int)gid[0] / km1;
            int s = S - km1 + j;
            conv_state[(size_t)c * km1 + j] =
                (s >= 0) ? mixed[(size_t)s * conv_dim + c] : bf16{0};
        });
    });
}

// Split mixed_qkv [S, conv_dim] into q,k (repeated x2 -> 32 heads) and v (32
// heads), all contiguous [S, n_v, d]. q in cols [0,key_dim), k in
// [key_dim,2*key_dim), v in [2*key_dim, conv_dim). repeat_interleave(2, head):
// dst[s, 2*h, d] = dst[s, 2*h+1, d] = src[s, h, d].
inline void extract_qkv_repeat(sycl::queue& q,
    const bf16* mixed, bf16* qout, bf16* kout, bf16* vout,
    int S, int n_k, int d_k, int d_v, int key_dim, int conv_dim) {
    int n_v = n_k * 2;                 // 32
    int total = S * n_v * d_k;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int dd  = (int)gid[0] % d_k;
            int hv  = ((int)gid[0] / d_k) % n_v;
            int s   = (int)gid[0] / (n_v * d_k);
            int h   = hv / 2;          // source key head (0..n_k-1)
            size_t base = (size_t)s * conv_dim;
            size_t dst  = (size_t)s * n_v * d_k + (size_t)hv * d_k + dd;
            qout[dst] = mixed[base + 0        + (size_t)h * d_k + dd];
            kout[dst] = mixed[base + key_dim  + (size_t)h * d_k + dd];
            // v has n_v heads already (no repeat): head hv maps directly.
            vout[(size_t)s * n_v * d_v + (size_t)hv * d_v + dd] =
                mixed[base + 2 * key_dim + (size_t)hv * d_v + dd];
        });
    });
}

// g[s, h] = -exp(A_log[h]) * softplus(a[s, h] + dt_bias[h]).
// softplus stable form: max(x,0) + log(1+exp(-|x|)).
inline void compute_g(sycl::queue& q, const bf16* a, const bf16* A_log,
                      const bf16* dt_bias, bf16* g, int S, int H) {
    int total = S * H;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int hd = (int)gid[0] % H;
            float x = bf16_to_float(a[(int)gid[0]]) + bf16_to_float(dt_bias[hd]);
            float sp = (x > 0.0f ? x : 0.0f) + sycl::log(1.0f + sycl::exp(-sycl::fabs(x)));
            g[(int)gid[0]] = float_to_bf16(-sycl::exp(bf16_to_float(A_log[hd])) * sp);
        });
    });
}

// DECODE recurrent gated delta rule. One work-item per (head, d_v) -> the
// thread owns column d_v of S[head] (128 rows of d_k). No cross-thread deps.
//   q,k,v: [n_v, d] BF16 (l2norm'd; q pre-scaled by 1/sqrt(d)). beta: [n_v].
//   graw:  [n_v] BF16 raw g (we exp() inside). S: [n_v, d_k, d_v] FP32 (in/out).
//   out:   [n_v, d_v] BF16.
inline void recurrent_gated_delta_decode(
    sycl::queue& q,
    const bf16* qv, const bf16* kv, const bf16* vv,
    const bf16* beta, const bf16* graw,
    float* S, bf16* out, int n_v, int d_k, int d_v)
{
    int total = n_v * d_v;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int d_v_i = (int)gid[0] % d_v;
            int hd    = (int)gid[0] / d_v;
            float eg  = sycl::exp(bf16_to_float(graw[hd]));
            float b   = bf16_to_float(beta[hd]);
            const bf16* qrow = qv + (size_t)hd * d_k;
            const bf16* krow = kv + (size_t)hd * d_k;
            const bf16* vrow = vv + (size_t)hd * d_v;
            float* Sc   = S + (size_t)hd * d_k * d_v + d_v_i;  // column d_v_i
            float kv_mem = 0.0f;
            for (int dk = 0; dk < d_k; ++dk)           // read-only pass for kv_mem
                kv_mem += Sc[dk * d_v] * eg * bf16_to_float(krow[dk]);
            float delta = (bf16_to_float(vrow[d_v_i]) - kv_mem) * b;
            float o = 0.0f;
            for (int dk = 0; dk < d_k; ++dk) {         // decay + rank-1 update + out
                float s = Sc[dk * d_v] * eg + bf16_to_float(krow[dk]) * delta;
                Sc[dk * d_v] = s;
                o += s * bf16_to_float(qrow[dk]);
            }
            out[(size_t)hd * d_v + d_v_i] = float_to_bf16(o);
        });
    });
}

// ===========================================================================
// Host-orchestrated chunked gated delta rule (prefill). Mirrors
// torch_chunk_gated_delta_rule (reference lines 245-323). Inputs are device
// BF16 buffers laid out [S, n_v, d]; we download, run per head in FP32, and
// upload core [S, n_v, d_v] BF16 + final S [n_v, d_k, d_v] FP32.
// q,k must arrive already l2norm'd and q scaled by 1/sqrt(d_k).
// ===========================================================================
inline void host_chunk_gated_delta_rule(
    GpuEngine& ctx,
    const bf16* qdev, const bf16* kdev, const bf16* vdev,
    const bf16* betadev, const bf16* gdev,
    float* ssm_state_dev, bf16* core_dev,
    int S, int n_v, int d_k, int d_v, int chunk_size)
{
    auto& q = ctx.queue;
    int pad  = (chunk_size - S % chunk_size) % chunk_size;
    int total = S + pad;
    int nch  = total / chunk_size;

    // Download device BF16 -> host FP32, layout [S, n_v, d].
    auto dl3 = [&](const bf16* dev, int d) {
        std::vector<bf16> tmp((size_t)S * n_v * d);
        q.memcpy(tmp.data(), dev, (size_t)S * n_v * d * sizeof(bf16)).wait();
        std::vector<float> o((size_t)S * n_v * d);
        for (size_t i = 0; i < tmp.size(); ++i) o[i] = bf16_to_float(tmp[i]);
        return o;
    };
    auto dl2 = [&](const bf16* dev) {
        std::vector<bf16> tmp((size_t)S * n_v);
        q.memcpy(tmp.data(), dev, (size_t)S * n_v * sizeof(bf16)).wait();
        std::vector<float> o((size_t)S * n_v);
        for (size_t i = 0; i < tmp.size(); ++i) o[i] = bf16_to_float(tmp[i]);
        return o;
    };
    std::vector<float> Q = dl3(qdev, d_k);
    std::vector<float> K = dl3(kdev, d_k);
    std::vector<float> V = dl3(vdev, d_v);
    std::vector<float> beta = dl2(betadev);
    std::vector<float> g   = dl2(gdev);

    // initial/final S on host, fp32 [n_v, d_k, d_v].
    std::vector<float> Shost((size_t)n_v * d_k * d_v, 0.0f);
    q.memcpy(Shost.data(), ssm_state_dev, (size_t)n_v * d_k * d_v * sizeof(float)).wait();

    std::vector<bf16>  Ohost((size_t)S * n_v * d_v, bf16{0});

    auto Qat = [&](int s, int h, int dk) { return Q[(size_t)s * n_v * d_k + (size_t)h * d_k + dk]; };
    auto Kat = [&](int s, int h, int dk) { return K[(size_t)s * n_v * d_k + (size_t)h * d_k + dk]; };
    auto Vat = [&](int s, int h, int dv) { return V[(size_t)s * n_v * d_v + (size_t)h * d_v + dv]; };

    for (int h = 0; h < n_v; ++h) {
        // Build padded per-head arrays (size total); padded tokens = 0.
        std::vector<float> Qh((size_t)total * d_k, 0.0f), Kh((size_t)total * d_k, 0.0f),
            Vh((size_t)total * d_v, 0.0f), bh(total, 0.0f), gh(total, 0.0f);
        for (int t = 0; t < S; ++t) {
            for (int dk = 0; dk < d_k; ++dk) {
                Qh[(size_t)t * d_k + dk] = Qat(t, h, dk);
                Kh[(size_t)t * d_k + dk] = Kat(t, h, dk);
            }
            for (int dv = 0; dv < d_v; ++dv) Vh[(size_t)t * d_v + dv] = Vat(t, h, dv);
            bh[t] = beta[(size_t)t * n_v + h];
            gh[t] = g[(size_t)t * n_v + h];
        }

        // v_beta = V*beta ; k_beta = K*beta  (per token).
        std::vector<float> v_beta((size_t)total * d_v, 0.0f), k_beta((size_t)total * d_k, 0.0f);
        for (int t = 0; t < total; ++t) {
            float bt = bh[t];
            for (int dv = 0; dv < d_v; ++dv) v_beta[(size_t)t * d_v + dv] = Vh[(size_t)t * d_v + dv] * bt;
            for (int dk = 0; dk < d_k; ++dk) k_beta[(size_t)t * d_k + dk] = Kh[(size_t)t * d_k + dk] * bt;
        }

        // S (initial) for this head. Local name `Sh` (state, per head) to avoid
        // shadowing the seq_len parameter `S`.
        std::vector<float> Sh(d_k * d_v);
        for (int i = 0; i < d_k * d_v; ++i) Sh[i] = Shost[(size_t)h * d_k * d_v + i];

        for (int c = 0; c < nch; ++c) {
            int base = c * chunk_size;
            // g_cum within chunk.
            std::vector<float> gc(chunk_size, 0.0f);
            {
                float acc = 0.0f;
                for (int j = 0; j < chunk_size; ++j) { acc += gh[base + j]; gc[j] = acc; }
            }
            // attn[a,b] = -(k_beta[a] . K[b]) * exp(gc[a]-gc[b])  for b<a, else 0.
            std::vector<float> attn((size_t)chunk_size * chunk_size, 0.0f);
            for (int a = 0; a < chunk_size; ++a) {
                for (int b = 0; b < a; ++b) {
                    float dm = std::exp(gc[a] - gc[b]);
                    float dot = 0.0f;
                    for (int dk = 0; dk < d_k; ++dk)
                        dot += k_beta[(size_t)(base + a) * d_k + dk] * Kh[(size_t)(base + b) * d_k + dk];
                    attn[(size_t)a * chunk_size + b] = -dot * dm;
                }
            }
            // Forward substitution: attn[i,j] += sum_r attn[i,r]*attn[r,j].
            for (int i = 1; i < chunk_size; ++i) {
                for (int j = 0; j < i; ++j) {
                    float s = 0.0f;
                    for (int r = 0; r < i; ++r) s += attn[(size_t)i * chunk_size + r] * attn[(size_t)r * chunk_size + j];
                    attn[(size_t)i * chunk_size + j] += s;
                }
            }
            for (int a = 0; a < chunk_size; ++a) attn[(size_t)a * chunk_size + a] += 1.0f;

            // value_intra = attn @ v_beta ; k_cumdecay = attn @ (k_beta * exp(gc)).
            std::vector<float> v_intra((size_t)chunk_size * d_v, 0.0f);
            std::vector<float> kcd((size_t)chunk_size * d_k, 0.0f);
            for (int a = 0; a < chunk_size; ++a) {
                for (int b = 0; b < chunk_size; ++b) {
                    float at = attn[(size_t)a * chunk_size + b];
                    if (at == 0.0f) continue;
                    for (int dv = 0; dv < d_v; ++dv)
                        v_intra[(size_t)a * d_v + dv] += at * v_beta[(size_t)(base + b) * d_v + dv];
                    float eg = std::exp(gc[b]);
                    for (int dk = 0; dk < d_k; ++dk)
                        kcd[(size_t)a * d_k + dk] += at * k_beta[(size_t)(base + b) * d_k + dk] * eg;
                }
            }

            // v_new[a] = v_intra[a] - kcd[a] @ S  (uses old S).
            std::vector<float> v_new((size_t)chunk_size * d_v, 0.0f);
            for (int a = 0; a < chunk_size; ++a) {
                for (int dv = 0; dv < d_v; ++dv) v_new[(size_t)a * d_v + dv] = v_intra[(size_t)a * d_v + dv];
                for (int dk = 0; dk < d_k; ++dk) {
                    float kv = kcd[(size_t)a * d_k + dk];
                    if (kv == 0.0f) continue;
                    for (int dv = 0; dv < d_v; ++dv)                         v_new[(size_t)a * d_v + dv] -= kv * Sh[(size_t)dk * d_v + dv];
                }
            }

            // core[a] = (Q[a]*exp(gc[a])) @ S  +  sum_{b<=a} (Q[a].K[b])*exp(gc[a]-gc[b]) * v_new[b].
            for (int a = 0; a < chunk_size; ++a) {
                if (base + a >= S) continue;           // padded -> leave Ohost zero
                std::vector<float> ai(d_v, 0.0f);
                float eg_a = std::exp(gc[a]);
                for (int dk = 0; dk < d_k; ++dk) {
                    float qv = Qh[(size_t)(base + a) * d_k + dk] * eg_a;
                    for (int dv = 0; dv < d_v; ++dv) ai[dv] += qv * Sh[(size_t)dk * d_v + dv];
                }
                for (int b = 0; b <= a; ++b) {
                    float dot = 0.0f;
                    for (int dk = 0; dk < d_k; ++dk)
                        dot += Qh[(size_t)(base + a) * d_k + dk] * Kh[(size_t)(base + b) * d_k + dk];
                    float at = dot * std::exp(gc[a] - gc[b]);
                    for (int dv = 0; dv < d_v; ++dv) ai[dv] += at * v_new[(size_t)b * d_v + dv];
                }
                for (int dv = 0; dv < d_v; ++dv)
                    Ohost[(size_t)(base + a) * n_v * d_v + (size_t)h * d_v + dv] =
                        float_to_bf16(ai[dv]);
            }

            // State update: S *= exp(gc[last]) ; S += sum_a outer(K[a]*exp(gc[last]-gc[a]), v_new[a]).
            float eg_last = std::exp(gc[chunk_size - 1]);
            for (int i = 0; i < d_k * d_v; ++i) Sh[i] *= eg_last;
            for (int a = 0; a < chunk_size; ++a) {
                float ec = std::exp(gc[chunk_size - 1] - gc[a]);
                for (int dk = 0; dk < d_k; ++dk) {
                    float kk = Kh[(size_t)(base + a) * d_k + dk] * ec;
                    if (kk == 0.0f) continue;
                    for (int dv = 0; dv < d_v; ++dv) Sh[(size_t)dk * d_v + dv] += kk * v_new[(size_t)a * d_v + dv];
                }
            }
        }

        for (int i = 0; i < d_k * d_v; ++i) Shost[(size_t)h * d_k * d_v + i] = Sh[i];
    }

    q.memcpy(ssm_state_dev, Shost.data(), (size_t)n_v * d_k * d_v * sizeof(float));
    q.memcpy(core_dev, Ohost.data(), (size_t)S * n_v * d_v * sizeof(bf16));
    q.wait();
}

// ===========================================================================
// Gated DeltaNet forward for one linear-attn layer.
//   hidden: [S, H] (already input_layernorm-ed). out: [S, H] (pre-residual).
//   past_len: cached tokens before this call (0 for the first prefill).
// ===========================================================================
inline void qwen_linear_attn_forward(
    GpuEngine& ctx, const QwenLinearAttn& w, QwenLinearAttnCache& cache,
    const bf16* hidden, bf16* out, int seq_len, int past_len, const QwenConfig& cfg)
{
    auto& q = ctx.queue;
    int H     = cfg.hidden_size;                 // 2048
    int n_k   = cfg.linear_num_key_heads;        // 16
    int n_v   = cfg.linear_num_value_heads;      // 32
    int d_k   = cfg.linear_key_head_dim;         // 128
    int d_v   = cfg.linear_value_head_dim;       // 128
    int kk    = cfg.linear_conv_kernel_dim;      // 4
    int key_dim   = n_k * d_k;                   // 2048
    int value_dim = n_v * d_v;                   // 4096
    int conv_dim  = 2 * key_dim + value_dim;     // 8192
    float scale   = 1.0f / std::sqrt((float)d_k);
    int S = seq_len;

    // 1. in_proj_qkv: [S, H] -> [S, conv_dim].
    GpuBuffer<bf16> mixed((size_t)S * conv_dim, q);
    matmul_bf16(hidden, S, H, w.in_proj_qkv.data(), conv_dim, mixed.data(), ctx);

    // 2. conv1d (depthwise causal k=4 + silu). Decode uses conv_state; prefill
    //    zero-left-pads and saves the last k-1 pre-conv inputs.
    GpuBuffer<bf16> conv_out((size_t)S * conv_dim, q);
    if (S == 1 && cache.has_state) {
        conv1d_causal_decode(q, mixed.data(), w.conv1d.data(),
                             cache.conv_state.data(), conv_out.data(), conv_dim, kk);
    } else {
        conv1d_causal_prefill(q, mixed.data(), w.conv1d.data(),
                             conv_out.data(), S, conv_dim, kk);
        save_conv_state(q, mixed.data(), cache.conv_state.data(), S, conv_dim, kk);
    }

    // 3. split + repeat q,k (->32 heads); v already 32 heads. -> [S, n_v, d].
    GpuBuffer<bf16> qbuf((size_t)S * n_v * d_k, q);
    GpuBuffer<bf16> kbuf((size_t)S * n_v * d_k, q);
    GpuBuffer<bf16> vbuf((size_t)S * n_v * d_v, q);
    extract_qkv_repeat(q, conv_out.data(), qbuf.data(), kbuf.data(), vbuf.data(),
                       S, n_k, d_k, d_v, key_dim, conv_dim);

    // 4. l2norm q,k (per head over d); scale q by 1/sqrt(d).
    l2norm(q, qbuf.data(), qbuf.data(), S * n_v, d_k, cfg.rms_norm_eps);
    l2norm(q, kbuf.data(), kbuf.data(), S * n_v, d_k, cfg.rms_norm_eps);
    scale_inplace(q, qbuf.data(), S * n_v * d_k, scale);

    // 5. z = in_proj_z(hidden) -> [S, value_dim] (gate for the output norm).
    GpuBuffer<bf16> zbuf((size_t)S * value_dim, q);
    matmul_bf16(hidden, S, H, w.in_proj_z.data(), value_dim, zbuf.data(), ctx);

    // 6. b,a -> beta=sigmoid(b); g=-exp(A_log)*softplus(a+dt_bias).
    GpuBuffer<bf16> bbuf((size_t)S * n_v, q);
    GpuBuffer<bf16> abuf((size_t)S * n_v, q);
    matmul_bf16(hidden, S, H, w.in_proj_b.data(), n_v, bbuf.data(), ctx);
    matmul_bf16(hidden, S, H, w.in_proj_a.data(), n_v, abuf.data(), ctx);
    sigmoid_inplace(q, bbuf.data(), S * n_v);
    GpuBuffer<bf16> gbuf((size_t)S * n_v, q);
    compute_g(q, abuf.data(), w.A_log.data(), w.dt_bias.data(), gbuf.data(), S, n_v);

    // 7. core gated delta rule.
    GpuBuffer<bf16> core((size_t)S * n_v * d_v, q);
    if (S == 1 && cache.has_state) {
        recurrent_gated_delta_decode(q, qbuf.data(), kbuf.data(), vbuf.data(),
                                     bbuf.data(), gbuf.data(),
                                     cache.ssm_state.data(), core.data(),
                                     n_v, d_k, d_v);
    } else {
        host_chunk_gated_delta_rule(ctx, qbuf.data(), kbuf.data(), vbuf.data(),
                                    bbuf.data(), gbuf.data(),
                                    cache.ssm_state.data(), core.data(),
                                    S, n_v, d_k, d_v, 64);
    }
    cache.has_state = true;

    // 8. gated rmsnorm: (norm * rmsnorm_D(core)) * silu(z), over d_v.
    gated_rmsnorm(q, core.data(), zbuf.data(), w.norm.data(), core.data(),
                  S * n_v, d_v, cfg.rms_norm_eps);

    // 9. out_proj: [S, value_dim] -> [S, H] (NVFP4).
    matmul_nvfp4(core.data(), S, value_dim, w.out_proj, out, ctx);
    q.wait();
}

#include "moe.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "linear_dispatch.hpp"
#include "fusions/prenorm.hpp"
#include "fusions/postnorm.hpp"
#include "fusions/int4_awq.hpp"
#include "../../common/gpu/expert_parallel.hpp"
#include "../../utils/profile.hpp"
#include "arena.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Small kernels local to the MoE block.
// ---------------------------------------------------------------------------

// rn[t][d] *= vec[d] * c   (router norm scale: per-channel scale + 1/sqrt(H))
static void scale_by_row_vec(sycl::queue& q, bf16* x, const bf16* vec,
                             float c, int seq, int H) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(seq, H), [=](sycl::id<2> id) {
            int t = id[0], d = id[1];
            float v = bf16_to_float(x[t * H + d]) * bf16_to_float(vec[d]) * c;
            x[t * H + d] = float_to_bf16(v);
        });
    });
}

// Xe[i][d] = src[idx[i]][d]   (gather token rows)
static void gather_rows(sycl::queue& q, const bf16* src, const int32_t* idx,
                        bf16* dst, int n, int H) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(n, H), [=](sycl::id<2> id) {
            int i = id[0], d = id[1];
            dst[i * H + d] = src[(size_t)idx[i] * H + d];
        });
    });
}

// out[idx[i]][d] += w[i] * src[i][d]   (weighted scatter-add).
// Within one launch each idx[i] is distinct (a token picks an expert once);
// the in-order queue serializes launches, so plain += is race-free.
static void scatter_add_weighted(sycl::queue& q, bf16* out, const bf16* src,
                                 const int32_t* idx, const float* w, int n, int H) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(n, H), [=](sycl::id<2> id) {
            int i = id[0], d = id[1];
            size_t o = (size_t)idx[i] * H + d;
            out[o] = float_to_bf16(bf16_to_float(out[o]) + w[i] * bf16_to_float(src[i * H + d]));
        });
    });
}

// a[i] = a[i] + b[i]
static void add_buf(sycl::queue& q, bf16* a, const bf16* b, int n) {
    add_inplace(q, a, b, n);
}

// ---------------------------------------------------------------------------
// Dense shared MLP (GeGLU): out = down(gelu(gate(x)) * up(x))
// ---------------------------------------------------------------------------
static void dense_mlp(GpuEngine& ctx, const DiffDenseMLP& w,
                      const bf16* x, bf16* out, int seq, int H, int inter) {
    auto& q = ctx.queue;
    auto& ar = diffarena::arena(ctx.index);
    if (!w.gate_up_proj_bf16.empty()) {
        auto gate_up_a = ar.alloc<bf16>((size_t)seq * 2 * inter);
        matmul_bf16(x, seq, H, w.gate_up_proj_bf16.data(), 2 * inter,
                    gate_up_a.data(), ctx);
        auto act_a = ar.alloc<bf16>((size_t)seq * inter);
        geglu_strided(q, gate_up_a.data(), act_a.data(), seq, inter);
        gate_up_a.reset();
        matmul_linear_weight(act_a.data(), seq, inter, w.down_proj, H, out, ctx);
        return;
    }
    if (!w.gate_up_proj_fp4.empty()) {
        if (!w.down_proj.nvfp4)
            throw std::runtime_error("NVFP4 dense fused gate/up requires NVFP4 down projection");
        int G = H / 16;
        auto gate_up_a = ar.alloc<bf16>((size_t)seq * 2 * inter);
        auto x_packed_a = ar.alloc<uint8_t>((size_t)seq * H / 2);
        auto x_scale_a  = ar.alloc<uint8_t>((size_t)seq * G);
        pack_bf16_to_nvfp4(q, x, x_packed_a.data(), x_scale_a.data(), seq, H,
                           w.gate_up_proj_fp4.input_global_scale);
        matmul_nvfp4_packed(x_packed_a.data(), x_scale_a.data(), seq, H,
                            w.gate_up_proj_fp4, gate_up_a.data(), ctx);
        x_packed_a.reset(); x_scale_a.reset();
        auto act_a = ar.alloc<bf16>((size_t)seq * inter);
        geglu_strided(q, gate_up_a.data(), act_a.data(), seq, inter);
        gate_up_a.reset();
        // down_proj goes through matmul_nvfp4 (bf16->pack->matmul). Pass stable
        // arena-backed pack workspaces so matmul_nvfp4 does NOT allocate a
        // transient sycl::malloc_device buffer (which would be freed at scope
        // exit and dangle at SYCL-graph-replay time) and does NOT wait. This
        // keeps the dense-MLP down_proj capture-safe inside a Nvfp4GraphSession.
        int dG = inter / 16;
        auto dp_packed_a = ar.alloc<uint8_t>((size_t)seq * inter / 2);
        auto dp_scale_a  = ar.alloc<uint8_t>((size_t)seq * dG);
        matmul_nvfp4(act_a.data(), seq, inter, w.down_proj.fp4, out, ctx,
                    dp_packed_a.data(), dp_scale_a.data());
        return;
    }

    auto gate_a = ar.alloc<bf16>((size_t)seq * inter);
    auto up_a   = ar.alloc<bf16>((size_t)seq * inter);
    matmul_linear_weight(x, seq, H, w.gate_proj, inter, gate_a.data(), ctx);
    matmul_linear_weight(x, seq, H, w.up_proj, inter, up_a.data(), ctx);
    geglu_inplace(q, gate_a.data(), up_a.data(), seq * inter);
    matmul_linear_weight(gate_a.data(), seq, inter, w.down_proj, H, out, ctx);
}

// ---------------------------------------------------------------------------
// Router: scores from the pre-normed router input `rn` (see
// fused_triple_prenorm), GPU matmul + host-side top-k.
// ---------------------------------------------------------------------------
struct RouterRoutes {
    std::vector<int> idx;
    std::vector<float> weight;
    diffarena::Alloc<int> idx_dev;
    diffarena::Alloc<float> weight_dev;
    bool device = false;

    RouterRoutes() = default;
    RouterRoutes(const RouterRoutes&) = delete;
    RouterRoutes& operator=(const RouterRoutes&) = delete;
    RouterRoutes(RouterRoutes&&) noexcept = default;
    RouterRoutes& operator=(RouterRoutes&&) noexcept = default;
};

// Compute router softmax + top-k on the GPU instead of downloading the per-layer
// scores to the host (a blocking copy that stalls the queue every MoE layer) and
// sorting there.  Numerically equivalent to the host path; default on, opt out
// with DIFF_ROUTER_GPU_TOPK=0.  Falls back to host routing when the device
// per-expert-scale buffer is unavailable (see router()).
static bool router_gpu_topk_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_ROUTER_GPU_TOPK");
        if (!env) return true;
        if (!std::strcmp(env, "0") || !std::strcmp(env, "off") ||
            !std::strcmp(env, "false") || !std::strcmp(env, "no"))
            return false;
        return true;
    }();
    return enabled;
}

static void router_topk_gpu(sycl::queue& q,
                            const bf16* scores,
                            const float* per_expert_scale,
                            int seq, int E, int top_k,
                            int* out_idx,
                            float* out_weight) {
    int local = 1;
    while (local < E) local <<= 1;
    if (local < 32) local = 32;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> probs(sycl::range<1>(E), h);
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(local), h);
        sycl::local_accessor<float, 1> topv(sycl::range<1>(top_k), h);
        sycl::local_accessor<int, 1> topi(sycl::range<1>(top_k), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)seq * local),
                              sycl::range<1>(local)),
            [=](sycl::nd_item<1> it) {
                int t = (int)it.get_group(0);
                int lid = (int)it.get_local_id(0);

                float mx = -3.402823466e38f;
                for (int e = lid; e < E; e += local) {
                    float v = bf16_to_float(scores[(size_t)t * E + e]);
                    if (v > mx) mx = v;
                }
                scratch[lid] = mx;
                it.barrier(sycl::access::fence_space::local_space);
                for (int stride = local >> 1; stride > 0; stride >>= 1) {
                    if (lid < stride && lid + stride < local) {
                        float v = scratch[lid + stride];
                        if (v > scratch[lid]) scratch[lid] = v;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                mx = scratch[0];

                float sum = 0.0f;
                for (int e = lid; e < E; e += local) {
                    float p = sycl::exp(bf16_to_float(scores[(size_t)t * E + e]) - mx);
                    probs[e] = p;
                    sum += p;
                }
                scratch[lid] = sum;
                it.barrier(sycl::access::fence_space::local_space);
                for (int stride = local >> 1; stride > 0; stride >>= 1) {
                    if (lid < stride && lid + stride < local)
                        scratch[lid] += scratch[lid + stride];
                    it.barrier(sycl::access::fence_space::local_space);
                }

                if (lid == 0) {
                    float inv_sum = scratch[0] > 0.0f ? 1.0f / scratch[0] : 0.0f;
                    for (int e = 0; e < E; ++e) probs[e] *= inv_sum;

                    float selected_sum = 0.0f;
                    for (int s = 0; s < top_k; ++s) {
                        float best = -1.0f;
                        int best_e = 0;
                        for (int e = 0; e < E; ++e) {
                            float p = probs[e];
                            if (p > best) {
                                best = p;
                                best_e = e;
                            }
                        }
                        topv[s] = best;
                        topi[s] = best_e;
                        selected_sum += best;
                        probs[best_e] = -1.0f;
                    }

                    float inv_selected = selected_sum > 0.0f ? 1.0f / selected_sum : 0.0f;
                    for (int s = 0; s < top_k; ++s) {
                        int e = topi[s];
                        out_idx[(size_t)t * top_k + s] = e;
                        out_weight[(size_t)t * top_k + s] =
                            topv[s] * inv_selected * per_expert_scale[e];
                    }
                }
            });
    });
}

static RouterRoutes router(GpuEngine& ctx, const DiffMoE& moe,
                           const bf16* rn, int seq, int H, int E, int top_k)
{
    auto& q = ctx.queue;
    RouterRoutes routes;
    auto scores_a = diffarena::arena(ctx.index).alloc<bf16>((size_t)seq * E);
    matmul_linear_weight(rn, seq, H, moe.router_proj, E, scores_a.data(), ctx);

    if (router_gpu_topk_enabled() && !moe.per_expert_scale_dev.empty()) {
        routes.idx_dev = diffarena::arena(ctx.index).alloc<int>((size_t)seq * top_k);
        routes.weight_dev = diffarena::arena(ctx.index).alloc<float>((size_t)seq * top_k);
        router_topk_gpu(q, scores_a.data(), moe.per_expert_scale_dev.data(),
                        seq, E, top_k, routes.idx_dev.data(), routes.weight_dev.data());
        routes.device = true;
        return routes;
    }

    std::vector<bf16> sh((size_t)seq * E);
    q.memcpy(sh.data(), scores_a.data(), (size_t)seq * E * sizeof(bf16)).wait();

    routes.idx.assign((size_t)seq * top_k, 0);
    routes.weight.assign((size_t)seq * top_k, 0.0f);

    std::vector<float> probs(E);
    std::vector<int> order(E);
    for (int t = 0; t < seq; ++t) {
        const bf16* row = sh.data() + (size_t)t * E;
        // softmax (fp32)
        float mx = -3.4e38f;
        for (int e = 0; e < E; ++e) { float v = bf16_to_float(row[e]); if (v > mx) mx = v; }
        float sum = 0.0f;
        for (int e = 0; e < E; ++e) { float p = std::exp(bf16_to_float(row[e]) - mx); probs[e] = p; sum += p; }
        for (int e = 0; e < E; ++e) probs[e] /= sum;
        // top-k by probability
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });
        float wsum = 0.0f;
        for (int s = 0; s < top_k; ++s) wsum += probs[order[s]];
        for (int s = 0; s < top_k; ++s) {
            int e = order[s];
            routes.idx[(size_t)t * top_k + s] = e;
            // renormalize to sum 1, then per-expert scale
            routes.weight[(size_t)t * top_k + s] = (probs[e] / wsum) * moe.per_expert_scale[e];
        }
    }
    return routes;
}

// ---------------------------------------------------------------------------
// Experts, hot/tail batched.  Profiling showed a single shared capacity blows
// up to 256 when one expert is hot (67% of all GEMM time, mostly padding).
// Instead: cold experts go into one batched GEMM pair at a tight tail capacity
// T, hot experts (count > T) each get an exact-size GEMM at their own weight
// offset (M rounded to 32 to bound the oneDNN primitive-cache shapes).
// ---------------------------------------------------------------------------
static constexpr int kTailCap = 64;

static int round_up(int v, int m) { return (v + m - 1) / m * m; }

static void experts(GpuEngine& ctx, const DiffMoE& moe,
                    const bf16* expert_in, bf16* out,
                    int seq, int H, int E, int top_k, int moe_inter,
                    const std::vector<int>& idx, const std::vector<float>& weight) {
    auto& q = ctx.queue;
    int A = seq * top_k;   // total (token, expert) assignments
    int T = std::min(kTailCap, round_up(seq, 8));   // tail cap (small seq: tighter)

    std::vector<int> count(E, 0);
    for (int a = 0; a < A; ++a) count[idx[a]]++;

    // DIFF_MOE_STATS=1: dump the expert-load distribution (for tuning kTailCap).
    static bool stats = std::getenv("DIFF_MOE_STATS") != nullptr;
    static int stats_calls = 0;
    if (stats && stats_calls++ < 80) {
        std::vector<int> c = count;
        std::sort(c.begin(), c.end());
        fprintf(stderr, "[moe] A=%d counts: max=%d p99=%d p95=%d p90=%d p75=%d p50=%d  >32:%d >64:%d >96:%d\n",
                A, c[E-1], c[E*99/100], c[E*95/100], c[E*90/100], c[E*3/4], c[E/2],
                (int)std::count_if(c.begin(), c.end(), [](int v){ return v > 32; }),
                (int)std::count_if(c.begin(), c.end(), [](int v){ return v > 64; }),
                (int)std::count_if(c.begin(), c.end(), [](int v){ return v > 96; }));
    }

    // Region layout in Xe/gu/act/Ye row space:
    //   [0, E*T)            tail slots, expert e at rows [e*T, e*T + min(count,T))
    //   [E*T, total_rows)   hot overflow, one padded run per hot expert
    struct Hot { int expert, rows_off, m; };
    std::vector<Hot> hot;
    std::vector<int> base(E);          // row offset of expert e's bucket
    std::vector<int> bucket_cap(E);    // capacity of that bucket
    int total_rows = E * T;
    for (int e = 0; e < E; ++e) {
        if (count[e] > T) {
            int m = round_up(count[e], 32);
            base[e] = total_rows; bucket_cap[e] = m;
            hot.push_back({e, total_rows, m});
            total_rows += m;
        } else {
            base[e] = e * T; bucket_cap[e] = T;
        }
    }

    // assign_token/assign_slot: the real rows to gather + where each (t, k)
    // assignment lives, for the gather and combine kernels.
    std::vector<int32_t> assign_token(A), assign_slot(A);
    std::vector<int> cursor(E, 0);
    for (int a = 0; a < A; ++a) {
        int e = idx[a];
        assign_token[a] = a / top_k;
        assign_slot[a]  = base[e] + cursor[e]++;
    }

    GpuBuffer<int32_t> atok_dev(A, q), aslot_dev(A, q);
    GpuBuffer<float>   aw_dev(A, q);
    atok_dev.upload(assign_token.data(), A);
    aslot_dev.upload(assign_slot.data(), A);
    aw_dev.upload(weight.data(), A);

    // Gather by assignment: zero the buffer (padding rows must be 0 for the
    // GEMM), then write only the A real rows.
    GpuBuffer<bf16> Xe((size_t)total_rows * H, q);
    q.memset(Xe.data(), 0, (size_t)total_rows * H * sizeof(bf16));
    {
        const int32_t* at = atok_dev.data();
        const int32_t* as = aslot_dev.data();
        bf16* xe = Xe.data();
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(A, H), [=](sycl::id<2> id) {
                int a = (int)id[0], d = (int)id[1];
                xe[(size_t)as[a] * H + d] = expert_in[(size_t)at[a] * H + d];
            });
        });
    }

    GpuBuffer<bf16> gu((size_t)total_rows * 2 * moe_inter, q);
    GpuBuffer<bf16> act((size_t)total_rows * moe_inter, q);
    GpuBuffer<bf16> Ye((size_t)total_rows * H, q);

    size_t gu_stride   = (size_t)2 * moe_inter * H;   // per-expert weight blocks
    size_t down_stride = (size_t)H * moe_inter;

    // Tail: one batched GEMM pair over all E experts at capacity T.
    // (Hot experts' tail slots are zero rows — wasted T rows each, negligible.)
    matmul_bf16_batched(Xe.data(), E, T, H,
                        moe.gate_up_proj.data(), 2 * moe_inter, /*transpose_W=*/true,
                        gu.data(), ctx);
    // Hot: exact-size GEMMs at the expert's weight offset.
    for (auto& hx : hot)
        matmul_bf16(Xe.data() + (size_t)hx.rows_off * H, hx.m, H,
                    moe.gate_up_proj.data() + (size_t)hx.expert * gu_stride,
                    2 * moe_inter,
                    gu.data() + (size_t)hx.rows_off * 2 * moe_inter, ctx);

    geglu_strided(q, gu.data(), act.data(), total_rows, moe_inter);

    matmul_bf16_batched(act.data(), E, T, moe_inter,
                        moe.down_proj.data(), H, /*transpose_W=*/true,
                        Ye.data(), ctx);
    for (auto& hx : hot)
        matmul_bf16(act.data() + (size_t)hx.rows_off * moe_inter, hx.m, moe_inter,
                    moe.down_proj.data() + (size_t)hx.expert * down_stride, H,
                    Ye.data() + (size_t)hx.rows_off * H, ctx);

    // Combine: out[t] = sum_k w[t,k] * Ye[assign_slot[t,k]]. One writer per
    // (t, d) — no atomics needed.
    {
        const int32_t* as = aslot_dev.data();
        const float* aw = aw_dev.data();
        const bf16* ye = Ye.data();
        int K = top_k;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(seq, H), [=](sycl::id<2> id) {
                int t = (int)id[0], d = (int)id[1];
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int slot = as[t * K + k];
                    acc += aw[t * K + k] * bf16_to_float(ye[(size_t)slot * H + d]);
                }
                out[(size_t)t * H + d] = float_to_bf16(acc);
            });
        });
    }
    q.wait();   // host vectors + temp device buffers freed at scope exit
}

// ---------------------------------------------------------------------------
// Dual FFN: hidden = (residual + post_ffn_ln(hs1 + hs2)) * layer_scalar
// ---------------------------------------------------------------------------
namespace {
struct FfnProfileLabels {
    const char* prenorm;
    const char* dense_mlp;
    const char* router;
    const char* experts_total;
    const char* postnorm;
};

const FfnProfileLabels& ffn_profile_labels(bool is_encoder, bool is_full) {
    static constexpr FfnProfileLabels labels[4] = {
        {
            "ffn.enc.sliding.prenorm",
            "ffn.enc.sliding.dense_mlp",
            "ffn.enc.sliding.router",
            "ffn.enc.sliding.experts_total",
            "ffn.enc.sliding.postnorm",
        },
        {
            "ffn.enc.full.prenorm",
            "ffn.enc.full.dense_mlp",
            "ffn.enc.full.router",
            "ffn.enc.full.experts_total",
            "ffn.enc.full.postnorm",
        },
        {
            "ffn.dec.sliding.prenorm",
            "ffn.dec.sliding.dense_mlp",
            "ffn.dec.sliding.router",
            "ffn.dec.sliding.experts_total",
            "ffn.dec.sliding.postnorm",
        },
        {
            "ffn.dec.full.prenorm",
            "ffn.dec.full.dense_mlp",
            "ffn.dec.full.router",
            "ffn.dec.full.experts_total",
            "ffn.dec.full.postnorm",
        },
    };
    return labels[(is_encoder ? 0 : 2) + (is_full ? 1 : 0)];
}
} // namespace

void dual_ffn_forward(
    GpuEngine& ctx,
    const DiffLayer& lw,
    bf16* hidden,
    int seq, int H, int intermediate,
    int num_experts, int top_k, int moe_intermediate,
    float rms_eps, float layer_scalar,
    bool is_encoder)
{
    auto& q = ctx.queue;
    size_t N = (size_t)seq * H;
    auto& ar = diffarena::arena(ctx.index);
    const FfnProfileLabels& prof = ffn_profile_labels(is_encoder, lw.is_full);
    const ExpertProfileLabels& exp_prof = expert_profile_labels(is_encoder, lw.is_full);

    // F2: one RMS reduction serves dense pre-norm, MoE pre-norm, router input.
    // Each input is freed as soon as its consumer is done (x1→dense, rn→router,
    // x2→experts) so the big MoE-expert workspace overlaps the freed storage.
    auto x1_a = ar.alloc<bf16>(N);
    auto x2_a = ar.alloc<bf16>(N);
    auto rn_a = ar.alloc<bf16>(N);
    auto mlp_a = ar.alloc<bf16>(N);
    diffarena::Alloc<bf16> moe_a;
    { DIFF_PROF(q, prof.prenorm);
      fused_triple_prenorm(q, hidden,
                           lw.pre_ffn_ln.data(), lw.pre_ffn_ln_2.data(),
                           lw.moe.router_scale.data(), 1.0f / std::sqrt((float)H),
                           x1_a.data(), x2_a.data(), rn_a.data(), seq, H, rms_eps); }

    // Path 1 — dense MLP
    { DIFF_PROF(q, prof.dense_mlp);
      dense_mlp(ctx, lw.mlp, x1_a.data(), mlp_a.data(), seq, H, intermediate); }
    x1_a.reset();

    // Path 2 — MoE
    RouterRoutes routes;
    { DIFF_PROF(q, prof.router);
      routes = router(ctx, lw.moe, rn_a.data(), seq, H, num_experts, top_k); }
    rn_a.reset();
    bool postnorm_fused = false;
    { DIFF_PROF(q, prof.experts_total);
      if (routes.device && diff_int4_fuse_expert_postnorm_enabled()) {
          ExpertPostnormFusion fusion;
          fusion.mlp_out = mlp_a.data();
          fusion.mlp_norm_weight = lw.post_ffn_ln_1.data();
          fusion.moe_norm_weight = lw.post_ffn_ln_2.data();
          fusion.combine_norm_weight = lw.post_ffn_ln.data();
          fusion.hidden = hidden;
          fusion.layer_scalar = layer_scalar;
          fusion.rms_eps = rms_eps;
          postnorm_fused = expert_parallel_forward_int4_fused_postnorm(
              ctx, lw.moe, x2_a.data(), seq, H, num_experts, top_k,
              moe_intermediate, routes.idx_dev.data(), routes.weight_dev.data(),
              fusion, exp_prof);
      }
      if (!postnorm_fused) {
          moe_a = ar.alloc<bf16>(N);
          if (routes.device) {
              expert_parallel_forward(ctx, lw.moe, x2_a.data(), moe_a.data(),
                                      seq, H, num_experts, top_k, moe_intermediate,
                                      routes.idx_dev.data(), routes.weight_dev.data(),
                                      exp_prof);
          } else {
              expert_parallel_forward(ctx, lw.moe, x2_a.data(), moe_a.data(),
                                      seq, H, num_experts, top_k, moe_intermediate,
                                      routes.idx, routes.weight, exp_prof);
          }
      } }
    x2_a.reset();

    // F3: hidden = (hidden + norm(norm(mlp,w1) + norm(moe,w2), w3)) * scalar
    if (!postnorm_fused) {
        DIFF_PROF(q, prof.postnorm);
        fused_dual_postnorm(q,
                            mlp_a.data(), lw.post_ffn_ln_1.data(),
                            moe_a.data(), lw.post_ffn_ln_2.data(),
                            lw.post_ffn_ln.data(),
                            hidden, layer_scalar, seq, H, rms_eps);
    }
}

#pragma once
// INT4-AWQ-only fusions for DiffusionGemma's fixed 256-position denoiser.
//
// Keep every optimization independently A/B selectable.  These switches are
// intentionally off by default until the focused benchmark has established a
// win on the target device:
//   DIFF_INT4_FUSE_DENSE_GATE_UP=1
//   DIFF_INT4_FUSE_EXPERT_POSTNORM=1
//   DIFF_INT4_FUSE_SELFCOND_ADD_NORM=1
//   DIFF_INT4_GROUPED_DPAS_MOE=1
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>
#include "../../../common/gpu/buffer.hpp"

inline bool diff_int4_env_enabled(const char* name) {
    const char* e = std::getenv(name);
    return e && std::strcmp(e, "0") && std::strcmp(e, "off") &&
           std::strcmp(e, "false") && std::strcmp(e, "no");
}

inline bool diff_int4_fuse_dense_gate_up_enabled() {
    static bool enabled = diff_int4_env_enabled("DIFF_INT4_FUSE_DENSE_GATE_UP");
    return enabled;
}

inline bool diff_int4_fuse_expert_postnorm_enabled() {
    static bool enabled = diff_int4_env_enabled("DIFF_INT4_FUSE_EXPERT_POSTNORM");
    return enabled;
}

inline bool diff_int4_fuse_selfcond_add_norm_enabled() {
    static bool enabled = diff_int4_env_enabled("DIFF_INT4_FUSE_SELFCOND_ADD_NORM");
    return enabled;
}

// Native single-GPU grouped W4A16 MoE kernel.  Off by default until the
// focused MoE kernel benchmark validates numerical tolerance and throughput.
inline bool diff_int4_grouped_dpas_moe_enabled() {
    static bool enabled = diff_int4_env_enabled("DIFF_INT4_GROUPED_DPAS_MOE");
    return enabled;
}

// inputs_embeds = rms_norm_no_scale(inputs_embeds + selfcond_delta).
// The BF16 round after the add is deliberate: it matches the original
// add_inplace -> rms_norm_no_scale numerical boundary exactly.
inline void fused_int4_selfcond_add_norm(
    sycl::queue& q, bf16* inputs_embeds, const bf16* selfcond_delta,
    int seq, int H, float eps)
{
    size_t local = std::min(256, H);
    while (local & (local - 1)) local--;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local, h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * local, local),
            [=](sycl::nd_item<1> it) {
                int tok = (int)it.get_group(0);
                int lid = (int)it.get_local_id(0);
                int lsz = (int)it.get_local_range(0);
                bf16* row = inputs_embeds + (size_t)tok * H;
                const bf16* delta = selfcond_delta + (size_t)tok * H;

                float ss = 0.0f;
                for (int d = lid; d < H; d += lsz) {
                    bf16 summed = float_to_bf16(
                        bf16_to_float(row[d]) + bf16_to_float(delta[d]));
                    row[d] = summed;
                    float v = bf16_to_float(summed);
                    ss += v * v;
                }
                lmem[lid] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz >> 1; s > 0; s >>= 1) {
                    if (lid < s) lmem[lid] += lmem[lid + s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float inv = sycl::rsqrt(lmem[0] / float(H) + eps);
                for (int d = lid; d < H; d += lsz)
                    row[d] = float_to_bf16(bf16_to_float(row[d]) * inv);
            });
    });
}

// Fuses the final INT4 expert pipeline boundary:
//
//   moe[t,d] = sum_k route_weight[t,k] * expert_out[slot[t,k],d]
//   hidden = (hidden + norm(norm(mlp,w1) + norm(moe,w2),w3)) * scalar
//
// The unfused path materializes moe[seq,H], then launches the existing dual
// postnorm kernel.  This kernel consumes the expert-sorted output directly,
// removes that intermediate, and reduces two launches to one.  Rounding the
// weighted expert sum through BF16 before the RMS reduction preserves the
// numerical boundary of the original combine kernel.
inline void fused_int4_expert_combine_postnorm(
    sycl::queue& q,
    const bf16* expert_out,
    const int32_t* slot,
    const float* route_weight,
    int top_k,
    const bf16* mlp_out,
    const bf16* mlp_norm_weight,
    const bf16* moe_norm_weight,
    const bf16* combine_norm_weight,
    bf16* hidden,
    float layer_scalar,
    int seq, int H, float eps)
{
    size_t local = std::min(256, H);
    while (local & (local - 1)) local--;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(local, h);
        // One BF16 MoE row is 5.5 KiB at H=2816, comfortably inside BMG's
        // 128 KiB SLM. Cache the route-weighted sum once; recomputing it for
        // each of the three RMS passes is markedly slower than the baseline.
        sycl::local_accessor<bf16, 1> moe_row((size_t)H, h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * local, local),
            [=](sycl::nd_item<1> it) {
                int tok = (int)it.get_group(0);
                int lid = (int)it.get_local_id(0);
                int lsz = (int)it.get_local_range(0);
                const bf16* mlp = mlp_out + (size_t)tok * H;
                bf16* residual = hidden + (size_t)tok * H;

                auto expert_sum = [&](int d) {
                    float acc = 0.0f;
                    for (int k = 0; k < top_k; ++k) {
                        int a = tok * top_k + k;
                        int s = slot[a];
                        if (s >= 0)
                            acc += route_weight[a] *
                                   bf16_to_float(expert_out[(size_t)s * H + d]);
                    }
                    return float_to_bf16(acc);
                };

                auto reduce = [&](float v) {
                    lmem[lid] = v;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int s = lsz >> 1; s > 0; s >>= 1) {
                        if (lid < s) lmem[lid] += lmem[lid + s];
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    float out = lmem[0];
                    it.barrier(sycl::access::fence_space::local_space);
                    return out;
                };

                float ss_mlp = 0.0f;
                float ss_moe = 0.0f;
                for (int d = lid; d < H; d += lsz) {
                    float a = bf16_to_float(mlp[d]);
                    bf16 cached = expert_sum(d);
                    moe_row[d] = cached;
                    float b = bf16_to_float(cached);
                    ss_mlp += a * a;
                    ss_moe += b * b;
                }
                float inv_mlp = sycl::rsqrt(reduce(ss_mlp) / float(H) + eps);
                float inv_moe = sycl::rsqrt(reduce(ss_moe) / float(H) + eps);

                float ss_sum = 0.0f;
                for (int d = lid; d < H; d += lsz) {
                    float a = bf16_to_float(mlp[d]) * inv_mlp *
                              bf16_to_float(mlp_norm_weight[d]);
                    float b = bf16_to_float(moe_row[d]) * inv_moe *
                              bf16_to_float(moe_norm_weight[d]);
                    float sum = a + b;
                    ss_sum += sum * sum;
                }
                float inv_sum = sycl::rsqrt(reduce(ss_sum) / float(H) + eps);

                for (int d = lid; d < H; d += lsz) {
                    float a = bf16_to_float(mlp[d]) * inv_mlp *
                              bf16_to_float(mlp_norm_weight[d]);
                    float b = bf16_to_float(moe_row[d]) * inv_moe *
                              bf16_to_float(moe_norm_weight[d]);
                    float delta = (a + b) * inv_sum *
                                  bf16_to_float(combine_norm_weight[d]);
                    residual[d] = float_to_bf16(
                        (bf16_to_float(residual[d]) + delta) * layer_scalar);
                }
            });
    });
}

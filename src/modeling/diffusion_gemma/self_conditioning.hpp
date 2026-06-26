#pragma once
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "weights.hpp"
#include "linear_dispatch.hpp"
#include "arena.hpp"

// inputs_embeds = post_norm(inputs_embeds + selfcond_ffn(pre_norm(soft)))
// `soft` (seq,H) is the previous step's soft-embedding signal; pass nullptr on
// the first denoising step (signal is zeros → post_norm(inputs_embeds)).
inline void self_conditioning_forward(
    GpuEngine& ctx, const DiffSelfCond& sc,
    bf16* inputs_embeds, const bf16* soft,
    int seq, int H, int inter, float rms_eps)
{
    auto& q = ctx.queue;
    if (soft != nullptr) {
        size_t N = (size_t)seq * H;
        auto& ar = diffarena::arena(ctx.index);
        auto normed = ar.alloc<bf16>(N);
        rms_norm(q, soft, sc.pre_norm.data(), normed.data(), seq, H, rms_eps);

        auto gate_up = ar.alloc<bf16>((size_t)seq * 2 * inter);
        if (!sc.gate_up_proj.empty()) {
            matmul_bf16(normed.data(), seq, H, sc.gate_up_proj.data(),
                        2 * inter, gate_up.data(), ctx);
        } else {
            matmul_linear_weight(normed.data(), seq, H, sc.gate_proj, inter,
                                 gate_up.data(), ctx);
            matmul_linear_weight(normed.data(), seq, H, sc.up_proj, inter,
                                 gate_up.data() + (size_t)seq * inter, ctx);
        }
        normed.reset();

        auto act = ar.alloc<bf16>((size_t)seq * inter);
        geglu_strided(q, gate_up.data(), act.data(), seq, inter);
        gate_up.reset();

        auto sc_out = ar.alloc<bf16>(N);
        matmul_linear_weight(act.data(), seq, inter, sc.down_proj, H, sc_out.data(), ctx);
        act.reset();
        add_inplace(q, inputs_embeds, sc_out.data(), (int)N);
    }
    // post_norm is scaleless RMSNorm.
    rms_norm_no_scale(q, inputs_embeds, inputs_embeds, seq, H, rms_eps);
}

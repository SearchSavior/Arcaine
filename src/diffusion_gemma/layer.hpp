#pragma once
#include "../common/gpu/buffer.hpp"
#include "../common/gpu/engine.hpp"
#include "../common/kernels/rms_norm.hpp"
#include "../common/kernels/elementwise.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
#include "attention.hpp"
#include "moe.hpp"
#include "../utils/profile.hpp"
#include "arena.hpp"

namespace diff_layer_detail {
inline const char* attn_input_norm_label(bool is_encoder, bool is_full) {
    if (is_encoder) return is_full ? "attn.enc.full.input_norm" : "attn.enc.sliding.input_norm";
    return is_full ? "attn.dec.full.input_norm" : "attn.dec.sliding.input_norm";
}

inline const char* attn_post_norm_label(bool is_encoder, bool is_full) {
    if (is_encoder) return is_full ? "attn.enc.full.post_norm" : "attn.enc.sliding.post_norm";
    return is_full ? "attn.dec.full.post_norm" : "attn.dec.sliding.post_norm";
}
} // namespace diff_layer_detail

// One transformer block = attention residual + dual FFN.  Shared structure for
// encoder (causal, writes KV) and decoder (bidirectional, reads KV) passes.
inline void diff_layer_forward(
    GpuEngine& ctx, const DiffLayer& lw, bf16* hidden,
    DiffLayerKv& kv, int seq, int pos, const DiffTextConfig& cfg,
    bool is_encoder)
{
    auto& q = ctx.queue;
    int H = cfg.hidden_size;
    float eps = cfg.rms_norm_eps;
    size_t N = (size_t)seq * H;

    // Attention sub-layer.  tmp/attn_out (and everything attention allocates
    // internally) are released at the end of this block, before the FFN runs —
    // so attention and FFN activations share arena storage instead of summing.
    {
        auto& ar = diffarena::arena(ctx.index);
        auto tmp_a = ar.alloc<bf16>(N);
        auto attn_a = ar.alloc<bf16>(N);
        bf16* tmp = tmp_a.data();
        bf16* attn_out = attn_a.data();
        { DIFF_PROF(q, diff_layer_detail::attn_input_norm_label(is_encoder, lw.is_full));
          rms_norm(q, hidden, lw.input_ln.data(), tmp, seq, H, eps); }

        if (is_encoder)
            encoder_attention_forward(ctx, lw, tmp, attn_out, kv, seq, pos, cfg);
        else
            decoder_attention_forward(ctx, lw, tmp, attn_out, kv, seq, pos, cfg);

        // F4: hidden = hidden + post_attention_layernorm(attn_out), one kernel.
        { DIFF_PROF(q, diff_layer_detail::attn_post_norm_label(is_encoder, lw.is_full));
          rms_norm_add_scale(q, attn_out, lw.post_attn_ln.data(),
                             hidden, /*scalar=*/1.0f, seq, H, eps); }
    }

    // Dual FFN (dense MLP + MoE), residual add, layer_scalar.
    float scalar = is_encoder ? lw.enc_layer_scalar : lw.dec_layer_scalar;
    dual_ffn_forward(ctx, lw, hidden, seq, H,
                     cfg.intermediate_size, cfg.num_experts, cfg.top_k_experts,
                     cfg.moe_intermediate_size, eps, scalar, is_encoder);
}

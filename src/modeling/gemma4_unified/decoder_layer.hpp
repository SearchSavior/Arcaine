#pragma once
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
#include "attention.hpp"
#include "ffn.hpp"
#include "config.hpp"
#include <variant>

// ctx: GPU context owning all data (hidden, kv cache, layer weights).
inline void decoder_layer_forward(
    GpuEngine& ctx,
    LayerWeights& lw,
    bf16* hidden,        // (seq, H) — modified in-place
    LayerKvCache& kv,
    int seq_len, int past_len,
    const ModelConfig& cfg,
    const int32_t* block_ids = nullptr
) {
    auto& q   = ctx.queue;
    int H     = cfg.text.hidden_size;
    float eps = cfg.text.rms_norm_eps;

    // --- Attention sub-layer ---
    // input_ln(hidden) → tmp for attention input
    GpuBuffer<bf16> tmp((size_t)seq_len * H, q);
    rms_norm(q, hidden, lw.input_ln.data(), tmp.data(), seq_len, H, eps);

    GpuBuffer<bf16> attn_out((size_t)seq_len * H, q);
    if (!lw.is_full) {
        sliding_attention_forward(ctx,
            std::get<SlidingAttnWeights>(lw.attn),
            tmp.data(), attn_out.data(), kv, seq_len, past_len, cfg.text, block_ids);
    } else {
        full_attention_forward(ctx,
            std::get<FullAttnWeights>(lw.attn),
            tmp.data(), attn_out.data(), kv, seq_len, past_len, cfg.text, block_ids);
    }

    // Fused: hidden += post_attn_ln(attn_out)
    //        tmp    = pre_ffn_ln(hidden)          [FFN input]
    rms_norm_add_rms_norm(q,
        attn_out.data(), lw.post_attn_ln.data(),
        hidden,
        lw.pre_ffn_ln.data(), tmp.data(),
        seq_len, H, eps);

    // --- FFN sub-layer ---
    GpuBuffer<bf16> ffn_out((size_t)seq_len * H, q);
    ffn_forward(ctx, lw.ffn, tmp.data(), ffn_out.data(), seq_len, H,
                cfg.text.intermediate_size);

    // Fused: hidden = (hidden + post_ffn_ln(ffn_out)) * layer_scalar
    rms_norm_add_scale(q,
        ffn_out.data(), lw.post_ffn_ln.data(),
        hidden, lw.layer_scalar,
        seq_len, H, eps);
}

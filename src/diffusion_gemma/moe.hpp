#pragma once
#include "../common/gpu/buffer.hpp"
#include "../common/gpu/engine.hpp"
#include "weights.hpp"

// Dual feed-forward block: dense shared MLP + sparse MoE, summed and combined.
// `hidden` (seq, H) is the post-attention residual; updated in place to the
// layer's FFN output (residual + combine) * layer_scalar.
void dual_ffn_forward(
    GpuEngine& ctx,
    const DiffLayer& lw,
    bf16* hidden,        // (seq, H) in/out
    int seq, int H,
    int intermediate,    // dense MLP intermediate (2112)
    int num_experts,     // 128
    int top_k,           // 8
    int moe_intermediate,// 704
    float rms_eps,
    float layer_scalar,
    bool is_encoder);

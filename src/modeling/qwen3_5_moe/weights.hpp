#pragma once

#include <variant>
#include <vector>
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/nvfp4.hpp"

// Qwen3.5-MoE device weights. NVFP4 projections are stored as Nvfp4Linear
// (identical scheme to diffusion_gemma: weight_packed U8, weight_scale F8_E4M3
// transposed to [K/16, N], dst_scale = input_global * weight_global). The
// unquantized BF16 params of Gated DeltaNet (in_proj_*, conv1d, A_log, dt_bias,
// norm) and the MoE router / shared_expert_gate stay as GpuBuffer<bf16>.

// Full attention (Qwen3_5MoeAttention): GQA 16:2, head_dim 256, partial RoPE
// (rotary_dim 64), q/k RMSNorm per head, sigmoid/swish output gate taken from
// the 2nd half of q_proj's 8192 outputs.
struct QwenFullAttn {
    Nvfp4Linear     q_proj;   // [8192, 2048]  (Q[4096] || output-gate[4096])
    Nvfp4Linear     k_proj;   // [512, 2048]   (2 KV heads * 256)
    Nvfp4Linear     v_proj;   // [512, 2048]
    Nvfp4Linear     o_proj;   // [2048, 4096]
    GpuBuffer<bf16> q_norm;   // [256]
    GpuBuffer<bf16> k_norm;   // [256]
};

// Gated DeltaNet (Qwen3_5MoeGatedDeltaNet), Mamba2-style. All in_proj params are
// BF16/unquantized; only out_proj is NVFP4.
struct QwenLinearAttn {
    GpuBuffer<bf16> in_proj_qkv;  // [8192, 2048]  q[16*128] || k[16*128] || v[32*128]
    GpuBuffer<bf16> in_proj_z;    // [4096, 2048]  gate
    GpuBuffer<bf16> in_proj_a;    // [32, 2048]    A-log decay input
    GpuBuffer<bf16> in_proj_b;    // [32, 2048]    beta input
    GpuBuffer<bf16> conv1d;       // [8192, 1, 4]  depthwise k=4
    GpuBuffer<bf16> A_log;        // [32]
    GpuBuffer<bf16> dt_bias;      // [32]
    GpuBuffer<bf16> norm;         // [128]
    Nvfp4Linear     out_proj;     // [2048, 4096]
};

// MoE block: 256 routed experts (SwiGLU, NVFP4, top-8) + always-on shared
// expert (DeepSeekMoE) with a per-token scalar sigmoid gate.
struct QwenMoE {
    GpuBuffer<bf16>            router_gate;          // [256, 2048]
    std::vector<Nvfp4Linear>   experts_gate_up;      // 256 x fused [1024, 2048]
    std::vector<Nvfp4Linear>   experts_down;         // 256 x [2048, 512]
    Nvfp4Linear                shared_gate_up;       // fused [1024, 2048]
    Nvfp4Linear                shared_down;          // [2048, 512]
    GpuBuffer<bf16>            shared_expert_gate;   // [1, 2048]
};

struct QwenLayer {
    int   layer_idx = 0;
    bool  is_full_attention = false;
    std::variant<QwenFullAttn, QwenLinearAttn> attn;
    QwenMoE          moe;
    GpuBuffer<bf16>  input_layernorm;          // [2048]
    GpuBuffer<bf16>  post_attention_layernorm; // [2048]
};

struct QwenWeights {
    GpuBuffer<bf16>        embed_tokens;  // [vocab, 2048]  (model.language_model.)
    GpuBuffer<bf16>        final_norm;     // [2048]         (model.language_model.norm)
    GpuBuffer<bf16>        lm_head;        // [vocab, 2048]  (top-level, untied)
    std::vector<QwenLayer> layers;
};

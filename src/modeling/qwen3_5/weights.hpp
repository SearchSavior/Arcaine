#pragma once

#include <variant>
#include <vector>

#include "../../runtime/gpu/buffer.hpp"
#include "../../runtime/quantization/fp8.hpp"
#include "../../runtime/quantization/int4.hpp"
#include "../../runtime/quantization/nvfp4.hpp"

// Dense projection weight: FP8 (NVFP4 checkpoint attention/GDN) or
// pack-quantized INT4 W4A16 (AWQ checkpoint).
using Qwen35Proj = std::variant<Fp8Linear, Int4Linear>;

struct Qwen35FullAttentionWeights {
    bool fused_projections = false;
    Qwen35Proj qkv_proj;
    Qwen35Proj q_proj;
    Qwen35Proj k_proj;
    Qwen35Proj v_proj;
    Qwen35Proj o_proj;
    GpuBuffer<bf16> q_norm;
    GpuBuffer<bf16> k_norm;
    GpuBuffer<bf16> k_cache_scale;
    GpuBuffer<bf16> v_cache_scale;
};

struct Qwen35LinearAttentionWeights {
    bool fused_projections = false;
    Qwen35Proj in_proj_qkvz;
    Qwen35Proj in_proj_qkv;
    Qwen35Proj in_proj_z;
    Qwen35Proj out_proj;
    GpuBuffer<bf16> in_proj_a;
    GpuBuffer<bf16> in_proj_b;
    GpuBuffer<bf16> in_proj_ba;
    GpuBuffer<bf16> conv1d;
    // [kernel, channel], used by the fused ESIMD M=1 DeltaNet path.
    GpuBuffer<bf16> conv1d_time_major;
    GpuBuffer<bf16> A_log;
    GpuBuffer<bf16> dt_bias;
    GpuBuffer<bf16> norm;
};

struct Qwen35MlpWeights {
    std::variant<Nvfp4Linear, Fp8Linear, Int4Linear> gate_up;
    std::variant<Nvfp4Linear, Fp8Linear, Int4Linear> down;
};

struct Qwen35LayerWeights {
    int index = 0;
    int gpu = 0;
    bool full_attention = false;
    std::variant<Qwen35FullAttentionWeights, Qwen35LinearAttentionWeights> mixer;
    Qwen35MlpWeights mlp;
    GpuBuffer<bf16> input_layernorm;
    GpuBuffer<bf16> post_attention_layernorm;
};

struct Qwen35VisionBlockWeights {
    GpuBuffer<bf16> norm1_weight;
    GpuBuffer<bf16> norm1_bias;
    GpuBuffer<bf16> norm2_weight;
    GpuBuffer<bf16> norm2_bias;
    GpuBuffer<bf16> qkv_weight;
    GpuBuffer<bf16> qkv_bias;
    GpuBuffer<bf16> proj_weight;
    GpuBuffer<bf16> proj_bias;
    GpuBuffer<bf16> fc1_weight;
    GpuBuffer<bf16> fc1_bias;
    GpuBuffer<bf16> fc2_weight;
    GpuBuffer<bf16> fc2_bias;
};

struct Qwen35VisionWeights {
    GpuBuffer<bf16> patch_weight;
    GpuBuffer<bf16> patch_bias;
    GpuBuffer<bf16> position_embedding;
    std::vector<Qwen35VisionBlockWeights> blocks;
    GpuBuffer<bf16> merger_norm_weight;
    GpuBuffer<bf16> merger_norm_bias;
    GpuBuffer<bf16> merger_fc1_weight;
    GpuBuffer<bf16> merger_fc1_bias;
    GpuBuffer<bf16> merger_fc2_weight;
    GpuBuffer<bf16> merger_fc2_bias;
};

struct Qwen35MtpWeights {
    GpuBuffer<bf16> fc;
    GpuBuffer<bf16> pre_fc_norm_embedding;
    GpuBuffer<bf16> pre_fc_norm_hidden;
    GpuBuffer<bf16> input_layernorm;
    GpuBuffer<bf16> post_attention_layernorm;
    GpuBuffer<bf16> q_proj;
    GpuBuffer<bf16> k_proj;
    GpuBuffer<bf16> v_proj;
    GpuBuffer<bf16> o_proj;
    GpuBuffer<bf16> q_norm;
    GpuBuffer<bf16> k_norm;
    GpuBuffer<bf16> gate_proj;
    GpuBuffer<bf16> up_proj;
    GpuBuffer<bf16> down_proj;
    GpuBuffer<bf16> norm;
};

struct Qwen35Weights {
    GpuBuffer<bf16> embed_tokens;
    GpuBuffer<bf16> final_norm;
    // FP8 (NVFP4 checkpoint) or plain BF16 (AWQ checkpoint, unquantized F16).
    std::variant<Fp8Linear, GpuBuffer<bf16>> lm_head;
    std::vector<Qwen35LayerWeights> layers;
    Qwen35VisionWeights vision;
    Qwen35MtpWeights mtp;
};

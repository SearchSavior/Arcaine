#pragma once

#include <variant>
#include <vector>

#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/fp8.hpp"
#include "../../common/gpu/nvfp4.hpp"

struct Qwen35FullAttentionWeights {
    bool fused_projections = false;
    Fp8Linear qkv_proj;
    Fp8Linear q_proj;
    Fp8Linear k_proj;
    Fp8Linear v_proj;
    Fp8Linear o_proj;
    GpuBuffer<bf16> q_norm;
    GpuBuffer<bf16> k_norm;
    GpuBuffer<bf16> k_cache_scale;
    GpuBuffer<bf16> v_cache_scale;
};

struct Qwen35LinearAttentionWeights {
    bool fused_projections = false;
    Fp8Linear in_proj_qkvz;
    Fp8Linear in_proj_qkv;
    Fp8Linear in_proj_z;
    Fp8Linear out_proj;
    GpuBuffer<bf16> in_proj_a;
    GpuBuffer<bf16> in_proj_b;
    GpuBuffer<bf16> conv1d;
    GpuBuffer<bf16> A_log;
    GpuBuffer<bf16> dt_bias;
    GpuBuffer<bf16> norm;
};

struct Qwen35MlpWeights {
    bool nvfp4 = false;
    std::variant<Nvfp4Linear, Fp8Linear> gate_up;
    std::variant<Nvfp4Linear, Fp8Linear> down;
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
    Fp8Linear lm_head;
    std::vector<Qwen35LayerWeights> layers;
    Qwen35VisionWeights vision;
    Qwen35MtpWeights mtp;
};

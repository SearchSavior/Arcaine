#pragma once
#include <variant>
#include <vector>
#include "../../common/gpu/buffer.hpp"

struct SlidingAttnWeights {
    GpuBuffer<bf16> q_proj;  // (4096, 3840)
    GpuBuffer<bf16> k_proj;  // (2048, 3840)
    GpuBuffer<bf16> v_proj;  // (2048, 3840)
    GpuBuffer<bf16> o_proj;  // (3840, 4096)
    GpuBuffer<bf16> q_norm;  // (256,)
    GpuBuffer<bf16> k_norm;  // (256,)
};

struct FullAttnWeights {
    GpuBuffer<bf16> q_proj;  // (8192, 3840)
    GpuBuffer<bf16> k_proj;  // (512,  3840) — K=V, no v_proj
    GpuBuffer<bf16> o_proj;  // (3840, 8192)
    GpuBuffer<bf16> q_norm;  // (512,)
    GpuBuffer<bf16> k_norm;  // (512,)
};

struct FfnWeights {
    GpuBuffer<bf16> gate_proj;  // (15360, 3840)
    GpuBuffer<bf16> up_proj;    // (15360, 3840)
    GpuBuffer<bf16> down_proj;  // (3840, 15360)
};

struct LayerWeights {
    bool is_full;
    std::variant<SlidingAttnWeights, FullAttnWeights> attn;
    FfnWeights ffn;
    GpuBuffer<bf16> input_ln;      // (3840,)
    GpuBuffer<bf16> post_attn_ln;  // (3840,)
    GpuBuffer<bf16> pre_ffn_ln;    // (3840,)
    GpuBuffer<bf16> post_ffn_ln;   // (3840,)
    float layer_scalar = 1.0f;
};

struct VisionWeights {
    GpuBuffer<bf16> patch_ln1_w;    // (6912,)
    GpuBuffer<bf16> patch_ln1_b;    // (6912,)
    GpuBuffer<bf16> patch_dense_w;  // (3840, 6912)
    GpuBuffer<bf16> patch_dense_b;  // (3840,)
    GpuBuffer<bf16> patch_ln2_w;    // (3840,)
    GpuBuffer<bf16> patch_ln2_b;    // (3840,)
    GpuBuffer<bf16> pos_embedding;  // (1120, 2, 3840)
    GpuBuffer<bf16> pos_norm_w;     // (3840,)
    GpuBuffer<bf16> pos_norm_b;     // (3840,)
    GpuBuffer<bf16> proj_w;         // (3840, 3840)
};

struct AudioWeights {
    GpuBuffer<bf16> proj_w;  // (3840, 640)
};

struct GlobalWeights {
    GpuBuffer<bf16>        embed_tokens;  // (262144, 3840)
    GpuBuffer<bf16>        final_norm;    // (3840,)
    std::vector<LayerWeights> layers;     // [48]
    VisionWeights          vision;
    AudioWeights           audio;
};

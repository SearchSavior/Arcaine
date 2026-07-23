#pragma once

#include <algorithm>

#include "config.hpp"
#include "../../common/gpu/buffer.hpp"

// Persistent per-device scratch. Buffers are role-based and reused across all
// layers; no layer-forward path allocates device memory.
struct Qwen35Workspace {
    int max_seq_len = 0;
    GpuBuffer<bf16> tmp0;  // gate_up / q_proj / mixed_qkv
    GpuBuffer<bf16> tmp1;  // activation / conv output / K
    GpuBuffer<bf16> tmp2;  // Q
    GpuBuffer<bf16> tmp3;  // gate / repeated K
    GpuBuffer<bf16> tmp4;  // V / DeltaNet core
    GpuBuffer<uint8_t> input_packed;
    GpuBuffer<uint8_t> input_scale;
    GpuBuffer<uint8_t> activation_packed;
    GpuBuffer<uint8_t> activation_scale;

    void init(const Qwen35Config& config, int max_seq, sycl::queue& queue) {
        max_seq_len = max_seq;
        const auto& c = config.text;
        int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
        int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        int conv_dim = 2 * key_dim + value_dim;
        int q_proj = c.num_attention_heads * c.head_dim * 2;
        size_t s = static_cast<size_t>(max_seq);
        tmp0 = GpuBuffer<bf16>(s * std::max({2 * c.intermediate_size, conv_dim, q_proj}), queue);
        tmp1 = GpuBuffer<bf16>(s * std::max(c.intermediate_size, conv_dim), queue);
        tmp2 = GpuBuffer<bf16>(s * value_dim, queue);
        tmp3 = GpuBuffer<bf16>(s * value_dim, queue);
        tmp4 = GpuBuffer<bf16>(s * value_dim, queue);
        input_packed = GpuBuffer<uint8_t>(s * c.hidden_size / 2, queue);
        input_scale = GpuBuffer<uint8_t>(s * c.hidden_size / 16, queue);
        activation_packed = GpuBuffer<uint8_t>(s * c.intermediate_size / 2, queue);
        activation_scale = GpuBuffer<uint8_t>(s * c.intermediate_size / 16, queue);
    }
};

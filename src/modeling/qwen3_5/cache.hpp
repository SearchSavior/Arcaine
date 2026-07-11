#pragma once

#include <vector>

#include "config.hpp"
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"

struct Qwen35KvLayerCache {
    GpuBuffer<bf16> key;
    GpuBuffer<bf16> value;
    int filled = 0;
    int capacity = 0;
};

struct Qwen35DeltaLayerCache {
    GpuBuffer<bf16> conv_state;
    GpuBuffer<float> recurrent_state;
    bool has_state = false;
};

struct Qwen35Caches {
    std::vector<Qwen35KvLayerCache> kv;
    std::vector<Qwen35DeltaLayerCache> delta;

    void init(const Qwen35Config& config, int max_seq_len, int split_layer) {
        const auto& c = config.text;
        kv.resize(c.num_hidden_layers);
        delta.resize(c.num_hidden_layers);
        int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
        int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        int conv_dim = 2 * key_dim + value_dim;
        for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
            int gpu = (layer < split_layer || GpuEngine::count() < 2) ? 0 : 1;
            auto& queue = GpuEngine::get(gpu).queue;
            if (c.is_full_attn(layer)) {
                size_t count = (size_t)max_seq_len * c.num_key_value_heads * c.head_dim;
                kv[layer].key = GpuBuffer<bf16>(count, queue);
                kv[layer].value = GpuBuffer<bf16>(count, queue);
                kv[layer].capacity = max_seq_len;
            } else {
                delta[layer].conv_state = GpuBuffer<bf16>(
                    (size_t)conv_dim * (c.linear_conv_kernel_dim - 1), queue);
                delta[layer].recurrent_state = GpuBuffer<float>(
                    (size_t)c.linear_num_value_heads * c.linear_key_head_dim *
                    c.linear_value_head_dim, queue);
                delta[layer].conv_state.zero();
                delta[layer].recurrent_state.zero();
            }
        }
    }

    void reset() {
        for (auto& layer : kv) layer.filled = 0;
        for (auto& layer : delta) {
            if (!layer.conv_state.empty()) {
                layer.conv_state.zero();
                layer.recurrent_state.zero();
            }
            layer.has_state = false;
        }
    }
};

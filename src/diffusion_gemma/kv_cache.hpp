#pragma once
#include <vector>
#include "../common/gpu/buffer.hpp"
#include "../common/gpu/engine.hpp"
#include "config.hpp"

// Encoder KV cache (causal, written by the encoder; read-only for the decoder).
struct DiffLayerKv {
    GpuBuffer<bf16> k;  // (max_seq, nkv, hd)
    GpuBuffer<bf16> v;  // (max_seq, nkv, hd)
    int filled = 0;
    int nkv = 0, hd = 0, max_seq = 0;
};

class DiffKvCache {
public:
    DiffKvCache() = default;
    DiffKvCache(const DiffConfig& cfg, int max_seq, int split_layer) {
        layers_.resize(cfg.text.num_hidden_layers);
        for (int l = 0; l < cfg.text.num_hidden_layers; ++l) {
            auto& kv = layers_[l];
            bool full = cfg.text.is_full_attention[l];
            kv.nkv = full ? cfg.text.num_global_kv_heads : cfg.text.num_kv_heads;
            kv.hd  = full ? cfg.text.global_head_dim     : cfg.text.head_dim;
            kv.max_seq = max_seq;
            int gpu = (l < split_layer) ? 0 : 1;
            auto& q = GpuEngine::get(gpu).queue;
            size_t n = (size_t)max_seq * kv.nkv * kv.hd;
            kv.k = GpuBuffer<bf16>(n, q);
            kv.v = GpuBuffer<bf16>(n, q);
        }
    }
    DiffLayerKv& layer(int l) { return layers_[l]; }
    void reset() { for (auto& kv : layers_) kv.filled = 0; }

    // Capacity of the allocated cache (max_seq slots, all layers, K and V).
    int    max_seq() const { return layers_.empty() ? 0 : layers_[0].max_seq; }
    size_t total_bytes() const {
        size_t b = 0;
        for (const auto& kv : layers_) b += (kv.k.count() + kv.v.count()) * sizeof(bf16);
        return b;
    }
    // Bytes one sequence position occupies across all layers (K and V).
    size_t bytes_per_token() const {
        int ms = max_seq();
        return ms ? total_bytes() / (size_t)ms : 0;
    }

private:
    std::vector<DiffLayerKv> layers_;
};

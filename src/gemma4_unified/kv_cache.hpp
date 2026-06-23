#pragma once
#include <vector>
#include <stdexcept>
#include "../common/gpu/buffer.hpp"
#include "config.hpp"

struct LayerKvCache {
    GpuBuffer<bf16> k;  // (max_seq_len, nkv_heads, head_dim)
    GpuBuffer<bf16> v;  // same shape — always separate (K != V in Gemma4Unified)
    GpuBuffer<bf16> k_decode;  // sliding decode layout: (nkv_heads, max_seq, head_dim)
    GpuBuffer<bf16> v_decode;  // sliding decode layout: (nkv_heads, head_dim, max_seq)
    int   filled   = 0;
    bool  kv_shared = false;  // unused, kept for compatibility

    // Number of KV heads and head_dim stored
    int nkv_heads = 0;
    int head_dim  = 0;
    int max_seq   = 0;
};

class KvCache {
public:
    KvCache() = default;

    // split_layer: layers [0, split_layer) go on GPU 0,
    //              layers [split_layer, L) go on GPU 1.
    // Defaults to all-GPU-0 (INT_MAX means no split).
    KvCache(const ModelConfig& cfg, int max_seq_len, int split_layer = INT_MAX) {
        max_seq_   = max_seq_len;
        layers_.resize(cfg.text.num_hidden_layers);

        for (int l = 0; l < cfg.text.num_hidden_layers; ++l) {
            auto& lkv = layers_[l];
            lkv.max_seq = max_seq_len;

            bool full = cfg.text.is_full_attention[l];
            lkv.kv_shared = full;

            if (!full) {
                // Sliding: 8 KV heads × 256 head_dim
                lkv.nkv_heads = cfg.text.num_kv_heads;
                lkv.head_dim  = cfg.text.head_dim;
            } else {
                // Full: 1 global KV head × 512 global_head_dim
                // K = k_norm(k_proj), V = v_norm(k_proj) — always separate
                lkv.nkv_heads = cfg.text.num_global_kv_heads;
                lkv.head_dim  = cfg.text.global_head_dim;
            }
            {
                int gpu_idx = (l < split_layer) ? 0 : 1;
                sycl::queue& q = GpuEngine::get(gpu_idx).queue;
                size_t n = (size_t)max_seq_len * lkv.nkv_heads * lkv.head_dim;
                lkv.k = GpuBuffer<bf16>(n, q);
                lkv.v = GpuBuffer<bf16>(n, q);
                if (!full) {
                    lkv.k_decode = GpuBuffer<bf16>(n, q);
                    lkv.v_decode = GpuBuffer<bf16>(n, q);
                }
            }
        }
    }

    LayerKvCache& layer(int l) { return layers_[l]; }
    const LayerKvCache& layer(int l) const { return layers_[l]; }
    int max_seq() const { return max_seq_; }
    void reset() { for (auto& lkv : layers_) lkv.filled = 0; }

    // Append seq_len new tokens' K (and optionally V) into layer l's cache.
    // k_new / v_new: (seq_len, nkv_heads, head_dim)
    void append(int l, const bf16* k_new, const bf16* v_new, int seq_len, sycl::queue& q) {
        auto& lkv = layers_[l];
        if (lkv.filled + seq_len > max_seq_)
            throw std::runtime_error("KV cache overflow at layer " + std::to_string(l));

        size_t row = (size_t)lkv.nkv_heads * lkv.head_dim;
        size_t offset = (size_t)lkv.filled * row;
        size_t nbytes  = (size_t)seq_len * row * sizeof(bf16);

        q.memcpy(lkv.k.data() + offset, k_new, nbytes);
        if (!lkv.kv_shared && v_new)
            q.memcpy(lkv.v.data() + offset, v_new, nbytes);
        lkv.filled += seq_len;  // CPU-side counter; GPU ordering via in-order queue
    }

    // Get pointer to cached K for layer l.
    bf16* k(int l) { return layers_[l].k.data(); }
    // Get pointer to cached V for layer l.
    bf16* v(int l) { return layers_[l].v.data(); }
    int len(int l) const { return layers_[l].filled; }

private:
    std::vector<LayerKvCache> layers_;
    int max_seq_ = 0;
};

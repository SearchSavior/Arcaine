#pragma once
#include <cstdint>
#include "../common/gpu/buffer.hpp"
#include "../common/gpu/ops.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"

// Compute attention output for one layer.
// hidden: (seq_len, hidden_size) — input and output
// tmp:    (seq_len, hidden_size) — scratch
// ctx:    GPU context; all buffers must live on ctx's device.
void sliding_attention_forward(
    GpuEngine& ctx,
    const SlidingAttnWeights& w,
    bf16* hidden,
    bf16* tmp,
    LayerKvCache& kv,
    int seq_len, int past_len,
    const TextConfig& cfg,
    const int32_t* block_ids = nullptr
);

void full_attention_forward(
    GpuEngine& ctx,
    const FullAttnWeights& w,
    bf16* hidden,
    bf16* tmp,
    LayerKvCache& kv,
    int seq_len, int past_len,
    const TextConfig& cfg,
    const int32_t* block_ids = nullptr
);

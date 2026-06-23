#pragma once
#include "../common/gpu/buffer.hpp"
#include "../common/gpu/engine.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"

// Encoder attention: causal, GQA, writes K/V into the cache at `past_len`.
// hidden: (seq, H) input (already input_layernorm'd); out: (seq, H).
void encoder_attention_forward(
    GpuEngine& ctx, const DiffLayer& lw,
    const bf16* hidden, bf16* out,
    DiffLayerKv& kv, int seq, int past_len,
    const DiffTextConfig& cfg);

// Decoder attention: bidirectional over [encoder KV ; canvas], read-only cache.
// Canvas positions start at absolute `enc_len`.
void decoder_attention_forward(
    GpuEngine& ctx, const DiffLayer& lw,
    const bf16* hidden, bf16* out,
    DiffLayerKv& enc_kv, int seq, int enc_len,
    const DiffTextConfig& cfg);

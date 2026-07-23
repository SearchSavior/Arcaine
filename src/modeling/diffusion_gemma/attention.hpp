#pragma once
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/nvfp4.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"

// Encoder attention: causal, GQA, writes K/V into the cache at `past_len`.
// hidden: (seq, H) input (already input_layernorm'd); out: (seq, H).
// `session`, when non-null and recording, causes the raw q.submit kernels and
// oneDNN matmuls below to be captured into that session's step graph (Phase 0
// confirmed oneDNN matmul is capture-safe; raw submits are captured by the
// active recording). Default nullptr preserves the pre-session behavior exactly.
void encoder_attention_forward(
    GpuEngine& ctx, const DiffLayer& lw,
    const bf16* hidden, bf16* out,
    DiffLayerKv& kv, int seq, int past_len,
    const DiffTextConfig& cfg,
    Nvfp4GraphSession* session = nullptr);

// Decoder attention: bidirectional over [encoder KV ; canvas], read-only cache.
// Canvas positions start at absolute `enc_len`. See encoder_attention_forward
// for the `session` capture semantics.
void decoder_attention_forward(
    GpuEngine& ctx, const DiffLayer& lw,
    const bf16* hidden, bf16* out,
    DiffLayerKv& enc_kv, int seq, int enc_len,
    const DiffTextConfig& cfg,
    Nvfp4GraphSession* session = nullptr);

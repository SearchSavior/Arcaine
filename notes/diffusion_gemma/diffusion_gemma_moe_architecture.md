# DiffusionGemma 26B-A4B: Block-Diffusion MoE Architecture

**Source**: HuggingFace Transformers 5.8.0.dev0 (`transformers/models/diffusion_gemma/`)
**Reference Model**: `google/diffusiongemma-26B-A4B-it` (`models/diffusiongemma-26B-A4B-it/`)
**HF class**: `DiffusionGemmaForBlockDiffusion`  •  `model_type: "diffusion_gemma"`
**Status**: Next implementation target for the SYCL/oneDNN engine
**Scope**: This document describes the architecture and inference algorithm precisely enough to re-implement it from scratch in the existing C++/SYCL engine. It assumes familiarity with [the Gemma4 Unified doc](../gemma4_unified_12b/GEMMA4_UNIFIED_ARCHITECTURE.md), since DiffusionGemma reuses most of that text-decoder design.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What Is (and Isn't) Reused from Gemma4 Unified](#2-what-is-and-isnt-reused-from-gemma4-unified)
3. [Configuration](#3-configuration)
4. [Weights Layout & Exact Shapes](#4-weights-layout--exact-shapes)
5. [The Transformer Block (Dual FFN: Dense MLP + MoE)](#5-the-transformer-block-dual-ffn-dense-mlp--moe)
6. [Attention: Encoder Mode vs Decoder Mode](#6-attention-encoder-mode-vs-decoder-mode)
7. [RoPE, Norms, Embedding Scale, Softcapping](#7-rope-norms-embedding-scale-softcapping)
8. [Self-Conditioning](#8-self-conditioning)
9. [Model Forward (one denoising forward pass)](#9-model-forward-one-denoising-forward-pass)
10. [Generation: Block-Diffusion Algorithm](#10-generation-block-diffusion-algorithm)
11. [Sampler, Temperature Schedule, Stopping](#11-sampler-temperature-schedule-stopping)
12. [Attention Masks](#12-attention-masks)
13. [Numeric Constants Reference](#13-numeric-constants-reference)
14. [Implementation Guide for the SYCL Engine](#14-implementation-guide-for-the-sycl-engine)

---

## 1. Executive Summary

DiffusionGemma is a **discrete text-diffusion** language model built on the **Gemma 4 26B-A4B Mixture-of-Experts** backbone (25.2B total params, ~3.8B active). Instead of autoregressive next-token decoding, it generates a fixed-size **canvas** of `canvas_length = 256` tokens *in parallel* and refines it over a small number of **denoising steps** (≤ 48). Long outputs are produced by stitching canvases together autoregressively (**block diffusion**).

A single set of transformer weights is used in **two modes**:

- **Encoder mode** — autoregressive, *causal* attention, **writes** the KV cache. Processes the prompt (and each finished canvas) to produce a read-only context KV cache. This is essentially a `Gemma4Model` text decoder.
- **Denoiser / Decoder mode** — *bidirectional* attention over the canvas, **reads** the encoder KV cache without updating it (cross-attention), and concatenates the canvas's own K/V so canvas positions attend to each other. Adds a **self-conditioning** signal from the previous step's logits.

```
prompt ──► ENCODER (causal, writes KV) ──► frozen context KV cache
                                                  │ (read-only cross-attn)
 random 256-tok canvas ──► DECODER (bidirectional) ──► logits[256, vocab]
        ▲                                                  │
        └──── accept / renoise (entropy-bound sampler) ◄───┘   ×  N denoising steps
                                                  │
                          denoised canvas ──► append to sequence ──► re-encode ──► next canvas
```

**Headline differences vs an AR LLM**: compute-bound (not memory-bound) at batch 1; many tokens per forward pass; needs a denoising loop, a canvas sampler, a temperature schedule, and adaptive stopping.

---

## 2. What Is (and Isn't) Reused from Gemma4 Unified

The text block is **structurally the Gemma4 text decoder** with the same attention idioms the engine already implements. Port-relevant facts:

**Reused as-is (already in `gemma4_unified/`):**
- BF16 everywhere; `GpuBuffer`, oneDNN matmul, embedding-lookup, RMSNorm, dual-RoPE, GQA batched attention, logit softcapping.
- **Q/K/V RMSNorm + attention `scale = 1.0`** (Q/K are normalized, so no `1/√d`). Same as Gemma4 Unified.
- **Hybrid sliding/full layers** in a **5:1** pattern; sliding `head_dim = 256`, full `head_dim = 512`.
- **Dual RoPE**: sliding θ=10000 (`rope_type="default"`, full head_dim), full θ=1000000 (`rope_type="proportional"`, `partial_rotary_factor=0.25`, over `global_head_dim=512`).
- **Full-attention layers share K and V** (no `v_proj`): `value = v_norm(k_proj(x))` with **no RoPE on V**. Sliding layers have a real `v_proj`.
- **`gelu_pytorch_tanh`**, `final_logit_softcapping = 30.0`, tied embeddings, `embed_scale = √hidden`.
- The **vision tower is identical to Gemma4's** (`model_type: "gemma4_vision"`, 27 layers, hidden 1152) and the multimodal embedder/`masked_scatter` placeholder logic is the same. *(Image-only here; no audio/video.)*

**New for DiffusionGemma (must be built):**
1. **Mixture-of-Experts** per layer (128 experts, top-8) **alongside** a dense shared MLP — a *dual* FFN that sums two paths.
2. **Encoder/decoder split** over one weight set: causal+KV-write vs bidirectional+KV-read with canvas-KV concat.
3. **Block-diffusion generation loop** (outer AR over canvases, inner denoising over steps).
4. **Entropy-bound sampler** (accept/renoise), **linear temperature schedule**, **stable-&-confident stopping**.
5. **Self-conditioning** FFN that injects the previous step's soft predictions.
6. Decoder **bidirectional attention masks** (canvas ↔ canvas + canvas → valid encoder KV).

**Dimensional changes vs the 12B unified model**: 30 layers (was 48), hidden **2816** (was 3840), dense-MLP intermediate **2112**, plus MoE. Sliding window **1024** (same).

---

## 3. Configuration

From `models/diffusiongemma-26B-A4B-it/config.json`. (`configuration_diffusion_gemma.py` class *defaults* differ — always trust the checkpoint's `config.json`.)

### Top-level `DiffusionGemmaConfig`
| Key | Value | Meaning |
|---|---|---|
| `model_type` | `diffusion_gemma` | registry key |
| `canvas_length` | **256** | block size for block diffusion |
| `image_token_id` | 258880 | multimodal placeholder |
| `boi_token_id` / `eoi_token_id` | 255999 / 258882 | begin/end of image |
| `eos_token_id` | `[1, 106]` | (generation uses `[1, 106, 50]`, see below) |
| `vision_soft_tokens_per_image` | 280 | |
| `tie_word_embeddings` | true | encoder embed = decoder embed = lm_head |

### `text_config` (`diffusion_gemma_text`)
| Key | Value |
|---|---|
| `vocab_size` | 262144 |
| `hidden_size` | **2816** |
| `intermediate_size` (dense MLP) | **2112** |
| `num_hidden_layers` | **30** |
| `num_attention_heads` | 16 (Q) |
| `num_key_value_heads` | 8 (sliding KV) |
| `num_global_key_value_heads` | **2** (full KV) |
| `head_dim` | 256 (sliding) |
| `global_head_dim` | 512 (full) |
| `num_experts` | **128** |
| `top_k_experts` | **8** |
| `moe_intermediate_size` | **704** (per-expert) |
| `sliding_window` | **1024** |
| `max_position_embeddings` | 262144 |
| `rms_norm_eps` | 1e-6 |
| `final_logit_softcapping` | 30.0 |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `use_bidirectional_attention` | `"vision"` |
| `attention_bias` | false |
| `pad/bos/eos token_id` | 0 / 2 / 1 |
| `rope_parameters.sliding_attention` | `{rope_type: default, rope_theta: 10000}` |
| `rope_parameters.full_attention` | `{rope_type: proportional, partial_rotary_factor: 0.25, rope_theta: 1000000}` |

**`layer_types`** (30 entries, 5:1 sliding:full). Full-attention layers are at indices **5, 11, 17, 23, 29** (every 6th; last layer forced to full). All others are sliding.

---

## 4. Weights Layout & Exact Shapes

Tensor names and shapes are from the real checkpoint (`model.safetensors.index.json`, 11 shards, BF16). The model nests as `model.encoder.*` and `model.decoder.*`, but **almost everything is tied**:

### Tying (one physical copy, many roles)
`DiffusionGemmaModel._tied_weights_keys` ties the **encoder language-model layers to the decoder layers**, the encoder `norm` to decoder `norm`, and the embedding table across encoder/decoder; and `lm_head.weight` ties to `decoder.embed_tokens.weight`. Net effect for a from-scratch implementation:

- **One** embedding table `[262144, 2816]` serves: encoder embed, decoder embed, `lm_head` (logits via `embedᵀ`), and the self-conditioning soft-embedding matmul.
- **One** set of 30 layer weight tensors, executed in encoder mode and decoder mode.
- **Exception — not tied:** each layer's `layer_scalar` **buffer** is stored separately for encoder vs decoder (`...encoder.language_model.layers.{i}.layer_scalar` and `...decoder.layers.{i}.layer_scalar`). They *can* differ; keep both.
- **Decoder-only** weights: `self_conditioning.*` and `decoder.norm` (tied) — the self-conditioning FFN has no encoder counterpart.

### Per-decoder-layer tensors (layer 0 = sliding; layer 5 = full)
| Tensor | Shape (sliding) | Shape (full) | Notes |
|---|---|---|---|
| `self_attn.q_proj.weight` | `[4096, 2816]` | `[8192, 2816]` | 16·256 vs 16·512 |
| `self_attn.k_proj.weight` | `[2048, 2816]` | `[1024, 2816]` | 8·256 vs 2·512 |
| `self_attn.v_proj.weight` | `[2048, 2816]` | *absent* | full layers reuse K proj as V |
| `self_attn.o_proj.weight` | `[2816, 4096]` | `[2816, 8192]` | |
| `self_attn.{q,k}_norm.weight` | `[256]` | `[512]` | RMSNorm over head_dim; `v_norm` has **no** weight (scaleless) |
| `mlp.gate_proj.weight` | `[2112, 2816]` | same | **dense shared MLP** (GeGLU) |
| `mlp.up_proj.weight` | `[2112, 2816]` | same | |
| `mlp.down_proj.weight` | `[2816, 2112]` | same | |
| `experts.gate_up_proj` | `[128, 1408, 2816]` | same | 128 experts × (2·704) × hidden (packed gate+up) |
| `experts.down_proj` | `[128, 2816, 704]` | same | 128 × hidden × 704 |
| `router.proj.weight` | `[128, 2816]` | same | expert scores |
| `router.scale` | `[2816]` | same | learnable, init ones |
| `router.per_expert_scale` | `[128]` | same | learnable, init ones |
| `input_layernorm.weight` | `[2816]` | same | |
| `post_attention_layernorm.weight` | `[2816]` | same | |
| `pre_feedforward_layernorm.weight` | `[2816]` | same | dense-MLP path |
| `pre_feedforward_layernorm_2.weight` | `[2816]` | same | MoE path |
| `post_feedforward_layernorm_1.weight` | `[2816]` | same | dense-MLP path |
| `post_feedforward_layernorm_2.weight` | `[2816]` | same | MoE path |
| `post_feedforward_layernorm.weight` | `[2816]` | same | final combine |
| `layer_scalar` | `[1]` | same | buffer, per enc/dec |

### Top-level / decoder-only
| Tensor | Shape |
|---|---|
| `model.decoder.embed_tokens.weight` | `[262144, 2816]` (tied everywhere) |
| `model.decoder.norm.weight` | `[2816]` |
| `model.decoder.self_conditioning.pre_norm.weight` | `[2816]` |
| `model.decoder.self_conditioning.gate_proj.weight` | `[2112, 2816]` |
| `model.decoder.self_conditioning.up_proj.weight` | `[2112, 2816]` |
| `model.decoder.self_conditioning.down_proj.weight` | `[2816, 2112]` |
| `model.encoder.vision_tower.*`, `model.encoder.embed_vision.*` | Gemma4 vision tower (unchanged) |

> `self_conditioning.post_norm` is a scaleless RMSNorm (no weight tensor, like `v_norm`).

---

## 5. The Transformer Block (Dual FFN: Dense MLP + MoE)

`DiffusionGemma{Encoder,Decoder}TextLayer` are **identical in structure** — they differ only in which attention module they hold (see §6). Each block runs attention, then **two parallel feed-forward paths** (a dense shared MLP and a sparse MoE), sums them, and applies a residual + `layer_scalar`.

```python
residual = h
h = input_layernorm(h)
h, _ = self_attn(h, position_embeddings, attention_mask, past_key_values)
h = post_attention_layernorm(h)
h = residual + h                      # ── attention residual

residual = h                          # post-attention hidden; feeds BOTH FFN paths
# Path 1 — dense shared MLP (GeGLU, intermediate=2112)
x1 = pre_feedforward_layernorm(h)
x1 = down_proj(gelu_tanh(gate_proj(x1)) * up_proj(x1))
hidden_states_1 = post_feedforward_layernorm_1(x1)

# Path 2 — sparse MoE (128 experts, top-8, intermediate=704)
flat = residual.reshape(-1, H)
route_in  = flat                                  # router sees the RAW residual
expert_in = pre_feedforward_layernorm_2(flat)     # experts see a normed residual
_, top_k_weights, top_k_index = router(route_in)  # see router below
x2 = experts(expert_in, top_k_index, top_k_weights).reshape(residual.shape)
hidden_states_2 = post_feedforward_layernorm_2(x2)

# Combine
h = hidden_states_1 + hidden_states_2
h = post_feedforward_layernorm(h)
h = residual + h                      # ── FFN residual
h = h * layer_scalar                  # per-layer learnable scalar (buffer)
```

Note both FFN paths branch from the **same** post-attention `residual`, each with its **own** pre-norm; the dense path is computed from the dense pre-norm output while the MoE path is computed from `pre_feedforward_layernorm_2`. The router, importantly, consumes the **un-normalized** residual (it applies its own internal norm).

### Router (`DiffusionGemmaTextRouter`)
```python
h = norm(h)                          # RMSNorm, scaleless (with_scale=False)
h = h * scale * (hidden_size ** -0.5)         # learnable per-channel scale + 1/√H
expert_scores = proj(h)                       # [tokens, 128]
probs = softmax(expert_scores, dtype=fp32)    # fp32 softmax
top_k_weights, top_k_index = topk(probs, k=8)
top_k_weights /= top_k_weights.sum(-1, keepdim=True)   # renormalize to sum 1
top_k_weights *= per_expert_scale[top_k_index]          # learnable per-expert gain
```

### Experts (`DiffusionGemmaTextExperts`)
Weights are 3-D batched: `gate_up_proj[E, 2·704, H]`, `down_proj[E, H, 704]`. For each token and each of its 8 selected experts `e`:
```python
gate, up = chunk(linear(x, gate_up_proj[e]), 2, dim=-1)   # each [.,704]
y = gelu_tanh(gate) * up
y = linear(y, down_proj[e])                                # [.,H]
out += top_k_weights[token, slot] * y                      # scatter-add over tokens
```
The reference loops experts and `index_add_`s contributions; on GPU this is a grouped/segmented GEMM. The dense MLP acts as an always-on "shared expert" (the README's "8 active / 128 total **and 1 shared**").

---

## 6. Attention: Encoder Mode vs Decoder Mode

Both share the Gemma4 attention body (per-head Q/K RMSNorm → RoPE → GQA, `scale=1.0`, softmax fp32). They differ **only** in KV-cache handling and causality.

**Common per-layer geometry** (same as §4 shapes):
- Sliding layer: 16 Q heads, 8 KV heads, head_dim 256, `sliding_window=1024`, real `v_proj`.
- Full layer: 16 Q heads, 2 KV heads, head_dim 512, **no `v_proj`** → `value_states = v_norm(k_proj(x))` (scaleless norm, **no RoPE on V**), `K = rope(k_norm(k_proj(x)))`.
- `q = rope(q_norm(q_proj(x)))`. RoPE `unsqueeze_dim=2` (applied in `[B, S, heads, dim]` layout before transpose).

### Encoder attention (`DiffusionGemmaEncoderTextAttention`)
- `is_causal = (use_bidirectional_attention != "all")` → here **True** (causal; vision spans made bidirectional via a block mask, exactly like Gemma4 Unified `use_bidirectional_attention="vision"`).
- **Updates** the KV cache: `key, value = past_key_values.update(key, value, layer_idx)`.
- This is the prompt/context pass. Output hidden states are discarded for generation; only the KV cache matters.

### Decoder attention (`DiffusionGemmaDecoderTextAttention`)
- `is_causal = False` **always** (bidirectional over the canvas).
- **Read-only** cache: it does *not* call `update()`. It reads the encoder's stored K/V for this layer and **prepends** them to the canvas's own K/V:
  ```python
  enc_K = past_key_values.layers[layer_idx].keys
  enc_V = past_key_values.layers[layer_idx].values
  K = cat([enc_K, canvas_K], dim=2)   # along sequence
  V = cat([enc_V, canvas_V], dim=2)
  ```
  So each canvas query attends to **all valid encoder KV** (cross-attention to the frozen context) **plus the entire canvas** (bidirectional self-attention). Sliding layers slice the encoder KV to the last `sliding_window` positions (see §12).

---

## 7. RoPE, Norms, Embedding Scale, Softcapping

- **RoPE** — two inverse-frequency tables, selected per layer type, computed in fp32:
  - *Sliding* (`default`): `inv_freq = 1/θ^(arange(0,256,2)/256)`, θ=10000. Full 256-dim rotation.
  - *Full* (`proportional`): over `global_head_dim=512`, θ=1000000, `partial_rotary_factor=0.25` → only the first `0.25·512/2 = 64` frequency pairs rotate; the remaining `192` pairs get `inv_freq = 0` (NoPE — identity). Output length is always full `head_dim`. `attention_scaling = 1.0`.
  - `rotate_half` convention: `[-x2, x1]` with `x1,x2` = first/second half of the head dim.
- **RMSNorm** (`DiffusionGemmaRMSNorm`, = `Gemma4RMSNorm`): compute in fp32, `x * pow(mean(x²)+eps, -0.5)`; if `with_scale`, multiply by `weight` (fp32). **Plain `* weight`, no `(1 + weight)` offset** — this matches the engine's existing `rms_norm.hpp` (`v *= weight[d]`), so no change is needed. Scaleless variants (`v_norm`, router `norm`, `self_conditioning.post_norm`) pass `weight = nullptr`. `eps = 1e-6`.
- **Embedding scale**: `embed_scale = hidden_size**0.5 = √2816 ≈ 53.07`, applied to token embeddings (and to soft-embeddings in self-conditioning). Computed/stored in BF16 → the usual rounding caveat (e.g. √2816 rounds in bf16).
- **Final logit softcapping** (in `lm_head`, fp32): `logits = 30 · tanh(lm_head(h)/30)`.

---

## 8. Self-Conditioning

Only the **decoder** has it. At denoising step *t*, the decoder is told what it predicted at step *t+1* by turning the previous step's logits into a soft embedding and folding it into the canvas input embeddings.

```python
inputs_embeds = embed_tokens(decoder_input_ids) * embed_scale     # [B, 256, H]

if self_conditioning_logits is not None:                 # not the first step
    p = softmax(self_conditioning_logits, fp32)          # [B, 256, vocab]
    soft = (p @ embed_tokens.weight) * embed_scale       # expected embedding, [B,256,H]
else:
    soft = zeros_like(inputs_embeds)                     # first step: no signal

# self_conditioning FFN (GeGLU, intermediate=2112)
normed   = pre_norm(soft)                                # RMSNorm (with scale)
sc       = down_proj(gelu_tanh(gate_proj(normed)) * up_proj(normed))
combined = inputs_embeds + sc
inputs_embeds = post_norm(combined)                      # RMSNorm, scaleless
```

The `self_conditioning_logits` fed in are the **processed** logits (post temperature/softcap) from the previous step, cast to the embedding dtype. The matmul `p @ embed_tokens.weight` reuses the tied embedding table.

---

## 9. Model Forward (one denoising forward pass)

`DiffusionGemmaForBlockDiffusion.forward` (= `DiffusionGemmaModel.forward` + LM head):

1. **(optional) Encode**: if `input_ids` is given (new uncached prompt/canvas tokens), run the **encoder** over them with causal masks and **write** the KV cache. During the inner denoising loop this is skipped — the cache is already populated.
2. **Decode**: run the **decoder** over `decoder_input_ids` (the current canvas, `[B, 256]`):
   - Build canvas input embeddings + self-conditioning (§8).
   - Decoder position ids continue **after** the encoder sequence: `arange(cache_seq_len, cache_seq_len + 256)`.
   - Per layer: bidirectional attention reading encoder KV + canvas KV (§6), then dual-FFN block (§5).
   - Final `decoder.norm` (RMSNorm).
3. **LM head**: `logits = lm_head(hidden)` then fp32 softcap `30·tanh(·/30)` → `[B, 256, vocab]`.

The decoder produces **no** KV cache (returns hidden states/logits only). The encoder KV cache is **frozen** across all denoising steps of a canvas.

---

## 10. Generation: Block-Diffusion Algorithm

`DiffusionGemmaGenerationMixin.generate` — outer autoregressive loop over canvases, inner diffusion loop over steps. (Batch is supported; the description below is per-sequence.)

```
max_new_canvases = ceil(max_new_tokens / canvas_length)        # default 256/256 = 1

for canvas_block in range(max_new_canvases):
    # 1.a ENCODE new tokens → grow frozen KV cache
    new_ids = full_prompt if prefill else last_256_generated
    encoder(new_ids, causal_masks, past_key_values)            # writes KV
    is_prefill = False

    # 1.b init denoiser state
    current_canvas = uniform_random(vocab, size=256)           # x_T  (pure noise)
    self_conditioning_logits = None
    finished_denoising = False
    diffusion_stopping.reset()

    # 1.c DENOISING LOOP — note cur_step counts DOWN: N, N-1, ..., 1
    for cur_step in reversed(range(1, max_denoising_steps + 1)):
        logits = decoder_forward(current_canvas, KV_cache,
                                 self_conditioning_logits, decoder_mask, decoder_pos)
        logits = temperature_schedule(logits, cur_step)        # §11
        probs  = softmax(logits, fp32)
        denoiser_canvas = multinomial(probs)                   # sampled tokens
        argmax_canvas   = argmax(logits)                       # the "draft" output

        accepted = sampler.accept_canvas(current_canvas, denoiser_canvas, logits)  # §11
        current_canvas = sampler.renoise_canvas(accepted)      # rejected → random
        finished_denoising |= diffusion_stopping(argmax_canvas, logits)            # §11
        self_conditioning_logits = logits                      # feed to next step

        if all(finished_denoising): break

    # 1.d append the denoised canvas (argmax) to the running sequence
    input_ids = cat([input_ids, argmax_canvas])

    # 1.e AR stopping: EOS in canvas or max length → pad finished rows
    input_ids, finished = finalize_canvas(input_ids, eos_ids, pad_id)
    if all(finished): break

    # 1.f advance positions by canvas_length for the next block
```

Key points:
- **Cache grows by 256 per block** (the finished canvas is encoded into KV before the next canvas starts). The *last* canvas is intentionally not cached.
- **Encoder vs decoder positions**: encoder positions cover the new tokens; decoder positions are `[cache_len, cache_len+256)`. After a block, `encoder_position_ids` become the previous `decoder_position_ids`.
- The reported per-sequence output token is the **argmax canvas** (most-likely token per position), not the multinomially-sampled `denoiser_canvas` (which is only used for acceptance).
- `tokens_per_forward = (#non-pad new tokens) / (#decoder forward passes)` is the diffusion efficiency metric.

---

## 11. Sampler, Temperature Schedule, Stopping

### Linear temperature schedule (`LinearTemperatureScheduleLogitsProcessor`)
At step `cur_step` of `N` (remember `cur_step` counts **down** N→1):
```
t = t_min + (t_max - t_min) * (cur_step / N)
logits = logits / t
```
So early steps (large `cur_step`) use high temperature `≈ t_max` (explore); late steps approach `t_min` (exploit). Defaults `t_min=0.4`, `t_max=0.8`, `N=48`.

### Entropy-Bound Sampler (`EntropyBoundSampler`)
- **Canvas init**: `uniform randint(0, vocab, size=[B,256])`.
- **Accept** (`accept_canvas`): per position compute entropy `H_i` of `Categorical(logits)`. Sort positions by ascending entropy; let the cumulative sum be `C_i`. Accept position *i* iff
  ```
  C_i - H_i  ≤  entropy_bound          # (sum of entropies seen so far, minus the current/max one)
  ```
  i.e. accept the most-confident prefix whose joint-MI upper bound stays under the bound. Scatter the mask back to original order; `accepted = where(mask, denoiser_canvas, current_canvas)`. Default `entropy_bound = 0.1`. (From arXiv:2505.24857.)
- **Renoise** (`renoise_canvas`): every **non-accepted** position is overwritten with a fresh uniform-random token (staying on the uniform-noise manifold the model was trained on). Accepted tokens can still change in later steps.

### Adaptive stopping (`StableAndConfidentStoppingCriteria`)
Stop a sequence's denoising when **both**:
- **Stable**: the argmax canvas is identical for the last `stability_threshold` steps (default **1**).
- **Confident**: mean per-position entropy of the processed logits `< confidence_threshold` (default **0.005**).

The outer **AR** stop uses standard EOS (`eos_token_id = [1, 106, 50]`) within the canvas and `max_length`; tokens after the first EOS in a finished canvas are replaced by `pad_token_id = 0`.

---

## 12. Attention Masks

### Encoder masks
Standard Gemma4: `create_causal_mask` for full layers and `create_sliding_window_causal_mask` for sliding layers, with the **vision block** bidirectional override when `use_bidirectional_attention="vision"` (vision spans, identified via `mm_token_type_ids` → `block_sequence_ids`, attend bidirectionally; text stays causal). The engine already implements this for Gemma4 Unified.

### Decoder masks (`create_diffusion_decoder_attention_mask`)
The mask has width `cache_len + canvas_length` and is **bidirectional**:
- **Full layers**: canvas rows attend to all *valid* (non-pad, populated) encoder KV positions **and** all 256 canvas positions. Shortcut: when not compiling and the attention mask is all-ones (no padding), return `None` (i.e., dense attend-all).
- **Sliding layers**: take the right slice of the encoder-KV mask covering the last `sliding_window` populated positions, then **pad `+canvas_length` columns of `True`** so the canvas is always fully visible to itself. (The implementation deliberately avoids the generic mask utils because of off-by-one assumptions: it wants KV length `sliding_window + query_length`, not `sliding_window − 1 + query_length`.)

ASCII (cache=8, two populated entries, canvas=4): every canvas row sees populated cache cols + all 4 canvas cols:
```
[i] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ │ ■ ■ ■ ■        ■ = attend, ⬚ = masked
```

---

## 13. Numeric Constants Reference

**Architecture** (config.json): `hidden=2816`, `layers=30`, `heads=16`, `kv_sliding=8`, `kv_full=2`, `head_dim_sliding=256`, `head_dim_full=512`, `experts=128`, `top_k=8`, `moe_inter=704`, `dense_inter=2112`, `sliding_window=1024`, `vocab=262144`, `rms_eps=1e-6`, `softcap=30.0`, `canvas_length=256`, RoPE θ sliding/full = 1e4 / 1e6, full `partial_rotary_factor=0.25`.

**Generation defaults** (`generation_config.json` + mixin defaults):
| Param | Value |
|---|---|
| `max_new_tokens` | 256 |
| `max_denoising_steps` | 48 |
| `sampler` | EntropyBound, `entropy_bound = 0.1` |
| `t_min` / `t_max` | 0.4 / 0.8 |
| `stability_threshold` | 1 |
| `confidence_threshold` | 0.005 |
| `eos_token_id` | `[1, 106, 50]` |
| `pad_token_id` | 0 |

**Tokens**: `bos=<bos>(2)`, `eos=<eos>(1)`, `pad=<pad>(0)`, `image_token_id=258880`. Chat template / tokenizer are Gemma4-style (`Gemma4Processor`, `chat_template.jinja`); the engine's existing Python tokenizer bridge applies.

---

## 14. Implementation Guide for the SYCL Engine

Suggested home: `src/diffusion_gemma/` (sibling to `gemma4_unified/`), registered as `model_type: "diffusion_gemma"`. Because the text block is Gemma4-shaped, much of `common/layers/attention_ops.hpp`, the dual-RoPE, RMSNorm, embedding, softcap, and the vision tower carry over directly.

**Reuse directly:**
- GQA attention with Q/K norm and `scale=1.0`; sliding/full split; dual RoPE (sliding default 256-dim θ=1e4, full proportional 512-dim θ=1e6 partial 0.25); full-layer K=V sharing with V un-RoPE'd.
- Embedding lookup + `√2816` scale; final RMSNorm; `30·tanh(·/30)` softcap; tied `lm_head = embedᵀ`.
- Vision tower (`gemma4_vision`) and `masked_scatter` placeholder merge — unchanged (image-only).
- safetensors loader; weight **tying** (load one layer-weight set; keep per-mode `layer_scalar`).

**Build new:**
1. **MoE block.** Router (`norm`(scaleless) → `*scale*1/√H` → `proj` → fp32 softmax → top-8 → renorm → `*per_expert_scale`) and grouped experts (`gate_up_proj[E,1408,H]` chunked into gate/up, GeGLU, `down_proj[E,H,704]`, weighted scatter-add). Run it **in parallel** with the dense MLP and sum via the two `post_feedforward_layernorm_{1,2}` paths (§5). This is the main new kernel; a grouped/segmented GEMM over the 8 selected experts per token.
2. **Encoder/decoder duality** over one weight set: encoder = causal + KV **write**; decoder = bidirectional + KV **read** + canvas-KV concat. The existing KV cache must support a *read-only* mode and concatenation of fresh canvas K/V in front-to-back order.
3. **Decoder masks**: bidirectional canvas + sliced encoder-KV visibility (§12). Mind the sliding off-by-one (`sliding_window + query_length`).
4. **Self-conditioning** FFN (§8), including the `softmax(prev_logits) @ embed` soft-embedding (reuses the embedding table).
5. **Generation driver**: the two nested loops (§10), the **entropy-bound sampler** (entropy sort + cumulative-bound accept + uniform renoise), the **linear temperature schedule** (count-down step), and **stable-&-confident** stopping (§11). Note canvas output is **argmax**, acceptance uses **multinomial**.

**Single-GPU note**: 25.2B params in BF16 ≈ 50 GB of weights (the 11 shards total ~50 GB). The current 2-GPU layer-split strategy in `gemma4_unified/model.cpp` is the natural template; with only ~3.8B active per token, compute per step is modest but **all 128 experts' weights must be resident**, so this is memory-capacity-bound on weights, not activations. Plan the split at a layer boundary as today (e.g. 15/15).

**Validation strategy**: reproduce HF `DiffusionGemmaForBlockDiffusion.generate` with fixed seed and greedy-ish settings; compare the **argmax canvas per denoising step** and the **encoder KV** for the first block before wiring the full loop. Behavior-preservation should be checked at the per-step logits level, since the sampler/stopping make end-to-end output seed-sensitive.

---

### Appendix: class map (HF → role)
| HF class | Role |
|---|---|
| `DiffusionGemmaForBlockDiffusion` | top model: `DiffusionGemmaModel` + `lm_head` + softcap + `generate` |
| `DiffusionGemmaModel` | encoder + decoder container; runs encode-then-decode |
| `DiffusionGemmaEncoderModel` | vision tower + `language_model` (causal, KV-write) — ≈ `Gemma4Model` |
| `DiffusionGemmaEncoderTextModel` | the encoder's text stack (30 dual-FFN layers) |
| `DiffusionGemmaDecoderModel` | bidirectional canvas decoder + self-conditioning (KV-read) |
| `DiffusionGemma{Encoder,Decoder}TextLayer` | dual-FFN block (§5) |
| `DiffusionGemma{Encoder,Decoder}TextAttention` | §6 |
| `DiffusionGemmaTextRouter` / `...Experts` | MoE |
| `DiffusionGemmaSelfConditioning` | §8 |
| `EntropyBoundSampler` / `LinearTemperatureScheduleLogitsProcessor` / `StableAndConfidentStoppingCriteria` | §11 |
```

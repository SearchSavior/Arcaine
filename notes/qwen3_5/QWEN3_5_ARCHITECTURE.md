# Qwen3.5 / Qwen3.5-MoE: Implementation Analysis (Dense + MoE, Vision + LM)

**Source**: HuggingFace `transformers==5.12.1` (pulled into this environment via pip)
**Reference checkpoints**: `Qwen/Qwen3.5-27B` (dense), `Qwen/Qwen3.5-35B-A3B` (MoE)
**Purpose**: Architecture reference to seed an Arcaine (SYCL + oneDNN) port. This document
describes *what the reference PyTorch implementation does* and *how the CUDA fast path is
wired*. It deliberately does **not** prescribe a SYCL design — only the math, data flow,
shapes, and dependencies another agent needs to implement it correctly.

> Citations use `transformers/models/<model>/<file>.py:<lines>` relative to the installed
> package (`/usr/local/lib/python3.11/dist-packages/transformers/...`). The `modular_*.py`
> files are the source of truth; the `modeling_*.py` files are auto-generated and are what
> actually runs. Both are cited where they diverge.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Model Family & Inheritance Map](#2-model-family--inheritance-map)
3. [Configuration System](#3-configuration-system)
4. [Hybrid Attention Schedule](#4-hybrid-attention-schedule)
5. [Tensor Shape Reference](#5-tensor-shape-reference)
6. [Dependencies: the conv1d op library and FLA](#6-dependencies-the-conv1d-op-library-and-fla)
7. [FORWARD PASS](#7-forward-pass)
   - 7.1 Top-level multimodal forward
   - 7.2 Vision encoder
   - 7.3 M-RoPE / 3D position ids
   - 7.4 Text decoder loop
   - 7.5 Gated DeltaNet (linear attention) — the core
   - 7.6 Full attention layer
   - 7.7 MLP / Sparse MoE block
8. [BACKWARD PASS](#8-backward-pass)
9. [CUDA Optimizations In Place](#9-cuda-optimizations-in-place)
10. [Weight Manifest](#10-weight-manifest)
11. [Porting Checklist](#11-porting-checklist)

---

## 1. Executive Summary

"Qwen3.5" (sometimes spoken as 3.5/3.6 — there is no separate `qwen3_6` module; the dense
and MoE text towers are `qwen3_5` and `qwen3_5_moe`) is a **hybrid linear-attention
transformer** with an optional **Qwen3-VL vision tower**. The defining feature is that the
text decoder is **not** a pure softmax-attention stack. Instead each layer is one of two
token mixers, chosen by a fixed schedule:

- **`linear_attention`** — a **Gated DeltaNet** (gated delta rule, a linear-attention /
  state-space hybrid) with a short causal **depthwise Conv1d** in front. This is the
  "conv1d op library" dependency: prefill/decode run through `causal_conv1d` +
  `flash-linear-attention` (FLA) CUDA kernels, with pure-PyTorch fallbacks.
- **`full_attention`** — standard GQA softmax attention with QK-norm, partial RoPE, and an
  output gate.

Default schedule is **3 linear : 1 full** (every 4th layer is full attention). The MoE
variant swaps every dense MLP for a 256-expert top-8 sparse block with a shared expert.

Both towers reuse the **Qwen3-Next** building blocks (`Qwen3NextGatedDeltaNet`,
`Qwen3NextAttention`, `Qwen3NextSparseMoeBlock`) and the **Qwen3-VL** multimodal scaffolding
(vision encoder, interleaved M-RoPE, `masked_scatter` token merge). Qwen3.5 is essentially
**Qwen3-Next text tower × Qwen3-VL multimodal shell**, minus DeepStack.

Key consequence for a port: there are **two completely different attention codepaths and two
different cache state types in the same model** (a rolling Conv1d state + a recurrent matrix
state for linear layers; a standard KV cache for full layers). Getting the hybrid cache and
the chunked delta-rule recurrence right is the bulk of the work.

---

## 2. Model Family & Inheritance Map

```
Qwen3-Next (text-only hybrid)                Qwen3-VL (softmax VLM)
  Qwen3NextGatedDeltaNet  ───────┐            Qwen3VLVisionModel
  Qwen3NextAttention             │            Qwen3VLModel (merge + M-RoPE)
  Qwen3NextSparseMoeBlock        │            Qwen3VLForConditionalGeneration
  Qwen3NextModel (decoder loop)  │              │
        │                        │             │
        ▼                        ▼             ▼
   ┌──────────────── Qwen3.5 (dense) ──────────────┐
   │ Qwen3_5GatedDeltaNet  (slims in_proj, see 7.5)│
   │ Qwen3_5Attention = Qwen3NextAttention         │
   │ Qwen3_5DecoderLayer (linear|full + dense MLP) │
   │ Qwen3_5TextModel    (hybrid decoder loop)     │
   │ Qwen3_5VisionModel  (Qwen3-VL minus DeepStack)│
   │ Qwen3_5Model / ...ForConditionalGeneration    │
   └───────────────────────────────────────────────┘
                         │
                         ▼
   ┌──────────────── Qwen3.5-MoE ──────────────────┐
   │ Qwen3_5MoeGatedDeltaNet = Qwen3_5GatedDeltaNet│
   │ Qwen3_5MoeSparseMoeBlock (256 exp, top-8 + sh)│
   │ Qwen3_5MoeDecoderLayer (linear|full + MoE MLP)│
   │ everything else mirrors the dense tower       │
   └───────────────────────────────────────────────┘
```

Source anchors:
- Dense modular: `transformers/models/qwen3_5/modular_qwen3_5.py` (imports at lines 38–57).
- MoE modular: `transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py` (imports 24–50).
- Next building blocks: `transformers/models/qwen3_next/modeling_qwen3_next.py`.
- VL scaffolding: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`.

The dense decoder layer hard-codes a dense `Qwen3_5MLP`
(`modular_qwen3_5.py:351-404`); the MoE layer hard-codes a `Qwen3_5MoeSparseMoeBlock` for
**every** layer (`modular_qwen3_5_moe.py:187-198`) — note `mlp_only_layers` and
`decoder_sparse_step` are deleted from the MoE config, so there is no "every-N-layers MoE"
gating like Qwen3-Next has; **all** layers are sparse.

---

## 3. Configuration System

Top-level config is a 2-way composite (no audio): `{vision_config, text_config}`
(`configuration_qwen3_5.py:143-189`). Special token ids are shared across dense/MoE:
`image_token_id=248056`, `video_token_id=248057`, `vision_start_token_id=248053`,
`vision_end_token_id=248054`.

### 3.1 Dense text config — `Qwen3_5TextConfig`
(`configuration_qwen3_5.py:27-113`)

| Field | Default | Notes |
|---|---|---|
| `vocab_size` | 248320 | larger than Qwen3-Next's 151936 |
| `hidden_size` | 4096 | |
| `intermediate_size` | 12288 | dense MLP |
| `num_hidden_layers` | 32 | |
| `num_attention_heads` | 16 | full-attn Q heads |
| `num_key_value_heads` | 4 | full-attn KV heads (GQA, 4× group) |
| `head_dim` | 256 | **explicit**, not hidden/heads |
| `linear_conv_kernel_dim` | 4 | depthwise causal conv kernel |
| `linear_key_head_dim` | 128 | DeltaNet K/Q head dim |
| `linear_value_head_dim` | 128 | DeltaNet V head dim |
| `linear_num_key_heads` | 16 | DeltaNet K/Q heads |
| `linear_num_value_heads` | 32 | DeltaNet V heads (2× key heads) |
| `partial_rotary_factor` | 0.25 | set in `__post_init__`, BC default |
| `rms_norm_eps` | 1e-6 | |
| `layer_types` | auto | see §4 |
| `max_position_embeddings` | 32768 | |

`__post_init__` (`configuration_qwen3_5.py:104-113`) builds `layer_types` from
`full_attention_interval` (default **4**):
```python
layer_types = ["linear_attention" if (i+1) % 4 else "full_attention"
               for i in range(num_hidden_layers)]
```
The MoE config does the identical thing (`configuration_qwen3_5_moe.py:112-121`).

In the **modular** form the dense text config subclasses `Qwen3NextConfig` and *deletes* the
MoE attributes via `AttributeError()` sentinels (`modular_qwen3_5.py:114-126`):
`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`,
`shared_expert_intermediate_size`, `decoder_sparse_step`, `norm_topk_prob`,
`mlp_only_layers`, `output_router_logits`, `router_aux_loss_coef`. The dense tower has **no
router and no aux loss**.

### 3.2 MoE text config — `Qwen3_5MoeTextConfig`
(`configuration_qwen3_5_moe.py:27-121`)

Diffs vs dense:

| Field | Default |
|---|---|
| `hidden_size` | 2048 (smaller) |
| `num_hidden_layers` | 40 |
| `num_key_value_heads` | 2 |
| `num_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `output_router_logits` | False |
| `router_aux_loss_coef` | 0.001 |

The linear-attention block dims (`linear_*`), `head_dim=256`, and `vocab_size=248320` are
identical to the dense tower. `intermediate_size`, `decoder_sparse_step`, `norm_topk_prob`,
`mlp_only_layers` are deleted in the modular (`modular_qwen3_5_moe.py:109-116`) — every
layer is a sparse block and the router normalizes unconditionally (see §7.7).

### 3.3 Vision config — `Qwen3_5VisionConfig`
(`configuration_qwen3_5.py:116-140`, modular `modular_qwen3_5.py:129-139`)

Subclass of `Qwen3VLVisionConfig` that **deletes `deepstack_visual_indexes`** — Qwen3.5 drops
DeepStack feature injection entirely.

| Field | Default | Notes |
|---|---|---|
| `depth` | 27 | ViT blocks |
| `hidden_size` | 1152 | vision width |
| `intermediate_size` | 4304 | vision MLP |
| `num_heads` | 16 | head_dim = 1152/16 = 72 |
| `in_channels` | 3 | |
| `patch_size` | 16 | |
| `temporal_patch_size` | 2 | video temporal patch |
| `spatial_merge_size` | 2 | 2×2 patch merge → ×4 width at merger |
| `out_hidden_size` | 3584 | projected to LM space |
| `num_position_embeddings` | 2304 | learned pos table; `num_grid_per_side = sqrt(2304)=48` |
| `hidden_act` | `gelu_pytorch_tanh` | |

> ⚠️ Dimension mismatch to watch: vision `out_hidden_size=3584` but dense LM `hidden_size=4096`
> (MoE LM `hidden_size=2048`). The default sub-configs are not internally consistent as a
> *runnable* pair; real checkpoints ship matched configs. Always read `out_hidden_size` and
> the text `hidden_size` from the actual checkpoint `config.json`, do not assume.

---

## 4. Hybrid Attention Schedule

`layer_types[i]` ∈ {`"linear_attention"`, `"full_attention"`}. With the default
`full_attention_interval=4` and 32 dense layers:

```
idx :  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | ...
type:  L  L  L  F | L  L  L  F | L  L  L  F | ...   (F at idx 3,7,11,15,19,23,27,31)
```
3 linear mixers then 1 full-attention mixer, repeating. MoE (40 layers) → full at
3,7,…,39. Per layer the decoder picks the mixer in `__init__`
(dense `modular_qwen3_5.py:351-362`, MoE `modular_qwen3_5_moe.py:187-198`):

```python
if layer_type == "linear_attention":
    self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
elif layer_type == "full_attention":
    self.self_attn = Qwen3_5Attention(config, layer_idx)
self.mlp = <dense Qwen3_5MLP | Qwen3_5MoeSparseMoeBlock>
self.input_layernorm        = Qwen3_5RMSNorm(...)
self.post_attention_layernorm = Qwen3_5RMSNorm(...)
```

The decoder loop builds **two masks** and dispatches per layer
(`Qwen3_5TextModel.forward`, `modular_qwen3_5.py:531-554`):
```python
causal_mask      = create_causal_mask(...)                      # for full_attention
linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
...
layer_mask = linear_attn_mask if layer_types[i]=="linear_attention" else causal_mask
```
`_update_linear_attn_mask` (inherited `modeling_qwen3_next.py:990-1002`) returns `None`
(i.e. "attend to everything, no zeroing") when decoding with cached state or when the 2D mask
is all-ones; otherwise it forwards the 2D padding mask so DeltaNet can zero padded tokens.
**Left padding** is assumed for linear-attention batches.

Both mixer types share **one** rotary embedding (`Qwen3_5TextRotaryEmbedding`,
`modular_qwen3_5.py:171-192`) computed once per forward; only full-attention layers consume
`position_embeddings`. RoPE here is **partial** (`partial_rotary_factor=0.25` → only the
first `0.25*256 = 64` dims of each head are rotated; see `apply_rotary_pos_emb`
`modeling_qwen3_next.py:180-215`, which splits `q_rot|q_pass`).

RoPE is also **M-RoPE** (multimodal 3D: temporal/height/width) with interleaved sections
`[11,11,10]` (`Qwen3_5TextRotaryEmbedding.__init__`, `modular_qwen3_5.py:174`). The text
position-id path expands to 4 rows `[text, t, h, w]` and splits text vs. multimodal
(`Qwen3_5TextModel.forward`, `modular_qwen3_5.py:517-529`). See §7.3.

---

## 5. Tensor Shape Reference

Notation: `B`=batch, `S`=sequence length, `H`=hidden. Dense defaults shown; sub in MoE
numbers where noted. Derived linear-attention dims:
- `key_dim   = linear_key_head_dim   * linear_num_key_heads   = 128 * 16 = 2048`
- `value_dim = linear_value_head_dim * linear_num_value_heads = 128 * 32 = 4096`
- `conv_dim  = key_dim*2 + value_dim   = 2048*2 + 4096 = 8192`
- full-attn QKV: Q proj emits `num_heads * head_dim * 2 = 16*256*2 = 8192` (Q **and** an
  output gate, split in half), K/V each `num_kv_heads * head_dim = 4*256 = 1024`.

| Tensor | Shape | Where |
|---|---|---|
| `input_ids` | `(B, S)` | |
| `inputs_embeds` | `(B, S, 4096)` | embed_tokens |
| **Linear attn** | | `Qwen3_5GatedDeltaNet.forward` |
| `mixed_qkv` (pre-conv) | `(B, conv_dim, S)` = `(B, 8192, S)` | transposed for Conv1d |
| `conv1d` weight | `(conv_dim, 1, kernel=4)` depthwise (`groups=conv_dim`) | |
| `query`,`key` (post split/reshape) | `(B, S, 16, 128)` → repeat→ `(B,S,32,128)` | `num_v/num_k=2` |
| `value` | `(B, S, 32, 128)` | |
| `g` (decay), `beta` | `(B, S, 32)` | per value head |
| `recurrent_state` (cache) | `(B, num_v_heads=32, head_k=128, head_v=128)` | matrix state |
| `conv_state` (cache) | `(B, conv_dim=8192, kernel-1=3)` | rolling conv window |
| DeltaNet output | `(B, S, 4096)` | `out_proj: value_dim→H` |
| **Full attn** | | `Qwen3NextAttention.forward` |
| `query_states`,`gate` | each `(B, 16, S, 256)` | Q proj chunked in 2 |
| `key/value_states` | `(B, 4, S, 256)` | GQA |
| KV cache (per full layer) | `(B, 4, S_kv, 256)` ×2 | standard |
| **MoE** | | |
| `router_logits` | `(B*S, 256)` | |
| `routing_weights` / `selected_experts` | `(B*S, 8)` | top-8 |
| `experts.gate_up_proj` | `(256, 2*512, 2048)` | 3D packed |
| `experts.down_proj` | `(256, 2048, 512)` | |
| **Vision** | | |
| `pixel_values` | `(num_patches, in_ch*t_patch*p*p)` = `(N, 3*2*16*16=1536)` | flattened patches |
| patch_embed (Conv3d) out | `(N, 1152)` | |
| post-merger `image_embeds` | `(N/4, 3584)` | spatial_merge 2×2 → /4 |
| `image_grid_thw` | `(num_images, 3)` | (t,h,w) in patch units |
| `cu_seqlens` | `(num_windows+1,)` int32 | packed attn boundaries |

---

## 6. Dependencies: the conv1d op library and FLA

The linear-attention fast path depends on **two external CUDA libraries**, both optional with
PyTorch fallbacks (`modeling_qwen3_next.py:47-62`):

1. **`causal_conv1d`** (Dao-AILab/causal-conv1d) — provides
   - `causal_conv1d_fn(x, weight, bias, activation, seq_idx)` — fused depthwise causal
     conv + activation over a full sequence (prefill / chunked decode).
   - `causal_conv1d_update(x, conv_state, weight, bias, activation)` — single-step decode;
     updates the rolling `conv_state` in place.
   This is the "conv1d op library" referenced in the task. It replaces `nn.Conv1d` with a
   kernel that (a) fuses the SiLU activation, (b) avoids materializing the `kernel-1`
   left-pad, and (c) does the in-place ring-buffer state update for decode.

2. **`flash-linear-attention` (FLA, `fla` package)** — provides
   - `chunk_gated_delta_rule(...)` — chunked parallel scan of the gated delta rule
     (prefill / multi-token).
   - `fused_recurrent_gated_delta_rule(...)` — sequential recurrence for single-token decode.
   - `FusedRMSNormGated` — fused RMSNorm + SiLU gate used as the DeltaNet output norm.

Resolution logic (`modeling_qwen3_next.py:344-346, 553-563`):
```python
is_fast_path_available = all((causal_conv1d_fn, causal_conv1d_update,
                              chunk_gated_delta_rule, fused_recurrent_gated_delta_rule))
self.causal_conv1d_fn       = causal_conv1d_fn                       # or None → nn.Conv1d path
self.causal_conv1d_update   = causal_conv1d_update or torch_causal_conv1d_update
self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
```
**For an Arcaine port the pure-PyTorch fallbacks are the executable spec**:
`torch_causal_conv1d_update` (`:349-364`), `torch_chunk_gated_delta_rule` (`:373-451`),
`torch_recurrent_gated_delta_rule` (`:454-495`). They are numerically equivalent to the CUDA
kernels and fully readable — port from these, validate against the kernels.

Full-attention layers additionally route through `ALL_ATTENTION_FUNCTIONS` (flash-attn / SDPA
/ eager) via `config._attn_implementation` (`modeling_qwen3_next.py:310-323`).

---

## 7. FORWARD PASS

### 7.1 Top-level multimodal forward — `Qwen3_5Model.forward`
(`modular_qwen3_5.py:591-664`)

```
input_ids ──embed_tokens──► inputs_embeds (B,S,H)
   │
   ├─ if pixel_values:       image_embeds = visual(pixel_values, image_grid_thw).pooler_output
   │     image_mask = (input_ids == image_token_id)
   │     inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)   # :628
   ├─ if pixel_values_videos: same path via get_video_features                    # :630-639
   │
   ├─ position_ids = compute_3d_position_ids(...)   # M-RoPE, see 7.3             # :641-650
   └─ language_model(inputs_embeds, position_ids, attention_mask, past_key_values)
```
Multimodal tokens are spliced into the text stream by **`masked_scatter`** on the embedding
tensor (inherited from `Qwen3VLModel`, `modeling_qwen3_vl.py:628`). The vision encoder runs
once; its merged tokens replace `image_token_id`/`video_token_id` placeholders. There is **no
cross-attention** — vision is fully tokenized into the LM sequence. (Unlike Qwen3-VL, Qwen3.5
drops DeepStack, so there is no per-layer visual feature injection.)

`Qwen3_5ForConditionalGeneration` (`modular_qwen3_5.py:680-685`) is the generation entry; it
inherits `Qwen3VLForConditionalGeneration` and only re-exposes `get_image_features` /
`get_video_features`. The LM head + loss live there.

### 7.2 Vision encoder — `Qwen3_5VisionModel.forward`
(`modular_qwen3_5.py:429-484`)

`Qwen3_5VisionModel` is `Qwen3VLVisionModel` with DeepStack deleted in `__init__`
(`modular_qwen3_5.py:433-436`: `del self.deepstack_visual_indexes`,
`del self.deepstack_merger_list`). Pipeline:

```
pixel_values (N, 1536)                                  # N flattened (t·h·w) patches
   │  Qwen3VLVisionPatchEmbed: Conv3d(3→1152, k=stride=[2,16,16])   modeling_qwen3_vl.py:80-97
   ▼
hidden_states (N, 1152)
   + pos_embeds   # bilinear-interpolated learned table, get_vision_bilinear_indices_and_weights
   │              # pos_embed: Embedding(2304, 1152), 48×48 grid, bilinear to (h,w)  :692-703
   ▼
rotary_pos_emb = rotary_pos_emb(get_vision_position_ids(grid_thw, merge=2))  # (N, head_dim/2)
emb = cat([rope, rope]); position_embeddings = (emb.cos(), emb.sin())        # :463-469
   │
   ├─ for blk in blocks (depth=27):   # Qwen3VLVisionBlock                    modeling_qwen3_vl.py:271-299
   │     h = h + attn(norm1(h), cu_seqlens, position_embeddings)             # packed varlen attn
   │     h = h + mlp(norm2(h))                                                # GELU-tanh MLP
   ▼
merged = merger(h)   # Qwen3VLVisionPatchMerger: norm → fc1 → GELU → fc2     modeling_qwen3_vl.py:114-127
   ▼
return BaseModelOutputWithPooling(last_hidden_state=h, pooler_output=merged) # :481-484
```

Vision attention (`Qwen3VLVisionAttention`, `modeling_qwen3_vl.py:188-268`) is **non-causal**
(`is_causal=False`) and **packed variable-length**: under flash-attn it passes `cu_seqlens`
directly (one segment per image/frame, `:225-242`); otherwise it `torch.split`s by segment and
runs SDPA/eager per segment (`:243-264`). `cu_seqlens` come from
`get_vision_cu_seqlens` (`vision_utils.py:35-50`): cumulative `h*w` repeated `t` times.

Patch merger (`modeling_qwen3_vl.py:114-127`) concatenates `spatial_merge_size² = 4` adjacent
patches → width `1152*4 = 4608`, LayerNorm, `fc1(4608→4608)`, GELU, `fc2(4608→3584)`. So the
LM receives `N/4` tokens at `out_hidden_size=3584`.

Vision RoPE position ids: `get_vision_position_ids` (`vision_utils.py:53-83`) produces
`(total_tokens, 2)` (row,col), reshaped through the `merge_size` blocking so the merge groups
are contiguous. The learned absolute pos-embed is bilinearly interpolated from the fixed
48×48 grid to the actual `(h,w)` via `get_vision_bilinear_indices_and_weights`
(`vision_utils.py:147-217`), returning 4 corner indices + weights `(4, total_thw)`.

### 7.3 M-RoPE / 3D position ids
(`Qwen3VLModel.get_rope_index`, `modeling_qwen3_vl.py:933-1024`;
`compute_3d_position_ids`, `:1111-1158`)

Text tokens get identical `(t,h,w)` indices (all three equal the running text position);
image/video tokens get a 3D grid where `t` advances per frame and `h,w` index the merged patch
grid (`get_vision_position_ids` member, `:875-931`). `mm_token_type_ids` (0=text, 1=image,
2=video) drives the grouping (`itertools.groupby`, `:993-1016`). Videos use timestamp tokens
between frames, so `video_grid_thw` is repeat-expanded per frame with `t` set to 1
(`:968-970`). The rotary module then **interleaves** the three sections via
`apply_interleaved_mrope` (`modeling_qwen3_vl.py:375-390`): layout goes from chunked
`[TTT…HHH…WWW]` to interleaved `[THWTHW…TT]` using `mrope_section` (`[11,11,10]` for 3.5).

`rope_deltas` is cached on the model so incremental decode reuses the offset
(`:1131-1154`).

### 7.4 Text decoder loop — `Qwen3_5TextModel.forward`
(`modular_qwen3_5.py:491-561`)

```python
inputs_embeds = embed_tokens(input_ids)                          # if not provided
past_key_values = DynamicCache(config) if use_cache else None    # hybrid cache, see §8
# position_ids → 4 rows [text,t,h,w]; split text_position_ids (row0) from M-RoPE rows 1:
causal_mask      = create_causal_mask(..., position_ids=text_position_ids)
linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
position_embeddings = self.rotary_emb(hidden_states, position_ids)   # M-RoPE cos/sin
for i, layer in enumerate(layers):
    layer_mask = linear_attn_mask if layer_types[i]=="linear_attention" else causal_mask
    hidden_states = layer(hidden_states, position_embeddings, attention_mask=layer_mask,
                          position_ids=text_position_ids, past_key_values=past_key_values, ...)
hidden_states = self.norm(hidden_states)     # final Qwen3_5RMSNorm
```

Each `Qwen3_5DecoderLayer.forward` (`modular_qwen3_5.py:364-404`) is a standard
pre-norm residual pair:
```
residual = h;  h = input_layernorm(h)
h = linear_attn(h, ...) | self_attn(h, ...)[0]    # token mixer
h = residual + h
residual = h;  h = post_attention_layernorm(h)
h = mlp(h)                                          # dense MLP or MoE block
h = residual + h
```

**RMSNorm note** (`Qwen3NextRMSNorm`, `modeling_qwen3_next.py:152-169`): weight is stored
**zero-centered** — `output * (1.0 + weight)` — and initialized to zeros
(`_init_weights`, `modular_qwen3_5.py:415-426`). Norm is computed in fp32 then cast back. A
port must add 1.0 to the loaded weight (or fold it in).

### 7.5 Gated DeltaNet (linear attention) — the core
`Qwen3_5GatedDeltaNet` (`modular_qwen3_5.py:195-334`), built on
`Qwen3NextGatedDeltaNet` (`modeling_qwen3_next.py:498-716`).

**Qwen3.5's specialization** (`modular_qwen3_5.py:195-210`): it *replaces* Qwen3-Next's two
fused input projections (`in_proj_qkvz`, `in_proj_ba` + the `fix_query_key_value_ordering`
de-interleave) with **four separate, already-ordered** projections:
```python
self.in_proj_qkv = nn.Linear(H, key_dim*2 + value_dim, bias=False)   # Q|K|V fused, contiguous
self.in_proj_z   = nn.Linear(H, value_dim, bias=False)               # output-gate input
self.in_proj_b   = nn.Linear(H, num_v_heads, bias=False)             # β (delta gate)
self.in_proj_a   = nn.Linear(H, num_v_heads, bias=False)             # α (decay) input
# fix_query_key_value_ordering raises — not needed (weights pre-ordered)   :209-210
```
This is a **weight-layout simplification only**; the math is identical to Qwen3-Next. A port
should target the Qwen3.5 layout (4 clean projections), not the Next interleaved one.

Forward (`modular_qwen3_5.py:212-334`), three regimes keyed on cache state + `seq_len`:

**(a) Common front-end** (all regimes):
```python
hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)  # zero pads
mixed_qkv = in_proj_qkv(h).transpose(1,2)        # (B, conv_dim=8192, S)
z = in_proj_z(h).reshape(B, S, -1, head_v_dim=128)
b = in_proj_b(h);  a = in_proj_a(h)              # (B, S, num_v_heads=32)
```

**(b) Causal depthwise Conv1d** (the conv1d op):
- *Single-token cached decode* (`use_precomputed_states and seq_len==1`, `:244-252`):
  `causal_conv1d_update(mixed_qkv, conv_state, conv1d.weight.squeeze(1), conv1d.bias, act)`
  — reads + writes the rolling `conv_state` ring buffer in place.
- *Prefill / chunked decode* (`:253-274`):
  - if cached, prepend `conv_state` so the conv sees correct left context (`:259`);
  - write next state: `conv_state = pad(mixed_qkv, (kernel-1 - len, 0))` (`:261-262`);
  - run `causal_conv1d_fn(x, weight, bias, activation, seq_idx)` (`:263-270`), or fallback
    `F.silu(conv1d(mixed_qkv)[..., :S])` (`:272`);
  - if cached, trim back to last `seq_len` (`:273-274`).
- The Conv1d is **depthwise** (`groups=conv_dim`, kernel=4, left-pad=kernel-1), applied to the
  concatenated Q|K|V channels **before** splitting into heads.

**(c) Split + gating params** (`:276-296`):
```python
mixed_qkv = mixed_qkv.transpose(1,2)
query, key, value = split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
query = query.reshape(B,S,-1,head_k=128); key likewise; value→(B,S,-1,head_v=128)
beta = b.sigmoid()                                        # (B,S,32)
g = -A_log.float().exp() * softplus(a.float() + dt_bias)  # log-decay (B,S,32), fp32
if num_v_heads // num_k_heads > 1:                        # 32/16 = 2
    query = query.repeat_interleave(2, dim=2); key likewise   # GQA-style head expand
```
`A_log` and `dt_bias` are per-(value-)head learnable params (init `A_log~log(U(0,16))`,
`dt_bias=ones`; `_init_weights` `modular_qwen3_5.py:416-420`). `g` is the **log** decay; the
kernels exponentiate internally.

**(d) Delta-rule scan**:
- *decode, seq_len==1* → `recurrent_gated_delta_rule(q,k,v, g, beta, initial_state=recurrent_state,
  output_final_state=True, use_qk_l2norm_in_kernel=True)` (`:298-308`).
- *prefill / chunked* → `chunk_gated_delta_rule(q,k,v, g, beta,
  initial_state=recurrent_state_or_None, output_final_state=cache?, use_qk_l2norm_in_kernel=True,
  cu_seqlens=cu_seq_lens_q)` (`:309-321`).

The recurrence (from the readable fallback `torch_recurrent_gated_delta_rule`,
`modeling_qwen3_next.py:454-495`), per timestep `t`, per head:
```
S_t = S_{t-1} * exp(g_t)                       # gated decay of the (head_k × head_v) state
kv  = (S_t * k_t).sum over key-dim            # read
Δ   = (v_t - kv) * β_t                          # delta correction
S_t = S_t + outer(k_t, Δ)                       # write
out_t = (S_t * q_t).sum over key-dim
```
with `q,k` L2-normalized first (`use_qk_l2norm_in_kernel=True` → `l2norm`,
`modeling_qwen3_next.py:367-370`) and `q` scaled by `1/sqrt(head_k_dim)`. The chunked variant
(`:373-451`) reformulates this as: intra-chunk lower-triangular correction solved by a
`chunk_size=64` forward substitution (`:418-421`), plus inter-chunk state carry — algebraically
equal, parallelized over chunks. **This chunk/recurrence equivalence is the single most
important thing to get right and to unit-test.**

**(e) State write + output norm** (`:323-333`):
```python
if cache: cache.update_recurrent_state(last_recurrent_state, layer_idx)
core_attn_out = norm(core_attn_out.reshape(-1, head_v), z.reshape(-1, head_v))  # RMSNormGated
output = out_proj(core_attn_out.reshape(B, S, -1))     # value_dim → H
```
`Qwen3NextRMSNormGated` (`modeling_qwen3_next.py:67-82`) is RMSNorm **then** multiply by
`SiLU(z)` — the output gate `z` comes from `in_proj_z`, *not* from the attention. On CUDA this
is FLA's `FusedRMSNormGated`.

### 7.6 Full attention layer — `Qwen3NextAttention.forward`
(`modeling_qwen3_next.py:255-329`)

GQA softmax attention with three Qwen-isms:
1. **Q projection is double-width and split into Q + output-gate** (`:267-298`):
   `q_proj: H → num_heads*head_dim*2`, chunked into `query_states` and `gate`; the final
   `attn_output = attn_output * sigmoid(gate)` (`:326`) before `o_proj`.
2. **QK-norm on the head dim** (`:279-301`): `Qwen3NextRMSNorm(head_dim)` applied to Q and K
   per head before RoPE.
3. **Partial RoPE** (`partial_rotary_factor=0.25`): only first 64 of 256 head dims rotated
   (`apply_rotary_pos_emb`, `:180-215`).
KV cache update is the standard `past_key_values.update(k, v, layer_idx)` (`:307-308`). Scaling
`head_dim**-0.5`. Attention backend chosen via `ALL_ATTENTION_FUNCTIONS`.

> `Qwen3_5Attention` is a bare subclass (`modular_qwen3_5.py:337-338`) — identical to
> Qwen3-Next.

### 7.7 MLP / Sparse MoE block

**Dense MLP** (`Qwen3NextMLP`, `modeling_qwen3_next.py:719-732`): standard SwiGLU
`down(silu(gate(x)) * up(x))`, `intermediate_size=12288`.

**MoE block** `Qwen3_5MoeSparseMoeBlock` (generated `modeling_qwen3_5_moe.py:795-814`):
```python
shared = shared_expert(x)                                   # dense SwiGLU, inter=512
logits, weights, idx = gate(x)                              # router, top-8
expert_out = experts(x, idx, weights)                       # routed experts
shared = sigmoid(shared_expert_gate(x)) * shared            # gated shared expert
out = expert_out + shared
```
- **Router** `Qwen3_5MoeTopKRouter` (`modeling_qwen3_5_moe.py:776-792`):
  `softmax(F.linear(x, weight))` over 256 experts in fp32 → `topk(8)` →
  **always** renormalize `top_value /= top_value.sum()` (no `norm_topk_prob` flag, unlike
  Qwen3-Next's router which gates on the config). Returns `(logits, scores, indices)`.
- **Experts** `Qwen3_5MoeExperts` = `Qwen3NextExperts` (`modeling_qwen3_next.py:735-772`):
  weights are **3D packed** tensors `gate_up_proj (256, 1024, 2048)` and
  `down_proj (256, 2048, 512)`. Forward builds a one-hot `expert_mask`, finds hit experts, and
  loops **only over experts that received tokens** (`:760-770`), gathering tokens, running
  `silu(gate)*up → down`, scaling by routing weight, and `index_add_`-scattering back. This is
  the reference (gather/scatter) path; the `@use_experts_implementation` decorator
  (`:735`) lets a fused/grouped-GEMM kernel replace it.
- **Shared expert** is a normal dense SwiGLU (`inter=512`) gated by a scalar
  `sigmoid(shared_expert_gate(x))` and **added** to the routed output.

The dense decoder layer's MLP returns a tensor directly; the Next-style layer also handles a
tuple return (`modeling_qwen3_next.py:878-881`) for router-logits capture. Qwen3.5-MoE
captures router logits via `_can_record_outputs` for the optional load-balancing aux loss
(`load_balancing_loss_func`, `modeling_qwen3_next.py:1005-1084`).

---

## 8. BACKWARD PASS

There is **no hand-written backward** anywhere in these models — training relies on autograd
plus the custom-op backward formulas registered by the kernel libraries. What a port must
reproduce is (a) which ops are differentiable, (b) where fp32 is forced, (c) the custom-op
gradients, and (d) what state is *not* in the autograd graph.

### 8.1 What carries gradients

- **All `nn.Linear` projections** (text, vision, experts incl. 3D-packed parameters), the
  **Conv1d weight/bias**, embeddings, the `lm_head`, all `RMSNorm`/`LayerNorm` weights, and
  the DeltaNet scalars **`A_log`** and **`dt_bias`** — standard autograd.
- The decay `g = -exp(A_log) * softplus(a + dt_bias)` (`modular_qwen3_5.py:293`) is computed
  in **fp32** specifically so the gradient to `A_log` doesn't underflow/`-inf` in fp16 — the
  inline comment flags this. A port's backward must keep this term in fp32.

### 8.2 Custom-op backward (the kernels)

- **`causal_conv1d_fn`** has a fused CUDA backward (grad wrt `x`, `weight`, `bias`). The
  PyTorch fallback `F.silu(conv1d(...))` is plain autograd and is the reference for gradients.
  Note the **decode** path uses `causal_conv1d_update`, which mutates `conv_state` **in place**;
  that in-place state update is a cache side effect, **not** part of the training graph
  (training runs the full-sequence `causal_conv1d_fn`, not the update kernel).
- **`chunk_gated_delta_rule`** (FLA) carries the backward for the gated delta-rule scan
  (grads wrt `q,k,v,g,beta` and the initial state). During training `output_final_state=False`
  unless cache is requested, and `initial_state=None` (`:309-321` passes
  `recurrent_state if use_precomputed_states else None`) — i.e. a *training* prefill starts
  from a zero state and does **not** thread a state gradient across calls. The chunked
  algorithm’s intra-chunk forward-substitution loop (`torch_chunk_gated_delta_rule:418-444`)
  is differentiable as written; the CUDA kernel implements the same Jacobian analytically.
- **`fused_recurrent_gated_delta_rule`** is the **decode** path (`seq_len==1`,
  `use_precomputed_states`) — used only under `@torch.no_grad()` generation, so its backward
  is effectively never exercised in training.
- **`FusedRMSNormGated`** backward covers the RMSNorm + SiLU-gate fusion; the fallback
  `Qwen3NextRMSNormGated` (`:67-82`) computes the same in fp32 and is the gradient reference.
- **Full-attention** backward is whatever the selected backend provides (flash-attn’s fused
  backward, or SDPA/eager autograd). The Q/gate split and `sigmoid(gate)` multiply
  (`:295-326`) are ordinary autograd.

### 8.3 MoE backward

The reference experts loop (`modeling_qwen3_next.py:748-772`) is autograd-friendly: the
`one_hot`/`expert_hit` selection runs under `torch.no_grad()` (`:755-758`) — only an index
computation, no gradient — while the per-expert `linear → silu*up → linear`, the
`* top_k_weights` scaling, and the `index_add_` scatter are all differentiable, so gradients
flow to the routed expert weights, the **router** (through `routing_weights`), the shared
expert, and `shared_expert_gate`. The **top-k selection itself is non-differentiable**
(straight-through via the selected probabilities only). The optional
**load-balancing aux loss** (`:1005-1084`, scaled by `router_aux_loss_coef=0.001`) adds a
gradient path to the router logits when `output_router_logits=True`.

A grouped-GEMM kernel replacing the loop (via `@use_experts_implementation`) must supply its
own backward producing identical grads to this gather/scatter reference.

### 8.4 Gradient checkpointing & state

Every decoder layer is a `GradientCheckpointingLayer` (`modular_qwen3_5.py:351`,
`modular_qwen3_5_moe.py:189`) and the vision blocks/merger likewise, so activations are
recomputed in backward by default when checkpointing is on. The **caches** (`conv_states`,
`recurrent_states`, KV) are inference-only state objects (`DynamicCache`,
`cache_utils.py:862-959`) and are **outside** the autograd graph — training does not populate
or backprop through them. `_is_stateful=True` / `_skip_keys_device_placement=["past_key_values"]`
mark them as non-parameter state.

---

## 9. CUDA Optimizations In Place

These are the GPU-specific fast paths an Arcaine implementation will be measured against
(architecture only — no SYCL prescription):

1. **Fused causal depthwise Conv1d** (`causal_conv1d_fn` / `_update`): fuses left-pad + grouped
   conv + SiLU; the decode kernel does an in-place ring-buffer `conv_state` update of shape
   `(B, 8192, 3)`, avoiding re-convolving history. Falls back to `nn.Conv1d` + slice.
2. **Chunked gated delta rule** (`chunk_gated_delta_rule`): block-parallel scan (`chunk_size=64`)
   that turns an inherently sequential recurrence into matmul-heavy chunk-local work + a small
   inter-chunk carry — this is what makes linear-attention prefill GPU-efficient. The
   `cu_seqlens` arg lets it handle **packed variable-length** sequences without padding.
3. **Fused recurrent gated delta rule** (`fused_recurrent_gated_delta_rule`): single-kernel
   per-step state update for decode (`seq_len==1`), reading/writing the `(B,32,128,128)`
   recurrent matrix state.
4. **`FusedRMSNormGated`**: one kernel for RMSNorm + SiLU gate (DeltaNet output).
5. **Flash-Attention varlen for vision**: packed `cu_seqlens` attention over all
   images/frames in one call (`modeling_qwen3_vl.py:225-242`); non-flash backends fall back to
   per-segment SDPA.
6. **Flash-Attention / SDPA for full-attention text layers** via `ALL_ATTENTION_FUNCTIONS`
   dispatch, with QK-norm and partial RoPE applied before the kernel.
7. **3D-packed expert weights + hit-only expert loop**: experts stored as
   `(num_experts, …)` tensors so a grouped/fused GEMM can replace the Python loop
   (`@use_experts_implementation`, `modeling_qwen3_next.py:735`); the reference skips experts
   that received zero tokens.
8. **Hub kernel hooks** on the VL text path: `@use_kernel_forward_from_hub("RMSNorm")`,
   `@use_kernel_func_from_hub("rotary_pos_emb")`, `@use_kernelized_func` (`modeling_qwen3_vl.py:
   393, 414, 440`) allow swapping in optimized RMSNorm/RoPE kernels at load time.
9. **fp32 islands**: RMSNorm, the decay `g`, router softmax, and RoPE freq computation are
   forced to fp32 (`maybe_autocast(enabled=False)`, `.float()` casts) regardless of model
   dtype — correctness-critical, not optional.
10. **Static cache addresses** for compile: conv/recurrent states call
    `torch._dynamo.mark_static_address` (`cache_utils.py:933, 940`) and the model sets
    `_can_compile_fullgraph` on the VL base — the hybrid cache is `torch.compile`-friendly.

---

## 10. Weight Manifest

Per-layer names (dense; `linear_attention` layers have `linear_attn.*`, `full_attention`
layers have `self_attn.*`):

```
model.embed_tokens.weight                         (248320, 4096)
# --- linear_attention layer i ---
model.layers.{i}.linear_attn.in_proj_qkv.weight   (key_dim*2+value_dim=8192, 4096)
model.layers.{i}.linear_attn.in_proj_z.weight     (value_dim=4096, 4096)
model.layers.{i}.linear_attn.in_proj_b.weight     (num_v_heads=32, 4096)
model.layers.{i}.linear_attn.in_proj_a.weight     (32, 4096)
model.layers.{i}.linear_attn.conv1d.weight        (conv_dim=8192, 1, 4)     # depthwise, no bias by default
model.layers.{i}.linear_attn.A_log                (32,)
model.layers.{i}.linear_attn.dt_bias              (32,)
model.layers.{i}.linear_attn.norm.weight          (head_v_dim=128,)         # RMSNormGated
model.layers.{i}.linear_attn.out_proj.weight      (4096, value_dim=4096)
# --- full_attention layer j ---
model.layers.{j}.self_attn.q_proj.weight          (num_heads*head_dim*2=8192, 4096)
model.layers.{j}.self_attn.k_proj.weight          (num_kv*head_dim=1024, 4096)
model.layers.{j}.self_attn.v_proj.weight          (1024, 4096)
model.layers.{j}.self_attn.o_proj.weight          (4096, 4096)
model.layers.{j}.self_attn.q_norm.weight          (head_dim=256,)
model.layers.{j}.self_attn.k_norm.weight          (256,)
# --- MLP (dense) ---
model.layers.{i}.mlp.{gate,up}_proj.weight        (12288, 4096)
model.layers.{i}.mlp.down_proj.weight             (4096, 12288)
# --- MLP (MoE, all layers) ---
model.layers.{i}.mlp.gate.weight                  (num_experts=256, hidden=2048)
model.layers.{i}.mlp.experts.gate_up_proj         (256, 2*512, 2048)
model.layers.{i}.mlp.experts.down_proj            (256, 2048, 512)
model.layers.{i}.mlp.shared_expert.{gate,up}_proj.weight (512, 2048)
model.layers.{i}.mlp.shared_expert.down_proj.weight      (2048, 512)
model.layers.{i}.mlp.shared_expert_gate.weight    (1, 2048)
# --- norms / head ---
model.layers.{i}.input_layernorm.weight           (H,)   # zero-centered, use (1+w)
model.layers.{i}.post_attention_layernorm.weight  (H,)
model.norm.weight                                  (H,)
lm_head.weight                                     (vocab, H)   # not tied (tie_word_embeddings=False)
# --- vision (prefix model.visual.) ---
visual.patch_embed.proj.{weight,bias}             Conv3d(3→1152, k=[2,16,16])
visual.pos_embed.weight                           (2304, 1152)
visual.blocks.{k}.norm1/norm2.{weight,bias}       LayerNorm(1152)
visual.blocks.{k}.attn.qkv.{weight,bias}          (3*1152, 1152)
visual.blocks.{k}.attn.proj.{weight,bias}         (1152, 1152)
visual.blocks.{k}.mlp.linear_fc1.{weight,bias}    (4304, 1152)
visual.blocks.{k}.mlp.linear_fc2.{weight,bias}    (1152, 4304)
visual.merger.norm.{weight,bias}                  LayerNorm(1152)  # pre-merge norm dim
visual.merger.linear_fc1.{weight,bias}            (4608, 4608)
visual.merger.linear_fc2.{weight,bias}            (3584, 4608)
```
Checkpoints may contain `mtp.*` (multi-token-prediction head) keys — ignored on load
(`_keys_to_ignore_on_load_unexpected`, `modular_qwen3_5.py:669`).

---

## 11. Porting Checklist

1. **Config**: parse nested `{vision_config, text_config}`; derive `layer_types` from
   `full_attention_interval` (default 4) if absent. Read `head_dim`, all `linear_*` dims,
   `partial_rotary_factor`, and `mrope_section` explicitly. Read vision `out_hidden_size` and
   text `hidden_size` from the actual checkpoint (do not assume the default sub-configs match).
2. **RMSNorm**: zero-centered weights → multiply by `(1 + w)`; compute in fp32.
3. **Linear attention** (the hard part): implement the **Qwen3.5** 4-projection layout
   (`in_proj_qkv/z/b/a`); depthwise causal Conv1d (kernel 4, left-pad 3) over the 8192 conv
   channels; `beta=sigmoid(b)`, `g=-exp(A_log)*softplus(a+dt_bias)` in fp32;
   L2-norm q,k; scale q by `1/sqrt(128)`; run the gated delta rule. Port the **chunked**
   algorithm from `torch_chunk_gated_delta_rule` and unit-test it against
   `torch_recurrent_gated_delta_rule` for equivalence. Output = `RMSNormGated(out, z)` then
   `out_proj`.
4. **Hybrid cache**: per-layer `conv_states (B,8192,3)` + `recurrent_states (B,32,128,128)` for
   linear layers; standard KV `(B,4,S,256)×2` for full layers. Decode: `seq_len==1` →
   conv-update + recurrent rule; chunked decode → prepend conv state, trim output.
5. **Full attention**: Q proj double-width → split Q/gate; QK-norm per head; partial RoPE
   (first 64 dims); `attn * sigmoid(gate)` before `o_proj`; GQA repeat 4×.
6. **MoE**: fp32 softmax router over 256 experts → top-8 → renormalize; 3D-packed experts
   gather/scatter (or grouped GEMM); shared expert gated by `sigmoid(shared_expert_gate)`,
   added.
7. **Vision**: Conv3d patch embed; bilinear-interpolated learned pos-embed (48×48 grid);
   2D vision RoPE; 27 non-causal varlen blocks (`cu_seqlens` per image/frame); 2×2 patch
   merger → 3584; **no DeepStack**.
8. **Merge**: `masked_scatter` vision tokens into `image_token_id`/`video_token_id` slots;
   build M-RoPE 3D position ids via `get_rope_index` + interleave with `mrope_section=[11,11,10]`.
9. **Validate** layer-by-layer against the HF reference (fallback kernels) on a tiny config
   before wiring the fused kernels.

---

**Document version**: 1.0
**Author**: architecture analysis for Arcaine SYCL port
**Reference**: `transformers==5.12.1`, `models/{qwen3_5, qwen3_5_moe, qwen3_next, qwen3_vl, qwen3_vl_moe}`

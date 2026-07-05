# Qwen-AgentWorld-35B-A3B-NVFP4 — Architecture & Porting Reference (Phases 1–3)

> Canonical, version-controlled analysis for porting **Qwen-AgentWorld-35B-A3B-NVFP4** to
> Arcaine (SYCL, Intel BMG G31). Covers target hardware, model inventory, architecture,
> the compressed-tensors **NVFP4** scheme, a full tensor map, and the Phase 5 kernel plan.
> **Phase 4 (NVFP4 dequant verification) is COMPLETE** — verified bit-exact against
> compressed-tensors + FlashInfer (cloned to `reference/`); see §10.3. The kernel arithmetic
> is already compliant; remaining NVFP4 work is loader-only. Phase 5 (kernel porting) is next;
> Phase 6 (validation) is deferred (no in-container golden reference available).
>
- **Checkpoint**: `/workspace/models/Qwen-AgentWorld-35B-A3B-NVFP4/model.safetensors` (22.1 GB, 124063 tensors)
- **HF reference**: `/workspace/reference/transformers/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (2280 lines)
- **Base model**: `Qwen/Qwen-AgentWorld-35B-A3B`, quantized to NVFP4 via llm-compressor (compressed-tensors `"nvfp4-pack-quantized"`)

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Target Hardware (BMG G31)](#2-target-hardware-bmg-g31)
3. [Identity & Scope](#3-identity--scope)
4. [Configuration (top-level dims)](#4-configuration-top-level-dims)
5. [Checkpoint & Tensor Map](#5-checkpoint--tensor-map)
6. [Architecture Overview](#6-architecture-overview)
7. [Full Attention (10 layers)](#7-full-attention-10-layers)
8. [Linear Attention — Gated DeltaNet (30 layers)](#8-linear-attention--gated-deltanet-30-layers)
9. [MoE Block (every layer)](#9-moe-block-every-layer)
10. [NVFP4 Quantization Scheme](#10-nvfp4-quantization-scheme)
11. [Phase 5 — Kernel Porting Plan](#11-phase-5--kernel-porting-plan)
12. [Phase 6 — Validation Status](#12-phase-6--validation-status)
13. [Reference Code Map](#13-reference-code-map)

---

## 1. Executive Summary

AgentWorld-35B-A3B is a **hybrid linear-/full-attention MoE** with **256 routed experts (top-8)
plus an always-on shared expert in every one of its 40 layers**, quantized to **NVFP4**
(4-bit weight, e4m3 per-group scale, two global fp32 scales). The port to Arcaine introduces
three substantive pieces of work:

1. **Gated DeltaNet (greenfield).** 30 of 40 layers use a Mamba2-style *gated delta rule*
   linear attention (depthwise `conv1d` k=4, recurrent decode with fp32 SSM state
   `S[bs,32,128,128]`, chunked prefill, chunk=64). `grep` finds **zero** scaffolding in
   `src/` — this is the largest new kernel effort.
2. **NVFP4 loader adaptation.** The checkpoint's packing differs from Arcaine's existing
   single-`dst_scale` NVFP4 path: `weight_scale` is stored `[N, K/16]` and there are **two**
   fp32 global scales (`input_global_scale`, `weight_global_scale`) that must be folded into
   the kernel's `dst_scale`. **Phase 4 VERIFIED bit-exact** (compressed-tensors + FlashInfer
   sources cloned to `reference/`): the kernel arithmetic is already compliant — e2m1 LUT,
   e4m3 layout, nibble order, and the `1/dst_scale` form all match. The remaining work is
   **loader-only** (repack `weight_scale`, fold globals → `dst_scale`, strip the
   `model.language_model.` key prefix). See §10.3–§10.4.
3. **256-expert + shared-expert dispatch.** Kernels are parametric in expert count, but the
   always-on shared expert with a **per-token scalar sigmoid gate** is new MoE wiring.

Full attention (10 layers) is comparatively standard (GQA 16:2, head_dim 256, per-head q/k
RMSNorm before partial RoPE, sigmoid output gate) and reuses existing primitives.

## 2. Target Hardware (BMG G31)

| Property | Value |
|---|---|
| Device | Intel Battlemage (BMG) G31 |
| EUs | 256 |
| `matrix` (joint_matrix) | **no** (`cl_khr_matrix` absent) |
| DPAS path | OpenCL `matrix-mad` / DPAS via `intel_gpu_dp4a`-style intrinsics |
| SLM per work-group | 128 KB |
| Subgroup sizes | 16, 32 |
| GPUs present | 2 physical devices |

Notes:
- `matrix=no` ⇒ no `joint_matrix`/`joint_matrix_mad` extensions. DPAS achieved via OpenCL
  `matrix-mad` / DPAS intrinsics; see `notes/artifacts/` probes (`dpas_mad_probe.cpp`,
  `nvfp4_dpas.cpp`).
- Hardware probe binary is **prebuilt** at `.agents/skills/check-available-hardware/get_device_props` — do not rebuild it.

## 3. Identity & Scope

- `model_type = qwen3_5_moe_text`; HF class `Qwen3_5MoeForCausalLM` (text path of the
  multimodal `Qwen3_5MoeForConditionalGeneration`).
- **TEXT-ONLY.** The checkpoint contains **zero** `mtp.*`, `model.visual.*`, or audio
  tensors. Reference `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`
  drops them. **MTP and the vision tower are OUT OF SCOPE.** M-RoPE / vision-tokenizer
  entries in config are vestigial; for text, M-RoPE degenerates to standard RoPE.
- Checkpoint state-dict keys are prefixed **`model.language_model.`** (the full multimodal
  container wraps the text model as `.language_model`); `lm_head.weight` is **top-level**
  (unprefixed). **The loader must strip `model.language_model.`** and handle `lm_head`
  separately. `embed_tokens` lives under the `model.language_model.` prefix.

## 4. Configuration (top-level dims)

| Field | Value |
|---|---|
| `hidden_size` | 2048 |
| `vocab_size` | 248320 |
| `num_hidden_layers` | 40 |
| `rms_norm_eps` | 1e-6 |
| `tie_word_embeddings` | false (`lm_head` separate, BF16) |
| `num_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `aux_loss_coef` | 0.001 |
| `full_attention_interval` | 4 |
| Linear-attn layers | 30 (idx ≠ 3 mod 4) |
| Full-attn layers | 10 (idx 3,7,…,39) |

Every layer is an MoE layer (no dense layers).

## 5. Checkpoint & Tensor Map

### 5.1 Inventory by dtype (124063 total)

| Dtype | Count | Role |
|---|---|---|
| `U8` | 30910 | `weight_packed` (2 e2m1 nibbles/byte) |
| `F8_E4M3` | 30910 | `weight_scale` (per-group, group=16) |
| `F32` | 61820 | `input_global_scale` + `weight_global_scale` (2 per quant Linear) |
| `BF16` | 423 | embeddings, norms, router, all linear-attn projections, q/k norms |

NVFP4 decomposition: each quantized Linear = 4 tensors
(`weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale`).

### 5.2 Quantized-Linear count breakdown (30910 weight_packed)

| Component | weight_packed count | Math |
|---|---|---|
| Routed experts (gate/up/down) | 30720 | 256 experts × 40 layers × 3 |
| Shared expert (gate/up/down) | 120 | 40 layers × 3 |
| Full-attn (q/k/v/o) | 40 | 4 projs × 10 layers |
| Linear-attn `out_proj` | 30 | 1 proj × 30 layers |
| **Total** | **30910** | |

### 5.3 Tensor mapping table

**Embeddings / output head (BF16, unprefixed/prefixed)**

| Tensor | Count | Dtype | Shape | Role |
|---|---|---|---|---|
| `lm_head.weight` | 1 | BF16 | [248320, 2048] | output projection (top-level, no prefix) |
| `model.language_model.embed_tokens.weight` | 1 | BF16 | [248320, 2048] | token embedding |
| `model.language_model.norm.weight` | 1 | BF16 | [2048] | final RMSNorm |

**Per-layer norms + MoE router (BF16)**

| Tensor | Count | Dtype | Shape | Role |
|---|---|---|---|---|
| `...layers.*.input_layernorm.weight` | 40 | BF16 | [2048] | pre-attn RMSNorm |
| `...layers.*.post_attention_layernorm.weight` | 40 | BF16 | [2048] | pre-MoE RMSNorm |
| `...layers.*.mlp.gate.weight` | 40 | BF16 | [256, 2048] | router logits (softmax→topk8→renorm) |
| `...layers.*.mlp.shared_expert_gate.weight` | 40 | BF16 | [1, 2048] | per-token scalar sigmoid gate for shared expert |

**Full attention (10 layers) — `self_attn.*`**

| Tensor | Count | Dtype | Shape | Role |
|---|---|---|---|---|
| `...self_attn.q_proj.{weight_packed,weight_scale,weight_global_scale,input_global_scale}` | 10 each | U8 / F8E4M3 / F32 / F32 | packed [8192,1024]; scale [8192,128]; globals [1] | q_proj: hidden→8192 = [Q 4096 \| output-gate 4096] |
| `...self_attn.k_proj.*` | 10 each | (same) | packed [512,1024]; scale [512,128] | k_proj: hidden→512 (2 kv_heads × 256) |
| `...self_attn.v_proj.*` | 10 each | (same) | packed [512,1024]; scale [512,128] | v_proj: hidden→512 |
| `...self_attn.o_proj.*` | 10 each | (same) | packed [2048,2048]; scale [2048,256] | o_proj: 4096→2048 (16 heads × 256) |
| `...self_attn.q_norm.weight` | 10 | BF16 | [256] | per-head q RMSNorm (before RoPE) |
| `...self_attn.k_norm.weight` | 10 | BF16 | [256] | per-head k RMSNorm (before RoPE) |

> q_proj output dim 8192 = `[Q (16×256=4096) | output-gate (4096)]`; the **sigmoid output gate**
> is taken from the second half of q_proj's output. o_proj input 4096 = 16 heads × 256.

**Linear attention / Gated DeltaNet (30 layers) — `linear_attn.*`**

| Tensor | Count | Dtype | Shape | Role |
|---|---|---|---|---|
| `...linear_attn.in_proj_qkv.weight` | 30 | BF16 | [8192, 2048] | projects hidden→ q/k/v concat (16×128 + 16×128 + 32×128 = 8192) |
| `...linear_attn.in_proj_z.weight` | 30 | BF16 | [4096, 2048] | gate projection (RMSNormGated) |
| `...linear_attn.in_proj_a.weight` | 30 | BF16 | [32, 2048] | decay-input projection → `A_log`-path |
| `...linear_attn.in_proj_b.weight` | 30 | BF16 | [32, 2048] | beta-input projection → `sigmoid` |
| `...linear_attn.conv1d.weight` | 30 | BF16 | [8192, 1, 4] | depthwise causal conv, kernel=4 (silu) |
| `...linear_attn.A_log` | 30 | BF16 | [32] | log-decay init (per head) |
| `...linear_attn.dt_bias` | 30 | BF16 | [32] | bias added to decay-input |
| `...linear_attn.norm.weight` | 30 | BF16 | [128] | RMSNormGated norm over head_v_dim=128 |
| `...linear_attn.out_proj.*` | 30 each | U8 / F8E4M3 / F32 / F32 | packed [2048,2048]; scale [2048,256]; globals [1] | out_proj: 4096→2048 (NVFP4) |

> Only `out_proj` is NVFP4-quantized in the linear-attn block; **all `in_proj_*`, `conv1d`,
> `A_log`, `dt_bias`, `norm` are BF16** (unquantized) — they require BF16 GEMMs, not the NVFP4 path.

**MoE routed experts (40 layers, 256 experts each) — `mlp.experts.*`**

| Tensor | Count | Dtype | Shape | Role |
|---|---|---|---|---|
| `...experts.*.gate_proj.{4-tuple}` | 10240 each | NVFP4 4-tuple | packed [512,1024]; scale [512,128]; globals [1] | SwiGLU gate: 2048→512 |
| `...experts.*.up_proj.{4-tuple}` | 10240 each | NVFP4 4-tuple | packed [512,1024]; scale [512,128]; globals [1] | SwiGLU up: 2048→512 |
| `...experts.*.down_proj.{4-tuple}` | 10240 each | NVFP4 4-tuple | packed [2048,256]; scale [2048,32]; globals [1] | down: 512→2048 |

**MoE shared expert (40 layers) — `mlp.shared_expert.*`** (same shapes as a routed expert)

| Tensor | Count | Dtype | Shape | Role |
|---|---|---|---|---|
| `...shared_expert.gate_proj.{4-tuple}` | 40 each | NVFP4 4-tuple | packed [512,1024]; scale [512,128]; globals [1] | always-on shared SwiGLU gate |
| `...shared_expert.up_proj.{4-tuple}` | 40 each | NVFP4 4-tuple | packed [512,1024]; scale [512,128]; globals [1] | always-on shared SwiGLU up |
| `...shared_expert.down_proj.{4-tuple}` | 40 each | NVFP4 4-tuple | packed [2048,256]; scale [2048,32]; globals [1] | always-on shared down |

### 5.4 Shape conventions (NVFP4)

For a quantized Linear `W[N, K]` (PyTorch `out, in`):
- `weight_packed`: U8 `[N, K/2]` — 2 e2m1 nibbles per byte
- `weight_scale`: F8_E4M3 `[N, K/16]` — one scale per output row × group-of-16
- `weight_global_scale`: F32 `[1]`
- `input_global_scale`: F32 `[1]` (dynamic/"local" — runtime per activation)

## 6. Architecture Overview

### 6.1 Hybrid 40-layer stack

40 decoder layers; **every layer is an MoE block**. Attention alternates by
`full_attention_interval = 4`:
- **Full attention** at layer indices `3, 7, 11, …, 39` → **10 layers** (`Qwen3_5MoeAttention`).
- **Linear attention** (Gated DeltaNet) at all other indices → **30 layers** (`Qwen3_5MoeGatedDeltaNet`).

### 6.2 Decoder layer data flow (pre-norm, sequential — NOT parallel attention/MoE)

```
h = x
a = AttnOrLinear(input_layernorm(x))      # full-attn OR gated-deltanet
h = h + a                                  # separate residual #1
m = MoE(post_attention_layernorm(h))
out = h + m                                 # separate residual #2
```

KV / state caches:
- **Full-attn layers**: standard KV cache `(bs, seq, 2 kv_heads, 256)`.
- **Linear-attn layers**: `conv_state (bs, 8192, 3)` + `recurrent_state S (bs, 32, 128, 128)` fp32.

## 7. Full Attention (10 layers)

`Qwen3_5MoeAttention` — GQA, partial RoPE, per-head RMSNorm, sigmoid output gate.

- **GQA**: `heads=16`, `kv_heads=2`, `head_dim=256`, group=8.
- **Projections** (NVFP4): q_proj `2048→8192` (Q 4096 ‖ output-gate 4096); k/v_proj `2048→512`; o_proj `4096→2048`.
- **Per-head RMSNorm** (`q_norm`, `k_norm`, shape [256]) applied to Q and K **before** RoPE.
- **RoPE**: `partial_rotary = 0.25` → `rotary_dim = 64` (32 pairs); `theta = 1e7`; NeoX
  `rotate_half` on the first 64 dims, 192 dims pass through. M-RoPE sections `[11,11,10]`
  are vestigial for text (forward feeds `arange` to all 4 position axes → standard RoPE).
- **Output gate**: `attn_out *= sigmoid(gate)` where `gate` = second half of q_proj output,
  applied **before** `o_proj`.
- Standard KV cache (eager path in reference: ref `621-643`).

## 8. Linear Attention — Gated DeltaNet (30 layers)

`Qwen3_5MoeGatedDeltaNet` — Mamba2-style **gated delta rule**. The biggest greenfield item.

### 8.1 Dimensions
- `k_heads = 16` (`head_k_dim = 128`) → `key_dim = 2048`
- `v_heads = 32` (`head_v_dim = 128`) → `value_dim = 4096`
- GQA 2:1: q,k are `repeat_interleave`×2 (16→32) to match v_heads.

### 8.2 Parameters (BF16, unquantized — except `out_proj`)
`in_proj_qkv[8192,2048]`, `in_proj_z[4096,2048]` (gate), `in_proj_a[32,2048]`,
`in_proj_b[32,2048]`, `conv1d.weight[8192,1,4]`, `A_log[32]`, `dt_bias[32]`,
`norm.weight[128]` (RMSNormGated). `out_proj[2048,4096]` is NVFP4.

### 8.3 Forward path
1. Pad attention mask.
2. `in_proj_qkv` → transpose → **depthwise causal `conv1d` (k=4, silu)**
   (decode: update `conv_state`; prefill: `causal_conv1d_fn`).
3. Split into `q[16,128]`, `k[16,128]`, `v[32,128]`.
4. `beta = sigmoid(in_proj_b)`; `g = -exp(A_log) * softplus(in_proj_a + dt_bias)` → `[bs, seq, 32]` (per-head decay).
5. `q, k` repeat×2 (16→32); L2-normalize `q, k`; `scale = 1/sqrt(128)`.

### 8.4 Core — gated delta rule
- **Decode (seq=1, recurrent)**: state `S[bs,32,128,128]` fp32. Per token:
  `S ← exp(g)·S`; `kv_mem = Sᵀ·k`; `delta = (v − kv_mem)·beta`; `S += k ⊗ delta`; `out = Sᵀ·q`.
  (ref `326-367`.)
- **Prefill (chunked)**: `chunk_size = 64` chunked delta rule (ref `245-323`).

### 8.5 Output
`RMSNormGated(core, z) = RMSNorm(x) · silu(z)` over `head_v_dim = 128` → `(bs, seq, 4096)`
→ `out_proj` → 2048. Cache: `conv_state (bs,8192,3)` + `S (bs,32,128,128)` per linear-attn layer.

## 9. MoE Block (every layer)

`Qwen3_5MoeSparseMoeBlock` — 256 routed experts (top-8) + always-on shared expert.

### 9.1 Router (`mlp.gate`, BF16 [256,2048])
`softmax` (fp32) → `topk(8)` → renormalize so weights sum to 1. `aux_loss_coef = 0.001`.

### 9.2 Routed experts (256, SwiGLU, all NVFP4)
Per hot expert: gather tokens → `down(silu(gate(x)) * up(x))` → scatter (`index_add`) → multiply
by `routing_weight`. `gate_proj`/`up_proj` `2048→512`; `down_proj` `512→2048`.

### 9.3 Shared expert (DeepSeekMoE, always-on)
Same SwiGLU shape as a routed expert (intermediate=512), NVFP4. Gated by a **per-token scalar**:
`shared_out = sigmoid(shared_expert_gate(x)) · shared_MLP(x)`, where `shared_expert_gate` is a
`Linear(2048→1)` (BF16). Final MoE output = `routed + shared`.

## 10. NVFP4 Quantization Scheme

### 10.1 Compressed-tensors `"nvfp4-pack-quantized"`
Each quantized Linear stores 4 tensors (see §5.4). Dequant for output element `n`:
```
y[n] = Σ_k ( e2m1(a[k]) · e4m3(a_scale_g[k_g]) )        # activation block-scaled
       · ( e2m1(w[n,k]) · e4m3(w_scale[n, k_g]) )       # weight block-scaled
       · ( input_global_scale · weight_global_scale )   # two fp32 globals
```
where `k_g = k // 16`. Symmetric; `input_global_scale` is **dynamic/"local"** (runtime, per
activation). **No single `dst_scale`** in the checkpoint — the two fp32 globals fold the e4m3
dynamic range.

### 10.2 Gap vs existing Arcaine kernels
- Existing path (e.g. `build/.../nvfp4_xe2v3_isolate.cpp:85`, oneDNN integration) ingests
  `weight_scale` in layout **`[K/16, N]`** (`W_sc[(k/16)*N + n]`) and a **single** `dst_scale`.
- Checkpoint stores **`[N, K/16]`** + two globals (no `dst_scale`).
- **Loader must**: (a) repack `weight_scale` from `[N, K/16]` → kernel layout `[K/16, N]`; (b)
  fold `input_global_scale · weight_global_scale` into the kernel's `dst_scale`.
- Bench struct (`nvfp4_roofline_bench.cpp:253`) **already has** `input_global_scale` /
  `weight_global_scale` fields (currently forced = 1.0) → partial plumbing exists.

### 10.3 ✅ Phase 4 — dequant verification (VERIFIED bit-exact vs compressed-tensors + FlashInfer)

Source refs (cloned to `reference/`):
- `reference/compressed-tensors/src/compressed_tensors/compressors/nvfp4/helpers.py`
- `reference/compressed-tensors/.../quantization/lifecycle/forward_helpers.py`
- `reference/compressed-tensors/.../quantization/utils/helpers.py`
- `reference/flashinfer/flashinfer/quantization/nvfp4_quantization_utils.py`
- Arcaine: `src/common/gpu/nvfp4.hpp`, `src/nvfp4_xe2v3_isolate.cpp`

**Exact dequant formula** (compressed-tensors `forward_helpers.py:255-263` `_dequantize` +
`calculate_qparams` `utils/helpers.py:86,101-102`). For a quant Linear `W[N,K]`:

```
# weight (static):  stored weight_scale = round_e4m3(weight_global_scale · amax_w_group / 6)
W[n,k] = e2m1(weight_packed[n,k]) · e4m3(weight_scale[n, k//16]) / weight_global_scale

# activation (dynamic="local", strategy=tensor_group, group=16): computed at runtime
a_scale_g       = round_e4m3(input_global_scale · amax_group / 6)      # per group of 16
a_q[n,k]        = round_e2m1( x[n,k] · input_global_scale / a_scale_g )  # then packed
A[n,k]          = e2m1(a_q[n,k]) · e4m3(a_scale_g) / input_global_scale

# full GEMM (matches Arcaine's result · 1/dst_scale form):
y[n] = Σ_k ( e2m1(a_q)·e4m3(a_scale_g)·e2m1(w)·e4m3(w_scale) ) · (1 / (input_global_scale · weight_global_scale))
     = [Σ_k ...] · (1 / dst_scale),   dst_scale = input_global_scale · weight_global_scale
```

The global scale **divides** the per-group scale (not multiplies it): during *quantize* the global
**amplifies** the raw scale before e4m3 rounding (`utils/helpers.py:101-102` `scales = global_scale * scales`),
then *dequant* divides it back (`forward_helpers.py:191-192,255-256` `scale = scale / global_scale`).
This is the NVFP4 precision trick: round e4m3 on the amplified value, recover the true scale on use.

**Verification table — every named trap, resolved:**

| # | Question | Answer | Source (compressed-tensors / Arcaine) |
|---|---|---|---|
| 1 | nibble order in `weight_packed` | **low-nibble = lower-K**, high-nibble = higher-K | ct `helpers.py:72` (`idx0 | idx1<<4`), `96-100`; arcaine `nvfp4.hpp:779` (`lo | hi<<4`) |
| 2 | e2m1 bit layout + LUT | bit3=sign, bits0-2=magnitude index; LUT `{0,0.5,1,1.5,2,3,4,6}` | ct `helpers.py:18-31,103-108`; arcaine `nvfp4.hpp:496-499` (identical LUT) |
| 3 | e4m3 bit layout | 1 sign / 4 exp / 3 mant, bias 7, max 448 | ct via `FLOAT8_E4M3_MAX=448`; arcaine `nvfp4.hpp:99-108,483-494` (bias 7, max 448) |
| 4 | weight_scale indexing | `[N, K/16]`, `k_group = k//16`, each scale covers 16 k | ct `base.py:93`; header confirms shape [N,K/16] |
| 5 | dynamic input scale | `a_scale_g = round_e4m3(global·amax/6)`; `a_q = round_e2m1(x·global/a_scale_g)` | ct `utils/helpers.py:86,101-102` + `forward.py:310-328`; arcaine `nvfp4.hpp:765-779` (identical) |
| 6 | global-scale combination | `dst_scale = input_global_scale · weight_global_scale`; kernel multiplies by `1/dst_scale` | ct `forward_helpers.py:255-256`; arcaine `result · 1/dst_scale` (diagnosis memo) |

**Conclusion**: Arcaine's existing NVFP4 **kernel arithmetic is already compliant** with
compressed-tensors — e2m1/e4m3 decode, nibble order, scale indexing, and the `1/dst_scale`
form all match. The `input_global_scale` factor is correctly applied in the activation packer
(`nvfp4.hpp:754,765,774`). **No kernel dequant changes required.** The remaining work is
loader-only (§10.4).

### 10.4 NVFP4 integration status — ALREADY IMPLEMENTED by existing diffusion_gemma loader

**Phase 4 conclusion (revised): the NVFP4 loader mechanics are fully solved and reusable.**
The existing `diffusion_gemma` (26B) model uses the **identical** compressed-tensors
`nvfp4-pack-quantized` scheme, and its loader + struct + dispatch already implement every
action item. The new Qwen3.5 model reuses them with different key prefixes — **no kernel or
struct changes required.**

Existing infrastructure (verified by code read):
- `Nvfp4Linear` struct — `src/common/gpu/nvfp4.hpp:35-52`: already has `input_global_scale`,
  `weight_global_scale`, `weight_packed`, `weight_scale` (transposed `[K/16,N]`), `dst_scale`
  (1-elem F32). Exactly the 4-tensor checkpoint scheme + folded `dst_scale`.
- `upload_nvfp4_scales_transposed` — `src/modeling/diffusion_gemma/loader.cpp:277-294`:
  already transposes `weight_scale` from checkpoint `[N, K/16]` → kernel `[K/16, N]`
  (comment: "model layout: (N, K/16)").
- `upload_nvfp4_linear` — `loader.cpp:296-321`: already reads both globals (F32[1]) and folds
  `dst_scale = input_global_scale * weight_global_scale` (line 317).
- `upload_nvfp4_linear_pair` — `loader.cpp:323-388`: fused gate+up (concat separate
  checkpoint gate/up into one `Nvfp4Linear` with `out=2*half`), folds + transposes.
- Expert dispatch — `src/common/gpu/expert_parallel.cpp`: per-expert `gate_up_proj_fp4` /
  `down_proj_fp4` (Nvfp4Linear), `input_global_scale` plumbed (lines 393,399,726,730),
  kernels parametric in expert count `E`.

What the new Qwen3.5 model module must do (wiring, not mechanics):
1. **Key prefixes**: call `upload_nvfp4_linear(sf, "model.language_model.layers.<i>.mlp.experts.<e>.<proj>", q)` for each quantized Linear. `lm_head` is top-level (prefix `"lm_head"`). No prefix-strip function needed — the prefix is just the string passed to `sf.get()`.
2. **Routed experts (256)**: fuse gate+up via `upload_nvfp4_linear_pair` → `gate_up_proj_fp4`; load down via `upload_nvfp4_linear` → `down_proj_fp4`. ~512 Nvfp4Linears. Dispatch is parametric in `E` (verify E=256 path; 26B used 128).
3. **Shared expert**: same 2 calls (gate+up fused + down) → 2 Nvfp4Linears per layer.
4. **Full-attn q/k/v/o (10 layers)**: 4× `upload_nvfp4_linear` each.
5. **Linear-attn `out_proj` (30 layers)**: 1× `upload_nvfp4_linear` each.
6. **BF16 (unquantized)**: `in_proj_qkv/z/a/b`, `conv1d`, `A_log`, `dt_bias`, `norm`, `q_norm`/`k_norm`, router `mlp.gate`, `shared_expert_gate`, embeddings, `lm_head` — plain BF16 uploads.

Net: the NVFP4 path is **de-risked and free**. The real Phase 5 work is the greenfield
Gated DeltaNet kernels, the full-attn output gate, and the shared-expert per-token gate.

## 11. Phase 5 — Kernel Porting Plan

Ordered by dependency / risk:

1. **NVFP4 loader adaptation** (shared across experts + shared + attn + linear_attn.out_proj).
   Repack `weight_scale` `[N,K/16]→[K/16,N]`, fold 2 globals → `dst_scale`, strip
   `model.language_model.` prefix, handle top-level `lm_head`. **Kernel arithmetic verified
   correct (Phase 4) — this is loader/integration work only.** Unblocks everything quantized.
2. **Gated DeltaNet (greenfield)** — the largest item:
   - depthwise `conv1d` (k=4, silu) with `conv_state` update (decode) / causal fn (prefill);
   - BF16 `in_proj_qkv/z/a/b` GEMMs (no NVFP4);
   - recurrent delta-rule decode (fp32 `S`) + chunked prefill (chunk=64);
   - SSM/conv cache management.
3. **Full attention**: GQA + per-head q/k RMSNorm + partial RoPE (text-degenerate M-RoPE) +
   sigmoid output gate. Mostly existing primitives.
4. **Shared-expert dispatch + per-token scalar sigmoid gate** (new MoE wiring).
5. **256-expert dispatch**: kernels are parametric in expert count `E`; verify gather/scatter
   + `routing_weight` multiply and `index_add` semantics.
6. **Hybrid layer scheduling**: route to full-attn vs linear-attn by `idx % 4 == 3`.

## 12. Phase 6 — Validation Status

**Deferred by user decision.** The container venv has `transformers==5.12.1` but **no**
`torch` / `compressed-tensors` / `llmcompressor` / `flashinfer` / `vllm` / `fla` → cannot run
an HF reference forward in-container for golden activations. Options to revisit if debugging stalls:
- CPU `torch` in the venv for small golden dumps;
- external env / vLLM run to capture reference activations.

## 13. Reference Code Map

`modeling_qwen3_5_moe.py` (2280 lines):

| Component | Lines |
|---|---|
| `TextRotaryEmbedding` / M-RoPE | 94–183 |
| `RMSNormGated` | 186–201 |
| `conv1d` (depthwise) | 221–236 |
| `l2norm` | 239–242 |
| chunked delta rule (prefill) | 245–323 |
| recurrent delta rule (decode) | 326–367 |
| `Qwen3_5MoeGatedDeltaNet` | 371–560 |
| `rotate_half` / `apply_rotary` | 563–606 |
| eager attention | 621–643 |
| `Qwen3_5MoeAttention` | 646–720 |
| `Qwen3_5MoeMLP` | 723–736 |
| `Qwen3_5MoeExperts` | 740–776 |
| `Qwen3_5MoeRouter` | 779–795 |
| `Qwen3_5MoeSparseMoeBlock` | 798–817 |
| `RMSNorm` | 820–836 |
| `Qwen3_5MoeDecoderLayer` | 840–896 |
| `TextModel.forward` | 1264–1331 |
| `ForCausalLM` | 1788–1889 |
| Vision pipeline | 935–1220 (**OUT OF SCOPE**) |

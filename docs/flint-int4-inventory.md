# Flint INT4 — Implementation Inventory

Status: inventory / pre-implementation (v2). Target: **DPC++/SYCL on the Level Zero backend**, no oneDNN, **no `joint_matrix`** (DPAS reached through ESIMD `xmx::dpas` or the SPIR-V intrinsic instead), single-batch inference of `RecursiveSI/flint-int4` on the Intel Arc Pro B70 in this container.

Sources of truth used for this inventory:

- Checkpoint at `/workspace/models/flint-int4` (host: `models/flint-int4`), 5 safetensors shards, 1751 tensors, **22.569 GB**.
- MLX reference: `reference/rec/engine/src/rec_mlx` (`recursive-oss/rec`, v0.1.0, commit `e53663b`). This repo is the primary kernel/data-flow reference.
- Hardware report: `/workspace/.agents/skills/check-available-hardware/get_device_props`.

No Arcaine GPU code is assumed reusable. Existing `src/common/gpu/int4*.hpp`, `quant_loader`, `nvfp4`, `q8_0` etc. are oneDNN-based or tied to other quant formats and are explicitly **out of scope** for this work. Everything below describes what must be built from scratch.

## 0. Resolved decisions

| Topic | Decision |
|---|---|
| API | DPC++/SYCL targeting Level Zero; direct `ze*` where it buys something; no oneDNN |
| DPAS access | Two implementations to be timed head-to-head: ESIMD `xmx::dpas` vs direct SPIR-V `SubgroupMatrixMultiplyAccumulateINTEL`; pick per kernel from measurements |
| CPU plumbing | Reuse Arcaine tokenizer + chat template; new sampler |
| Decode structure | Per-layer L0 launches first; persistent megakernel later behind an env flag |
| Prefill | Both bf16 and int8 paths from day one, A/B behind an env var |
| KV | bf16 now; 8-bit K/V behind an env flag later |
| Oracle | Static analysis of the MLX repo + the shipped sglang wheel; no runtime golden logits |
| Milestone | Full path: loader, prefill, decode, top-k/top-p sampling, server interface |
| Reference added | `reference/level-zero` (loader v1.34, spec 1.19.12) |

---

## 1. Model identity and shape

| Field | Value |
|---|---|
| HF repo | `RecursiveSI/flint-int4` (`recursive-oss/rec` is the MLX engine) |
| Architecture | `FlintForCausalLM`, `model_type: flint` |
| Layers | 49 (`num_hidden_layers`) |
| Hidden | 2048 |
| Attention | 32 Q heads, 4 KV heads (GQA 8:1), head_dim 128, scale = 1/√128 |
| Dense MLP | layers `[0, 1]`, intermediate 6144 |
| MoE | layers `2..48` (47 layers), 256 experts, top-6, moe intermediate 512, shared expert 1024 |
| Router | sigmoid scoring, `norm_topk_prob: true`, no router bias tensor (expert_bias defaults to zero) |
| Vocab | 150144, untied embed / lm_head |
| Positions | max 262144, RoPE θ=10000 on **sliding layers only** |
| Sliding window | 513 |
| Attention sinks | learnable per-head scalar (`softmax_offset`), present in all 49 layers |
| Quantization | MLX affine, group 64; 4-bit projections/experts/lm_head; **8-bit embedding** |
| KV format (manifest) | `bf16` |
| Total / active params | ≈40B total, ≈2.2B active (matches the "40B-A2.5B" label) |

### 1.1 The two non-obvious config quirks

1. **"full_attention" layers are the No-RoPE layers.** `mlx_export.full_attention_layers` == `no_rope_layer_indices` == `[0,1,5,9,13,17,21,25,29,33,37,41,45,48]` (14 layers). The reference derives `is_global = not use_rope` (`reference/rec/engine/src/rec_mlx/quantized.py:52-61`), so:
   - 14 global layers: no RoPE, full causal context.
   - 35 sliding layers: RoPE (NeoX-style halves), 513-token window.
2. **All 49 layers use attention sinks**, sliding and global alike. Sink is a per-head logit appended to the softmax: `softmax([scores..., sink])[:len(scores)]`, equivalent to an always-present key with weight 1.

### 1.2 Layer map

```
layer   type      mlp
0       global    dense 6144
1       global    dense 6144
2..48   sliding   MoE 256/top6 + shared 1024
        except the 14 global indices above, which are global but still MoE
```

Global indices: 0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48. All others sliding.

---

## 2. Weight format (MLX affine INT4)

Trap: **`biases` in this checkpoint is the affine zero-point, not a linear bias.** There are no linear biases anywhere (`attention_bias: false`, all projections bias-free).

Reconstruction, per group of 64 along the input dimension:

```
w[n, k] = scales[n, k/64] * q[n, k] + biases[n, k/64]      # bf16 scale/zero-point
q       = unsigned nibble (0..15) for 4-bit, unsigned byte (0..255) for 8-bit embed
```

Packing (verified against `reference/rec/engine/src/rec_mlx/kernels/megakernel.py:855`):

- U8 tensor, row-major `[out, K/2]` bytes for 4-bit (or `K` bytes for 8-bit).
- Element `k` lives in byte `k/2`; even `k` = low nibble, odd `k` = high nibble.
- `scales` / `biases`: bf16 `[out, K/64]`; for `switch_mlp` they are 3-D `[experts, out, K/64]`.

### 2.1 Per-layer tensors (all 49 layers)

| Tensor | Dtype / shape (4-bit) | Notes |
|---|---|---|
| `input_layernorm.weight` | BF16 [2048] | |
| `post_attention_layernorm.weight` | BF16 [2048] | |
| `self_attn.q_proj.{weight,scales,biases}` | U8 [4096,1024] / BF16 [4096,32] | |
| `self_attn.k_proj.*` | U8 [512,1024] / BF16 [512,32] | |
| `self_attn.v_proj.*` | U8 [512,1024] / BF16 [512,32] | |
| `self_attn.o_proj.*` | U8 [2048,2048] / BF16 [2048,64] | K=4096 |
| `self_attn.q_norm.weight`, `k_norm.weight` | BF16 [128] | per-head RMSNorm |
| `self_attn.softmax_offset` | BF16 [32] | sink, fp32 in reference |
| dense layers only: `mlp.{gate,up}_proj.*` | U8 [6144,1024] / BF16 [6144,32] | |
| dense layers only: `mlp.down_proj.*` | U8 [2048,3072] / BF16 [2048,96] | K=6144 |
| MoE layers: `mlp.gate.weight` | BF16 [256,2048] | router, not quantized |
| MoE layers: `mlp.shared_expert.{gate,up}_proj.*` | U8 [1024,1024] / BF16 [1024,32] | |
| MoE layers: `mlp.shared_expert.down_proj.*` | U8 [2048,512] / BF16 [2048,16] | K=1024 |
| MoE layers: `mlp.switch_mlp.{gate,up}_proj.*` | U8 [256,512,1024] / BF16 [256,512,32] | per-expert 3-D |
| MoE layers: `mlp.switch_mlp.down_proj.*` | U8 [256,2048,256] / BF16 [256,2048,8] | per-expert, K=512 |
| global: `model.embed_tokens.*` | **U8 [150144,2048]** / BF16 [150144,32] | 8-bit, 1 byte/weight |
| global: `lm_head.*` | U8 [150144,1024] / BF16 [150144,32] | 4-bit |
| global: `model.norm.weight` | BF16 [2048] | |

### 2.2 Size budget

| Category | GB |
|---|---|
| MoE/MLP weights (packed) | 19.16 |
| MoE/MLP scales + zero-points | 2.38 |
| Attention weights + scales | 0.52 |
| Embedding (8-bit) | 0.31 |
| lm_head (4-bit) | 0.15 |
| norms / sinks | <0.01 |
| **Total** | **22.569** |

One MoE layer is ~384 MB; one dense layer ~19 MB; attention ~7.3 MB.

---

## 3. Reference data flow

### 3.1 The MLX "native" path (used for prefill, and as the correctness oracle)

`reference/rec/engine/src/rec_mlx/model.py`:

| Piece | Lines |
|---|---|
| `ModelArgs` | 14-37 |
| `MixedQuantizedKVCache` (8-bit K/V path, optional) | 40-105 |
| `quantized_attention_with_sinks` | 108-139 |
| `Attention` (Q/K norm, RoPE only when `use_rope`, sinks) | 142-213 |
| `MLP`, `Router`, `SparseMoeBlock` | 216-254 |
| `DecoderLayer` | 257-271 |
| `FlintTransformer` / `Model` / `make_cache` | 274-360 |

Per-token/per-chunk flow:

```
h = embed(tokens)
for layer in 0..48:
    n  = RMSNorm(h, input_layernorm)
    q  = Wq n -> [32,128]     k = Wk n -> [4,128]     v = Wv n -> [4,128]
    q  = RMSNorm_head(q, q_norm); k = RMSNorm_head(k, k_norm)
    if layer not in no_rope: q,k = RoPE(q,k, pos)
    k,v = cache.update_and_fetch(k,v)
    out = softmax_sink(q·kᵀ * scale, causal/window mask) · v
    h  = h + Wo out
    n2 = RMSNorm(h, post_attention_layernorm)
    if dense: h += Wdown( silu(Wgate n2) * (Wup n2) )
    else:
        s = sigmoid(router(n2))                      # fp32
        c = s + expert_bias                          # zeros here
        idx = topk(c, 6); w = s[idx]; w /= sum(w)    # bf16 rounding
        h += sum_e w_e * Wdown_e( silu(Wgate_e n2) * Wup_e n2 )
             + shared_expert(n2)
h = RMSNorm(h, model.norm); logits = lm_head(h)
```

Prefill: `QATSession.prefill` (`reference/rec/engine/src/rec_mlx/quantized.py:98-151`) chunks the prompt at 2048/4096 tokens (`kernels/device.py`), runs the native model on the chunk, writes bf16 K/V into one shared storage, and only keeps the last token's logits.

### 3.2 The resident decode megakernel (the performance reference)

`reference/rec/engine/src/rec_mlx/kernels/megakernel.py` is a single Metal kernel that runs **all 49 layers for one token in one launch**, with 40 persistent SIMD groups (one per Q/K/V head: 32 Q + 4 K + 4 V), device-wide barriers between phases (`sync_all`, 137-159).

| Element | Lines |
|---|---|
| Host layout / scratch / alignment | 9-74 |
| `KVView` (4-block paged KV, 65536 positions/block) | 115-135 |
| `Matrix` + `weight_matrix` (base offsets per projection) | 161-170 |
| `load16` / `dot16` (4-bit dot with in-place nibble trick) | 171-196 |
| `q4` / `q4_local` (4-row × subgroup GEMV) | 197-232 |
| `rms` / `rms_256_w4` | 233-247, 726-744 |
| `head_norm` (per-head RMS + RoPE in one pass) | 248-266 |
| `attention_partial` / `attention_partial_path_w53` (split-K online softmax + sinks) | 320-384, 619-693 |
| `attention_merge_vec4_w2` (partial reduction normalizer) | 276-318 |
| `route_dot_bf16`, `route_top` / `route_top_lane_scores` | 406-435, 696-725 |
| `gateup_all`, `down_all`, `expert_team_*` (expert-specialized workgroups) | 437-482, 746-820 |
| Fused body (embed → 49 layers → head) | 823-975 |
| Host packing + metadata + schedule | 983-1301 |
| KV capacity growth | 1302-1337 |

Notable techniques worth carrying over conceptually (not code):

1. **Weights packed once into per-4-layer banks** (`FullModel.__init__`, 1008-1109): alignment 256 B, order `weight, scales, biases`, all offsets stored in a 50×64 uint32 metadata table. This turns every layer into base+offset addressing and allows a single descriptor table.
2. **Bit-trick int4 dot** (`load16`/`dot16`, 171-196): x is pre-scaled by 1/16,1/256,1/4096 so packed `<u16` high nibbles can be multiplied without shifting; group scale/zero-point applied as `s*a + sum(x)*b` in fp32.
3. **Split-K attention with sinks** (`attention_partial_path_w53`): `parts` = 32 (short) up to 1024 (long context); each part keeps running max/denominator, sink is injected as `hi = sink`, `z = 1` in part 0. Merge renormalizes partials. Rounded (bf16) partials once `count >= 1024` to halve scratch bandwidth.
4. **Expert teams** (`expert_team_*`): four groups per routed expert role, eight for the shared expert; each team owns a slice of the 512/1024 intermediate. One device-wide handoff (`expert_team_handoff`) per layer between gate/up and down for the six routed experts + shared.
5. **Own-rank routing** (`kernels/rank_routing.py`): each expert's workgroup computes only its own rank of the top-6 selection, removing a serialized top-k over 256 logits (only group 0 publishes the normalized weights).
6. **Batched resident sampling** (`kernels/sampling.py`, `resident_block.py`): the megakernel loops 8 tokens internally and does the argmax in-kernel, feeding the next token back without a host sync.
7. **Trace buffer**: `trace[50][2048]` records the residual stream after every layer for parity debugging (body line 959).

### 3.3 KV cache

- Global (No-RoPE) layers: 14 layers × [2 K/V, 4 heads, 262144, 128] bf16 = **7.52 GB** at full context, paged in 64K-position blocks (max 4 blocks) — `reference/rec/engine/src/rec_mlx/cache.py:47-118`.
- Sliding layers: 513-slot ring buffers, 35 layers × ~1.05 MB = **36.8 MB** (`cache.py:153-258`).
- The MLX-native path can quantize KV to 8/8 bit (`MixedQuantizedKVCache`) but the shipped resident engine uses bf16 KV; the model manifest says `kv_format: bf16`. Start with bf16, keep int8 KV behind an env flag.

---

## 4. Hardware target: Arc Pro B70

| Property | Value |
|---|---|
| Architecture | `intel_gpu_bmg_g31` (Battlemage Xe2) |
| Xe cores / vector engines / HW threads | 32 / 256 / 2048 |
| Clock | 2800 MHz |
| Global memory | 34.24 GB |
| SLM per work-group | 128 KB |
| Max work-group | 1024 |
| Sub-groups | 16, 32 |
| Aspects | fp16, fp64, USM shared/device, **matrix (DPAS) = yes**, ext_intel_matrix = yes |
| Backends | Level Zero and OpenCL (we target Level Zero) |

Implications:

- 32 Xe cores maps well onto the 40-group megakernel structure: one Xe core can host one or two work-groups, with the same phase-barrier pattern implemented via `atomic_ref` + device-scope fences (no Metal `simdgroup` shuffle, but sub-group shuffles are available).
- DPAS exists but `joint_matrix` is off-limits. Valid access path in this toolchain: `sycl/ext/intel/esimd/xmx/dpas.hpp` (`xmx::dpas`), plus the SPIR-V `SubgroupMatrixMultiplyAccumulateINTEL` builtin. There is **no int4 DPAS**, so 4-bit weights must be expanded to bf16 (or int8 with s8 activations) inside the GEMM tile.
- 34.24 GB total. Budget: 22.57 GB weights + 7.52 GB KV at 262K + ~1 GB weights-staging/scratch = ~31.1 GB, leaving ~3 GB. At the practical mbench depths (≤32K) KV is only 0.94 GB, so there is comfortable headroom.
- 128 KB SLM is enough for a 64×64 bf16 B-tile (8 KB) plus A-tiles and double buffering; not enough for whole 2048-wide rows.

---

## 5. Kernel inventory to build

Grouped by stage. "M=1" = decode, "M=L" = prefill. All new code, no oneDNN.

### 5.1 Shared primitives

| Kernel | Math | Notes |
|---|---|---|
| `rmsnorm` | y = x·rsqrt(mean(x²)+1e-6)·w, bf16 | 2048 elements; one work-group, SLM reduction, then broadcast scale |
| `rmsnorm_head_rope` | per-head RMS over 128, then rotate-half RoPE | fuse into QKV epilogue; skip rotation on the 14 global layers; π/2 split layout (pair i with i+64) |
| `q4_gemv` | out[n] = Σ_g s_g·Σ_k x_k q_{n,k} + b_g·Σ_g x | M=1 core kernel; template on N, K, and optional expert base offset |
| `q8_embed` | row gather + `s*q + b` | 8-bit embedding; 1 row |
| `topk_router` | 256 sigmoid logits + bias → top-6, renormalize | single work-group, sub-group reductions; exact tie-break must match reference (`>` with lower index first) |
| `silu_mul` | silu(g)·u, bf16 | in MoE/dense epilogue |

`q4_gemv` variants needed: q/k/v fused (one pass over x, three outputs), gate/up fused, down, o_proj, lm_head (N=150144, memory-heavy with top-k/top-p fused or separate).

### 5.2 Decode (single token)

| Kernel | Shape / dispatch | Reference |
|---|---|---|
| embed | 1 row of 2048 | megakernel:852-859 |
| per layer: qkv GEMV (fused) | N=4096+512+512, K=2048 | `q4`, 197-232; body 873-888 |
| q/k norm + RoPE fused | 32+4 heads × 128 | `head_norm`, 248-266 |
| KV append | 8 heads × 128 bf16 | body 884-887; cache.py 97-118 |
| attention decode (split-K + sinks) | 32 heads, 4 KV heads, parts 32..1024 | 619-693, 276-318 |
| o_proj GEMV | N=2048, K=4096 | body 905 |
| post-norm + router | 2048 / 256×2048 bf16 | 406-435 |
| expert gate/up GEMV | 6 experts + shared: 6×(512×2048) + 1024×2048 | `expert_team_gateup`, 746-777 |
| expert down + weighted sum | 6×(2048×512) + 2048×1024 | `expert_team_down`, 778-798 |
| final norm + lm_head | N=150144, K=2048 | body 971-972 |

Two implementation strategies, decide after a spike:

1. **Per-phase kernel launches per layer** (~8 launches × 49 = ~390 launches/token). Simple, debuggable; at ~2-4 µs/launch this is 1-1.5 ms/token of pure launch overhead on top of ~2 ms of weight streaming. Acceptable initially.
2. **Persistent megakernel** like rec: one launch per token, 32 work-groups, device-scope barrier between phases, 8-token internal loop with in-kernel argmax. This is the end-state for throughput but needs a barrier/forward-progress design validated on Level Zero.

Recommendation: build (1) first, keep the fused-per-layer structure (one kernel per layer stage, not per projection), then promote to (2) behind an env var (e.g. `ARCAINE_FLINT_PERSISTENT=1`) per the AGENTS A/B rule.

### 5.3 Prefill (M = chunk, 2048/4096 tokens)

| Kernel | Notes |
|---|---|
| affine-int4 GEMM | C[M,N] = A[M,K]·dequant(W)[N,K]ᵀ; group 64; accumulate fp32, apply `s`/`b` per 64-chunk; bf16 and int8 variants both required, A/B behind an env var; DPAS via ESIMD `xmx::dpas` with int4→bf16/int8 expansion in SLM. int8 path quantizes activations per group (a deliberate numerical deviation) |
| causal/window mask + flash attention | sinks, GQA 8:1, bf16 KV, window 513 for sliding layers; online softmax; M up to 4096, N up to context |
| RoPE (only on sliding layers) | applied to Q,K after per-head RMSNorm |
| MoE routing + grouping | top-6 per token, sort/scatter to per-expert buckets, counts |
| grouped expert GEMM | 256 experts × [tokens_e, 2048]×[2048,512] and [tokens_e, 512]×[512,2048]; plus shared expert on all tokens; scatter-add back |
| lm_head | only last chunk's last token needed for generation; for scoring/logprobs, full vocab |

Prefill is compute-bound (weights reused across L tokens); decode is bandwidth-bound (1.19 GB of active weights per token). They need different kernels — do not try to share one GEMM.

### 5.4 Sampling

Reference defaults: `temperature 1.0, top_p 0.95, top_k 20` (`generation_config.json`), plus `do_sample: true`; rec also supports greedy, presence penalty, and grammars. For the first implementation: logits → CPU (300 KB/token) → temperature/top-k/top-p in C++, one sync per token. Add a GPU-side argmax/Gumbel path only with the persistent megakernel work.

---

## 6. Numerical parity checklist

The reference is **bf16-centric with fp32 accumulation**, and its rounding points are deliberate. Match these or logits will drift:

1. Projection inputs are bf16; `dot16` accumulates fp32; result cast to bf16 before the next stage.
2. Group scale/zero-point applied as `s·(Σ x·q) + b·Σx`, not by materializing bf16 weights (rec's `dot16`).
3. Router dot: accumulator is cast to bf16 (`a=float(bfloat16_t(a))`) before sigmoid — `kernels/mixed_bits.py:24-60`.
4. Expert weighting: `down_e` is cast to bf16, multiplied by bf16 weight, and the six-term sum rounds to bf16 after every addition (`mixed_bits.py`).
5. Router selection uses `sigmoid` scores for weights but `sigmoid + expert_bias` for ranking; ties prefer the higher index in `route_top` (420-435) but the lower index in `route_top_lane_scores` (696-725). With zero bias and bf16-rounded scores, ties are rare; still document which variant we match.
6. Sink semantics: softmax includes the sink logit exactly once, with weight 1.0, for every layer.
7. RoPE only on the 35 sliding layers; NeoX half-split; `theta = pos · base^(-pair/64)`.
8. Attention partials are fp32 below 1024 tokens and bf16-rounded above; this is a performance/accuracy trade-off, not a semantic requirement.
9. Residual adds and the running hidden state are bf16.
10. KV is bf16 (manifest), K and V written after RoPE and norm.
11. `trace[50][2048]` residual dump is the cheapest way to bisect a parity failure layer-by-layer. Build the equivalent from day one.

Verification targets:

- Layer-by-layer trace comparison against a PyTorch reference (see §8).
- Single-prompt logits, then greedy decode equality over 32+ tokens.
- mbench end-to-end at prompt depths 512, 1024, 2048, 4096 (AGENTS convention).

---

## 7. Memory and performance model

Decode reads per token (active weights only):

| Item | MB/token |
|---|---|
| Attention (q,k,v,o) ×49 | 360 |
| Dense layers 0-1 | 38 |
| 47 MoE layers: shared + 6 routed + router | 640 |
| lm_head (full) | 154 |
| **Total** | **≈1.19 GB** |

At a B70-class HBM/GDDR bandwidth (measure with a streaming kernel; ~600 GB/s assumed) the decode ceiling is ~2.0 ms/token ≈ **500 tok/s**, before random-access penalties across 22.6 GB. Expert GEMVs are tiny (512×2048 and 2048×512, ~0.5 MB each), so latency hiding and work distribution matter more than raw FLOPs. Realistic first target: 100-200 tok/s single batch; optimize toward 300+.

KV at full 262K context is 7.52 GB; the practical mbench depths are negligible (≤32K → 0.94 GB).

---

## 8. Gaps and prerequisites

1. **No in-repo numerical oracle.** `reference/transformers` (`b70d02fc72`) has no Flint model, and there is no MLX-capable machine to run `rec`. Per the operator's decision, parity is established by static analysis only: the MLX implementation, the shipped `sglang/flint_sglang-0.1.4` wheel, and `fa4-learned-sink-fix.patch` are the references. A PyTorch `FlintForCausalLM` is out of scope unless external golden logits are supplied.
2. **Tokenizer / chat template.** Resolved: reuse Arcaine's CPU tokenizer and Jinja chat-template plumbing; the sampler is new. The pure-Level-Zero mandate covers GPU compute only.
3. **Loader.** A new MLX-affine safetensors loader is required: U8 nibble packing, bf16 per-group scale/zero-point, 3-D per-expert tensors, 5 shards, no index file. Existing `quant_loader` targets GGUF/AWQ/NVFP4 and is not applicable.
4. **KV dequant option.** If long-context memory becomes tight, add 8-bit K/V (`MixedQuantizedKVCache`, model.py:40-105) behind an env flag; it changes attention math, so it must be A/B tested against bf16 KV.
5. **Build isolation.** The top-level CMake links `DNNL::dnnl` into every executable. The Flint target must compile without including or calling dnnl; add a dedicated pure-SYCL verification target so "no oneDNN" is mechanically checkable.
6. **Context limit.** Start at 8192 (mbench conservative path), then 32768, then 262144 once parity holds.

---

## 9. Proposed module scaffold

New directory `src/modeling/flint/` (nothing reused from existing architectures):

```
config.hpp            config.json subset, layer schedule, sanity checks
weights.hpp           tensor descriptor table (name -> shard, dtype, shape, quant recipe)
loader.cpp/.hpp       sharded safetensors reader + MLX-affine repack into device buffers
plan.hpp              per-layer metadata table (base offsets, flags: global/rope/moe/dense)
kernels/rmsnorm.hpp   rmsnorm, per-head rmsnorm + rope
kernels/q4.hpp        q4_gemv (+ fused qkv / gateup variants), q8 embedding gather
kernels/attention.hpp decode split-K attention with sinks, KV append, prefill flash attention
kernels/moe.hpp       router/top-6, expert gateup/down, shared expert
kernels/prefill.hpp   affine-int4 GEMM (bf16 + int8, ESIMD DPAS), grouped expert GEMM, masks
kernels/sampling.hpp  temperature/top-k/top-p
model.cpp/.hpp        prefill chunk loop, decode loop, trace buffer
register.cpp          architecture registration ("flint")
```

Buffers: USM device allocations for weights (one arena, 256 B-aligned, descriptor table), KV (global arena + per-sliding-layer rings), and a small activation/scratch arena. No re-packing at load time beyond alignment; the checkpoint layout is already the execution layout.

---

## 10. Remaining questions

Resolved (see §0): CPU tokenizer/chat-template reuse, per-layer decode first, both prefill precisions, bf16 KV, static-analysis oracle, full-path milestone including sampling and the server interface.

Remaining:

1. **Naming:** the earlier reply to "what should I call you?" was empty; I'll use "operator" until told otherwise.
2. **Static-analysis oracle:** with no MLX-capable hardware, what is the acceptance bar for numerical parity? Proposal: (a) per-kernel unit tests against hand-computed cases and MLX formulas, (b) int4 decode round-trip checks against tensor bytes read directly from the checkpoint, (c) residual-trace similarity only between our own prefill and decode paths, since there is no external golden run. Confirm this is enough, or supply reference logits from an external Flint run.
3. **Context target for the first full-path milestone:** 8192 (recommended) or 4096? Full 262K KV is 7.52 GB and fits, but bring-up should start smaller.
4. **Sampler determinism:** match `generation_config.json` defaults exactly (temperature 1.0, top-p 0.95, top-k 20) and expose greedy; do you also need the rec-style 8-token in-kernel sampling later, or is host sampling per token acceptable for v1?

---

## 11. Toolchain spike (done)

A minimal end-to-end Level Zero run was validated in the container on the B70, in `.scratchpad/l0_spike/`:

- **Headers/loader:** `/usr/include/level_zero/ze_api.h` (spec 1.17), `libze_loader.so`, `pkg-config level-zero`. The added `reference/level-zero` is loader v1.34 / spec 1.19.12 and is a useful API reference.
- **Device:** `Intel(R) Arc(TM) Pro B70 Graphics`, 256 EUs, queue group 0 = compute+copy+metrics (flags 0x7), group 1 = copy only.
- **Kernel flow:** module create → kernel create → group size 64 → USM **shared** allocations → launch → queue sync → verify. 1024/1024 correct.
- **Gotcha:** device-USM allocations plus pageable host `zeCommandListAppendMemoryCopy` produced an all-zero result in the first spike even though every call returned success. Shared USM worked immediately. Use pinned host memory or shared staging for uploads, and treat copy success as unverified until read back.
- **ocloc is absent** from the image (only `ocloc_api.h` in the intel-llvm source; `intel-opencl-icd`, `libigc2`, `libigdfcl2` are installed). It is not needed for the DPC++ route. If we ever want AOT binaries or standalone SPIR-V validation, the `intel-ocloc` package must be added.
- **SPIR-V fallback path works:** `clang++ -x cl -target spir64 -emit-llvm -c` + `llvm-spirv` (both at `/opt/intel/oneapi/2026.1/bin/compiler/`) emits a module that `zeModuleCreate` accepts. Kept as a diagnostic path, not the primary implementation route.
- **ESIMD DPAS is available:** `/opt/intel/oneapi/compiler/2026.1/include/sycl/ext/intel/esimd/xmx/dpas.hpp`. No `joint_matrix` usage required.

### 11.1 Experiment: ESIMD vs direct SPIR-V DPAS

Goal: find the fastest legal DPAS path per kernel before committing the prefill design.

- **Path A — ESIMD:** `sycl::ext::intel::esimd::xmx::dpas` (sub-group 16, bf16 or int8), DPC++ TU compiled with `icpx -fsycl` for `intel_gpu_bmg_g31`.
- **Path B — direct SPIR-V:** hand-written OpenCL C (or inline `__spirv_SubgroupMatrixMultiplyAccumulateINTEL`) compiled with `clang -x cl -target spir64 -emit-llvm` + `llvm-spirv`, built with `-spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate`, loaded as its own module.
- **A/B control:** same tile shapes (64×64×64 bf16 and 64×128×64 int8), same packing, same epilogue; select with `ARCAINE_FLINT_DPAS_PATH=esimd|spirv`. Do not mix paths inside one kernel.
- **Measure:** single-tile throughput, sustained GEMM throughput at prefill shapes (M=512/2048/4096), code size, register/scratch cost, and compile time. Run both through the same in-repo benchmark binary.
- **Decision rule:** default to ESIMD unless SPIR-V wins by a stable margin (≥5%) on the shapes that matter; keep the loser as a validated reference kernel for numeric comparison.

---

## 12. Next implementation steps

1. Scaffold `src/modeling/flint/` and a minimal CMake target that does not include dnnl.
2. New sharded safetensors + MLX-affine loader; verify nibble unpack, scales/zero-points, and the 3-D expert tensors by reconstructing a few rows and checking sizes/hashes against the raw bytes.
3. Implement and unit-test `rmsnorm`, `rmsnorm_head_rope`, `q4_gemv`, `q8_embed` at M=1.
4. Bring up layers 0-1 (dense, global, no RoPE) with a residual trace, then layer 2 (first MoE + sliding + RoPE), then all 49.
5. Add prefill GEMMs (bf16 and int8), KV append/masks, then wire sampling and the server interface.
6. Promote decode to the persistent kernel behind `ARCAINE_FLINT_PERSISTENT`, with an A/B benchmark against the per-layer path.

---

## 13. Implementation status

Implemented in `src/modeling/flint/` (all new code, no Arcaine GPU paths reused):

| File | Contents |
|---|---|
| `config.hpp` | config/generation parsing, layer schedule, rope/global predicates |
| `weights.hpp`, `loader.cpp` | independent sharded safetensors reader; MLX affine-int4 repack into one 256 B-aligned USM arena; nibble/scales/zero-point validation and device readback checks |
| `kernels.hpp` | decode M=1: int8 embedding, RMSNorm, q4 GEMV (fused qkv calls, residual epilogue), per-head RMS+RoPE, KV append, split-free online-softmax attention with sinks and GQA, router/top-6, expert gate/up/down, bf16-rounded expert combine, dense MLP, lm_head |
| `prefill.hpp` | batched prefill: embedding, RMSNorm, head norm+RoPE, KV append, 8-token weight-reuse affine GEMM in **bf16 and int8** (int8 quantizes activations per 64-group), router, shared/routed expert GEMMs with gather/scatter row indices, combine |
| `model.cpp/.hpp` | prefill/decode orchestration, KV caches (bf16, global arena + 513-slot rings), trace buffer, residual-stream semantics |
| `sampling.hpp` | new temperature / top-k / top-p / greedy sampler |
| `register.cpp` | `model_type: flint` registration for `arcaine_server` / `arcaine_mbench` |

Build targets: `flint_main` (self-tests + generation; links **no** DNNL — verified with `ldd`), and Flint is part of `ARCH_SRCS` so `arcaine_server` and `arcaine_mbench` dispatch to it.

Environment variables:

| Variable | Effect |
|---|---|
| `ARCAINE_FLINT_MAX_LAYERS=N` | load only layers `[0,N)` (fast bring-up) |
| `ARCAINE_FLINT_PREFILL=batched` | use the batched prefill path (default is token-wise) |
| `ARCAINE_FLINT_PREFILL_INT8=1` | int8 prefill GEMM path (requires batched prefill) |
| `ARCAINE_FLINT_DEBUG_SYNC=1` | per-stage `wait_and_throw` diagnostics |
| `ARCAINE_FLINT_TRACE=path` | dump the residual stream after every layer |

Verification performed:

- 8 kernel self-tests pass (`flint_main --tests-only`): q4 GEMV, RMSNorm, batched RMSNorm, embed+rmsnorm, decode attention with sinks, router top-6, bf16 prefill GEMM, int8 prefill GEMM.
- Loader validates tensor dtypes/shapes and verifies device readback of a norm and the embedding scales.
- Full 49-layer token-wise and batched prefill produce identical greedy tokens on the smoke prompt for both bf16 and int8 paths.
- End-to-end through `arcaine_server`: `{"role":"user","content":"The capital of France is"}` → `The capital of France is **Paris**.` with `prompt_tokens=7`.
- `arcaine_mbench --model .../flint-int4 -p 8 -n 8 -d 0` runs through the registry path.

Measured on the B70 (2K context):

| Path | Result |
|---|---|
| decode | ~20–21 ms/token (~47–50 tok/s), per-layer launches |
| prefill, token-wise, 52 tokens | 1.76 s |
| prefill, batched, 52 tokens | 1.24 s |
| prefill, token-wise, 7 tokens | 0.15–0.44 s |

Shared-code changes required by the checkpoint template:

- `third_party/minja/include/minja/minja.hpp`: added the standard Jinja `replace` filter. Minja only implemented `replace` as a method, so the Flint template threw and its capability probe wrongly classified it as requiring typed content.
- `src/common/preprocess/chat_template.cpp`: for `model_type == flint`, flatten OpenAI `[{type:text,text:...}]` content to a plain string before rendering. The server normalizes every message to typed parts, which this template prints verbatim.

Deferred (not implemented in this pass):

1. Persistent single-launch decode megakernel (per-layer path is the baseline; hook point is `FlintModel::step`).
2. ESIMD `xmx::dpas` vs direct SPIR-V DPAS A/B (the batched prefill currently uses vector FMA/integer ALU, not DPAS).
3. 8-bit KV cache behind an env flag (bf16 KV is implemented).
4. Batched flash-style attention (prefill attention calls the decode kernel per token) and a true tiled DPAS GEMM.
5. 32K / 262K context validation (2K exercised; ring/global KV sizing already honours `--max-seq`).

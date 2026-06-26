# vllm-xpu-kernels: SYCL/Xe Kernel Optimizations for the Qwen3.5 Path

**Source**: `github.com/vllm-project/vllm-xpu-kernels` @ `main` (fetched into this env).
**Companion to**: `QWEN3_5_ARCHITECTURE.md` — that doc describes the PyTorch reference and the
CUDA fast path; this one maps each Qwen3.5 component to an **existing Intel-GPU SYCL kernel**
and extracts the optimization techniques worth reusing in Arcaine.

> This is the most directly relevant prior art for Arcaine: it is SYCL/DPC++ + **oneDNN**,
> targets **Battlemage** (Xe2) and PVC, uses CUTLASS-SYCL (`cute`) with DPAS matrix engines,
> and already implements the exact GatedDeltaNet + MoE + paged-attention stack Qwen3.5 needs.
> Citations are `path:line` inside the vllm-xpu-kernels tree.

---

## Table of Contents

1. [What this repo is](#1-what-this-repo-is)
2. [Component → kernel map](#2-component--kernel-map)
3. [Gated DeltaNet (linear attention) — the crown jewel](#3-gated-deltanet-linear-attention--the-crown-jewel)
4. [Full attention: FMHA chunk-prefill + paged-decode](#4-full-attention-fmha-chunk-prefill--paged-decode)
5. [Fused QK-norm + RoPE, and M-RoPE](#5-fused-qk-norm--rope-and-m-rope)
6. [MoE: grouped GEMM + routing kernels](#6-moe-grouped-gemm--routing-kernels)
7. [oneDNN matmul integration (quantized GEMM)](#7-onednn-matmul-integration-quantized-gemm)
8. [Cross-cutting Xe2 techniques](#8-cross-cutting-xe2-techniques)
9. [Reuse guidance for Arcaine](#9-reuse-guidance-for-arcaine)

---

## 1. What this repo is

A PyTorch-custom-op extension that registers XPU kernels into the dispatcher
(`import vllm_xpu_kernels._C`). Written in **SYCL/DPC++ + oneDNN**, built with CMake/Ninja,
oneAPI 2025.3, PyTorch 2.12+xpu (`README.md`). Two relevant build dimensions:

- **Arch tiers** gated by `VLLM_XPU_ENABLE_XE2` (`gdn_attn_interface.cpp:9`) and runtime
  device probes `is_bmg_g21` / `is_bmg_g31` / `is_bmg` / `is_pvc`
  (`csrc/xpu/torch_bindings.cpp:116-125`). **BMG = Battlemage (Xe2)**, the Arcaine target;
  PVC = Ponte Vecchio (Xe-HPC). The high-perf kernels live under `csrc/xpu/*/xe_2/`.
- **Compile-time attention-variant selection** (`KERNEL_CONFIGURATION.md`): chunk-prefill and
  paged-decode kernels are templated on `(headsize, paged, causal, local, sink, lse)` /
  `(qgroup, headsize, pagesize, causal, local, sink)` and only the listed tuples are built.
  The "default" preset explicitly covers **Llama / Qwen / DeepSeek**. Lesson for a from-scratch
  port: these axes are the real specialization surface — head size, causal, **sliding-window
  ("local")**, **attention sink**, and **LSE** output are separate compiled variants, not
  runtime branches.

Registered op surface relevant to Qwen3.5 (`csrc/xpu/torch_bindings.cpp`):
`gdn_attention`, `cutlass_grouped_gemm_interface`, `apply_rotary_emb`,
`multimodal_rotary_embedding`, `deepseek_scaling_rope`, `topk_topp_sampler`, plus the oneDNN
GEMMs `fp8_gemm`, `fp8_gemm_w8a16`, `fp4_gemm`, `int4_gemm_w4a16`, `int4_gemm_w4a8`
(`:14-38`). Note: the **fused QK-norm+RoPE** kernel lives in the common `csrc/` bindings, not
the XPU-only table.

---

## 2. Component → kernel map

| Qwen3.5 component (from architecture doc) | vllm-xpu-kernels source | Technique headline |
|---|---|---|
| `Qwen3_5GatedDeltaNet` (linear attention) | `csrc/xpu/gdn_attn/` (whole dir) | conv1d + chunked delta rule on DPAS; decode/prefill/spec split |
| └ depthwise causal Conv1d (`causal_conv1d`) | `gdn_attn/xe_2/chunk_causal_conv1d_{,tiled_}xe2.hpp` | SLM-staged, tiled vs untiled by token count; fused l2norm |
| └ chunked gated delta rule | `gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp` (+ `gemm.hpp`) | `chunk_size=64`, CUTLASS-SYCL `cute` MMA tiles |
| └ q/k L2-norm | `gdn_attn/xe_2/l2norm_kernel.hpp` | fused into conv1d when dims fit, else standalone |
| └ single-token decode recurrence | `gdn_attn/gated_delta_rule.hpp` (native) | per-step state update, no chunking |
| `Qwen3NextAttention` (full attention, GQA) | `csrc/xpu/attn/xe_2/{chunk_prefill,paged_decode}*` | paged-KV FMHA, varlen, local/sink/lse |
| └ QK-norm + partial RoPE | `csrc/fused_qknorm_rope.cpp` | one kernel: RMSNorm(head) + RoPE |
| M-RoPE (3D interleaved) | `csrc/xpu/sycl/multimodal_rope.cpp` | section-cumsum lookup, vec4 |
| RoPE (NeoX/GPT-J) | `csrc/xpu/sycl/{apply_rotary_emb,rotary_embedding.hpp}` | |
| `Qwen3_5MoeSparseMoeBlock` experts | `csrc/xpu/grouped_gemm/` (+ `csrc/moe/`) | CUTLASS grouped GEMM, int4/mxfp4 experts |
| └ router top-k (softmax/sigmoid) | `csrc/moe/topk.cpp`, `grouped_topk.cpp` | subgroup-reduction scoring |
| └ permute / scales / scatter | `csrc/moe/{fused_moe_prologue,moe_align_sum_kernels,moe_gather,remap_hidden_states}.cpp` | gather/scatter without host sync |
| Quantized projections (w8a8/w8a16/w4a16/w4a8/fp4) | `csrc/xpu/onednn/*_gemm_*.h` + `onednn_matmul.cpp` | oneDNN matmul primitives + cache |
| Vision encoder | *(no Qwen3-VL-specific kernel)* | reuse FMHA (non-causal varlen) + oneDNN GEMM |

Two gaps to note up front: (a) there is **no DeltaNet/Mamba backward** here (inference repo,
same as the CUDA libs); (b) there is **no vision-tower-specific kernel** — the ViT would reuse
the generic varlen FMHA + GEMMs.

---

## 3. Gated DeltaNet (linear attention) — the crown jewel

`gdn_attention` (`gdn_attn/gdn_attn_interface.cpp:16-66`) is a near-1:1 match to the Qwen3.5
`Qwen3_5GatedDeltaNet.forward`. Its tensor contract names the exact reference tensors:
`projected_states_qkvz`, `projected_states_ba`, `conv_state` (`[batch, width-1, conv_dim]`),
`ssm_state` (= the recurrent state, `[batch, num_v_heads, head_v_dim, head_k_dim]`),
`conv_weights`, `A_log`, `dt_bias`, `activation` (`:16-66`). It even accepts the **interleaved
`qkvz`/`ba` layout** (Qwen3-Next packing) and does the `fix_query_key_value_ordering` split
**inside** the conv1d kernel, rather than as separate projections.

### 3.1 Three execution regimes (matches the reference's cache-state branching)

The interface splits work into **prefill / decode / spec-decode** and chooses a path
(`:279-525`):

- **Prefill (chunked), `num_prefills>0`, XE2**: `chunk_causal_conv1d_*_xe2` →
  (optional standalone `l2norm`) → `chunk_gated_delta_rule_xe2` (`:410-518`). This is the
  `torch_chunk_gated_delta_rule` analogue.
- **Decode / small (`NATIVE_LAUNCHER`)**: `gdn::causal_conv1d` + `gdn::gated_delta_rule`
  native kernels (`:342-402`) — the per-step recurrence (`torch_recurrent_gated_delta_rule`
  analogue).
- **Speculative decode (`spec_token>0`)**: same two native calls but driven by
  `spec_*` index tensors and `num_accepted_tokens` (`:279-339`) — a vLLM feature with no
  transformers analogue; safe to ignore for a first Arcaine port.

The chunk-vs-native switch is purely a **token-count heuristic**:
`use_tiled = non_spec_token >= conv1d_tile_size` and the chunk path only runs when
`num_prefills>0` (`:441, 410`). Decode (1 token/seq) always takes the native recurrence.

### 3.2 Causal Conv1d — two SLM-staged variants

`chunk_size_xe2 = 64` (`gdn_attn_utils.h:8`) — **identical to the reference `chunk_size=64`**.

**Tiled** (`chunk_causal_conv1d_tiled_xe2.hpp`): the high-throughput prefill conv.
- `wg_size=64` (4 subgroups × `sub_group_size=16`), `elems_per_item=4` →
  **`feats_per_wg = 256` features per workgroup** (`:30-32, 123`).
- Grid `(num_tiles * num_feat_chunks, num_k_heads)`, tile length `conv1d_tile_size=8`
  (`:10, 18`). The conv channels (8192 in Qwen3.5) are split into 256-wide feature chunks so
  a WG holds its slice in **SLM**: `slm_data_elems = (TileT + Width - 1) * feats_per_wg`
  staging the conv window + left-context halo, plus `slm_meta` (5 ints) and a `norm_slm`
  region (`:119-127`). This is the in-kernel realization of the reference's "prepend conv_state
  / left-pad" logic — the halo of `Width-1` rows is the rolling `conv_state`.
- **Fused l2norm**: when `2*head_k_dim <= feats_per_wg (256)` the Q/K L2-norm is folded into
  the conv epilogue (`fuse_l2norm`, interface `:439-443, 469`); otherwise a standalone
  `l2norm` kernel runs (`:498-500`). For Qwen3.5 `head_k_dim=128` → `2*128=256 ≤ 256`, so
  **l2norm fuses** in the tiled path. Worth replicating: it removes a full-tensor round trip.
- Width is a compile-time template (`TILED_WIDTH_DISPATCH` for width 1–5,
  `chunk_causal_conv1d_tiled_xe2.hpp:834-849`) and `reorder_input` is a second template axis
  (`:855-859`) — kernel specialization over the kernel size (4 for Qwen3.5) and input layout.

**Untiled** (`chunk_causal_conv1d_xe2.hpp`): `sub_group_size=32`, `elems_per_item=4`, entire
QKV of a token-group in one WG (`:14-15`); used for small token counts where tiling overhead
dominates. l2norm always fuses here (interface `:442-443`).

### 3.3 Chunked gated delta rule on DPAS

`chunk_gated_delta_rule_kernels_xe2.hpp` is CUTLASS-SYCL (`cute`) (`:33-53`). Key points for a
port:

- **On-device decay**: `chunk_prepare_kernel` computes
  `g = -exp(A_log) * softplus(a + dt_bias)` in fp32 (`act_softplus` with the `>20` linear
  cutoff, `:44-50`; applied `:78-110`) — exactly the reference formula and the same fp32
  requirement flagged in the architecture doc.
- **Four GEMM tile policies** (`:19-42`) map to the four matmuls of the chunked algorithm:
  - `compute_A` = `64x64x32` SG layout `2x2` — the intra-chunk `K·Kᵀ` attention matrix.
  - `inverse` = `16x16x16` — the lower-triangular `(I − tril)⁻¹` solve (the reference's
    sequential forward-substitution loop, done here as a small dense inverse).
  - `compute_wu` = `64x64x32` `2x1` — the `W`/`U` (k_cumdecay, v_new) products.
  - `fwd_o` = `64x64x32` `4x2` — the forward output `q·state + attn·v_new`.
- DPAS via `TiledMMA` + `partition_sg_fragment_{A,B}` and **block-2D loads with software
  prefetch depth 3** and `barrier_arrive`/`barrier_scope` double-buffering
  (`gemm.hpp:60-130`). This is the standard cute pipeline; the takeaway is that the delta-rule
  inner products are cast as **bf16/fp16 DPAS GEMMs** with an fp32 accumulator, not scalar
  loops.
- Chunk padding: the interface pads work to a multiple of `chunk_size_xe2` per sequence
  (`padding_size = batch_size*(chunk_size_xe2-1)`, allocates zero-padded q/k/v, `:411-433`) and
  carries `b`/`a` as **fp32** `[num_v_heads, tokens]` (transposed vs q/k/v) (`:428-433`).
- **No host gather/scatter**: an optional `token_indx_ptr` lets the kernels read
  `mixed_qkvz`/`mixed_ba` and write `z`/`core_attn_out` at interleaved global slots directly
  (`:404-409, 414-417`).

### 3.4 Native (decode) recurrence

`gated_delta_rule.hpp` (833 lines) + `causal_conv1d.hpp` (1144 lines) implement the
single-token path: `causal_conv1d_update`-style in-place ring-buffer conv-state update, then
the per-step `S = S·exp(g); kv=(S·k).sum; Δ=(v−kv)β; S+=k⊗Δ; out=(S·q).sum`. These are the
files to port first for a correctness baseline (they're scalar/subgroup, no cute/DPAS).

---

## 4. Full attention: FMHA chunk-prefill + paged-decode

`csrc/xpu/attn/xe_2/` is a CUTLASS-SYCL flash-attention with **paged KV cache**. Signature
(`fmha_xe2.h`, `cutlass_chunk_prefill_xe2`):
`query[seq_q,heads,hd]`, `key_cache/value_cache[num_block,block_size,heads,hd]`, `block_table`,
`cu_seqlens_q/k`, `k_scale/v_scale` (fp8 KV), `sm_scale`, `sm_sink`, `window_size_left/right`,
flags `is_varlen/paged/causal/local/sink`, optional `softmax_lse` (`fmha_xe2.h:3-26`).

What maps to Qwen3.5 full-attention layers:
- **GQA** via `qgroup` (paged-decode config axis, `KERNEL_CONFIGURATION.md`); Qwen3.5 dense
  is 16 Q : 4 KV.
- **`is_local` + `window_size_left/right`** is sliding-window attention — not used by Qwen3.5
  (its full layers are global) but present for Qwen2.5/Gemma-style models.
- **`is_sink` + `sm_sink`** = attention sinks; **`softmax_lse`** = log-sum-exp output for
  splitting/merging (`csrc/attention/merge_attn_states.cpp` merges partial softmax states).
- **fp8 paged KV** via `k_scale`/`v_scale` — dequant in the mainloop.
- Files: `kernel/chunk_prefill_kernel.hpp`, `kernel/paged_decode_kernel.hpp`,
  `collective/chunk_prefill_{mainloop,epilogue,scheduler}.hpp` — the usual cutlass collective
  split (tile scheduler + mainloop + epilogue).

Note the output gate `attn * sigmoid(gate)` and QK-norm that Qwen3-Next adds are **not** inside
this FMHA — they're applied by the caller (gate) and by `fused_qknorm_rope` (norm). A port must
keep those outside the attention kernel, as the reference does.

---

## 5. Fused QK-norm + RoPE, and M-RoPE

`csrc/fused_qknorm_rope.cpp` is a direct match to the Qwen3 full-attention preamble (QK-norm +
RoPE), ported from a CUDA kernel (`:1-3`). One kernel, one pass over the fused QKV buffer:
- `SUB_GROUP_SIZE=32`, `NUM_ELEMS_PER_THREAD = head_dim/32`, requires `head_dim % 64 == 0`
  (`:24-27`) — Qwen3.5 `head_dim=256` ✓.
- One **subgroup per (token, qk-head)**: `global_sg_id` decodes to `(token_idx, head)` and
  splits Q vs K by `local_head_idx < num_heads_q` (`:55-66`). Each subgroup RMS-normalizes its
  head (lane holds `head_dim/32` elems, subgroup-reduces the sum-of-squares) then applies RoPE
  from `cos_sin_cache[position_ids]` — fusing the two head-dim reductions the reference does
  separately (`q_norm`, then `apply_rotary_pos_emb`). `IS_NEOX` is a template axis.
- V heads are skipped (only Q/K normalized+rotated), matching the reference.

**M-RoPE** `csrc/xpu/sycl/multimodal_rope.cpp` matches the Qwen3.5 interleaved 3D RoPE:
- `MROPE_MAX_SECTIONS=4`; precomputes cumulative `section_end[]` from `mrope_section`
  (`:11-51`) so each rotary index does an O(sections) lookup to pick the temporal/height/width
  frequency band — the kernel-side version of `apply_interleaved_mrope`.
- `VEC_SIZE=4`, GPT-J/NeoX template; processes 2 rotation offsets per vec4 (`:53-55`).
`deepseek_scaling_rope.cpp` is the YaRN/scaling variant (not needed for Qwen3.5 defaults).

---

## 6. MoE: grouped GEMM + routing kernels

For `Qwen3_5MoeSparseMoeBlock` (256 experts, top-8, shared expert):

**Grouped GEMM** `cutlass_grouped_gemm_interface` (`grouped_gemm/grouped_gemm_interface.h`):
```
ptr_A, ptr_B, ptr_scales?, ptr_bias?, ptr_D, rows_per_expert, N, K, num_experts,
is_B_int4, is_B_mxfp4
```
- One batched/grouped GEMM over **all experts**, driven by `rows_per_expert` (the token counts
  after routing) — this is the fused replacement for the reference's Python "loop over hit
  experts." Implemented in `xe_2/grouped_gemm_xe2.*` (cutlass) and a `xe_default` fallback.
- **Quantized experts built in**: `is_B_int4` / `is_B_mxfp4` with `ptr_scales` — i.e. the 3D
  packed expert weights can be int4 or MXFP4, dequantized in the GEMM. The cutlass collective
  (`collective/gemm/moe_*`) carries a dtype policy + callbacks + array-mma + tile scheduler
  specialized for the ragged per-expert tile counts.

**Routing / scatter kernels** (`csrc/moe/`):
- `topk.cpp` — softmax **and** sigmoid scoring (`sigmoid_typed`, custom transposable softmax,
  `:17-26`), top-k via subgroup reductions (`reqd_sub_group_size(WARP_SIZE)`, `:43-44`). Covers
  both Qwen (softmax→topk→renorm) and DeepSeek (sigmoid) routers.
- `grouped_topk.cpp` / `fused_grouped_topk` — group-limited routing (DeepSeek-V2/3 expert
  groups); Qwen3.5 uses plain top-k but the kernel is a superset.
- `fused_moe_prologue.cpp` — fuses permutation + scale gather into a workspace
  (`token_selected_experts`, `token_final_scales`, `:6-11`) so the GEMM reads contiguous
  per-expert blocks.
- `moe_align_sum_kernels.cpp` (align token counts to tile size), `moe_gather.cpp`,
  `remap_hidden_states.cpp`, `init_expert_map.cpp` — the gather/scatter plumbing that keeps the
  whole route→GEMM→unroute pipeline on-device.

The shared expert and its `sigmoid(shared_expert_gate)` gate have no dedicated kernel — they're
a normal dense GEMM + activation (reuse oneDNN matmul + the SiLU-mul activation kernels in
`csrc/activation.cpp` / `csrc/quantization/fused_kernels/fused_silu_mul_*`).

---

## 7. oneDNN matmul integration (quantized GEMM)

`csrc/xpu/onednn/` is the most directly liftable piece for Arcaine, since Arcaine is already
oneDNN-based. `onednn_matmul.cpp` + headers expose:
- `fp8_gemm` (w8a8), `fp8_gemm_w8a16`, `fp4_gemm` (w4a4 / NVFP4-style with A_scale+B_scale),
  `int4_gemm_w4a16`, `int4_gemm_w4a8` (`torch_bindings.cpp:14-38`,
  headers `fp8_gemm_w8a8.h`, `fp8_gemm_w8a16.h`, `fp4_gemm_w4a4.h`, `int4_gemm_w4a16.h`,
  `int4_gemm_w4a8.h`).
- 2D/3D inputs, 2D weights; int4 weights required in **NT format** (`onednn_matmul.cpp:23-28`);
  default out dtype fp16 (`:48`).
- `onednn_runtime.{h,cpp}` + `lru_cache.h` — a **primitive/descriptor LRU cache** so repeated
  GEMM shapes don't re-create oneDNN primitives. `onednn_ext.h` wraps engine/stream from the
  SYCL queue.

This is exactly the W4A16/W8A16/NVFP4 matmul surface Arcaine already wants (the README lists
NVFP4 DiffusionGemma support) — the descriptor-cache pattern and the scale/zero-point wiring
are reusable for Qwen3.5's quantized projections and MoE experts.

---

## 8. Cross-cutting Xe2 techniques

Patterns that recur and are worth standardizing in Arcaine's kernel layer:

1. **CUTLASS-SYCL (`cute`) for all matmul-shaped work** — DPAS `TiledMMA`, block-2D copies,
   software prefetch (depth 3), `barrier_arrive` double-buffering (`gdn .../gemm.hpp:97-130`).
   Reused for delta-rule, FMHA, and grouped GEMM.
2. **Compile-time specialization over runtime branching** — kernel tuples templated on
   head size / width / causal / local / sink / lse / dtype / NeoX, selected by config file
   (`KERNEL_CONFIGURATION.md`; conv width dispatch `chunk_causal_conv1d_tiled_xe2.hpp:834-849`).
3. **SLM staging with explicit halos** — conv windows + left-context loaded to shared local
   memory; `(TileT + Width - 1) * feats_per_wg` sizing (`:124`).
4. **Subgroup as the reduction unit** — `reqd_sub_group_size(16|32)`; one subgroup per
   (token, head) for norm/RoPE; subgroup reductions for top-k and RMS.
5. **Fuse adjacent elementwise reductions into the producer** — l2norm folded into conv1d when
   dims fit (`gdn_attn_interface.cpp:439-443`); QK-norm folded into the RoPE kernel.
6. **Index-tensor-driven gather/scatter, never host sync** — `token_indx`, `query_start_loc`,
   `state_indices`, `rows_per_expert`, `block_table` thread the irregular work entirely on
   device.
7. **fp32 islands preserved** — decay `g`, softplus, RMS variance, router softmax all fp32 even
   when activations are bf16/fp16 (`chunk_gated_delta_rule_kernels_xe2.hpp:44-50, 78-79`).
8. **oneDNN primitive LRU cache** for shape-stable GEMMs (`onednn/lru_cache.h`).
9. **Token-count heuristics pick the kernel** — tiled vs untiled conv, chunk vs native delta
   rule (`gdn_attn_interface.cpp:441`), so prefill and decode hit different code without a
   separate API.

### 8.1 Matrix-engine (DPAS) programming model — and the `joint_matrix` question

**Verified by reading the matrix kernels in full** (`gdn_attn/xe_2/gemm.hpp` 1–490,
`chunk_gated_delta_rule_kernels_xe2.hpp` 1–1631, `grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp`
1–372, `attn/xe_2/collective/chunk_prefill_mainloop.hpp`): **none of this repo's kernels call
the SYCL `joint_matrix` API directly.** Every matmul-shaped kernel (GDN delta rule, grouped
GEMM, FMHA) drives the Xe **DPAS** (Dot-Product-Accumulate-Systolic) units through the **`cute`
MMA abstraction from SYCL-TLA** — Intel's CUTLASS-for-SYCL, pulled at build time via
FetchContent from `github.com/intel/sycl-tla` (`CMakeLists.txt:339-371`).

**Important boundary**: the DPAS *atom* (`XE_DPAS_TT`), the 2D-block copy atoms, and
`cute::gemm` are **defined in the sycl-tla dependency, not in this repo** (sycl-tla is not
vendored in the source tree — it is fetched during the build). So whether the final lowering
emits the SYCL `joint_matrix` intrinsics or targets the DPAS instruction through another path
is a property of sycl-tla that **cannot be determined from this repository alone**. What is
verifiable here: the application kernels program the matrix engine exclusively through cute
atoms. The stack is:

```
cute::gemm(mma, A, B, C)                       # issue
   └─ TiledMMA  =  TiledMMAHelper<             # tiling of the atom over the WG
          MMA_Atom< XE_DPAS_TT<8, float, Elt> >,   # the DPAS atom
          Layout<WGTile>,                      # e.g. Shape<_64,_64,_32>
          SGLayout >::TiledMMA                 # e.g. Shape<_2,_2,_1> subgroups
```

Concretely (`chunk_gated_delta_rule_kernels_xe2.hpp:1275`,
`grouped_gemm_xe2_interface.hpp:93`):
```cpp
auto op = XE_DPAS_TT<8, float, Element_non_CV>{};        // DPAS atom
using MMA = TiledMMAHelper<MMA_Atom<decltype(op)>,
                           Layout<WGTile>, SGLayout>::TiledMMA;
```
- **`XE_DPAS_TT<8, float, Elt>`** is the cute "atom" describing one DPAS instruction: systolic
  **depth 8** (8 K-steps accumulated per issue), **fp32 accumulator** (`float`), and bf16/fp16
  **input** element `Elt` (`Element_non_CV` = the non-const operand type). `TT` = both operands
  row/transposed-major as the atom expects. This is the source-level handle to the same Xe
  hardware instruction the SYCL `joint_matrix` API lowers to — cute just exposes it as a
  typed atom rather than through `joint_matrix_load/mad/store`.
- **`TiledMMAHelper<atom, WGTile, SGLayout>`** replicates the atom across a workgroup: `WGTile`
  is the MNK macro-tile (`Shape<_64,_64,_32>` for the GDN compute GEMMs) and `SGLayout` is how
  many **subgroups** cover M×N (`2x2`, `2x1`, `4x2`, or `1x1` for the 16³ inverse — see the
  four policies in §3.3). `size(mma)` then gives the workgroup thread count used to launch
  (`chunk_gated_delta_rule_kernels_xe2.hpp:1318`).
- **Per-thread fragments**: `thr_mma.partition_sg_fragment_A/B(...)` allocate the register
  fragments each work-item feeds to the atom; `cute::gemm(mma, tCrA, tCrB, tCrC)` issues the
  DPAS over them (`gemm.hpp:88-92, 133`). The accumulator `tCrC` lives in registers across the
  K-loop.
- **Operand loads are 2D block loads with VNNI packing**, not scalar loads: the copy atoms are
  `XE_LOAD_2D` / `XE_LOAD_2D_VNNI` and `XE_2D_U16x8x16_LD_N` / `XE_2D_U32x8x16_LD_N`
  (`U16` = bf16/fp16 operands, `U32` = fp32 accumulator store `XE_2D_U32x8x16_ST_N`). VNNI
  pre-interleaves the K dimension into the layout the systolic array consumes. These are built
  via `get_block_2d_copy_A/B/C/D(mma, tensor)` (`gemm.hpp:81-82, 162`).
- **Software-pipelined K-loop**: `make_block_2d_prefetch` issues 2D prefetches `prefetch_dist=3`
  tiles ahead, with `barrier_arrive(scope)` / `barrier_wait(scope)` double-buffering around
  `copy → reorder → cute::gemm` (`gemm.hpp:97-136`). `reorder(tArA, tCrA)` shuffles the loaded
  fragment into the atom's expected lane layout before the MAD.
- **int4 experts** go through a specialized caller `XE_GEMM_4BITS_CALLER(GroupSize)` for
  group sizes 32/64/128/256 (`grouped_gemm_xe2.hpp:178-196`) — same DPAS atom, with an
  in-fragment dequant of the 4-bit B operand against `ptr_scales`.
- **Arch gating with a correctness workaround**: the chunked delta rule launches **five**
  kernels — `chunk_prepare` (decay cumsum), `chunk_compute_A`, the triangular **inverse**,
  `chunk_compute_wu`, `chunk_fwd_o` (`kernel_launcher`, `:1249-1501`). The inverse has **two
  implementations chosen at runtime**: on Battlemage (`is_bmg()`) a blocked 4×4 DPAS inverse
  using 16×16 MMA tiles (`chunk_inverse_opt_kernel`, `:388`, `:1348`); on PVC a **native
  non-MMA SLM kernel** because "PVC has acc issue of sycl tla" for this op (`:1382-1411`). A
  port must keep the same numerically-stable fallback option.
- **Register-sharing fusions** (read in `gemm.hpp`): `gemm_TTS_fused_2A` (`:381-489`) issues
  two DPAS ops sharing one B fragment in registers — used in `chunk_fwd_o` to compute `W×S` and
  `Q×S` from a single `S[dv]` load (`:1106-1123`); `gemm_TTS_k_multi` (`:282-379`) folds a
  per-K scale vector (`beta`/`g`) into the A fragment before the MAD.
- **FMHA uses two DPAS tilings**: `chunk_prefill_mainloop.hpp` is templated on `TiledMMAQK_`
  (Q·K) and `TiledMMAPV_` (P·V) separately (`:73-74`), the classic flash-attention two-GEMM
  pipeline with "P in registers" (`XeDefault<Stages>`, `:46-47`), synchronized with raw
  inline-asm split barriers `sbarrier.signal/wait` + `lsc_fence.ugm` (`:53-62`).
- **Grouped-GEMM tile policy is chosen by average tokens-per-expert**: `A_avg_M = total_M /
  num_experts` selects `w{4,8,16}a16_policy_m_{8,16,32}` (and by N for w16a16)
  (`grouped_gemm_xe2_interface.hpp:287-368`) — small per-expert batches (MoE decode) use
  M-skinny tiles. int4 packs 2 weights/byte (`B_K = size(2)*2`, `:203-205`).

**Why it matters for Arcaine**: if you want the same DPAS throughput you have two options —
(a) depend on CUTLASS-SYCL `cute` and reuse these `XE_DPAS_TT` atoms + tile policies verbatim
(what this repo does), or (b) write the matmuls against the SYCL
`sycl::ext::oneapi::experimental::matrix::joint_matrix` API yourself
(`joint_matrix_load` with VNNI layout → `joint_matrix_mad` → `joint_matrix_store`), which
targets the identical DPAS instruction but is lower-level and you re-implement the tiling /
prefetch pipeline by hand. The repo deliberately chose (a); the `WGTile`/`SGLayout` policies in
§3.3 are the tuned configuration you'd otherwise have to rediscover. Note oneDNN (option in
§7) hides DPAS entirely behind its matmul primitive — for plain projections that's the least
work.

---

## 9. Reuse guidance for Arcaine

Priority order for a Qwen3.5 bring-up, given Arcaine = SYCL + oneDNN + Battlemage (Xe2):

1. **Lift `csrc/xpu/onednn/` almost verbatim.** It's already SYCL+oneDNN and gives w8a8/w8a16/
   w4a16/w4a8/fp4 GEMM + a primitive cache — covers every Qwen3.5 linear projection and the MoE
   experts (int4/mxfp4). Smallest effort, highest coverage.
2. **Port the GDN native path first** (`gated_delta_rule.hpp` + `causal_conv1d.hpp`) for a
   correct, un-tuned linear-attention baseline; validate against the transformers fallback
   kernels at `chunk_size=64`.
3. **Then adopt the XE2 chunked GDN** (`chunk_causal_conv1d_tiled_xe2.hpp` +
   `chunk_gated_delta_rule_kernels_xe2.hpp` + `gemm.hpp`) for prefill throughput — this is the
   single biggest perf lever and the hardest to get right (the four-GEMM chunk decomposition and
   the fused l2norm). The `cute` tile policies (§3.3) are a ready-made starting config.
4. **FMHA**: reuse `attn/xe_2` chunk-prefill + paged-decode for the 1-in-4 full-attention
   layers; you only need the `(head=256, causal=true)` variant (no local/sink) for Qwen3.5.
5. **MoE routing + grouped GEMM**: `csrc/moe/*` + `grouped_gemm/xe_2` give the full
   route→GEMM→unroute pipeline with quantized experts; map the router to the softmax+renorm
   `topk.cpp` path.
6. **RoPE/norm**: `fused_qknorm_rope.cpp` (full-attn preamble) and `multimodal_rope.cpp`
   (M-RoPE `[11,11,10]`) drop in directly.

Caveats:
- These are **inference-only** (no backward) — same limitation as the architecture doc.
- **Spec-decode** paths (`spec_*` tensors) are vLLM-specific; skip for Arcaine v1.
- The kernels assume vLLM's **paged KV + cu_seqlens varlen packing** and ragged index tensors;
  Arcaine must produce the same `query_start_loc` / `state_indices` / `block_table` layouts or
  adapt the launchers.
- Licensing: kernel headers are **BSD-3-Clause** (Intel) (e.g.
  `chunk_gated_delta_rule_kernels_xe2.hpp:1-31`); the repo top-level is Apache-2.0. Check
  per-file headers before vendoring.

---

**Document version**: 1.0
**Reference**: `vllm-project/vllm-xpu-kernels@main`, `csrc/xpu/{gdn_attn,attn,grouped_gemm,onednn,sycl}`, `csrc/moe`, `csrc/fused_qknorm_rope.cpp`
**Companion**: `QWEN3_5_ARCHITECTURE.md`

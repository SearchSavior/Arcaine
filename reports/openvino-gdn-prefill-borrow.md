# OpenVINO GPU plugin GatedDeltaNet vs vllm-xpu-kernels — prefill-attack report

Date: 2026-08-08
Scope: OpenVINO's Intel GPU plugin `gated_delta_net` / `paged_gated_delta_net`
OpenCL kernels (Qwen3.5 linear attention), compared against the vllm-xpu-kernels
chunked Xe2 suite and Arcaine's `gated_delta_chunk*`, to find borrowable
patterns for attacking **prefill** in Arcaine.

## Provenance

| Repo | Checkout | Path |
| --- | --- | --- |
| OpenVINO | `0f453eb8` (shallow clone) | `reference/openvino` |
| vllm-xpu-kernels | `de8eae1` | `reference/vllm-xpu-kernels` |
| Arcaine | working tree | `src/modeling/qwen3_5_moe/kernels/gated_delta_chunk*` |

---

## 1. OpenVINO GPU plugin inventory (3 kernels, all OpenCL)

```
src/plugins/intel_gpu/src/graph/impls/ocl_v2/
  gated_delta_net_ref.cl        (362 l) sequential recurrence, non-paged
  paged_gated_delta_net_ref.cl  (191 l) sequential recurrence + paged state table
  paged_gated_delta_net_opt.cl  (278 l) vectorized (K_VEC_SIZE 8) paged variant
```

Host sides: `gated_delta_net_ref.cpp`, `paged_gated_delta_net.cpp` (ref+opt
generators, JIT constants, dispatch data). Graph primitives:
`src/graph/{gated_delta_net,paged_gated_delta_net}.cpp`; core op defs in
`src/core/src/op/*.cpp` + `src/core/dev_api/openvino/op/paged_gated_delta_net.hpp`.
Selection: `src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp:441`
recognizes `GatedDeltaNet`/`PagedGatedDeltaNet`; ops factory in
`src/plugins/intel_gpu/src/plugin/ops/paged_gated_delta_net.cpp`.

### 1.1 The kernel algorithm (both paged variants; `paged_gated_delta_net_opt.cl`)

Grid: `global = (sequences, v_heads, v_blocks × subgroup_size)`,
`local = (1,1,subgroup_size)`. One WG per (sequence, head, v-block), where
`V_BLOCK_SIZE = 4` (fixed in `get_v_block_size`, `paged_gated_delta_net.cpp:16-19`)
and `SUBGROUP_SIZE = 16` on Xe2/BMG (`:29-34`).

Per WG, per v-block (4 value columns):
- Load the **recurrent state** `[K_HEAD_DIM × V_HEAD_DIM]` block into registers:
  `state[V_BLOCK_SIZE][K_VEC_COUNT]` with `K_VEC_SIZE=8` (f16/f32, `:36-67`),
  i.e. 4×1 = 4 float8 = 32 floats/lane → the v-block×K-slice tile.
- `K_VEC_COUNT = (K_HEAD_DIM/SUBGROUP_SIZE)/K_VEC_SIZE = (128/16)/8 = 1`
  for Qwen3.5 — so each lane holds the full 128-K vector as 1 float8 per v-col.
- Sequential token loop (`:159`): for each token load q/k (l2norm fused),
  `g=exp(gate)`, `beta`; then the standard gated-delta recurrence per v-col:
  decay state by g → `h_k = Σ state·k` (subgroup-reduced) → `update = (v−h_k)·β`
  → state += k·update → `out = Σ state·q` (subgroup-reduced).
- **Block-boundary state swap** (`:257-277`): every `cache_interval` tokens
  (or at sequence end), write the current state to the block in the paged state
  table pointed to by `block_indices[block_begin+slot]` and load the next one.

### 1.2 The paged state design (the key differentiator)

`recurrent_state_table [num_blocks, v_heads, v_head_dim, k_head_dim]` is a
**block table**. Per sequence:
- `subsequence_begins[seq..seq+1]` — token range for this sequence
- `block_indices_begins[seq]` — index into `block_indices`
- `block_indices[...]` — ordered list of state-table blocks for this sequence
- `past_lens[seq]`, `cache_interval[seq]` — how many tokens were cached per
  block and the interval between state checkpoints

The kernel: `prev_nums = past_len % interval` → `tokens_to_next_boundary` →
process tokens in chunks, **write-back state at every interval boundary to the
next block**. This is a **state-checkpoint paging** scheme: the SSM state is
persisted/restored across KV-cache eviction, prefill-continuation, and decode
steps without recomputation.

### 1.3 Host dispatch / JIT

`PagedGatedDeltaNetOptGenerator::get_jit_constants` (`paged_gated_delta_net.cpp:76-102`)
bakes `K_HEAD_DIM/V_HEAD_DIM/V_BLOCK_SIZE/SUBGROUP_SIZE/K_VEC_SIZE/FUSE_QK_L2NORM/
SCALE_FACTOR` into the kernel at compile time (no runtime branching). Dispatch
data (`:120-184`) passes strides/offsets as scalars. Ref vs opt selected by
`m_force_ref_path` (opt = `K_VEC_SIZE=8`).

---

## 2. Key architectural comparison (OpenVINO vs vllm-xpu vs Arcaine)

| Aspect | OpenVINO GPU plugin | vllm-xpu-kernels | Arcaine (current) |
| --- | --- | --- | --- |
| Language | OpenCL (.cl → SPIR-V) | SYCL + cute/DPAS (Xe2) | SYCL (scalar SIMT + hand SPIR-V DPAS + oneDNN hybrid) |
| Prefill strategy | **Sequential** token loop, per (seq,head,v-block) | **Chunked** (C=64): 5-kernel DPAS pipeline | **Chunked**: hybrid (oneDNN Gram + T-solve + DPAS state) |
| Decode | Same sequential kernel (paged) | Reference SIMT kernel | Fused ESIMD decode |
| State mgmt | **Paged block table** (checkpoint every `cache_interval`) | Dense `ssm_state [batch,heads,d,d]`; chunk carry | Dense fp32 master; SLM resident per WG |
| Vectorization | `K_VEC_SIZE=8` (f16/f32 block reads) | cute block-2D LSC + DPAS | DPAS frags / scalar |
| GEMM engine | None — pure FMA/DPAS-free SIMT dot products | DPAS (XMX) via `XE_DPAS_TT` | DPAS (XMX) via SPIR-V intrinsic + oneDNN |
| L2-norm | Fused into conv/GDN (SCALE_FACTOR) | Fused into conv or standalone l2norm | Separate kernel / fused decode |

**The critical difference for prefill:** OpenVINO does **not** use the chunked
algorithm at all — it runs the same O(T) sequential recurrence for prefill that
it uses for decode (each WG walks the whole token range of its sequence). It
relies on per-(seq,head,v-block) parallelism (sequences × heads × v_blocks WGs)
and `K_VEC_SIZE=8` vectorization. vllm's Xe2 chunked path is the only one with a
true **chunked-parallel** prefill (A=K·Kᵀ, T-solve, then chunk recurrence).

---

## 3. What Arcaine can borrow for prefill (from OpenVINO)

Even though OpenVINO's prefill is sequentially slower in principle, it has
patterns that directly strengthen Arcaine's chunked/hybrid path:

### 3.1 The paged state-table pattern (HIGH value)
OpenVINO's `block_indices`/`cache_interval`/`past_lens` state-checkpoint scheme
is exactly what Arcaine needs to make **long-context prefill + decode
continuation** cheap:
- Persist the SSM state per KV-block so a continued prefill or a decode step
  restarts from the last checkpoint instead of re-scanning the whole prefix.
- Arcaine's `qwen3_5_moe` hybrid currently keeps the fp32 `ssm_state` dense
  `[n_v, 128, 128]` resident in SLM for a whole prefill but **re-uploads from
  global per chunk in the scalar path**. Adopting a block-checkpoint layout
  (write state to the block table at interval boundaries) removes the global
  round-trip and enables resumable prefill/decode.
- **Concrete**: port the `block_indices`/`cache_interval` logic into
  `state_pass` (hybrid) — instead of one WG per (head, dv-half) looping all
  chunks, add a checkpoint write every C-chunk boundary to a block table
  indexed like OpenVINO's. This also future-proofs the decode path (resume from
  block N).

### 3.2 K_VEC_SIZE=8 block-vectorized state load/store (MEDIUM value)
OpenVINO's `BLOCK_READN/BLOCK_WRITEN(..., 8, ...)` (`paged_gated_delta_net_opt.cl:58-68`)
load/store the state as float8 per lane with `K_VEC_COUNT=1` (K=128, SG=16,
vec=8). This is a clean register-resident state layout:
`state[v_idx][kc]` = 4 v-cols × 1 float8. Arcaine's hybrid stages K as fp16 SLM
and S as fp32 SLM; OpenVINO keeps the **S block fully in registers** per lane
(32 floats), avoiding SLM for the state entirely. For the decode/continuation
kernel (not prefill's DPAS state pass), this is a direct borrow.

### 3.3 Fused q/k l2norm + scale inside the recurrence (LOW effort)
`FUSE_QK_L2NORM` + `SCALE_FACTOR = 1/sqrt(k_dim)` is folded into `prepare_qk`
(`gated_delta_net_ref.cl:72-107`) — no separate norm pass, one less global
round-trip. Arcaine's hybrid already does l2norm in a separate kernel; fusing it
into the pad_qk stage (which already touches every q/k element) removes a pass.
Note vllm's chunked suite already fuses l2norm into the conv kernel
(`chunk_causal_conv1d_xe2.hpp`), so this is catching Arcaine up to both.

### 3.4 Per-sequence parallelism for the serial phase (MEDIUM)
OpenVINO's grid is `(sequences, heads, v_blocks)` — the token loop is
**per-sequence**. For decode/continuation (where Arcaine's hybrid serializes
chunks per head), splitting the grid by sequence with an independent state
carry per sequence is a cheap parallelism win for multi-sequence decode. For
pure single-sequence prefill this does not help (already chunk-parallel in the
hybrid), but it matters for batched decode.

### 3.5 What NOT to borrow
- **Do not** replace the chunked prefill with OpenVINO's sequential loop —
  Arcaine's hybrid (oneDNN batched Gram + parallel T-solve + DPAS state pass)
  is strictly better for prefill, and vllm's `fwd_o` chunked recurrence
  (gemm_TTS_fused_2A / _k_multi) beats OpenVINO's FMA dot products on XMX.
- **Do not** adopt OpenCL — Arcaine is SYCL; the patterns to port are the
  *data layout and paging logic*, not the kernel language.

---

## 4. Recommended prefill-attack plan (Arcaine)

1. **Port the state-block checkpointing (3.1)** into `state_pass`:
   `QWEN35_GDN_PAGED_STATE=1` — block table + interval writes. This is the
   highest-leverage change: it converts the dense fp32 master round-trip into
   checkpointed persistence and unlocks resumable decode.
2. **Fuse l2norm into pad_qk (3.3)** — `QWEN35_GDN_FUSE_L2NORM=1` — remove the
   standalone norm pass (currently a separate kernel in the hybrid).
3. **Register-resident S block (3.2)** for the decode/continuation kernel:
   `K_VEC_SIZE=8` state layout, no SLM for S.
4. **Per-sequence grid split (3.4)** for batched decode.
5. Keep the chunked DPAS prefill (hybrid) as the compute core; apply the
   borrows to the state-carry and norm edges.

### Bench / A-B (per AGENTS.md)
- `arcaine_kbench qwen35-gdn --device 1` (perf: core_cos, state_max_rel)
- mbench `-p 512,1024,2048,4096` for prefill; `-d 512,1024,2048,8192` for decode
- Env vars: `QWEN35_GDN_IMPL`, `QWEN35_GDN_PAGED_STATE`, `QWEN35_GDN_FUSE_L2NORM`

---

## 5. Correctness / notes

- OpenVINO's `paged_gated_delta_net_opt` requires `K_HEAD_DIM % 16 == 0 &&
  V_HEAD_DIM % 16 == 0` and falls back to `K_VEC_SIZE=1` otherwise
  (`:50-52`). Qwen3.5 (128/128) is on the vectorized path.
- `V_BLOCK_SIZE=4` is hardcoded to GRF size (`get_v_block_size`); the state
  register footprint is 4×8 = 32 floats/lane.
- OpenVINO's state writeback at `cache_interval` boundaries is in-place on the
  block table (no separate copy), which is exactly what a checkpoint scheme
  wants.

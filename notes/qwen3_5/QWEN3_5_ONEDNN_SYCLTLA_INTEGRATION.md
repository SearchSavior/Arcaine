# Qwen3.5 Dense on Arcaine: oneDNN + SYCL-TLA Integration Plan

**Audience**: Arcaine kernel devs building the Qwen3.5 **dense** path (SYCL + oneDNN, Battlemage
`bmg_g21/g31`).
**Companions**: `QWEN3_5_ARCHITECTURE.md` (the PyTorch reference + component list),
`QWEN3_5_XPU_KERNELS.md` (how the Intel-GPU kernels and DPAS lowering actually work).
**Thesis**: oneDNN owns every standard dense GEMM; **SYCL-TLA** (`intel/sycl-tla`, the Intel
CUTLASS-for-SYCL) provides the matrix-engine primitives for the two kernels oneDNN cannot
express as a single primitive — the **fused flash-attention** and the **Gated-DeltaNet linear
attention**. Both run on one `sycl::queue` over shared USM and lower to the same DPAS
instruction, so they compose with no copies.

---

## 1. Why a hybrid (not one or the other)

- **oneDNN** is a primitive library: `matmul`, `convolution`, `softmax`, `layer_normalization`,
  `eltwise`, `binary`, `reorder`, `reduction`, plus a Graph API. Its `matmul` is the right tool
  for `[M,K]·[K,N]` with DPAS, post-op fusion, scales/zero-points, and an internal primitive
  cache. It has **no single primitive** for: the gated delta-rule recurrence, a causal Conv1d
  with rolling state, RMSNorm-gated, interleaved partial M-RoPE, or a *fused* flash-attention
  with online softmax (its Graph-API SDPA is coarse and does not touch the GDN side at all).
- **SYCL-TLA** is not a primitive library — it is a set of `cute` device building blocks (DPAS
  MMA atoms, 2D block copies, in-register reorder/dequant, split barriers) plus collective
  templates. It is what you use to *write* the bespoke kernels, with direct DPAS control.

So: oneDNN for the boring GEMMs (most of the FLOPs, least of the effort), SYCL-TLA atoms for the
two custom kernels that carry Qwen3.5's novelty.

### Interop mechanism (the part that makes it free)
Both bind to the same SYCL context/queue and operate on the same USM allocations:
```cpp
// oneDNN side (see llama.cpp ggml-sycl/gemm.hpp:43-44 for the working pattern)
dnnl::engine eng = dnnl::sycl_interop::make_engine(sycl_device, sycl_context);
dnnl::stream  s  = dnnl::sycl_interop::make_stream(eng, queue);   // same queue as TLA kernels
dnnl::memory  m(md, eng, usm_ptr);                                // wraps the SAME pointer
```
SYCL-TLA kernels are plain `queue.submit(...)` over the same USM pointers. Ordering is the
queue's; no host sync, no device copies at the oneDNN↔TLA seam — only a possible **layout
reorder** (see §5).

---

## 2. oneDNN's territory — the dense GEMMs

Use `dnnl::matmul` (batched/3D where applicable) for every projection and MLP. oneDNN handles
the DPAS, the bf16/fp16/fp8/NVFP4 dtypes (Arcaine already uses NVFP4 matmul), per-channel /
per-group scales + zero-points for quantized weights, and post-op fusion.

| Qwen3.5 dense GEMM | oneDNN call | Fusion (post-ops) |
|---|---|---|
| `q_proj` (double-width → Q + output-gate) | `matmul` | split halves after; gate applied later |
| `k_proj`, `v_proj`, `o_proj` | `matmul` | bias if present |
| GDN `in_proj_qkv`, `in_proj_z/b/a`, `out_proj` | `matmul` | — |
| MLP `gate_proj` | `matmul` | **`eltwise swish` (SiLU) post-op** |
| MLP `up_proj` | `matmul` | — |
| MLP `down_proj` | `matmul` | preceded by `binary mul` (gate⊙up) |
| `lm_head` | `matmul` | **`eltwise tanh` post-op** for final-logit softcap (scaled) |
| Vision: patch-embed (Conv3d→GEMM), QKV, MLP fc1/fc2, merger fc1/fc2 | `matmul` (+`conv` for patch embed) | GELU-tanh post-op on MLP/merger |

**SwiGLU concretely**: `tmp = matmul(x, Wgate)` with `swish` post-op; `up = matmul(x, Wup)`;
`act = binary_mul(tmp, up)`; `y = matmul(act, Wdown)`. (Or a single custom SwiGLU kernel if you
want to fuse the two input GEMMs — but the oneDNN path is the no-effort default.)

**Quantized projections** (if running W4A16/W8A16/NVFP4 weights): keep them in oneDNN with
weight scales/zero-points; you do **not** need SYCL-TLA's mixed-input mainloop unless you want
in-kernel dequant fused into a *custom* GEMM. oneDNN already exposes the quantized matmul Arcaine
uses for DiffusionGemma.

oneDNN standalone primitives also cover the easy non-GEMM ops if you don't fuse them:
`softmax`, `eltwise` (SiLU/GELU/tanh), `binary`, `reorder`, `reduction`. RMSNorm specifically is
**not** a clean oneDNN primitive (its `layer_normalization` computes mean+variance, not the
mean-square-only RMSNorm) — treat RMSNorm as custom (§4).

---

## 3. SYCL-TLA's territory — the bespoke kernels

Two kernels oneDNN cannot host as one primitive. Build them from these SYCL-TLA primitives
(reference implementations live in vllm-xpu-kernels `csrc/xpu/{attn,gdn_attn}`):

### 3.1 Fused flash-attention (full-attention layers)
- **Primitives**: `MMA_Atom<XE_DPAS_TT<8,float,bf16/fp16>>` ×2 (one tiling for `Q·Kᵀ`, one for
  `P·V`); `XE_LOAD_2D`/`_VNNI` + `make_block_2d_copy_*` for Q/K/V tiles; `XE_PREFETCH_2D` for
  L1 prefetch; `barrier_arrive`/`barrier_wait` for the pipeline; `Xe_Reorder` if the KV-cache is
  quantized (in-register dequant before the MMA).
- **Template to lift**: the FMHA collective (`cutlass/.../collective/chunk_prefill_mainloop.hpp`)
  — two `TiledMMA`s with the **online softmax kept in registers** ("P in registers" `XeDefault`
  policy), varlen via `cu_seqlens`.
- **Stays outside the kernel** (caller's job): QK-norm + partial M-RoPE (custom, §4); the
  `attn_output * sigmoid(gate)` output gate (elementwise, §4).

### 3.2 Gated-DeltaNet linear attention (the novel layers)
- **Causal Conv1d (+ rolling state, + L2-norm)**: SLM-staged tile with an explicit `Width-1`
  halo = the conv state; fuse the q/k **L2-norm** into the conv epilogue. Use `XE_LOAD_2D` /
  SLM `local_accessor`; no DPAS here (it's depthwise). oneDNN `convolution` is a *fallback* for
  the bare conv but cannot fuse the state update or the l2norm.
- **Chunked gated delta rule**: the 5-kernel DPAS pipeline (prepare → compute_A → inverse →
  compute_wu → fwd_o) using `XE_DPAS_TT` 64×64×32 tiles, the `gemm_TTS`/`gemm_STS`/
  `gemm_TTS_fused_2A`/`gemm_TTS_k_multi` helpers, `Xe_Reorder` for fragment→VNNI, split barriers,
  and SLM for the `g/β/A` scratch. **No oneDNN equivalent exists.** Keep the BMG-DPAS vs
  PVC-native-SLM split for the triangular inverse.
- **Decay + gating math** (`g = -exp(A_log)·softplus(a+dt_bias)`, `β = sigmoid(b)`) and the
  **RMSNorm-gated** output: custom elementwise, fp32 (§4).

### The SYCL-TLA primitive inventory (reusable in any custom kernel)
| Primitive | Header | Use |
|---|---|---|
| `XE_DPAS_TT` MMA atom + `TiledMMA`/`TiledMMAHelper` | `cute/arch/mma_xe.hpp`, `atom/mma_traits_xe.hpp` | the matrix multiply in FA & GDN |
| 2D block copy: `XE_LOAD_2D`/`_VNNI`/`_TRANSPOSE`, `XE_STORE_2D`, `XE_PREFETCH_2D` + `make_block_2d_copy_*` | `cute/arch/copy_xe_2d.hpp`, `atom/copy_traits_xe_2d.hpp` | register-resident operand loads, VNNI, L1 prefetch |
| `Xe_Reorder` (fused convert + VNNI; int4/fp8/e2m1/ue8m0 dequant) | `cute/arch/reorder_xe.hpp` | KV/weight dequant inside a kernel |
| Split barriers `barrier_arrive`/`barrier_wait` | `cute/util/xe_split_barrier.hpp` | workgroup software pipeline |
| Collective mainloops (dense / mixed-input / FMHA) | `cutlass/gemm/collective/xe_mma*.hpp`, FMHA collective | lift wholesale as kernel skeletons |
| `cute` layout algebra + tile schedulers (data-parallel / stream-K / persistent) | `cute/*`, `cutlass/gemm/kernel/xe_*scheduler*` | tiling the custom kernels |

---

## 4. The leftover custom kernels (neither oneDNN nor TLA matmul)

Small SYCL kernels (subgroup reductions, elementwise); no matrix engine. SYCL-TLA's subgroup
helpers are optional conveniences here.
- **RMSNorm** and **RMSNorm-gated** (`out = rms(x)·(1+w)`, gated variant `·SiLU(z)`), fp32
  reduction. Note the weight is stored zero-centered → multiply by `(1+w)`.
- **QK-norm + partial/interleaved M-RoPE** (one subgroup per (token,head); fuse the two head-dim
  passes — see vllm-xpu `fused_qknorm_rope.cpp`, SIMD32). `mrope_section = [11,11,10]`.
- **Output gate** `attn_out *= sigmoid(gate)` and the GDN decay/β math (fp32).
- **embed_tokens** gather; **token/placeholder scatter** for multimodal (not in pure-dense).

---

## 5. The oneDNN ⇄ SYCL-TLA seam — layouts & hand-off

The only correctness/perf concern at the boundary is **operand layout**, because DPAS wants
VNNI-packed B and the two libraries pick their own internal layouts.

- **oneDNN → custom TLA kernel** (e.g. projections feed the FA/GDN kernel): oneDNN writes
  row/col-major USM; the TLA kernel's `make_block_2d_copy_*` + `Xe_Reorder` handle VNNI packing
  *on load*, so usually **no explicit reorder** is needed — just make sure the oneDNN output
  obeys the 2D-block alignment (base 64 B, row pitch %4 B; see `QWEN3_5_XPU_KERNELS.md` §8.3).
- **custom TLA kernel → oneDNN** (e.g. attention output feeds `o_proj`): emit a plain
  row-major tile (`XE_STORE_2D`), which oneDNN matmul consumes directly.
- **When a reorder is unavoidable** (a packed/VNNI weight needs a different layout than oneDNN
  expects), use **oneDNN's `reorder` primitive** (stays in the oneDNN graph, cached) rather than
  a bespoke kernel.
- **dtype at the seam**: keep activations in bf16/fp16 through the chain; oneDNN matmul with f32
  accumulate and the DPAS atom's f32 accumulator agree numerically. Avoid silent f16 KQ overflow
  by accumulating attention in f32 (the DPAS atom already does).
- **Alignment is a hard gate**: oneDNN `matmul` (`MainloopXeL1Staged::can_implement`) and the 2D
  block messages both require 128-bit copy alignment / 64 B base; size your KV-cache and weight
  buffers accordingly or the fast paths silently won't apply.

---

## 6. Build / dependency notes

- **oneDNN**: already in Arcaine's stack (source build, SYCL CPU+GPU runtime). Enable the
  primitive cache; set `fpmath_mode::f16`/`bf16` for the matmuls.
- **SYCL-TLA**: header-only `cute`/CUTLASS-SYCL, BSD-3-Clause (vendoring is fine, check per-file
  headers). It is a heavy CUTLASS-scale template dependency; pull at a pinned revision (vllm-xpu
  pins `cd763790…`) and target `intel_gpu_bmg_g21`. You only need the `cute` device headers
  (`include/cute`, `include/cutlass/gemm/collective`, `…/kernel`), not the host build.
- Compile both with the same oneAPI/`icpx`; SYCL-TLA's DPAS path needs
  `-fsycl-targets=intel_gpu_bmg_g21` and the large-GRF property (`grf_size<256>`).

---

## 7. Bring-up order (recommended)

1. **All projections + MLP + lm_head on oneDNN** (with SwiGLU/softcap post-ops). This stands up
   the whole dense graph minus the two attention families.
2. **Full-attention via a SYCL-TLA FMHA kernel** (lift the vllm-xpu `attn/xe_2` collective);
   wire QK-norm+RoPE+output-gate as the custom pre/post kernels. Validate vs the HF reference.
3. **GDN linear attention** (lift vllm-xpu `gdn_attn`): conv1d+l2norm kernel, then the chunked
   delta-rule pipeline. This is the highest-effort piece; unit-test the chunk↔recurrence
   equivalence at `chunk_size=64`.
4. **RMSNorm / gating** custom kernels throughout.
5. Only then consider replacing any oneDNN GEMM with a custom TLA GEMM (e.g. to fuse dequant for
   quantized weights via `xe_mma_mixed_input.hpp`) — measure first; oneDNN is usually already at
   DPAS peak for plain GEMM.

---

## 8. Summary table

| Qwen3.5 dense component | Implement with | Primitive(s) |
|---|---|---|
| All projections, MLP, lm_head, vision GEMMs | **oneDNN** | `matmul` + post-ops (swish/tanh/binary), scales |
| SwiGLU | **oneDNN** | matmul swish post-op + binary-mul |
| Quantized weight GEMMs (W4A16/NVFP4/FP8) | **oneDNN** | `matmul` + scales/zero-points |
| RMSNorm / RMSNorm-gated | custom | subgroup-reduce SYCL kernel (fp32) |
| QK-norm + partial M-RoPE, output gate | custom | subgroup SYCL kernel |
| Full attention (QK·softmax·PV) | **SYCL-TLA** | `XE_DPAS_TT`×2 + 2D-copy + reorder + barriers + online softmax |
| GDN causal Conv1d (+state,+l2norm) | custom + SYCL-TLA | 2D-copy/SLM halo; fuse l2norm |
| GDN chunked gated delta rule | **SYCL-TLA** | `XE_DPAS_TT` 5-kernel pipeline (no oneDNN equiv) |
| Layout hand-off / weight repack | **oneDNN** `reorder` or `Xe_Reorder` | only where VNNI/layout mismatch |

**Bottom line**: oneDNN gets ~all the dense FLOPs (projections/MLP/lm_head/vision) for almost no
effort and at DPAS peak; SYCL-TLA's `XE_DPAS_TT` + 2D-copy + reorder + barrier atoms are exactly
what you need to hand-write the **flash-attention** and **Gated-DeltaNet** kernels that oneDNN
cannot express — and because both share the queue/USM/DPAS, they compose into one Qwen3.5 dense
forward pass without copies.

---

**Document version**: 1.0
**Reference**: `intel/sycl-tla@cd763790`, `vllm-project/vllm-xpu-kernels@main`
(`csrc/xpu/{attn,gdn_attn}` reference kernels), oneDNN matmul interop pattern from
`ggml-org/llama.cpp` `ggml/src/ggml-sycl/gemm.hpp`.
**Companions**: `QWEN3_5_ARCHITECTURE.md`, `QWEN3_5_XPU_KERNELS.md`

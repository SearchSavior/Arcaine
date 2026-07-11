# llama.cpp `ggml-sycl`: Lowering to DPAS/SPIR-V — Feasibility & Performance Report

**Source**: `github.com/ggml-org/llama.cpp` @ `master` (fetched into this env), backend
`ggml/src/ggml-sycl/`.
**Companion**: `notes/qwen3_5/QWEN3_5_XPU_KERNELS.md` (§8.2/§8.3 — how Intel `sycl-tla` actually
emits DPAS via inline vISA / `__spirv_SubgroupMatrixMultiplyAccumulateINTEL`). That doc is the
"how to lower" reference; this doc is the "what to lower and where" analysis for ggml-sycl.
**Question (from the brief)**: llama.cpp's SYCL backend is not using the XMX (Xe Matrix /
DPAS) cores; prefill flash-attention does the work in software at large cost. Is it feasible to
lower the hot paths to DPAS via SPIR-V and apply close-to-hardware techniques, and what would
have to change in ggml?

**Answer up front**: **Confirmed, and yes.** ggml-sycl uses the matrix engine **nowhere** in
its own kernels. Every attention matmul and every quantized matmul runs on the vector/EU ALU
pipe — flash-attention as scalar FMA, quantized GEMM as a **software-emulated** `dp4a`. The
work is self-contained in the SYCL backend; the changes are additive device kernels plus two
dispatch hooks, with **no change to ggml core or the GGUF quant formats**. Below is the
code-verified current state, the specific lowering opportunities, and the change surface.

---

## 1. Current state — verified by reading the kernels

### 1.1 The matrix engine is used nowhere in ggml-sycl's own code
`grep joint_matrix` over `ggml/src/ggml-sycl/` returns **nothing**. The only matrix-engine
access in the whole backend is through **oneDNN**, and it is optional and off by default:
- `DnnlGemmWrapper` (`gemm.hpp:23-89`) wraps `dnnl::matmul` — oneDNN internally targets DPAS.
- It is gated by `GGML_SYCL_DNNL`, which `CMakeLists.txt:126` sets to **0** unless `DNNL` is
  found at configure time.
- It is only used for **non-quantized** matmul: `ggml_sycl_op_mul_mat_sycl` (bf16/f16/f32,
  `ggml-sycl.cpp:2443-2538`) and `ggml_sycl_mul_mat_batched_sycl` (f16 batched,
  `:3375-3409`).

So even in the best case (DNNL enabled), **only f16/f32/bf16 dense GEMM** touches XMX. Quantized
models and all of flash-attention do not.

### 1.2 `SYCL_USE_XMX` is a misnomer inherited from the CUDA port
`common.hpp:105-109`:
```c
// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#if !defined(GGML_SYCL_FORCE_MMQ)
    #define SYCL_USE_XMX
#endif
```
Where it is referenced (`mmq.cpp:1343` etc., `ggml-sycl.cpp:4205-4207`) it only selects smaller
MMQ tile dimensions (the `*_AMPERE` config: `MMQ_X=4, MMQ_Y=32`) and caps the MMQ batch to
`MMQ_MAX_BATCH_SIZE=32` — parameters copied verbatim from the NVIDIA tensor-core path. It emits
**no matrix instruction**. The `*_AMPERE / *_PASCAL / *_RDNA` naming throughout `mmq.cpp` is all
carried over from the CUDA/HIP backend.

### 1.3 Quantized matmul = software-emulated `dp4a` (this is the big one)
The quantized dot products (Q5_0/Q5_1/Q5_K and every other quant) bottom out in `dpct::dp4a`,
which is **pure scalar C++** — there is not even a hardware DP4A, let alone DPAS
(`dpct/helper.hpp:1859-1868`):
```cpp
inline auto dp4a(T1 a, T2 b, T3 c) {
  dot_product_acc_t<T1,T2> res = c;
  auto va = extract_and_sign_or_zero_extend4(a);
  auto vb = extract_and_sign_or_zero_extend4(b);
  res += va[0]*vb[0]; res += va[1]*vb[1];
  res += va[2]*vb[2]; res += va[3]*vb[3];
  return res;
}
```
Every `vec_dot_q*_q8_1_impl` calls this in an unrolled loop. For **Q5** specifically
(`vecdotq.hpp:722-801`): the kernel reconstructs the 5-bit value by shuffling the `qh` 5th bit
into the packed 4-bit `qs` int, then issues `dpct::dp4a(vi, u, sumi)` against the Q8_1
activations, then scales by the block deltas `d`/`dm`. So a Q5 weight × Q8 activation dot is
**4 scalar int MACs per dp4a × (unroll)** on the EU pipe.

This is the path for:
- **MMVQ** (`mmvq.cpp`) — mat×vec for **decode** (batch ≤ `MMVQ_MAX_BATCH_SIZE`).
- **MMQ** (`mmq.cpp`, `mul_mat_q<...>`) — mat×mat for **prefill** (batch ≤ 32), SLM-tiled but
  still `vec_dot_*_q8_1_mul_mat` → `dp4a`.
- **DMMV** (`dmmv.cpp`) — dequantize then f32 vec dot, for some decode cases.

Battlemage's DPAS array can do **int8 8×16×32** matrix-multiply-accumulate natively (the
`XE_8x16x32_S32S8S8S32_TT` atom / `dpas.s8.s8.8` instruction documented in the sycl-tla report).
The Q8_1×Q-quant integer GEMM is *exactly* that shape after dequant-to-int8 — and it is being
done as scalar adds instead.

### 1.4 Flash attention = software FMA, no matrix engine (and an explicit TODO)
Dispatch (`fattn.cpp:95-223`) has exactly **two** kernels and a comment marking the gap:
```c
// Todo: Use the XMX kernel if possible:                       // fattn.cpp:192
// If there are no tensor cores available, use the generic tile kernel:
```
- `BEST_FATTN_KERNEL_VEC` — decode (`Q->ne[1] ≤ 1..2`), `fattn-vec.hpp`.
- `BEST_FATTN_KERNEL_TILE` — **everything else, including all prefill**, `fattn-tile.hpp`.

There is no XMX/tensor-core kernel; prefill always falls to the software tile kernel. Inside
`fattn-tile.hpp`:
- **KQ = K·Qᵀ** (`flash_attn_tile_iter_KQ`, `:325-395`): K/V staged into SLM
  (`flash_attn_tile_load_tile`), then the dot is a triple-nested loop of
  `ggml_sycl_mad(KQ_acc, K_k[..], Q_k[..])` — **per-element FMA** in registers (`:380-389`).
- **online softmax** (`flash_attn_tile_iter`, `:472-573`): `warp_reduce_max`,
  `sycl::native::exp`, running max/sum rescale — software, scalar.
- **VKQ = V·P** (`:581-654`): again per-element `VKQ[..].x() += V_k[..].x()*KQ_k[..].x()` FMA.

So both GEMMs inside the prefill attention kernel are scalar FMA on the vector pipe. On a B-series
GPU this leaves the systolic DPAS array idle during the single most FLOP-heavy phase of prefill —
precisely the cost the brief describes.

### 1.5 Dispatch map (`ggml_sycl_mul_mat`, `ggml-sycl.cpp:4170-4255`)
| Case | Path | Hardware |
|---|---|---|
| f16, permuted, batch 1 (KQ decode) | `mul_mat_vec_p021` | vector |
| f16, non-contig, batch 1 (KQV decode) | `mul_mat_vec_nc` | vector |
| **f16, batch > 1 (KQ/KQV prefill)** | `mul_mat_batched_sycl` → **oneDNN** | **DPAS (if DNNL on)** |
| quantized, batch 1, dmmv-eligible | `op_mul_mat<dequantize_mul_mat_vec>` | vector |
| quantized, batch ≤ 32 (decode/small) | `op_mul_mat<mul_mat_vec_q>` (MMVQ) | scalar dp4a |
| **quantized, prefill** | MMQ (`mul_mat_q`) | **scalar dp4a** |
| flash attention (all) | tile / vec | **scalar FMA** |

The only DPAS today is the f16-prefill GEMM via oneDNN, *if* it was compiled in. Everything
quantized and all attention is software.

---

## 2. The lowering opportunity

Three hot paths, in priority order, can be lowered to DPAS. The "how" is in the companion
sycl-tla report; the salient point is **you do not need `joint_matrix` the C++ API** — the
portable hardware handle is the SPIR-V cooperative-matrix intrinsic
`__spirv_SubgroupMatrixMultiplyAccumulateINTEL` (what `joint_matrix` and sycl-tla both lower to),
and Battlemage exposes DPAS atoms for **bf16/fp16 (16×16×16)** and **int8 (8×16×32)**.

### 2.1 Flash-attention prefill → DPAS (highest impact)
Replace the `fattn-tile.hpp` scalar QK/PV with a fused-MHA collective:
- **KQ** (`Q·Kᵀ`) and **PV** (`P·V`) are bf16/fp16 GEMMs → the `f16/bf16` DPAS atom. K/V are
  already staged as `half2` in SLM (`fattn-tile.hpp:194-312`), so the operands are in the right
  dtype; what changes is the inner product becomes a `joint_matrix`/SPIR-V MMA over 16×16×16
  tiles instead of the `ggml_sycl_mad` triple loop.
- Keep the **online-softmax** epilogue in registers (it is elementwise + warp reductions; it
  does not need the matrix engine). This is exactly the structure of the sycl-tla FMHA
  (`chunk_prefill_mainloop.hpp`: two `TiledMMA`s for QK and PV, softmax in registers — companion
  report §8.1).
- Wire it into the existing `// Todo: Use the XMX kernel if possible:` slot
  (`fattn.cpp:192`): add `BEST_FATTN_KERNEL_XMX` ahead of `…_TILE` when the device reports XMX
  and head dim/dtype are supported.
- Expected win is largest at **long prefill** (the KQ/PV cost is O(seq²·d) and is the bulk of
  prefill compute); this is the case the brief calls out.

### 2.2 Quantized GEMM (MMQ/MMVQ) → int8 DPAS
The Q8_1 activation × quantized weight integer GEMM is the int8-DPAS shape. Two sub-options:
1. **Dequant-to-int8 + int8 DPAS**: unpack Q5/Q4/Q8 weights to int8 in registers/SLM (the
   dequant bit-tricks already exist in `dequantize.hpp` / `vecdotq.hpp`), then feed the
   `XE_8x16x32_S32S8S8S32` DPAS instead of `dp4a`. sycl-tla even has **in-register fused
   dequant+VNNI** reorder sequences for int4/int8/fp8 (companion report §8.3) that fold the
   unpack into the load→MMA pipeline.
2. **Dequant-to-bf16 + bf16 DPAS**: simpler, slightly more bandwidth, reuses the f16 atom; good
   first step.

Replace `dpct::dp4a` (the scalar emulation, `helper.hpp:1859`) at minimum with the hardware
path on Xe — but the real win is restructuring `mul_mat_q` to issue block-level DPAS rather than
per-thread 4-wide dot products.

### 2.3 Dense f16/f32 GEMM — already covered by oneDNN, make it default
`ggml_sycl_op_mul_mat_sycl` / `…_batched_sycl` already route to oneDNN→DPAS. The gap is that
`GGML_SYCL_DNNL` defaults **off** (`CMakeLists.txt:126`). For Battlemage, enabling oneDNN (or a
sycl-tla GEMM) by default closes the dense-GEMM gap with no new kernel.

---

## 3. What would have to change — and where (the ggml surface)

The brief's hypothesis ("changes to ggml") is correct but the scope is narrower than it sounds:
**all changes are inside `ggml/src/ggml-sycl/`**. The ggml *core* (`ggml.c`, tensor/op
definitions) and the **GGUF quant block formats are untouched** — the kernels consume the same
`block_q5_0/1`, `block_q5_K`, `block_q8_1` layouts (`quants.hpp`). Concretely:

1. **New kernels** (additive, no API change):
   - `fattn-xmx.hpp` — DPAS fused-MHA prefill kernel (model on sycl-tla's `chunk_prefill_*`).
   - A DPAS path in `mmq.cpp` (e.g. `mul_mat_q_dpas`) for the int8/bf16 MMA, behind a real
     capability check.
2. **Dispatch hooks** (two small edits):
   - `fattn.cpp` — fill the `// Todo: Use the XMX kernel if possible:` branch
     (`:192`): return `BEST_FATTN_KERNEL_XMX` when `device has XMX && supported(DKQ,DV,dtype)`.
   - `ggml_sycl_mul_mat` (`ggml-sycl.cpp:4170`) — route quantized prefill to the DPAS MMQ when
     available, and consider defaulting the f16 batched path to oneDNN/sycl-tla on BMG.
3. **A real device capability probe**: today `SYCL_USE_XMX` is a compile-time misnomer and
   `cc`/`VER_GEN12/13` (`common.hpp:100-101`) are placeholder "todo for hardware optimize"
   constants. Add an actual XMX/architecture query (BMG `intel_gpu_bmg_g21/g31`,
   `ext_oneapi_architecture`) — `ggml-sycl.cpp:144` already calls
   `device.ext_oneapi_architecture_is(...)` for the reorder feature, so the mechanism exists.
4. **Lowering mechanism**: either (a) depend on Intel `sycl-tla` (`cute` `XE_DPAS_TT` atoms +
   the FMHA/GEMM collectives — most reuse, heaviest dep), or (b) hand-write the SPIR-V
   `__spirv_SubgroupMatrixMultiplyAccumulateINTEL` MMA with VNNI 2D-block loads (no extra dep,
   more code). Either targets the same DPAS instruction (companion report §8.2). For a ggml
   upstream PR, (b) is more palatable (no CUTLASS-scale dependency); for an Arcaine-internal
   fork, (a) is faster.

**Non-goals / cautions for a port**:
- The block-q reorder optimization (`should_reorder_tensor`, `reorder_qw_q*`,
  `ggml-sycl.cpp:3509-3700`) already repacks weights for the SoA MMVQ path — a DPAS MMQ wants a
  **VNNI**-friendly weight layout, so this is the natural place to add a DPAS-specific reorder
  rather than inventing a new buffer path.
- Numerics: the tile kernel scales Q down by 4 to avoid f16 KQ overflow without `v_dot2_f32_f16`
  (`fattn-tile.hpp:481-485`); a DPAS f16 path with f32 accumulate (the atom's accumulator is
  f32) removes that hazard — a correctness *improvement*, but validate against the tile kernel.
- MMQ batch is capped at 32 today (`MMQ_MAX_BATCH_SIZE`); a DPAS MMQ should lift that, since the
  whole point is throughput at large batch/prefill.

---

## 4. Bottom line

| Path | Today | Hardware used | DPAS-lowerable? |
|---|---|---|---|
| FA prefill (tile) | scalar FMA QK/PV + SW softmax | EU/FPU | **Yes — biggest win** (bf16 16×16×16) |
| FA decode (vec) | scalar FMA | EU/FPU | marginal (vec-bound, low arith intensity) |
| Quantized prefill (MMQ) | **scalar-emulated `dp4a`** | EU ALU | **Yes** (int8 8×16×32, or bf16) |
| Quantized decode (MMVQ) | scalar-emulated `dp4a` | EU ALU | partial (bandwidth-bound) |
| Dense f16/f32 GEMM | oneDNN (if `GGML_SYCL_DNNL`) | **DPAS (optional)** | already; default it on |

The brief's intuition holds: the SYCL backend leaves the matrix engine almost entirely unused,
prefill flash-attention and quantized GEMM are the two paths paying the largest software tax, and
both are classic DPAS shapes. The lowering is **feasible within ggml-sycl alone** (no core/format
changes), the mechanism is the SPIR-V DPAS MMA (with or without sycl-tla), and the highest-value
first target is a DPAS fused-MHA prefill kernel slotted into the existing `fattn.cpp` TODO.

---

**Document version**: 1.0
**Reference**: `ggml-org/llama.cpp@master`, `ggml/src/ggml-sycl/{fattn*.hpp,fattn.cpp,mmq.cpp,mmvq.cpp,dmmv.cpp,vecdotq.hpp,dequantize.hpp,gemm.hpp,common.hpp,ggml-sycl.cpp,dpct/helper.hpp}`
**Companion**: `notes/qwen3_5/QWEN3_5_XPU_KERNELS.md` (DPAS/SPIR-V lowering mechanism)

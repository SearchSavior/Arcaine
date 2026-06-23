# NVFP4 Tiled DPAS Expert Matmul — Working Scratchpad

Goal: replace the scalar grouped NVFP4 expert matmul
(`matmul_nvfp4_grouped_custom`) with a tiled kernel that drives the Battlemage
XMX/DPAS systolic array, so a custom expert path can actually compete with
oneDNN's `jit:gemm:any` FP4 microkernel instead of regressing ~6.5x.

Context: `DIFF_NVFP4_EXPERT_KERNEL=hybrid` is the current best (groups activation
packing + GeGLU, keeps oneDNN FP4 matmuls). The scalar `custom` kernel is correct
but far too slow. See `NVFP4_INFERENCE_NOTES.md` Optimization Experiments.

---

## Hardware (verified 2026-06-14)

`./build/gpu_check` + probes on the local box:

- 2x `Intel(R) Graphics [0xe223]`, **architecture = `intel_gpu_bmg_g31`**
  (Battlemage G31, Xe2). NOTE: device id maps to **g31, not g21**.
- 32 Xe-cores/GPU, 8 vector engines/Xe-core (256 total), 8 HW threads/VE.
- SLM = 128 KB / work-group. Max WG size = 1024. Sub-group sizes = {16, 32}.
- 32.5 GB VRAM/GPU.

## DPAS reachability — the key constraint

XMX shape (from forcing a `joint_matrix` compile): **bf16 A/B, f32 acc, tile
`M<=8, N=16, K=16`**. `N=8` is rejected by ocloc ("unsupported number of
columns: 8, supported values: 16"). So Battlemage XMX is 16-wide (xmx16-style).

**Blocker:** the portable SYCL `joint_matrix` path is *not usable* on this stack:
- `device.has(aspect::ext_intel_matrix)` returns **0** on BOTH level_zero and
  opencl backends.
- `joint_matrix_fill/load` therefore throw at runtime:
  `"no matrix hardware on the target device, joint_matrix is not supported"`,
  even when AOT codegen (`-fsycl-targets=intel_gpu_bmg_g31`) accepts the kernel.
- `info::device::matrix_combinations` returns an empty list (JIT and AOT).

This is a **libsycl/runtime gap** (oneAPI 2025.3 + NEO 26.05): bmg_g31 is not
flagged as having the matrix aspect. The hardware *does* have DPAS — oneDNN's
`jit:gemm:any` uses it on this exact GPU.

**Reachable DPAS path:** the compute-runtime OpenCL platform advertises
`cl_intel_subgroup_matrix_multiply_accumulate` (+ `_tf32`),
`cl_intel_subgroup_2d_block_io`, `cl_intel_bfloat16_conversions`. So DPAS is
reachable through the Intel sub-group matrix-mad builtin / SPIR-V op
`OpSubgroupMatrixMultiplyAccumulateINTEL`, which bypasses the SYCL aspect guard.
This is the same mechanism oneDNN and ggml-sycl use.

### Decision
Do NOT build on SYCL `joint_matrix` (blocked). Two candidate routes:
1. **Intel sub-group MMA builtin** via `SYCL_EXTERNAL` SPIR-V intrinsic
   `__spirv_SubgroupMatrixMultiplyAccumulateINTEL`, AOT-targeted to
   `intel_gpu_bmg_g31`. Highest control, matches oneDNN's access path.
2. Re-check after a runtime/driver bump that flips `ext_intel_matrix` to 1; then
   `joint_matrix` (shape M8/N16/K16) becomes the clean portable route.

Proceeding with route 1, but keep the tile math identical to route 2 so we can
swap the inner mad if the aspect is fixed.

---

## Numerics (mirror the scalar kernel exactly)

From `matmul_nvfp4_grouped_custom` / `pack_bf16_to_nvfp4_grouped`:

    C[r,n] = ( sum_g sum_{i in group g} e2m1_a * e2m1_w * a_scale_g * w_scale_{g,n} )
             / dst_scale          where dst_scale = input_global * weight_global

Folding the per-group FP8 scale into the dequantized bf16 operands gives a plain
bf16 matmul:

    A_bf16[r,k] = e2m1(a_packed) * fp8(a_scale[r, k/16])
    B_bf16[k,n] = e2m1(w_packed) * fp8(w_scale[k/16, n])
    C[r,n]      = (A_bf16 @ B_bf16) / dst_scale

Precision note: e2m1 has <=3 sig bits, fp8 e4m3 scale 3 mantissa bits; the
product fits bf16 (7 mantissa) with negligible error. Accumulation is f32 in the
DPAS accumulator — same as scalar. Validate against scalar/oneDNN anyway.

Weight layout already on device (`Nvfp4Linear`):
- `weight_packed`: u8, logical `[N, K]` packed low-nibble-first, i.e. `[N, K/2]`.
- `weight_scale`: f8_e4m3, transposed to `[K/16, N]`.
- `dst_scale`: 1 element f32 = input_global*weight_global.

For DPAS we need B = Wᵀ as `[K, N]`. Weight is stored `[N, K]`, so the staging
step transposes while dequantizing into SLM (or use 2d_block_io transpose).

---

## Tiling plan (route 1)

Per expert bucket (grouped, like hybrid): rows = active tokens for that expert,
N = out_features (2*moe_inter for gate/up = 4224; H=2816 for down), K = in (2816
or 704).

- Work-group computes a `BM x BN` output tile for one expert; sub-group (SG=16)
  owns a `8 x 16` DPAS fragment, looping K in steps of 16.
- Stage per K-step into SLM as bf16: dequant A tile `[BM,16]` and B tile
  `[16,BN]` (transposed from weight `[N,K]`), folding FP8 group scales.
- Accumulate f32; at the end multiply by `1/dst_scale` and store bf16 to `Ye`.
- Reuse the existing grouped row→slot/expert indirection so output lands in the
  same `Ye` slots the combine kernel expects.

Open questions to settle in code:
- BM/BN choice vs SLM budget and the small/skewed expert M distribution
  (most buckets are tiny — kTailCap=64). A DPAS tile wants M multiples of 8;
  rounding tiny buckets up wastes work. May need a min-M threshold below which we
  fall back to oneDNN (i.e. DPAS only for hot experts).
- Whether to dequant-to-SLM vs. use `cl_intel_subgroup_2d_block_io` to load
  packed FP4 directly then unpack in registers.
- Activation packing: can likely reuse `pack_bf16_to_nvfp4_grouped`, or skip
  packing entirely and feed dequantized bf16 A straight to DPAS (activations are
  bf16 on input anyway — packing them to FP4 only helps oneDNN's FP4 matmul; for
  our own kernel we may keep A in bf16 and only dequant the FP4 *weights*). This
  could be a real win: weight-only FP4, bf16 activations, bf16 DPAS.

### Revised numeric for weight-only-FP4 variant
If A stays bf16 (no activation quant), then:

    C[r,n] = ( A_bf16 @ B_bf16 ) / weight_global    (no input_global, no a_scale)
    B_bf16[k,n] = e2m1(w_packed) * fp8(w_scale[k/16,n])

This drops the whole activation pack/scale path and the input_global_scale. Most
promising first target — simpler and removes a kernel.

---

## Integration

- New env value `DIFF_NVFP4_EXPERT_KERNEL=dpas` dispatched in
  `expert_parallel.cpp run_shard` alongside `hybrid`/`custom`.
- Kernel header: `src/common/gpu/nvfp4_dpas.hpp`.
- AOT: diffusion_gemma target must add `-fsycl-targets=intel_gpu_bmg_g31` (or
  keep JIT for everything else and AOT only this TU). Confirm build impact.

## Validation plan

1. Standalone micro-test: random A,W for one real shape (M=64,K=704,N=2816 down;
   M,K=2816,N=4224 gate/up). Compare DPAS-kernel C vs scalar
   `matmul_nvfp4_grouped_custom` and vs oneDNN `matmul_nvfp4_packed`. Require
   cosine > 0.999 / small MAE.
2. End-to-end: `tools/bench_nvfp4_variants.sh` add a `dpas` case; compare
   forward/s vs `hybrid` (target: beat 3.34 forward/s) and check generated text
   sanity.

## VALIDATED DPAS RECIPE (2026-06-14)

`tools/dpas_mad_probe.cpp` runs a real bf16 8x16x16 DPAS mad on the GPU and
matches the host reference exactly (`max_err=0`). This is the working access
path around the blocked `joint_matrix`:

**Intrinsic** (declare yourself; device-only, needs a host stub to link):
```cpp
using v8s = short __attribute__((ext_vector_type(8)));   // NOT sycl::vec (sret breaks it)
using v8i = int   __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));
SYCL_EXTERNAL v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, v8s A, v8i B, v8f C, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return v8f{}; }
#endif
```

- **Operands literal = `0x3000`** (`MatrixAPackedBFloat16INTEL |
  MatrixBPackedBFloat16INTEL`). Must be a compile-time constant. The backend
  validates it and prints the only legal combinations — for K=16/f32/i16-A/i32-B
  these are `0x3000` (bf16) or `0xC00` (fp16); anything else fails the build.
- **Tile = M8 x N16 x K16, sub-group size 16.** Per lane `l`:
  - A operand `v8s`: 8 bf16 bits = `A[m][k=l]` for m=0..7 (A column k=l).
  - B operand `v8i`: 8 ints VNNI-packed = for j=0..7, low=`B[2j][n=l]`,
    high=`B[2j+1][n=l]` (B column n=l, two K per int).
  - C/result `v8f`: 8 floats = `C[m][n=l]` for m=0..7.
- **Build flags:** AOT `-fsycl-targets=intel_gpu_bmg_g31` **and**
  `-Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate`.
  (The ext must be explicitly enabled or llvm-spirv refuses to translate.)
- `-Wpsabi` AVX-ABI warnings on the host side are harmless (device-only call).

Build-system implication: the TU using this needs the AOT target + spirv-ext
flag. Cleanest is to isolate the DPAS kernel in its own TU with per-file options
in CMake, leaving the rest of `diffusion_gemma` as-is (JIT). TODO confirm whether
mixed AOT(one TU)+JIT(rest) links cleanly, else AOT the whole target.

## Status & results (2026-06-14)

Kernel is **correct and integrated, but not yet faster than `hybrid`.**

- Standalone validation (`tools/dpas_gemm_test.cpp`): both modes cos=0.999999
  vs host reference on all real expert shapes.
- End-to-end: correct output ("...is **Paris**.") for both quant modes.
- Builds via **JIT** — no AOT target needed, just the link-time spirv-ext flag.

Real full generation (`--steps 48 --max-tokens 2048`, longish prompt; the
right benchmark, see NVFP4_INFERENCE_NOTES env-combinations note):

| kernel | forward/s | decode tok/s |
|---|---:|---:|
| `hybrid` (oneDNN FP4) | 2.82 | 28.8 |
| `dpas` weight-only, pre-K-split | 0.60 | 5.7 |
| `dpas` weight-only, **K-split** | **1.32** | 12.2 |
| `dpas` full, **K-split** | 0.95 | 11.1 |

K-splitting (multiple sub-groups per output tile over strided K-slices, f32
partials reduced in SLM; `ksplit_factor`, `DIFF_NVFP4_DPAS_OCC` to tune the
thread target) was a **2.2x win** for weight-only — occupancy was the dominant
bottleneck for tiny MoE buckets. Now ~2.1x slower than `hybrid` (was ~4.7x).
The earlier fast f8/f4 decode (drop `sycl::exp2`, see `e4m3_fast`) gave +26-29%.

**Negative result — gate/up+GeGLU fusion (reverted).** Fusing the GeGLU into the
gate/up epilogue (two DPAS accumulator chains per sub-group emitting `act`
directly, no `gu` buffer, no GeGLU pass) measured **~15% slower** (weight-only
1.32 -> 1.12 forward/s; full 0.95 -> 0.91). Reason confirmed by the traffic
breakdown below: intermediates are ~1% of bytes, so the saving is tiny, while
the dual-accumulator register/SLM pressure cuts occupancy. The down-projection
fusion would save even less (act ~0.4% of traffic) at more risk, so it was not
pursued. Kept the separate K-split path.

### MoE roofline + the orchestration-overhead win (2026-06-15)

Forward-pass profile (`DIFF_PROFILE=1`, profiler in `profile.cpp`) on the hybrid
path: **experts = 74%** of GPU time, attn 11%, dense MLP 6%, router 4.5%,
lm_head 4.4%. So the expert path is correctly the target.

Roofline: VRAM bandwidth measured at **536 GB/s**. Expert weights are ~11.4 GB
read/pass (all 128 experts active: 256 canvas tokens x top_k 8 / 128 = ~16
tokens/expert), so the bandwidth floor is ~21 ms/pass vs measured ~240 ms —
**~11x headroom**. Cause: the MoE shatters 256 tokens into 128 skinny M~16
GEMMs; oneDNN can't batch NVFP4 per-expert scales, so it's launch/occupancy
bound, not bandwidth bound.

Sub-profiling `experts` showed it was **~56% matmul + ~44% orchestration**. The
orchestration was almost entirely **per-call device allocation**: `run_shard`
allocated ~16 `GpuBuffer`s per layer per pass, and `sycl::malloc_device`+free
costs **~450 us/pair** here. Fix: a **persistent per-GPU `ExpertWorkspace`**
(`ws_ensure`, grow-on-demand, intentionally leaked to dodge static-destruction
order). Result: experts phase **10.55 s -> 5.60 s (-47%)** on the profiled run;
`setup+gather` 1.39 s -> 0.24 s and the ~3 s allocator gap eliminated. This is
kernel-agnostic — it speeds up hybrid, dpas, and the bf16 path alike.

**Real full-generation impact (the headline result):**

| `hybrid` (default path) | forward/s | cumulative |
|---|---:|---:|
| baseline | 2.82 | 1.00x |
| + persistent `ExpertWorkspace` | 4.58 | **1.62x** |
| + per-layer `LayerScratch` (dense MLP, dual-FFN, attn wrapper) | 4.81 | **1.71x** |
| + attention internals (Q/K/V, scores, ctx) | 5.10 | **1.81x** |

Default-path decode **28.8 -> 48.9 tok/s**. The win was allocator churn, not the
matmul — found by profiling, not kernel work. `LayerScratch` (`scratch.hpp`)
applies the persistent-buffer pattern to the dense paths and attention. The
attention internals route through shared `transpose_q`/`scatter_ctx` in
`common/layers/attention_ops.hpp`; converted via a non-breaking additive
`transpose_q_into` overload (gemma4 still builds, behavior unchanged). The
attention conversion added +6% (4.81 -> 5.10) — worth more than first estimated.

### Fused-MoE de-risk microbench (2026-06-15) — thesis REFUTED, real lever found

`tools/dpas_moe_grouped_bench.cpp` (gate shape E=128, M=16, K=2816, N=1408,
weight-only). Compared per-expert loop vs one grouped launch vs layout/dequant
variants:

| variant | GB/s | % of 536 |
|---|---:|---:|
| per-expert loop (current DPAS) | 45.7 | 8.5% |
| grouped (one launch) | 38.9 | 7.3% |
| grouped + coalesced weight layout | 52.5 | 9.8% |
| coalesced + LUT dequant | **90.2** | 16.8% |
| coalesced, NO dequant (ceiling) | 477.8 | 89.1% |

Conclusions, all measurement-backed:
- **Fused/grouped MoE does NOT help** — one launch is *slower* than the per-expert
  loop. Launch/occupancy was never the bottleneck. The ~11x "bandwidth headroom"
  was illusory: the kernel runs at ~10% of bandwidth because it is **dequant-ALU
  bound**, not memory bound (no-dequant variant hits 89% of bandwidth).
- **The real lever is a cheaper FP4->bf16 dequant.** A coalesced/blocked weight
  layout + a precomputed `(f8_scale_byte, e2m1_nibble) -> bf16` LUT (256x16 = 8 KB,
  L1-resident) replaces all per-element e2m1/e4m3/f2bf math with one table load
  and ~doubles throughput to 90 GB/s — **above oneDNN's measured ~79 GB/s**.

### coal+LUT integrated into the DPAS kernel (2026-06-15)

Built it: lazy per-GPU coalesced `weight_coal` copy (`ensure_weight_coal`) + a
global 8 KB `(scale,nibble)->bf16` LUT (`dequant_lut`), wired into both DPAS
kernels. Correct output, +~8 GB VRAM (the coalesced copies).

| DPAS path (full-gen) | before | after coal+LUT | speedup |
|---|---:|---:|---:|
| weight-only | 1.60 | **2.64** | 1.65x |
| full | ~0.95 | 1.74 | ~1.8x |

The microbench's ~2x matmul win translated to the path (1.60 -> 2.64). **But it
does NOT beat hybrid (5.10).** Correction to the earlier "coal+LUT 90 > oneDNN
79" claim: that 79 was a soft estimate from a pre-workspace profile; end-to-end,
oneDNN's FP4 matmul (in hybrid) is ~2x faster than even the optimized DPAS
kernel. The isolated microbench overstated our position.

**Verdict on the custom DPAS path:** fully optimized (K-split + coalesced + LUT)
it reaches 2.64 forward/s — a validated, correct fallback, but ~half of hybrid.
oneDNN's jit:gemm FP4 path is genuinely better end-to-end. `hybrid` (oneDNN +
the persistent workspace, 5.10, 1.81x over baseline) is the recommended default;
the DPAS work's lasting value is the reusable DPAS-access recipe, the profiler,
and the workspace/allocation fixes — not out-matmuling oneDNN.

### Why hybrid (oneDNN) wins — traffic breakdown
These are thin GEMMs (M=8..64), so they are **weight-bandwidth bound**. Per
expert bucket: gate_up weight ~5.9 MB + down weight ~3.0 MB of FP4 = ~8.9 MB,
vs ~145 KB of bf16 intermediates (gu/act/Ye). Weights are ~98% of traffic and
fusion cannot reduce them. oneDNN's `jit:gemm` also emits DPAS, but reads those
weights near bandwidth-optimally (coalesced 2D block loads, blocked layout) and
overlaps dequant with loads via software pipelining. Our kernel's remaining gap
is **uncoalesced FP4 weight reads** (adjacent lanes read columns `K/2` bytes
apart) + non-overlapped scalar dequant — not the mad, and not fusion.

Remaining gap vs `hybrid` is **not** the mad — it's:

1. **Occupancy.** Grid is `(M/8, N)` with one sub-group per 8x16 tile. MoE
   buckets are tiny (M=8..64), so gate/up at M=8 launches only `N/16 ≈ 264`
   sub-groups = ~13% of the 2048 HW threads. oneDNN's FP4 kernel keeps the
   machine full by splitting K and blocking internally.
2. **No weight reuse / SLM.** Each tile re-reads + re-dequantizes weights from
   global; ~12M scalar dequant ops per gate/up matmul with no staging.
3. **Per-bucket launch overhead.** `localE*2` tiny kernel launches per shard.

### Optimization roadmap (next phase, to actually beat hybrid)
- **K-splitting for occupancy:** multiple sub-groups accumulate partial K tiles
  into SLM/global with a reduction, so tiny-M buckets fill the machine. Biggest
  expected win.
- **SLM weight staging:** dequant a weight K-panel once into SLM, reuse across
  the M rows of the bucket (matters more once buckets are batched).
- **Batch buckets:** one launch over all active rows with a row->expert map
  (like the grouped pack), instead of per-bucket launches.
- **2D block loads** (`cl_intel_subgroup_2d_block_io`) for coalesced weight/act
  reads.

## Task log

- [x] Verify hardware + XMX shape (M8 N16 K16), find aspect gap, find builtin route.
- [x] Probes saved: `tools/matrix_probe.cpp`, `tools/jm_probe.cpp`.
- [x] Extend `gpu_check.cpp` to report matrix/DPAS reachability.
- [x] Prototype `__spirv_SubgroupMatrixMultiplyAccumulateINTEL` bf16 micro-mad,
      AOT bmg_g31, validate vs host. **DONE — `tools/dpas_mad_probe.cpp`, max_err=0.**
- [x] CMake: JIT works (no AOT needed) — spirv-ext at **link** options. TU is
      `src/common/gpu/nvfp4_dpas.{hpp,cpp}`.
- [x] Single-tile -> full GEMM: loop K in 16-steps, f32 acc, bf16 store. Weight
      FP4 dequant->bf16 VNNI B operand per lane.
- [x] Two quant modes behind `DIFF_NVFP4_DPAS_QUANT=weight|full`.
- [x] Per-bucket dispatch + `DIFF_NVFP4_EXPERT_KERNEL=dpas` in expert_parallel.
- [x] Standalone test vs host (`tools/dpas_gemm_test.cpp`, cos 0.999999);
      end-to-end correct output both modes.
- [x] Fast bit-decode for f8/f4 (drop sycl::exp2): +26-29%.
- [x] Document env combinations in NVFP4_INFERENCE_NOTES.
- [x] **K-splitting for occupancy** — 2.2x win (0.60 -> 1.32 forward/s). Now
      ~2.1x slower than hybrid (was ~4.7x). `ksplit_factor` + `DIFF_NVFP4_DPAS_OCC`.
- [x] Cross-stage fusion (gate/up+GeGLU) — **tried, reverted, ~15% slower**.
      Intermediates are ~1% of traffic; fusion can't beat a weight-bound GEMM.
- [ ] If pursued further, the only real lever left is **coalesced FP4 weight
      reads** (blocked/transposed weight layout or `cl_intel_subgroup_2d_block_io`)
      + software-pipelined dequant — i.e. reimplementing what oneDNN already does.
      Likely not worth it vs keeping `hybrid` default. `dpas` stays a validated
      fallback at 1.32 forward/s.
- [ ] Add `dpas` cases to `tools/bench_nvfp4_variants.sh`.

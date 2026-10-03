# Artifacts: reaching Battlemage DPAS through SPIR-V (NVFP4 expert matmul)

This folder preserves a custom NVFP4 expert matmul that ran on the Intel
Battlemage (Xe2) **XMX/DPAS** systolic array. It was **removed from the
production code path** because oneDNN's `jit:gemm` FP4 matmul is ~2x faster
end-to-end (see "Verdict" below). What's worth keeping is the **procedure for
reaching DPAS on a stack where the portable SYCL `joint_matrix` API is
runtime-blocked** — i.e. the descent to the raw SPIR-V matrix-mad builtin. These
files are the trail.

Hardware: 2x `Intel(R) Graphics [0xe223]`, architecture `intel_gpu_bmg_g31`
(Battlemage G31, Xe2). oneAPI 2025.3, NEO compute-runtime 26.05. VRAM 536 GB/s.

## The descent, step by step

Each probe answered one unknown; run them with `icpx -fsycl` (+ the AOT/ext flags
noted). The order below is the order the unknowns fell.

### 1. `matrix_probe.cpp` — does the runtime report matrix support?
Queries `info::device::matrix_combinations`. **Result: empty** under both JIT and
AOT. The portable query tells us nothing here.

### 2. `jm_probe.cpp` — can `joint_matrix` even run?
A minimal `joint_matrix` bf16 multiply. Two findings:
- The backend (ocloc) **rejects N=8** ("unsupported number of columns: 8,
  supported values: 16") — so the XMX tile is **M<=8, N=16, K=16** for bf16.
- At runtime it throws **`"no matrix hardware on the target device"`**, because
  `device.has(aspect::ext_intel_matrix)` is **`0`** on this stack (both Level-Zero
  and OpenCL backends) — even though the hardware has XMX and oneDNN uses it.
  The guard lives in libsycl; not bypassable.

So the clean portable path (`joint_matrix`) is dead here. But disassembling the
compiled kernel (`llvm-spirv -to-text`) showed modern `joint_matrix` lowers to
`SPV_KHR_cooperative_matrix` (opaque cooperative-matrix types) — unusable without
the blocked wrappers. The reachable path is the **older INTEL extension**, whose
op takes plain vector types: `cl_intel_subgroup_matrix_multiply_accumulate`,
advertised by the OpenCL platform. This is the same mechanism oneDNN/ggml use.

### 3. `dpas_mad_probe.cpp` — nail the SPIR-V intrinsic
Declare `__spirv_SubgroupMatrixMultiplyAccumulateINTEL` ourselves and let the
compiler errors teach the exact form (the operand sweep is in the file). The
recipe that produces a bit-exact result (`max_err=0`):

```cpp
// Native Clang vectors, NOT sycl::vec (a struct return forces an sret pointer
// that breaks the op's operand count).
using v8s = short __attribute__((ext_vector_type(8)));
using v8i = int   __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));
SYCL_EXTERNAL v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, v8s A, v8i B, v8f C, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;                       // device: provided by the extension
#else
    { return v8f{}; }       // host stub: never runs, satisfies the host link
#endif
```
- **Operands = `0x3000`** (`MatrixAPackedBFloat16INTEL | MatrixBPackedBFloat16INTEL`)
  and must be a **compile-time constant**. The backend validates it and prints the
  legal set (`0x3000` bf16, `0xC00` fp16) for K=16/f32/i16-A/i32-B.
- **Tile = M8 x N16 x K16, sub-group size 16.** Per lane `l`: A operand = 8 bf16
  bits `A[m][k=k0+l]` (column k0+l); B operand = 8 ints VNNI-packed `B[k][n=n0+l]`
  (column n0+l, 2 K per int); C/result = 8 f32 `C[m][n0+l]`.
- **Build flags:** `-fsycl-targets=intel_gpu_bmg_g31` (AOT) and
  `-Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate`.
  The extension must be explicitly enabled or `llvm-spirv` refuses to translate.
  It also **JITs** at runtime (no AOT target) as long as the ext is enabled at the
  device-link step.

### 4. `dpas_gemm_test.cpp` — full GEMM, validated
Builds the K-looped weight-only and full-NVFP4 GEMMs from the single mad and
checks against a host reference on the real expert shapes (cos 0.999999).

### 5. `dpas_moe_grouped_bench.cpp` — the de-risk that killed the idea
Bandwidth microbench at the gate shape (E=128, M=16, K=2816, N=1408). This is the
file that turned the whole effort around:

| variant | GB/s | % of 536 |
|---|---:|---:|
| per-expert loop | 45.7 | 8.5% |
| grouped (one launch) | 38.9 | 7.3% |
| grouped + coalesced layout | 52.5 | 9.8% |
| coalesced + LUT dequant | 90.2 | 16.8% |
| coalesced, NO dequant (ceiling) | 477.8 | 89.1% |

- The "fused MoE for bandwidth" thesis is **refuted** — one grouped launch is
  *slower* than the loop. Launch/occupancy was never the bottleneck.
- With dequant removed, the same kernel hits **89% of bandwidth** → the kernel is
  **dequant-ALU bound, not memory bound**. The "~11x headroom" was illusory.
- The real lever is a cheaper dequant: a coalesced/blocked weight layout + a
  `(f8_scale, e2m1_nibble) -> bf16` LUT (8 KB, L1-resident) ~doubles throughput.

### 6. `nvfp4_dpas.hpp` / `nvfp4_dpas.cpp` — the production kernel (as removed)
The integrated version: K-split for occupancy, lazily-built coalesced weight copy
(`ensure_weight_coal`), the global dequant LUT (`dequant_lut`), weight-only and
full modes. This is the end-to-end proof the SPIR-V path works in the real model.

## Verdict (why it was removed)

Fully optimized (K-split + coalesced + LUT) the kernel reached, on full
generation: **dpas 2.67 forward/s, dpas-full 1.76** — versus **hybrid (oneDNN +
the persistent-workspace fix) at 5.16**. These are weight-bandwidth-bound thin
MoE GEMMs (256 canvas tokens split across 128 experts = M~16 each), the matmul is
dequant-bound, and oneDNN does that dequant better. There is no path for this
custom kernel to beat oneDNN here, so it was cut to reclaim the maintenance
surface (a `weight_coal` field on the shared `Nvfp4Linear`, the `-spirv-ext` link
flag, the dispatch branches, +8 GB VRAM when enabled).

## If you revive it

It could win where DPAS is **not** dequant-starved — a compute-bound, large-M
GEMM (e.g. prefill at large batch), not these thin MoE matmuls. Or if a driver
update flips `aspect::ext_intel_matrix` on, the cleaner `joint_matrix` path (tile
M8/N16/K16) becomes viable and you can swap the inner mad. To re-integrate:
restore `nvfp4_dpas.{hpp,cpp}` to `src/common/gpu/`, the `Nvfp4Kernel::Dpas`
dispatch branch in `expert_parallel.cpp`, the `weight_coal` field on
`Nvfp4Linear`, the TU in `CMakeLists.txt`, and the `-spirv-ext` link flag.

See also `../diffusion_gemma/NVFP4_DPAS_KERNEL.md` (full chronological log) and
`../diffusion_gemma/NVFP4_INFERENCE_NOTES.md` (env/runtime config).

# OpenVINO GDN: fused vs non-fused path, DPAS/XMX — full investigation

Date: 2026-08-08. Device: BMG (Xe2), oneAPI 2026.1. OpenVINO master 0f453eb8 (2026-08-07).

## Summary

OpenVINO master's **fused** GDN primitive is a single-stage OCL kernel with no
DPAS. But the user's XMX expectation is correct for the **non-fused** path: the
loop-body GEMMs route through oneDNN's `jit:gemm` (GemmStone DSL, ngen-based
in-process JIT) which emits DPAS on Xe2. Measured head-to-head on BMG:

| path | p=512 | p=1024 | p=2048 | p=4096 |
|---|---:|---:|---:|---:|
| fused OCL (master's shipped kernel) | 3.5k | 3.4k | 3.4k | 3.3k tok/s |
| **non-fused FULL** (complete loop body) | 36.9k | 37.9k | 37.8k | 37.6k |
| non-fused, gemm-only (of the full) | — | — | — | ~72% of full |
| non-fused, batched over T (N=T) | 5.6M | 7.6M | 9.5M | 10M (invalid) |
| Arcaine hybrid | 208k | 217k | 222k | 227k |
| Arcaine xmx | 193k | 198k | 205k | 211k |

## How the DPAS path works (source-verified)

1. Qwen3.5's GDN is exported as a `Loop` whose body is the linear-attention
   recurrence built from `Multiply`/`ReduceSum`/`Exp`/`ScatterUpdate`.
2. `GatedDeltaNetFusion` (transformations_pipeline.cpp:672) collapses that loop
   into a single `GatedDeltaNet`/`PagedGatedDeltaNet` primitive — **unconditionally**,
   no runtime toggle.
3. Fused dispatch: `PagedGatedDeltaNetOptImpl` registers exactly ONE stage —
   the `paged_gated_delta_net_opt.cl` kernel. Compiled via OCL-source path with
   exact build flags (`-cl-mad-enable -cl-std=CL2.0`): **zero DPAS**, scalar
   FP32 mul/add inner loop.
4. Non-fused dispatch (if fusion doesn't match): the loop-body ops become
   `gemm`/`reduce` primitives whose registries include `OV_GPU_WITH_ONEDNN`
   impls (`impls/onednn/gemm_onednn.cpp`, `reduce_onednn.cpp`). These require
   `supports_immad` = the DPAS capability flag (`CL_DEVICE_FEATURE_FLAG_DPAS_INTEL`
   / `ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS`), which is true on BMG.
5. `gemm_onednn` creates real oneDNN primitives → the `onednn_gpu` submodule's
   GEMM JIT (`jit_xe_hp_systolic.cpp`, 21 DPAS refs; GemmStone DSL on ngen)
   emits DPAS **in-process, bypassing IGC entirely**. `DNNL_VERBOSE=1` confirms
   `jit:gemm:any` for the bf16 matmuls.

## Honest non-fused measurement

The complete loop body per token: decay (`S *= exp(g)`), `h_k = S@k_t` (oneDNN
GEMV), `delta = (v-h_k)*beta`, rank-1 update `S += k_t·delta`, `o = S@q_t`
(oneDNN GEMV) — 4 kernel launches/token, sequential dependency on S.

- **36.9–37.9k tok/s**, flat vs T (sequential chain).
- Breakdown at p=4096 (108.9 ms): 2× GEMV matmuls ≈ 78.5 ms (72%, N=1 matmul is
  launch-bound), elementwise decay+update ≈ 30 ms (28%).
- The 10M tok/s batched number is invalid for GDN: it puts all T in the matmul
  N dimension with a shared S, which deletes the recurrence.

## Conclusions

- The fused kernel (master's shipped GDN) is the slow option: 3.4k tok/s.
- Non-fused (DPAS GEMV per token): ~11x faster than fused (37k), confirming the
  user's XMX intuition — but only when the recurrence is GEMM-decomposed.
- Arcaine hybrid still beats non-fused by ~5.5-6x (208-227k vs 37k): chunking
  over T + parallel Gram solve wins over per-token DPAS GEMVs. The sequential
  dependency caps any per-token strategy at ~38k on this shape.
- If OpenVINO wanted fast GDN prefill it should fuse the dot products into
  chunked oneDNN GEMM calls (as Arcaine does), not the OCL sequential kernel.

## Artifacts (ngen_lab/ovino/)

- `gdn_onednn_full.cpp` — complete non-fused loop body benchmark (the honest number)
- `gdn_onednn_seq.cpp` — gemm-only sequential (breakdown)
- `gdn_onednn_gemm.cpp` — batched-over-T (invalid but shows GEMM ceiling)
- `gdn_seq_scalar.cl` + `ovino_scalar` — fused-OCL-equivalent scalar kernel
- Build: `icpx -fsycl -fno-sycl-id-queries-fit-in-int -O2 -I/opt/onednn/include
  <f>.cpp -o build_jit/ngen_lab/<f> -L/opt/onednn/lib -ldnnl`

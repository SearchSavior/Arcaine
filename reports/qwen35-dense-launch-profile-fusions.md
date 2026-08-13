# Qwen3.5 dense: launch profiler + fusion optimization pass

Date: 2026-08-12. Hardware: Intel Arc Pro B70 (34.2GB), oneAPI 2026.1 +
/opt/onednn. Container: arcaine-dev-run-0a46febbd53b. Model:
models/cyankiwi_Qwen3.6-27B-AWQ-INT4.

## 1. Launch profiler (new)

`src/runtime/profiling/launch_prof.hpp` (header-only, namespace `launchprof`),
env `ARCAINE_LAUNCH_PROFILE`. Launch census + event-based per-launch GPU time:
- `GpuEngine` builds the SYCL queue with `enable_profiling` when the env is on
  (`mark_queue_profiling` flag); per-kernel times are read from submit events
  WITHOUT waiting (no pipeline serialization, unlike the wait-based DIFF_PROFILE
  timers).
- ~40 instrumented sites: matmul_int4 pieces, bf16 GEMMs (count-only — oneDNN
  `execute` returns void), every qwen3_5 kernel, cache memcpys, uploads, logits
  D2H. `forward()` does `reset()` at entry + `report()` at exit.
- oneDNN-internal kernel timing: `DNNL_VERBOSE=2` (zero code).
- `report()`: per-name count + total/avg GPU ms, sorted. Approx oneDNN GEMM
  time = forward wall time − profiled GPU time.

## 2. Census findings (prefill 512, one forward)

- ~1511 launches (full instrumentation), decode ~1303/token.
- **256 int4 GEMMs + 256 rowsum + 256 corr GEMMs + 256 corr-subtracts per
  forward (4 projections × 64 layers)**; `int4_corr_sub` alone = 67µs avg,
  17.1ms total (29% of profiled GPU) — pure C-buffer round-trip traffic.
- Decode: only 1.36ms of ~40ms/token is visible → ~38ms is inside oneDNN s4
  GEMMs (weight-bandwidth-bound, confirmed).
- Prefill: esimd GDN 19.9ms, conv 8.1ms, extract_qkv 4.9ms, xmx attention
  4.9ms, copy_strided 1.6ms.

## 3. Optimizations (all env-gated, default off, verified bit-exact unless noted)

| env | change | launch savings | perf (B70, pp512@d / tg128@d) |
|---|---|---|---|
| ARCAINE_QWEN35_INT4_CORR_FUSED | zp-corr subtract folded into consumers (split/swiglu/conv/add_inplace); per-projection corr computed via matmul_int4_zp_corr + corr work buffer | −256/forward, −208/token | pp +4.0-4.5%, tg +0.7-0.9% |
| ARCAINE_QWEN35_SPLIT_WRITE_CACHE | fused qkv split writes K/V straight into the KV cache | −32/forward, −32/token | +0.4% pp, +0.5% tg |
| ARCAINE_QWEN35_SPLITKV_FUSED_EPILOGUE | splitkv combine writes O·sigmoid(gate) to query buffer in place | −32/token (decode) | +0.2-0.4% tg |
| ARCAINE_QWEN35_GDN_FUSIONS | conv+state, l2norm×2+scale, sigmoid+compute_g → 1 launch each | −144/forward (chunked) | neutral-to-+0.4% |
| ARCAINE_QWEN35_LOGITS_F32 | lm_head GEMM dst f32 (full-precision logits; kills bf16→f32 kernel) | −1/token | precision win; f32-gemm ≥ bf16-gemm speed |

**Combined (all five): pp512 @ d2048/4096/8192 = 1668/1555/1360 t/s (baseline
1589/1483/1303, +5.0/4.8/4.4%), tg128 = 24.46/23.19/21.00 (baseline
24.05/22.85/20.72, +1.7/1.5/1.4%).**

Bit-exactness: all fusions bf16-round the `C − corr` difference exactly like the
subtract kernel; verified with bit-mismatch=0 kernel gates in
`arcaine_kbench qwen35-int4-fusion` (split q/k, swiglu, add_inplace,
update_conv_state_corr, conv_causal_state out+state, l2norm_scale_k q/k,
sigmoid_beta_compute_g). e2e: greedy generation token-identical baseline vs
full stack.

## 4. Dequant-weight path: memory-infeasible on B70 (documented, reverted)

Pre-dequantized weights (fold zp, run bf16/s8 GEMM) measured 1.35-1.48x faster
at M=512 in kbench (`bench_dequant_bf16`), but the 27B dense model needs
~48GB (bf16) / ~24GB (s8) of dequantized weights vs 34.2GB device and ~19.4GB
baseline usage. USM allocations silently spill to system memory → 10-1000x
kernel slowdowns e2e (measured: pp512 22 t/s). Per-layer staging math is a
wash. Kept as a kbench-only A/B gate for bigger-memory targets; `Int4Linear::
ensure_dequantized_bf16` retained for it.

## 5. Promotion recommendation

O2 (corr fusion), O3 (cache write), O4 (splitkv epilogue): clear bit-exact
wins → promote to default. O5: neutral but bit-exact launch reduction →
promote or keep opt-in. LOGITS_F32: numeric change (logits max_abs 0.0069 vs
bf16-rounded) → keep opt-in pending user decision. Envs stay default-off until
promoted.

## 6. Known issue (pre-existing, unrelated to this work)

This checkpoint/container currently generates degenerate "!!!!" sequences for
the capital-of-France prompt on BOTH baseline and fused paths (bit-identical).
The memory-note "Paris" smoke was from an earlier session/build; not
investigated here (out of scope).

## Commands

Build: `docker exec -i arcaine-dev-run-0a46febbd53b sh -c 'cd /workspace &&
cmake --build build -j$(nproc)'`
Bench: `LD_LIBRARY_PATH=/opt/onednn/lib:/opt/intel/oneapi/2026.1/lib
ZE_AFFINITY_MASK=0 ARCAINE_QWEN35_INT4_CORR_FUSED=1
ARCAINE_QWEN35_SPLIT_WRITE_CACHE=1 ARCAINE_QWEN35_SPLITKV_FUSED_EPILOGUE=1
ARCAINE_QWEN35_GDN_FUSIONS=1 ARCAINE_QWEN35_LOGITS_F32=1 ./build/arcaine_mbench
--model models/cyankiwi_Qwen3.6-27B-AWQ-INT4 -p 512 -n 128 -d 2048,4096,8192
-r 1 --device 0`
Profile: add `ARCAINE_LAUNCH_PROFILE=1`.

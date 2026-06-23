# DiffusionGemma NVFP4 Inference Notes

Working notes for adding support for
`models/diffusiongemma-26B-A4B-it-NVFP4`.

## Model Format Findings

- The quantized model is a single `model.safetensors` file plus an index.
- `config.json` declares `quantization_config.format =
  "nvfp4-pack-quantized"`.
- Quantization was produced with `llm-compressor` / `compressed-tensors`.
- The recipe targets `Linear` modules but ignores:
  - embeddings / lm head
  - self-attention
  - routers
  - self-conditioning
  - vision tower
- In the text decoder this means attention, router, embedding, lm head, layer
  norms, layer scalars, and self-conditioning remain BF16. Dense MLP and expert
  MLP weights are NVFP4.

## Safetensors Layout

The NVFP4 file has these dtype counts:

| dtype | tensor count |
|---|---:|
| `BF16` | 897 |
| `F32` | 23220 |
| `F8_E4M3` | 11610 |
| `U8` | 11610 |

For a dense decoder MLP linear, example
`model.decoder.layers.0.mlp.gate_proj`:

| tensor suffix | dtype | shape | meaning |
|---|---|---:|---|
| `weight_packed` | `U8` | `[2112, 1408]` | packed FP4 values for logical `[2112, 2816]` |
| `weight_scale` | `F8_E4M3` | `[2112, 176]` | one scale per output row and 16 input channels |
| `weight_global_scale` | `F32` | `[1]` | global weight multiplier |
| `input_global_scale` | `F32` | `[]` | activation quantization global scale |

For an expert linear, example
`model.decoder.layers.0.experts.0.gate_proj`:

| tensor suffix | dtype | shape | meaning |
|---|---|---:|---|
| `weight_packed` | `U8` | `[704, 1408]` | packed FP4 values for logical `[704, 2816]` |
| `weight_scale` | `F8_E4M3` | `[704, 176]` | one scale per output row and 16 input channels |
| `weight_global_scale` | `F32` | `[1]` | global weight multiplier |
| `input_global_scale` | `F32` | `[]` | activation quantization global scale |

The BF16 model stores experts as consolidated tensors:
`experts.gate_up_proj` and `experts.down_proj`. The NVFP4 model stores each
expert and projection separately:
`experts.{expert_id}.{gate_proj,up_proj,down_proj}.*`.

## oneDNN Findings

- The installed oneDNN is `3.12.0`.
- oneDNN exposes `memory::data_type::f4_e2m1`.
- oneDNN exposes `memory::data_type::f8_e4m3`.
- A GPU `matmul` primitive can be created for packed FP4 weights without scale
  attributes.
- For grouped weight scales, the primitive descriptor accepts:
  - weights logical descriptor: `{K, N}`, `f4_e2m1`, `format_tag::ba`
  - scale attribute: `DNNL_ARG_WEIGHTS`, mask `3`, groups `{16, 1}`,
    scale dtype `f8_e4m3`, quantization mode `static_sazp`
- Scale masks with only dimension `0` (`mask = 1`, groups `{16}`) crashed the
  local oneDNN build during primitive descriptor creation, so the implementation
  should use the two-dimensional mask above.

## Dequantization Cross-Check

Compared `model.decoder.layers.0.mlp.gate_proj` from the NVFP4 model against
the BF16 model:

- Packed FP4 nibble order is low nibble first.
- `weight_scale` values decode as FP8 E4M3.
- The dequantized weight approximation matches the BF16 reference when using:

```text
weight ~= fp4(weight_packed) * fp8_e4m3(weight_scale) / weight_global_scale
```

For an 8x128 slice from layer 0 gate projection:

| variant | MAE | RMSE | cosine |
|---|---:|---:|---:|
| low nibble first, divide by global | 0.00248 | 0.00330 | 0.9954 |
| high nibble first, divide by global | 0.03778 | 0.04750 | 0.0500 |

## oneDNN Probe Log

- BF16 matmul execution is healthy and selects `jit:gemm:any`.
- Unscaled FP4 matmul execution succeeds for:
  - `bf16 x f4_e2m1 -> bf16`
  - `f32 x f4_e2m1 -> f32`
  - `f4_e2m1 x f4_e2m1 -> f32`
- Grouped FP4 weight scales execute successfully when using:
  - scale attribute: `DNNL_ARG_WEIGHTS`, mask `3`, groups `{16, 1}`
  - quantization mode: `static_sazp`
  - scale dtypes tested: `f32`, `f8_e4m3`, `e8m0`
- `query::exec_arg_md` for `DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS` returns
  an empty descriptor, but the runtime still consumes the scale argument.
- Scale values affect output as expected in a synthetic `M=1,K=64,N=1` test:
  `0.5 -> 16`, `1.0 -> 32`, `2.0 -> 64`.
- oneDNN consumes grouped scales in group-major order `[K / 16, N]`. The model
  stores scales in row-major `[N, K / 16]`, so scales must be transposed during
  upload.
- oneDNN packed FP4 nibble order is low nibble first.

## oneDNN Batched Expert Probes

For a synthetic 3D batched matmul with weights laid out like current expert
GEMMs (`{B, K, N}` logical weights, stored as `{B, N, K}`):

| scale setup | result |
|---|---|
| no scales | primitive created (`ocl:ref:any`) |
| `mask=6`, groups `{16,1}` | primitive created; scales shared across batch |
| `mask=7`, groups `{1,16,1}` | empty primitive descriptor |
| `mask=3`, groups `{16,1}` | primitive created, but mask does not encode per-output scales for `{B,K,N}` |
| `mask=2`, groups `{16}` | local oneDNN floating-point exception |

Implication: oneDNN does not appear to support the needed per-expert,
per-output, per-16-input scale tensor as one batched FP4 matmul. Quantized
experts should use per-expert 2D matmuls, or a custom grouped kernel, rather
than the existing BF16 batched expert primitive.

## Global Scale Handling

`DNNL_ARG_DST` host output scales use an inverse convention in the tested
matmul path. With packed FP4 values and weight scales producing an unscaled
result of `32`:

| dst host scale | output |
|---:|---:|
| `0.5` | `64` |
| `1.0` | `32` |
| `2.0` | `16` |

This matches the compressed-tensors dequantization formula: use the stored
`weight_global_scale` as the oneDNN destination scale to compute
`fp4 * weight_scale / weight_global_scale` without a separate post-matmul
kernel.

## Full NVFP4 Matmul Path

A full FP4 source + FP4 weights matmul works in oneDNN and selects
`jit:gemm:any` for tested shapes. Required attributes:

- source descriptor: `{M, K}`, `f4_e2m1`, `ab`
- source scales: `DNNL_ARG_SRC`, mask `3`, groups `{1, 16}`, dtype
  `f8_e4m3`, static scales
- weight descriptor: `{K, N}`, `f4_e2m1`, `ba`
- weight scales: `DNNL_ARG_WEIGHTS`, mask `3`, groups `{16, 1}`, dtype
  `f8_e4m3`, static scales
- destination descriptor: `{M, N}`, `bf16`, `ab`
- destination host scale can be provided with `DNNL_ARG_DST`

Synthetic full-NVFP4 tests pass for these real model dimensions:

| M | K | N | role | impl |
|---:|---:|---:|---|---|
| 1 | 2816 | 2112 | dense gate/up | `jit:gemm:any` |
| 8 | 2816 | 2112 | dense gate/up | `jit:gemm:any` |
| 1 | 2112 | 2816 | dense down | `jit:gemm:any` |
| 8 | 704 | 2816 | expert down | `jit:gemm:any` |

oneDNN reorder does **not** create a BF16-to-FP4 activation packing primitive
for either dynamic or static grouped scales. Activation packing must be a custom
SYCL kernel.

## Implementation Notes

Implemented runtime support uses a mixed path:

- BF16 tensors continue through the existing loader and BF16 oneDNN matmuls.
- Dense MLP linears are represented as `DiffLinearWeight`, which dispatches to
  BF16 or NVFP4 at execution time.
- Expert weights are loaded either from the legacy consolidated BF16 tensors or
  from per-expert NVFP4 projections.
- NVFP4 experts currently run as per-expert 2D matmuls. This avoids unsupported
  batched per-expert scale layouts in oneDNN.
- Activation packing is a custom SYCL kernel that computes one FP8 E4M3 scale
  per row and per 16 input channels, then packs two E2M1 FP4 values per byte in
  low-nibble-first order.
- The runtime passes source scales as `{M, K / 16}`, weight scales as
  `{K / 16, N}`, and a one-element destination scale equal to
  `input_global_scale * weight_global_scale`.

The project uses oneDNN headers whose C++ API supports:

```cpp
attr.set_scales(DNNL_ARG_SRC, 3, {1, 16}, dt::f8_e4m3);
attr.set_scales(DNNL_ARG_WEIGHTS, 3, {16, 1}, dt::f8_e4m3);
attr.set_scales(DNNL_ARG_DST, 0, {}, dt::f32);
```

The newer convenience API names used in early probes (`set_host_scale`,
`memory::desc::host_scalar`, and the explicit quantization mode argument) are
not present in the local oneDNN headers used by this build. A bounded probe
confirmed that the old API plus a one-element f32 USM destination-scale memory
keeps the same inverse convention: `0.5 -> 64`, `1.0 -> 32`, `2.0 -> 16` in the
synthetic FP4 matmul test.

`matmul_nvfp4` currently waits for completion before returning because it owns
its temporary packed activation and activation-scale buffers. This is correct
for the first inference path; a future optimization should pass reusable
workspaces from the caller to regain asynchronous overlap.

## Placement

NVFP4 auto-placement now targets a single GPU by default. When
`quantization_config.format` is `nvfp4-pack-quantized`, unresolved placement
options are rewritten as:

- `layers=single`
- `experts=layer-owner`

Explicit CLI placement flags still override this. The dry-run estimator is also
quantization-aware: it counts NVFP4 linears as packed FP4 weights plus FP8
per-group scales, while attention, routers, embeddings, norms, and
self-conditioning remain BF16.

A bounded default run, with no `--layers` or `--experts` flags, confirmed the
single-GPU target:

```text
[model] 2 GPU(s); 30 layers, split at 30; experts=layer-owner
[load] layer 25/30 (sliding) -> GPU 0
[perf] decode  :    11.3 tok/s effective  (11 tokens in 0.97 s)
[perf]              1.03 forward passes/s  (1 passes, 11.0 tok/forward)
[perf] GPU0    :  16.9 / 32.5 GB VRAM
[perf] GPU1    :   0.0 / 32.5 GB VRAM
```

## Smoke Test

After implementation, a bounded one-step run against the quantized weights
completed successfully:

```text
./build/diffusion_gemma \
  --model models/diffusiongemma-26B-A4B-it-NVFP4 \
  --prompt Hello \
  --max-tokens 1 \
  --steps 1 \
  --no-stream \
  --no-print-placement \
  --experts shard
```

Observed output:

```text
[load] 47337 tensors across 1 shards
[load] done (split at layer 14)
[main] load: 8.8 s
[main] generated 9 tokens in 2.0 s
[perf] GPU0    :   8.4 / 32.5 GB VRAM
[perf] GPU1    :   7.2 / 32.5 GB VRAM
```

The first attempted smoke run used `--max-seq 128` and failed before loading
weights because the decoder canvas length is 256. Leaving the default max-seq
capacity fixed that.

## Optimization Log

After switching NVFP4 auto-placement to single GPU, the first optimization pass
focused on reducing activation packing, launch count, and per-matmul setup:

- `Nvfp4Linear` now owns a persistent one-element destination scale buffer.
  This removes a device allocation/upload from every NVFP4 matmul.
- The activation E4M3 scale encoder was changed from a brute-force search over
  positive FP8 codes to a direct rounding encoder. A host-side sample check
  matched the old brute-force encoder across sampled linear/log ranges.
- `matmul_nvfp4_packed` was added so callers can pack an activation once and
  reuse it for multiple oneDNN matmuls.
- Expert down projections now use caller-owned packed activation workspaces and
  enqueue matmuls asynchronously inside the shard, relying on the existing
  end-of-shard wait as the lifetime fence.
- Dense and expert gate/up projections are fused at load time by concatenating
  packed weight rows and building a combined `[K / 16, 2N]` scale tensor for
  oneDNN. This replaces two gate/up matmul launches with one.

Bounded smoke timings on the single-GPU NVFP4 default path:

| state | command shape | decode | forward passes |
|---|---|---:|---:|
| single-GPU baseline | `--steps 1 --max-tokens 1` | `11.3 tok/s effective` | `1.03 forward/s` |
| fast pack + shared gate/up pack | `--steps 1 --max-tokens 1` | sampler committed 0 tokens | `1.24 forward/s` |
| async expert down | `--steps 2 --max-tokens 1` | `7.8 tok/s effective` | `1.73 forward/s` |
| fused gate/up | `--steps 2 --max-tokens 1` | `8.9 tok/s effective` | `1.99 forward/s` |
| fused gate/up | `--steps 1 --max-tokens 1` | `14.3 tok/s effective` | `1.43 forward/s` |


## Env-Scoped Optimization Experiments

Added runtime toggles so the NVFP4 path can probe Xe-oriented layout, expert
bucket choices, and custom expert kernels without changing normal CLI behavior:

| env var | values | default | purpose |
|---|---|---|---|
| `DIFF_NVFP4_WEIGHT_LAYOUT` | `raw`, `any`, `xe`, `reorder` | `raw` | Select raw model-packed weights or a oneDNN `format_tag::any` weights descriptor. `xe` and `reorder` are aliases for `any`. |
| `DIFF_NVFP4_EXPERT_ROUND` | `1`, `8`, `16`, `32`, `64` | `8` | Round each active expert bucket's `M` before the per-expert NVFP4 matmuls. |
| `DIFF_NVFP4_EXPERT_KERNEL` | `hybrid`, `grouped-pack`, `pack`, `custom`, `grouped`, `1` | unset | `hybrid` groups activation packing and GeGLU while keeping oneDNN FP4 matmuls; `custom` replaces expert matmuls with a scalar grouped kernel. |
| `DIFF_NVFP4_VERBOSE` | any value | off | Print oneDNN primitive implementation and concrete weight descriptor details. |
| `DIFF_FORCE_DENOISE_STEPS` | any value | off | Benchmark-only: ignore adaptive stability stopping so `--steps N` runs exactly `N` decode passes. |

### Env combinations

- `DIFF_NVFP4_EXPERT_KERNEL` selects one expert path; the variants are mutually
  exclusive. Unset = per-expert oneDNN FP4 matmuls (the baseline).
- `DIFF_NVFP4_EXPERT_ROUND` applies to `custom`/`hybrid`.
- `DIFF_NVFP4_WEIGHT_LAYOUT` affects only the oneDNN matmul paths.
- `DIFF_FORCE_DENOISE_STEPS` is orthogonal and composes with any kernel; use it
  for apples-to-apples per-pass throughput. For real-world signal prefer a full
  generation (`--steps 48 --max-tokens 2048`, longish prompt, no force).

**Benchmarking:** prefer `build/diffusion_bench` (llama-bench style) — it loads
the model once and sweeps kernels in-process with warmup + reps + mean/stddev:
`./build/diffusion_bench --model <dir> --kernels default,hybrid,custom --reps 5
[--md]`. It uses the runtime `set_nvfp4_kernel()` API rather than the env vars,
so no per-config model reload. See `src/diffusion_bench.cpp`.

A tiled Battlemage XMX/DPAS expert kernel (reached via the SPIR-V Intel
sub-group matrix-mad builtin, since portable SYCL `joint_matrix` is
runtime-blocked on this stack) was explored and **removed** — it is correct but
~2x slower than `hybrid` end-to-end (the MoE matmul is dequant-bound, and oneDNN
does the dequant better). The full code, the SPIR-V descent procedure, and the
de-risk measurements are preserved under `docs/artifacts/`.

The reorder experiment is applied in `matmul_nvfp4_packed`: the primitive cache is
keyed by `(gpu, M, K, N, layout)`, and when `DIFF_NVFP4_WEIGHT_LAYOUT=any` the
first call for a weight lazily reorders `Nvfp4Linear::weight_packed` into the
primitive's concrete `weights_desc`. This is the right insertion point for a
probe because oneDNN is allowed to make the concrete `any` descriptor
primitive-dependent. If a future oneDNN starts selecting a real opaque Xe-blocked
format, the same transform should move earlier to load time so the raw and
reordered copies are not both resident.

Current oneDNN 3.12.0 does not select such a blocked format for the tested NVFP4
matmuls. With `DIFF_NVFP4_VERBOSE=1`, representative descriptors are still dense
`{K, N}` row-major layouts:

```text
[nvfp4] gpu=0 M=256 K=2816 N=4224 layout=any impl=jit:gemm:any weight_bytes=5947392 dims=(2816,4224) strides=(4224,1)
[nvfp4] gpu=0 M=256 K=2112 N=2816 layout=any impl=jit:gemm:any weight_bytes=2973696 dims=(2112,2816) strides=(2816,1)
[nvfp4] gpu=0 M=8 K=704 N=2816 layout=any impl=jit:gemm:any weight_bytes=991232 dims=(704,2816) strides=(2816,1)
```

A repeatable one-off benchmark script now lives at
`tools/bench_nvfp4_variants.sh`. It now defaults to a full-generation benchmark
shape: `MAX_TOKENS=256`, `STEPS=48`, and `FORCE_DENOISE_STEPS=1` so every case
runs exactly 48 decode passes.

```bash
OUT_DIR=/tmp/nvfp4_bench_full \
TIMEOUT_SECONDS=1800 \
./tools/bench_nvfp4_variants.sh
```

Full-generation forced-48-pass run on 2026-06-14 with the local Arc GPU setup and
`models/diffusiongemma-26B-A4B-it-NVFP4`:

| case | env | decode | forward passes | GPU0 VRAM | wall |
|---|---|---:|---:|---:|---:|
| `baseline` | `DIFF_FORCE_DENOISE_STEPS=1` | `0.6 tok/s effective` | `3.06 forward/s` | `16.8 / 32.5 GB` | `29.852 s` |
| `hybrid` | `DIFF_FORCE_DENOISE_STEPS=1 DIFF_NVFP4_EXPERT_KERNEL=hybrid` | `0.6 tok/s effective` | `3.34 forward/s` | `16.8 / 32.5 GB` | `28.557 s` |
| `hybrid_round32` | `DIFF_FORCE_DENOISE_STEPS=1 DIFF_NVFP4_EXPERT_KERNEL=hybrid DIFF_NVFP4_EXPERT_ROUND=32` | `0.6 tok/s effective` | `3.34 forward/s` | `16.8 / 32.5 GB` | `28.589 s` |
| `round32` | `DIFF_FORCE_DENOISE_STEPS=1 DIFF_NVFP4_EXPERT_ROUND=32` | `0.5 tok/s effective` | `3.09 forward/s` | `16.8 / 32.5 GB` | `29.851 s` |
| `weight_any` | `DIFF_FORCE_DENOISE_STEPS=1 DIFF_NVFP4_WEIGHT_LAYOUT=any` | `0.6 tok/s effective` | `2.94 forward/s` | `25.7 / 32.5 GB` | `31.890 s` |
| `any_round32` | `DIFF_FORCE_DENOISE_STEPS=1 DIFF_NVFP4_WEIGHT_LAYOUT=any DIFF_NVFP4_EXPERT_ROUND=32` | `0.5 tok/s effective` | `2.93 forward/s` | `25.9 / 32.5 GB` | `31.934 s` |
| `custom_matmul` | `DIFF_FORCE_DENOISE_STEPS=1 DIFF_NVFP4_EXPERT_KERNEL=custom` | `0.1 tok/s effective` | `0.46 forward/s` | `16.8 / 32.5 GB` | `118.738 s` |

Conclusion: the first useful custom-kernel win is the `hybrid` expert path, not
the full scalar matmul replacement. `hybrid` groups NVFP4 activation packing and
GeGLU across active expert rows, then keeps oneDNN's FP4 matmul microkernel. It
improved forced full-generation throughput from `3.06` to `3.34` forward/s
(about 9%). `DIFF_NVFP4_EXPERT_ROUND=32` is neutral once hybrid is enabled. The
full scalar grouped custom matmul is correct enough to run a generation, but is
far too slow because it gives up oneDNN/Xe FP4 matmul acceleration. The oneDNN
`any` reorder path remains a regression: it creates a second dense weight copy,
raises GPU0 memory by roughly 9 GB, and is slower than raw weights.

## Remaining Work

- Replace per-call NVFP4 activation workspaces with caller-owned reusable
  buffers so quantized matmuls can be asynchronous again.
- Replace the scalar custom expert matmul with a tiled Xe/DPAS-oriented kernel
  before considering it as a replacement for oneDNN FP4 matmuls. The first scalar
  version is much slower than oneDNN.
- Add a small automated matmul test that compares NVFP4 dequantized outputs
  against BF16 reference slices for dense and expert projections.

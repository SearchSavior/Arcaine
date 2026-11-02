# Flint native joint_matrix AWQ prefill

Enable the native W4A16 prefill path with
`ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas` (subgroup 16) or
`ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas-sg32` (subgroup 32).
Both kernels explicitly request **256 GRFs** through the Intel
`grf_size<256>` kernel property.

This selector takes precedence over `ARCAINE_FLINT_AWQ_XMX` in the shared
`awq_xmx::gemm` dispatch. It covers AWQ attention/dense/shared projections
and the legacy per-expert projection path during batched prefill. Set
`ARCAINE_FLINT_PREFILL=batched` when exercising the model.
The separately selected grouped routed-expert kernels retain their own
implementation. MLX INT8 prefill, BF16 weights and decode are outside this path.

Unset the selector or set it to `0` to use the existing dispatch.
Invalid selector values throw an error. The isolated
`prefill-awq-joint-matrix` benchmark requires an explicit selector and
rejects unsupported launches without falling back. The model can fall back
to vector AWQ on a rejected native call, following its existing dispatch.

## Kernel and numerical behavior

The native defaults use a 32x64 output tile at SG16 and a 64x64 tile at
SG32. Each work-group has four subgroups, using 256 GRFs. Each subgroup
keeps several 8x16x16 atoms in registers and reuses A/B fragments across
independent FP32 accumulators. The K stage is 64 elements, one AWQ group.

Packed staging is now the default. Weight bytes are fetched as 16-byte
vectors, dequantized to BF16, and written as paired BF16 values through
32-bit SLM stores. Staged activations use four BF16 values per 64-bit store,
with memcpy preserving support for BF16-aligned inputs.

For full tiles, a 64-byte-aligned input and no input row gather, the default
global-A specialization loads activation fragments directly through
`joint_matrix_load`. Only B occupies SLM (8 KiB). Other calls use the packed
activation staging path, which needs 12 KiB at 32x64 or 16 KiB at 64x64.
Output scatter, arbitrary output strides and in-place residuals work in
both paths. The epilogue stores directly using matrix element coordinates.
There is no weight repacking or persistent scratch allocation.

A/B controls:

- `ARCAINE_FLINT_AWQ_JOINT_MATRIX_TILE=32x64` or `64x64` selects a tile
  explicitly; `16x32` keeps the original kernel and SLM epilogue.
- `ARCAINE_FLINT_AWQ_JOINT_MATRIX_STAGE=scalar` keeps the previous tiled
  implementation. `packed` is the default for the newer tiles.
- `ARCAINE_FLINT_AWQ_JOINT_MATRIX_GLOBAL_A=0` disables direct activation
  matrix loads while preserving packed staging. `1` is the default; shape,
  gather and alignment checks still apply. The benchmark reports the
  requested setting; a tail or gathered call uses the staged fallback.

The kernel computes
`bf16(float(code-zero) * float(scale))`, accumulates BF16 products in FP32,
then rounds the projection to BF16 before adding an optional BF16 residual.
The FP32 accumulator remains live across the full K dimension: it is not
rounded to BF16 between AWQ groups or matrix tiles. BF16 multiplication
with FP32 accumulation is preserved by every staging variant.
It supports M/N tails, input gather, output scatter, arbitrary output row
strides, independently padded expert weight/scale/zero strides, and
`residual == C`. Row maps must contain valid indices and unique output rows.

Requirements: an in-order queue, Intel matrix-capable GPU supporting the
BF16/FP32 8x16x16 atom, the selected subgroup width, positive M/N, K divisible
by 64, and distinct input/output buffers. Capability checks are cached per
host thread and device. Rejected calls submit no work.

## DPAS versus DPASW

DPASW is a DG2/XeHP instruction, not an instruction on the Arc Pro B70's Xe2
architecture. See Intel's
[ISA instruction list](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/6_instructions.md).
Both native variants here emit DPAS; subgroup 32 does not mean DPASW.
The `dpasw` selector therefore reports an explicit error. A DG2-only
implementation would require a separate supported atom and hardware validation.

On oneAPI 2026.1.1 / IGC 2.40.13 the SG32 32x64 composite matrix failed at
JIT compilation because its SLM load/store helpers were unavailable. The
8x16 atom used here passed both subgroup variants. ISA dumps confirmed
SIMD16/SIMD32, `grf_count: 256`, and BF16 `dpas.8x8` instructions.

## Build and correctness checks

Run from the host through the development container:

```bash
docker exec arcaine-dev-run-0b4ca842ff71 \
  cmake --build /workspace/build --target arcaine_kbench arcaine_mbench arcaine_server -j 4

docker exec arcaine-dev-run-0b4ca842ff71 env \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas \
  /workspace/build/arcaine_kbench flint-check

docker exec arcaine-dev-run-0b4ca842ff71 env \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas-sg32 \
  /workspace/build/arcaine_kbench flint-check
```

The expanded Flint check suite passed 287 checks for each of fourteen
retained subgroup/tile/staging/global-A launch configurations on the B70.
Native AWQ cases compare against CPU and vector references, including
gather/scatter, residual aliasing, repeated calls, output padding/canaries,
expert offsets, distinct parameter strides, N=35/48/64/129 and K=64/128/192/256.
An M=129 case crosses multiple work-group row tiles while repeating in-place
residual updates through gather/scatter maps. An M=64 case exercises the
global-A path with scattered in-place residual output. Additional cases use
one AWQ group and inputs offset by one BF16 to verify alignment fallbacks. Invalid native calls are also checked for rejection without output writes.

## Isolated projection A/B

```bash
docker exec arcaine-dev-run-0b4ca842ff71 env \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas \
  /workspace/build/arcaine_kbench flint-kernels \
    --case prefill-awq-joint-matrix --m 512 --n 2048 --k 2048 --iters 20 --check

docker exec arcaine-dev-run-0b4ca842ff71 env \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas-sg32 \
  /workspace/build/arcaine_kbench flint-kernels \
    --case prefill-awq-joint-matrix --m 512 --n 2048 --k 2048 --iters 20 --check

docker exec arcaine-dev-run-0b4ca842ff71 env \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX=0 ARCAINE_FLINT_AWQ_XMX=0 \
  /workspace/build/arcaine_kbench flint-kernels \
    --case prefill-awq --m 512 --n 2048 --k 2048 --iters 20 --check

docker exec arcaine-dev-run-0b4ca842ff71 env \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX=0 ARCAINE_FLINT_AWQ_XMX=1 \
  IGC_allowDecompose2DBlockFuncs=0 \
  /workspace/build/arcaine_kbench flint-kernels \
    --case prefill-awq-xmx --m 512 --n 2048 --k 2048 --iters 20 --check
```

Initial implementation measurements, 2026-10-29, Arc Pro B70, oneAPI
2026.1.1, driver 1.17.39395+13,
M=512 / N=2048 / K=2048, three warmups, 20 iterations:

| Path | ms/iteration | Effective TFLOP/s |
|---|---:|---:|
| Vector AWQ | 3.595246 | 1.195 |
| CuTe AWQ XMX | 0.125154 | 34.317 |
| Original joint_matrix SG16, 16x32, 256 GRFs | 1.741327 | 2.466 |
| Original joint_matrix SG32, 16x32, 256 GRFs | 1.541514 | 2.786 |

These are host wall times per complete GEMM submission, including staging
and output conversion, with compilation/warmup excluded. CPU sampled checks
and vector comparisons passed for all paths. Native timings are experimental;
CuTe is substantially faster on this projection. No full-model throughput or
generation correctness claim is made from these isolated measurements.

## Register tiling iteration

Matched device-event measurements after tiling, on the same B70,
M=512 / N=2048 / K=2048, three warmups, 50 iterations:

| Subgroup | Output tile | Host ms/iteration | GPU ms/iteration |
|---|---|---:|---:|
| 16 | original 16x32 | 1.551329 | 1.550738 |
| 16 | 32x64 | 0.604945 | 0.603838 |
| 16 | 64x64 | 0.771981 | 0.771461 |
| 32 | original 16x32 | 1.211860 | 1.211250 |
| 32 | 32x64 | 0.771897 | 0.771371 |
| 32 | 64x64 | 0.571684 | 0.571187 |

The selected defaults improved device time by 2.57x at SG16 and 2.12x at
SG32 in this matched run. These are isolated projection measurements;
they do not establish full-model throughput.

Enable exact kernel timing with
`ARCAINE_FLINT_AWQ_JOINT_MATRIX_PROFILE=1`. The native GEMM optionally returns
the event from its actual `q.submit`, and the benchmark reports its mean
`command_end-command_start` separately from host wall time.

```bash
docker exec arcaine-dev-run-0b4ca842ff71 env \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas-sg32 \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX_TILE=64x64 \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX_PROFILE=1 \
  /workspace/build/arcaine_kbench flint-kernels \
    --case prefill-awq-joint-matrix --m 512 --n 2048 --k 2048 --iters 50 --check
```

For a matched reference, repeat with `ARCAINE_FLINT_AWQ_JOINT_MATRIX_TILE=16x32`.
Use `dpas` with `32x64` to reproduce the SG16 default.

Both optimized defaults also passed CPU sampled and vector checks on expert
and dense projections (N/K = 512/2048, 2048/512, 6144/2048 and 2048/6144)
and on H=2048 projections with M=1024, 2048 and 4096. The existing CuTe path
passed its expanded correctness fixture as well.

## Packed staging and global-A iteration

Matched device-event measurements, M=512 / N=2048 / K=2048, three warmups,
50 iterations, 256 GRFs and packed B staging throughout:

| Subgroup / tile | A staged in SLM | A loaded globally | Improvement |
|---|---:|---:|---:|
| SG16 / 32x64 | 0.784056 ms | 0.299046 ms | 2.62x |
| SG32 / 64x64 | 0.601260 ms | 0.169294 ms | 3.55x |

CPU sampled and vector checks passed for all four runs. These are isolated
projection timings. The CuTe result above is the earlier reference, not a
new baseline run.

```bash
docker exec arcaine-dev-run-0b4ca842ff71 env \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas-sg32 \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX_TILE=64x64 \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX_STAGE=packed \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX_GLOBAL_A=1 \
  ARCAINE_FLINT_AWQ_JOINT_MATRIX_PROFILE=1 \
  /workspace/build/arcaine_kbench flint-kernels \
    --case prefill-awq-joint-matrix --m 512 --n 2048 --k 2048 --iters 50 --check
```

Set `GLOBAL_A=0` using the full variable name above for the staged-A
comparison; use `STAGE=scalar` for the preceding tiled implementation.
The arithmetic remains BF16 operands and an FP32 accumulator in every case.

The final defaults passed CPU/vector checks for expert and dense projections,
K=192, and batches through M=4096. Both subgroup widths also passed a
full-reference K=192 run with IGC_allowDecompose2DBlockFuncs=0.

Final global-A ISA dumps confirm grf_count=256 and SIMD16/SIMD32, with
dpas.8x8 operands typed f for the destination/accumulator and bf for A/B.
The benchmark, model benchmark and server targets all build successfully.

## Warm model benchmark at depth 32,000

2026-10-30, full 49-layer mattbucci_Flint-AWQ checkpoint, one Arc Pro B70
(GPU 0), native SG32 projections with packed/global-A defaults, grouped
routed MoE and the existing GPU decode path. One discarded warmup and one
timed repetition, 32 fixed-token decode steps, max_seq=32128. The GPU prefix
checkpoint avoids repeating the untimed 32k cache fill.

| Workload | Starting KV depth | Tokens/s | ms/token |
|---|---:|---:|---:|
| Prefill 512 | 0 | 1866.26 | 0.536 |
| Prefill 1024 | 0 | 1688.79 | 0.592 |
| Prefill 2048 | 0 | 1498.70 | 0.667 |
| Prefill 4096 | 0 | 1283.24 | 0.779 |
| Decode 32 | 32000 | 61.91 | 16.152 |

Timed decode took 0.517 s. Native projection dispatch reported 343 hits and
zero fallbacks per prefill chunk. Loading the 22.687-GB arena took 43.0 s
and was excluded from the measurements. These are repeated BOS-token model
benchmark inputs, not sampled text generation.

Exact environment and command:
[run script](../.scratchpad/flint-joint-matrix-warm-32k-20261030.sh).
Full output:
[benchmark log](../.scratchpad/flint-joint-matrix-warm-32k-20261030.log).

The script runs arcaine_mbench with
-p 512,1024,2048,4096 -d 32000 -n 32 -w 1 -r 1 --max-seq 32128.

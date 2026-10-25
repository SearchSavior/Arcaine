# Flint grouped W4A16 MoE

The AWQ batched-prefill path has an opt-in grouped routed-expert branch:
`ARCAINE_FLINT_AWQ_GROUPED_MOE=1`. Enable batched prefill with
`ARCAINE_FLINT_PREFILL=batched`. Other projections can independently use
`ARCAINE_FLINT_AWQ_XMX=1` (with `IGC_allowDecompose2DBlockFuncs=0`).

Grouped dispatch covers batched routed experts. The dedicated GPU decode
kernels, command graphs and performance switches are described in
[GPU decode](decode.md).

## Architecture and tensor mapping

Sources: `models/mattbucci_Flint-AWQ/config.json`,
`src/modeling/flint/loader.cpp`, and the MoE math in
`reference/rec/engine/src/rec_mlx/model.py`. The existing full-model data flow is
documented in [the Flint inventory](../flint-int4-inventory.md).

The checkpoint has H=2048, I=512, E=256, top-k=6, and 47 MoE layers.
All kernel dimensions come from QTensor/config metadata.

| Checkpoint projection under model.layers.L.mlp.experts.e | QTensor | Logical bank |
|---|---|---|
| gate_proj.{qweight,scales,qzeros} | egate | [E,I,H] |
| up_proj.{qweight,scales,qzeros} | eup | [E,I,H] |
| down_proj.{qweight,scales,qzeros} | edown | [E,H,I] |

The loader normalizes AWQ packing once: codes are unsigned nibbles in
`[E,N,K/2]` bytes, even K in the low nibble. FP16 scales and unpacked uint8
zero points are `[E,K/64,N]`. Each weight is
`bf16(float(code-zero) * float(scale))`. The grouped kernels read the same
banks as the existing GEMMs; they allocate no duplicate weights or repacks.

For M input rows, route slot p identifies token `p/top_k` and rank
`p%top_k`. The device counts slots per expert, builds expert and 8-row tile
prefix sums, and scatters slot IDs into an expert-major permutation.
Atomic scatter order may vary; output addressing always uses the original
slot ID, preserving rank order in the existing weighted combine.

Two subgroup-16 DPAS kernels then compute:

1. Gate and up together: gather eight activation rows directly from
   `input[p/top_k,:]`, dequantize both weight fragments to BF16, accumulate in
   FP32, round gate/up to BF16, then store
   `bf16(bf16(silu(bf16(gate))) * bf16(up))` at `activation[p,:]`.
2. Down: gather `activation[p,:]`, accumulate dequantized BF16 weights in
   FP32, and store BF16 output at `routed[p,:]`.

Each work-group owns an 8x16 output tile and finds its expert through the
device tile prefix sums. Launching at most `ceil(M*top_k/8)+E-1` row tiles
avoids downloading a tile count. Empty experts and partial tiles are handled
on device. The caller must supply valid router IDs and an in-order queue;
the model checks tensor geometry and GPU support before using this path.

The routed branch submits six commands: count clear, histogram, prefix,
scatter, fused gate/up+SiLU, and down. It has no host wait, routing download,
row-map upload, activation compaction copy, or per-expert GEMM submission.
Debug synchronization remains available through `ARCAINE_FLINT_DEBUG_SYNC`.

Scratch is caller-owned and reused across layers/chunks: `4*E+2` metadata
integers, an `M*top_k` pair permutation, and `M*top_k*I` BF16 activations.
The pair permutation reuses `b_outs`. Batch resizing drains outstanding work
and releases the previous batch allocations.

## Kernel benchmark

Build and run inside the development container (current name:
`arcaine-dev-run-0b4ca842ff71`):

```bash
docker exec arcaine-dev-run-0b4ca842ff71 bash -lc '
  cd /workspace &&
  cmake --build build --target arcaine_kbench flint_main -j 4 &&
  ARCAINE_FLINT_AWQ_GROUPED_MOE=1 ARCAINE_FLINT_AWQ_XMX=1 \
  IGC_allowDecompose2DBlockFuncs=0 \
  ./build/arcaine_kbench flint-awq-grouped-moe \
    --m 512 --hidden 2048 --inter 512 --experts 256 --top-k 6 \
    --routing balanced --iters 10 --check
'
```

Set `ARCAINE_FLINT_AWQ_GROUPED_MOE=0` for the existing host/per-expert
baseline. The benchmark honors `ARCAINE_FLINT_AWQ_XMX` when selecting legacy
GEMMs. `--routing skewed` mixes a hot expert set with distributed tokens;
`--routing concentrated` sends every token to the same top-k experts.

Timed work includes route construction and routed gate/up, SiLU, and down,
including the legacy host routing copies. It excludes initialization,
correctness readbacks, router scoring, shared experts, and residual combine.
`--check` compares every routed output against the existing per-expert path
(atol=0.01, rtol=0.01) and verifies counts, offsets, tile counts, expert
membership, and a complete slot permutation.

## Validation

Measured on 2026-10-23, Level Zero, Intel Arc Pro B70 (Xe2/BMG G31,
32 Xe cores, subgroup sizes 16/32, 128 KiB SLM). The implementation uses
the raw SPIR-V BF16 DPAS intrinsic, FP32 accumulation, and no joint_matrix.

At M=512, H=2048, I=512, E=256, top-k=6, ten timed iterations:

| Path | Routing | ms / routed MoE branch |
|---|---|---:|
| Existing CuTe/XMX + host grouping | balanced | 40.3294 |
| Grouped | balanced | 3.6998 |
| Grouped | skewed | 3.6397 |
| Grouped | concentrated | 3.1646 |

The balanced comparison is about 10.9x. This is a kernel-path measurement,
not an end-to-end speedup. Routing host traffic falls from
`3*M*top_k*sizeof(int)` (36,864 bytes here) to zero per MoE layer.
The grouped balanced timing was measured with the XMX legacy selector unset;
that selector only affects the comparison path, not the grouped implementation.

All 6,291,456 routed output elements matched CuTe/XMX exactly for balanced,
skewed, and concentrated routing. Comparison against the vector baseline
also passed tolerance (maximum absolute difference 0.015625).
Smaller checks covered M=1/17/65, ragged 8-row tiles, empty experts, and skew.
The existing `flint_main --tests-only` suite passed with AWQ XMX enabled,
including its independent CPU references for AWQ math.

A 17-token local-checkpoint integration check compared all 100,352 residual
values after layers 1 through 49: bitwise equal and finite. This loaded the
complete 22.687 GB AWQ arena and passed the loader's full tensor-consumption
checks (109,612 checkpoint tensors).

## Full-model throughput command

The model benchmark measures repeated BOS-token inputs, separately from the
routed-kernel timing above. Use one discarded warmup, three timed repetitions,
and 32 decode steps at each KV depth:

```bash
docker exec arcaine-dev-run-0b4ca842ff71 bash -lc '
  cd /workspace &&
  ARCAINE_FLINT_PREFILL=batched ARCAINE_FLINT_AWQ_XMX=1 \
  ARCAINE_FLINT_AWQ_GROUPED_MOE=1 IGC_allowDecompose2DBlockFuncs=0 \
  ./build/arcaine_mbench --model /workspace/models/mattbucci_Flint-AWQ \
    --device 0 -p 512,1024,2048,4096 -d 512,1024,2048,4096 \
    -n 32 -w 1 -r 3 --max-seq 4224
'
```

# INT4-AWQ fusion benchmark and usage

## Focused kernel-path benchmark

Run inside `arcaine-dev-run-69fca6164f91`:

```bash
cd /workspace
./scripts/benchmark_diffusion_int4_fusions.sh 100
```

This does not load the model and does not run inference. It benchmarks the
actual fixed-shape adjacent kernel sequences used by a 256-position denoiser.

Measured on Intel BMG G31:

| Subchain | Baseline | Fused | Speedup | Correctness |
|---|---:|---:|---:|---|
| dense gate/up + GeGLU (`M=256,K=2816,N=2112`) | 0.429 ms | 0.113 ms | 3.807× | bit-exact, 540672 elements |
| expert combine + dual postnorm (`seq=256,H=2816,top-k=8`) | 0.039 ms | 0.029 ms | 1.357× | bit-exact, 720896 elements |
| self-condition add + RMSNorm (`seq=256,H=2816`) | 0.010 ms | 0.006 ms | 1.572× | bit-exact, 720896 elements |

These are subchain measurements, not an estimate of whole-model speedup.

## Load-only integration check

```bash
cd /workspace
DIFF_INT4_FUSE_DENSE_GATE_UP=1 \
DIFF_INT4_FUSE_EXPERT_POSTNORM=1 \
DIFF_INT4_FUSE_SELFCOND_ADD_NORM=1 \
DIFF_BENCH_FORCE_FULL_CANVAS=1 \
DIFF_FORCE_DENOISE_STEPS=1 \
./build/diffusion_bench \
  --model models/diffusiongemma-26B-A4B-it-AWQ-INT4 \
  --device 0 --layers single --experts layer-owner \
  -p 512 -n 256 -ds 48 -w 0 -r 0
```

`DIFF_BENCH_FORCE_FULL_CANVAS=1` disables commit-time EOS truncation for the
benchmark only. EOS remains in the sampled vocabulary and the denoising
trajectory is unchanged; all 256 finalized positions are counted in TG.
`DIFF_FORCE_DENOISE_STEPS=1` disables adaptive early stopping so every canvas
executes exactly 48 denoiser forwards.

The complete baseline/fused sweep is automated by:

```bash
cd /workspace
./scripts/benchmark_diffusion_int4_e2e_ab.sh
```

This loaded 35,777 tensors from four shards, placed all 30 layers and experts
on GPU 0, and reserved 0.25 GiB for the activation arena.

## Full A/B throughput command

For a later end-to-end measurement, use the required prompt sweep and retain
the fixed 48 denoising steps / 256-token canvas. Run once without the three
variables for baseline, then once with them set to `1`:

```bash
cd /workspace
DIFF_INT4_FUSE_DENSE_GATE_UP=1 \
DIFF_INT4_FUSE_EXPERT_POSTNORM=1 \
DIFF_INT4_FUSE_SELFCOND_ADD_NORM=1 \
./build/diffusion_bench \
  --model models/diffusiongemma-26B-A4B-it-AWQ-INT4 \
  --device 0 --layers single --experts layer-owner \
  -p 512,1024,2048,4096 -n 256 -ds 48 -w 1 -r 5 --md
```

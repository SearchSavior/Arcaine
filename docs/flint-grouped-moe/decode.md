# Flint GPU decode

The opt-in resident decode path uses eight-token SYCL command graphs. Token
selection, RNG, embedding, position updates, KV access, projections, routing,
expert math, attention and the vocabulary head execute on the GPU. Generation
copies token IDs and a small state record to the host once per block. It does
not download logits or upload a token on every decode step. The first block
includes graph capture and compilation; use repeated requests for warm timing.

This path is integrated with `arcaine_mbench`. The server does
not yet use the resident generation API. Stochastic sampling supports top-k
1..64; the parallel and distributed kernels support k<=20. Greedy is supported.

## Implementation

- `gpu_decode.cpp/.hpp`: resident state, graph capture, exact stable top-k and
  nucleus sampling, device RNG and embedding. `ARCAINE_FLINT_DECODE_SAMPLE_GROUPS`
  accepts 8, 16, 32 or 64; the default 1 uses a single workgroup. Distributed
  selection computes partition top-k, then merges their union on the GPU.
- `flash_decode.hpp`: eight grouped query heads share a KV head. Long contexts
  stage KV tiles in local memory; short sliding rings read through the device
  cache. FP32 online softmax includes the attention sink, and a separate merge
  combines context partitions. `ARCAINE_FLINT_FLASH_DECODE_SINGLE_EXP=1` computes
  one exponential per key and selects the other rescaling weight as one.
- `awq_decode.hpp`: packed W4 loads with BF16 dequantization and FP32 dot products.
- `decode_router.hpp`: parallel expert scoring and stable top-k, using attention
  scratch after the attention merge completes.
- `bf16_decode.hpp`: packed BF16 vector loads for the vocabulary projection.

All performance switches are read before graph capture; keep them fixed for a
model instance. `ARCAINE_FLINT_DECODE_GLOBAL_ATTN_SPLIT` can override the common
split count for global layers. Counts 4, 8, 16, 32, 64, 128 and 256 are supported.
Higher counts trade shorter partials for more merge work and scratch memory.

`ARCAINE_FLINT_MBENCH_GPU_CHECKPOINT=1` saves the sliding KV rings and logits
with device-to-device copies after a depth prefill, then restores them before
each timed decode repetition. Global KV caches append beyond the saved prefix;
future positions are masked and overwritten before use. Prefix setup and
restore are excluded from decode timing. No KV data is copied to the host.

## Validated measurements

The complete [64K depth sweep](decode-64k-20261025.md) measured 78.39 tokens/s
at 512 and 50.71 tokens/s at 65,536 with single-exponential Flash softmax.


One Intel Arc Pro B70, full 49-layer mattbucci_Flint-AWQ, 2026-10-24. Fixed-token
model decode, 32 steps, one warmup and three timed repetitions:

| KV depth | tokens/s | ms/token |
|---:|---:|---:|
| 512 | 78.11 | 12.802 |
| 4096 | 75.20 | 13.298 |
| 8192 | 71.53 | 13.980 |
| 12000 | 68.08 | 14.688 |
| 16384 | 68.45 | 14.609 |
| 32000 | 60.76 | 16.459 |

These measurements use Flash16, split16, packed AWQ, parallel routing and the
vec4 head; they precede the single-exponential and distributed-sampler changes.
Fixed-token decode excludes sampling. The corresponding log is
`.scratchpad/flint-gpu-decode-deep-checkpoint-20261024.log`.

On the isolated vocabulary sampler (150144 logits, k=20), the serial GPU path
measured 13.4453 ms, single-workgroup parallel 0.5918 ms, and 64-workgroup
distributed 0.0901 ms. All tested k=2/6/20, top-p=0.01/0.95/1 combinations matched
16 sampled token IDs and RNG state exactly against the serial GPU reference.
Stable ties and EOS tails passed as well. This is a sampler measurement,
separate from complete model throughput.

With distributed sampling and single-exponential Flash softmax, warm real
generation measured 379.0/379.4 ms for 32 tokens (about 84 tokens/s). All six
serial/distributed GPU requests produced the same token IDs and final RNG
state. See `.scratchpad/flint-gpu-generation-distributed-ab-20261024.log`.

Experimental XMX Flash, native exponential, parallel partition merge, subgroup
router top-k, subgroup sampler merge and parallel expert down remain disabled
by default: measurements have not established an end-to-end gain.

## Reproduce

Run inside the development container:

```bash
cd /workspace
export ARCAINE_FLINT_GPU_PREFILL=1 ARCAINE_FLINT_GPU_DECODE=1
export ARCAINE_FLINT_DECODE_VEC=4 ARCAINE_FLINT_DECODE_AWQ_VEC=1
export ARCAINE_FLINT_DECODE_ROUTER=1 ARCAINE_FLINT_DECODE_ATTN_SPLIT=16
export ARCAINE_FLINT_FLASH_DECODE=16
export ARCAINE_FLINT_DECODE_PARALLEL_SAMPLE=1 ARCAINE_FLINT_DECODE_SAMPLE_GROUPS=64
export ARCAINE_FLINT_AWQ_XMX=1 IGC_allowDecompose2DBlockFuncs=0
cmake --build build --target arcaine_kbench arcaine_mbench -j 4

# Exact GPU sampler comparison and isolated timing.
./build/arcaine_kbench flint-decode-sampling --vocab 150144 --top-k 20 --iters 200 --check

# Fixed-token model decode at depth, with device-only prefix restore.
ARCAINE_FLINT_MBENCH_GPU_CHECKPOINT=1 ./build/arcaine_mbench \
  --model /workspace/models/mattbucci_Flint-AWQ --device 0 \
  -p 512,1024,2048,4096 -d 512,4096,8192,12000,16384,32000 \
  -n 32 -w 1 -r 3 --max-seq 32128

# Real sampled generation; serial/distributed GPU token and RNG parity.
ARCAINE_FLINT_FLASH_DECODE_SINGLE_EXP=1 ./build/arcaine_mbench flint-generate \
  /workspace/models/mattbucci_Flint-AWQ --ctx 4224 \
  --max-new 32 --runs 3 --sample-ab
```

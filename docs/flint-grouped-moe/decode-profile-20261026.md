# Flint B70 decode profile — 2026-10-26

Global attention accounts for the context-dependent slowdown. QKV projection
is the largest fixed GPU cost, with compiler-generated private-memory traffic
in its current implementation. No per-token host copy appears in the resident
fixed-token decode trace.

## Hardware and method

Hardware was verified with the project tool:

~~~bash
docker exec arcaine-dev-run-0b4ca842ff71 bash -lc \
  'cd /workspace && .agents/skills/check-available-hardware/get_device_props'
~~~

Device 0 is Intel Arc Pro B70, intel_gpu_bmg_g31: 32 Xe cores, 256 vector engines,
2,048 hardware threads, 34,242,297,856 bytes of VRAM, 128 KiB SLM per work-group,
subgroup sizes 16/32. Two physical GPUs are installed; this run selects device 0.
The imported hardware helper's B60 bandwidth fallback is excluded.

Intel PTI 1.1 collected kernel and copy timestamps from the existing full
49-layer Flint AWQ model and eight-token resident graphs. Each variant/depth
used a 32-token warmup and one measured 32-token repetition. The trace contains
384 complete tokens across vector, native-exp, and adaptive two-pass variants at
depths 512 and 65,536. Only the second repetition is analyzed. Prefix setup and
GPU restoration are excluded. Tokens are fixed BOS inputs; sampling is excluded.

**Profiling is intrusive.** The table sums GPU kernel durations, not ordinary
end-to-end latency. PTI introduces about 4.4–4.6 ms/token between kernel timestamps,
plus host/graph-submission overhead. Those gaps do not establish normal launch
overhead. The previous untraced three-repetition adaptive measurements remain
78.52 tokens/s at 512 and 53.69 tokens/s at 65,536. This is a diagnosis, not a new
throughput baseline.

## GPU time breakdown

Adaptive two-pass uses vector attention at shallow depth and separate QK/PV
kernels for global layers at 64K.

| Stage | 512 GPU ms/token | 65,536 GPU ms/token | Kernels/token at 65,536 |
| --- | ---: | ---: | ---: |
| Global attention | 0.349 | 6.229 | 42 |
| Sliding attention | 0.878 | 0.884 | 70 |
| QKV projection | 4.289 | 4.300 | 49 |
| MoE gate/up | 2.221 | 2.148 | 47 |
| MoE down/combine | 1.226 | 1.212 | 47 |
| Output projection | 1.168 | 1.129 | 49 |
| LM head | 1.041 | 1.033 | 1 |
| Router | 0.826 | 0.831 | 94 |
| RMSNorm | 0.248 | 0.260 | 99 |
| QK norm/RoPE/KV write | 0.146 | 0.184 | 49 |
| Dense MLP | 0.186 | 0.156 | 4 |
| Embedding | 0.002 | 0.002 | 1 |
| Token bookkeeping | 0.003 | 0.003 | 2 |
| Total kernel execution | 12.583 | 18.371 | 554 |

Global attention grows by 5.880 ms/token, accounting for essentially all of the
increase in summed kernel time. The 35 sliding layers remain near 0.88 ms/token.
QKV stays near 4.3 ms/token. MoE gate/up and down/combine total 3.36 ms/token at 64K.

For the 14 global layers at 64K:

| Phase | GPU ms/token |
| --- | ---: |
| QK scores | 2.846 |
| Probability/value accumulation | 2.748 |
| Partition merge | 0.635 |

Global attention totals 6.907 ms with vector, 6.440 ms with native exp, and
6.229 ms with two-pass. Two-pass adds 14 dispatches/token, reaching 554 versus
540. Its larger merge costs 0.635 ms versus 0.176 ms for vector, offsetting part
of the main-kernel improvement.

The earlier packed-V isolated benchmark passed reference checks at 17, 32,000,
and 65,536 positions. At 65,536 it took 0.3584 ms, versus the earlier unpacked
two-pass result of approximately 0.4058 ms. Packed V is still benchmark-only;
model integration and maintenance cost have not been measured.

## Copies and dispatches

Each measured 32-token repeated-decode interval has a 40-byte H2D state transfer
before it and a 40-byte D2H transfer after it, taking approximately 4 and 7–8
microseconds of recorded device time. There are zero copy records inside the
32-token compute interval. Prefix restoration is D2D outside the timed interval.

The 540/554 counts are GPU kernel dispatches. The ordinary repeated-decode loop
makes four host submissions of eight-token graphs for 32 tokens. This trace
cannot isolate normal dispatch overhead because PTI adds timing operations.

This copy result applies to decode_repeated_device. Interactive generate_device
also reads IDs and state once per eight-token block; it does not download logits
or upload each token. Norms, router stages, QKV postprocessing, and attention
merges remain possible fusion targets, subject to ordinary A/B timing.

## QKV investigation

The fused qkv_gemv selects a reference to one of three captured QTensor objects
based on the output row. In the compiler dump, that capture survives as an
addressable private allocation. ISA explicitly stores and reloads descriptor
fields in private memory. Metadata reports private_size: 11008 for fused QKV;
the separate q4_gemv has no private allocation entry. Both use 128 GRFs and SIMD32.
This was verified with the production compiler flags, not inferred from timing.

The diagnostic uses Q=4,096, K=512, V=512 output rows; input width 2,048; AWQ
group size 64; packed-four loads; subgroup 32. It compares existing fused QKV
against three existing q4_gemv calls. No new arithmetic kernel was introduced.

The final test cycles through 49 distinct weight banks (about 281 MB including
scales/zeros), exceeding L2 capacity. A graph contains all 49 banks, is warmed
once, and timed for ten replays. Compiler flags match arcaine_mbench:
-std=gnu++17 -fsycl -fno-sycl-id-queries-fit-in-int -O2.

| Existing implementation | Graph wall time per QKV bundle |
| --- | ---: |
| Fused QKV, one dispatch | 94.118 µs |
| Three GEMVs, three dispatches | 48.623 µs |

All 250,880 BF16 outputs across 49 banks matched bit-for-bit. Three GEMVs reduce
isolated QKV time by 48.3%. This is a synthetic diagnostic, not an end-to-end
model speedup. Single-bank O3 exploratory results also favored splitting
(56.793 versus 36.561 µs), but the table uses production flags and 49 banks.

The strongest next experiment is fused QKV with register-resident descriptors:
direct Q/K/V branches or explicit field selection avoiding a dynamically selected
descriptor reference. Keep the original behind a descriptive environment flag,
verify output agreement and private-memory removal, then run full-model A/B.
The evidence identifies a strong lead; it does not prove that private memory
alone explains the entire timing difference.

## Reproduction

All commands below run inside the development container at /workspace.
The complete profile script is .scratchpad/flint-decode-profile-20261026.sh:

~~~bash
#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export ARCAINE_FLINT_GPU_PREFILL=1 ARCAINE_FLINT_GPU_DECODE=1
export ARCAINE_FLINT_DECODE_VEC=4 ARCAINE_FLINT_DECODE_AWQ_VEC=1
export ARCAINE_FLINT_DECODE_ROUTER=1 ARCAINE_FLINT_DECODE_ROUTER_SG_TOPK=0
export ARCAINE_FLINT_DECODE_ATTN_SPLIT=16 ARCAINE_FLINT_DECODE_GLOBAL_ATTN_SPLIT=256
export ARCAINE_FLINT_FLASH_DECODE=16 ARCAINE_FLINT_FLASH_DECODE_SINGLE_EXP=1
export ARCAINE_FLINT_FLASH_DECODE_NATIVE_EXP=0 ARCAINE_FLINT_FLASH_DECODE_PARALLEL_MERGE=0
export ARCAINE_FLINT_FLASH_DECODE_XMX=0 ARCAINE_FLINT_DECODE_PARALLEL_DOWN=0
export ARCAINE_FLINT_DECODE_PARALLEL_SAMPLE=1 ARCAINE_FLINT_DECODE_SAMPLE_GROUPS=64
export ARCAINE_FLINT_DECODE_SAMPLE_SG_MERGE=0 ARCAINE_FLINT_MBENCH_GPU_CHECKPOINT=1
export ARCAINE_FLINT_AWQ_XMX=1 IGC_allowDecompose2DBlockFuncs=0
export ARCAINE_FLINT_MBENCH_ATTN_AB=2
export ARCAINE_FLINT_FLASH_DECODE_TWO_PASS=auto ARCAINE_FLINT_FLASH_DECODE_BLOCK=0
export ARCAINE_FLINT_FLASH_DECODE_CONTIGUOUS=0
export ARCAINE_FLINT_PROFILE_FILE=/workspace/.scratchpad/flint-decode-profile-20261026.tsv
export ZE_ENABLE_TRACING_LAYER=1
stdbuf -oL -eL env LD_PRELOAD=/workspace/.scratchpad/flint-pti-profile.so ./build/arcaine_mbench \
  --model /workspace/models/mattbucci_Flint-AWQ --device 0 \
  -p 512,1024,2048,4096 \
  -d 512,65536 \
  -n 32 -w 1 -r 1 --max-seq 65664
~~~

Compile the collector first:

~~~bash
icpx -std=c++17 -shared -fPIC -O2 .scratchpad/flint-pti-profile.cpp \
  -I/opt/intel/oneapi/2026.1/include -L/opt/intel/oneapi/2026.1/lib \
  -Wl,-rpath,/opt/intel/oneapi/2026.1/lib -lpti_view -ldl \
  -o .scratchpad/flint-pti-profile.so

python3 .scratchpad/analyze-flint-pti.py \
  .scratchpad/flint-decode-profile-20261026.tsv \
  --output .scratchpad/flint-decode-profile-20261026.json
~~~

Reproduce QKV diagnostics:

~~~bash
icpx -std=gnu++17 -fsycl -fno-sycl-id-queries-fit-in-int -O2 \
  -I/workspace/src -I/workspace/third_party \
  .scratchpad/flint-qkv-diagnostic.cpp -o .scratchpad/flint-qkv-matched-diagnostic

ZE_AFFINITY_MASK=0 ARCAINE_FLINT_PROFILE_QKV_BANKS=49 \
  ARCAINE_FLINT_PROFILE_QKV_SPLIT=0 .scratchpad/flint-qkv-matched-diagnostic
ZE_AFFINITY_MASK=0 ARCAINE_FLINT_PROFILE_QKV_BANKS=49 \
  ARCAINE_FLINT_PROFILE_QKV_SPLIT=1 .scratchpad/flint-qkv-matched-diagnostic
~~~

Artifacts in .scratchpad/:

- flint-b70-device-props-20261026.txt: project hardware report.
- flint-pti-profile.cpp and .so: opt-in collector.
- flint-decode-profile-20261026.tsv: raw timestamps and kernel-name dictionary.
- flint-decode-profile-20261026.log: complete profiled run.
- analyze-flint-pti.py: segmentation and stage classification.
- flint-decode-profile-20261026.json and flint-decode-profile-analysis.txt: results.
- flint-qkv-diagnostic.cpp: isolated existing-kernel comparison.
- flint-qkv-fused-matched.log and flint-qkv-split-matched.log: final QKV measurements.
- igc-qkv-matched-20261026/: production-flags compiler dumps; dump-run timing excluded.

No production inference code or skill documents were changed.

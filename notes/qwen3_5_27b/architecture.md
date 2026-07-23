# Qwen3.5 27B NVFP4 implementation notes

## Target

The implementation targets `unsloth_Qwen3.6-27B-NVFP4` on two Intel
Battlemage GPUs (`intel_gpu_bmg_g31`). Each device has 32 Xe cores, 128 KiB
SLM per workgroup, subgroups 16/32, and 32.5 GiB device memory. Decoder layers
0-31 are resident on GPU 0 and layers 32-63 on GPU 1.

## Decoder architecture

- Hidden size 5120, 64 layers, vocabulary 248320.
- Every fourth layer is full causal GQA: 24 query heads, 4 KV heads,
  head dimension 256.
- The other 48 layers use gated DeltaNet: 16 key heads, 48 value heads,
  key/value head dimensions 128, causal convolution, recurrent state, and
  learned gates.
- MLP intermediate size 17408. Layers 0-55 use NVFP4 weights; layers 56-63
  use E4M3 FP8 weights. Attention, DeltaNet projections, and LM head use E4M3.
- Partial MRoPE uses theta 1e7 and sections `[11, 11, 10]`.

KV and DeltaNet state are persistent per layer. The attention implementation
uses online causal GQA and does not materialize the score matrix. DeltaNet's
recurrent state is kept on device. Its production path stores the 128x128
FP32 state in value-major order and keeps two complete state rows in each
ESIMD work-item's registers across the sequence. Q/K are staged once per token
through 2 KiB of SLM. This avoids both the scalar inner loop and the baseline's
64 KiB per-head SLM allocation.

## Quantized execution

FP8 linear layers remain resident in E4M3 and execute through oneDNN with
per-channel BF16 weight scales. NVFP4 weights remain packed. The Xe2 path uses
SPIR-V DPAS tiles and fuses gate/up projection, SwiGLU, and activation repacking
before the down projection. Set `ARCAINE_QWEN35_NVFP4_DPAS=0` for the oneDNN
path. Measurements on BMG G31 currently favor oneDNN for the dense MLP, so it
is the default while the custom kernel remains available with
`ARCAINE_QWEN35_NVFP4_DPAS=1`.

Full attention uses a head-256 flash-style XMX kernel: DPAS computes both QK
and softmax-times-V in 8x16 tiles while online softmax avoids materializing the
score matrix. `ARCAINE_QWEN35_XMX_ATTENTION=0` restores the scalar baseline for
A/B measurements. `ARCAINE_QWEN35_SUBGROUP_ATTENTION=1` selects the intermediate
reduction-only experiment and is not the production path.

Gated DeltaNet uses the register-resident ESIMD recurrence by default. It is
the vector/reduction portion of linear attention, not an XMX matrix tile. Set
`ARCAINE_QWEN35_ESIMD_DELTA=0` to restore the scalar/SIMT baseline and its
key-major state layout. The two layouts must not be mixed within a live cache.
The focused benchmark uses the checkpoint's exact 48-head, K=128, V=128 shape:

```bash
docker exec arcaine-dev-run-5283f9726339 bash -lc \
  'cd /workspace && ARCAINE_QWEN35_DELTA_PREFILL=512,1024,2048,4096 \
  ARCAINE_QWEN35_DELTA_TOKENS=32 ARCAINE_QWEN35_DELTA_WARMUP=1 \
  ARCAINE_QWEN35_DELTA_RUNS=5 ./scripts/benchmark_qwen35_deltanet_ab.sh'
```

On BMG G31 this reduces the recurrent-core time from 6.84 to 1.07 ms at
512 tokens, 13.88 to 2.48 ms at 1024, and 55.41 to 10.05 ms at 4096. A
32-step sequential decode drops from 0.0356 to 0.00484 ms/token. Maximum BF16
output error against the baseline was 8e-6 over the full sweep.

For M=1 decode, `ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE=1` (the default)
also fuses causal conv1d, Q/K normalization and query scaling, beta/decay gate
evaluation, recurrent update, and z extraction. Its convolution state is
time-major; the scalar fallback selected with `=0` retains the original
channel-major layout, so the setting must not change inside a live cache.
`ARCAINE_QWEN35_FUSED_BA_PROJECTION=1` (also default) concatenates the small
beta/a weights and replaces two 48-output BF16 matmuls with one 96-output
matmul.

The architecture-shaped decode-fusion benchmark is:

```bash
docker exec arcaine-dev-run-a82a51b2b852 bash -lc \
  'cd /workspace && ARCAINE_QWEN35_DELTA_DECODE_PROMPT=1 \
  ARCAINE_QWEN35_DELTA_DECODE_TOKENS=32 \
  ARCAINE_QWEN35_DELTA_DECODE_WARMUP=1 \
  ARCAINE_QWEN35_DELTA_DECODE_RUNS=5 \
  ./scripts/benchmark_qwen35_delta_decode_fusion_ab.sh'
```

On BMG G31, the complete post-projection decode sequence drops from 0.0563
to 0.0194 ms/token with exact BF16 core and z output. End-to-end decode reaches
13.38, 12.26, and 11.34 tokens/s at KV depths 0, 512, and 1024.

The focused benchmark is:

```bash
docker exec arcaine-dev-run-5283f9726339 bash -lc \
  'cd /workspace && ./build/qwen35_nvfp4_mlp_bench \
  --model /workspace/models/unsloth_Qwen3.6-27B-NVFP4 \
  --p 1,128,512,1024 --w 1 --r 5 --kernels onednn,xe2-dpas --layer 0'
```

The end-to-end benchmark command, including warmup and all required prompt
lengths, is:

```bash
docker exec arcaine-dev-run-5283f9726339 bash -lc \
  'cd /workspace && ./build/arcaine_mbench \
  --model /workspace/models/unsloth_Qwen3.6-27B-NVFP4 \
  --p 512,1024,2048,4096 --n 32 --w 1 --r 1 -d 0,512,1024'
```

## Vision path

The vision tower implements temporal/spatial patch embedding, learned
position interpolation, two-dimensional rotary embedding, per-frame online
attention, 27 transformer blocks, and the four-patch merger into the 5120-wide
language representation. Image preprocessing implements Qwen smart resize,
bicubic resize, normalization, temporal duplication, and merge-major token
ordering. Multimodal position IDs and image-token scatter are handled by the
model runner.

## Strict checkpoint contract

Configuration parsing validates the nested text, vision, processor, and
generation configuration. Loading validates tensor shapes and dtypes and
refuses to finish if a checkpoint tensor was not consumed. All 1968 checkpoint
tensors are mapped, including the 15 MTP tensors. MTP weights are resident but
speculative MTP scheduling is not yet exposed by the generation loop.

The complete generated checkpoint inventory is in `tensors.tsv`.

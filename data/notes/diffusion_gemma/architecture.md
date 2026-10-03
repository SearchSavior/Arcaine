# DiffusionGemma INT4-AWQ decode architecture contract

This is the optimization-facing companion to
[`diffusion_gemma_moe_architecture.md`](diffusion_gemma_moe_architecture.md).
The checkpoint remains the source of truth; this file records the fixed decode
workload and hardware facts that constrain the INT4-AWQ kernels.

## Target and fixed workload

- Device: Intel BMG G31 (Xe2), 32 Xe-cores / 256 vector engines.
- Subgroups: 16 and 32.
- Work-group limit: 1024 work-items.
- Shared local memory: 128 KiB per work-group.
- FP16/BF16: supported.
- XMX access: Intel matrix-mad / DPAS builtin is available; the SYCL
  `joint_matrix` aspect is not advertised.
- Deployment: one Level Zero GPU.
- Canvas: exactly 256 positions.
- Denoising: exactly 48 forward passes per canvas.

The denoiser executes 30 layers per pass. Each layer is:

1. input RMSNorm;
2. attention (INT4 Q/K/V/O projections in the AWQ checkpoint);
3. post-attention norm + residual;
4. shared dense GeGLU MLP (left F16 by this AWQ checkpoint);
5. router (left F16) + 128 INT4 experts, top-8;
6. dual FFN postnorm + residual + layer scalar.

The checkpoint uses compressed-tensors `pack-quantized`, symmetric INT4,
group size 32. Packed U4 nibbles use implicit zero point 8. The loader XORs
each byte with `0x88`, producing two's-complement s4 for oneDNN. Runtime
weight scales are BF16 and transposed from `[N,K/group]` to `[K/group,N]`.

## Existing and introduced fusion boundaries

The implementation already joins INT4 sliding Q/K/V and full Q/K projections,
then performs norm/RoPE/KV staging in fused postprocessing kernels. It also
fuses the three FFN prenorms, the original dual postnorm, and the vocab-wide
softcap/softmax/sample path.

The INT4-AWQ work adds three independently gated boundaries:

| Environment flag | Original sequence | Fused sequence | Per-canvas launch reduction |
|---|---|---|---:|
| `DIFF_INT4_FUSE_DENSE_GATE_UP=1` | F16 gate GEMM + F16 up GEMM | one row-concatenated gate/up GEMM | 30 × 48 = 1440 |
| `DIFF_INT4_FUSE_EXPERT_POSTNORM=1` | weighted expert combine + dual postnorm | one SLM-cached combine/postnorm kernel | 30 × 48 = 1440 |
| `DIFF_INT4_FUSE_SELFCOND_ADD_NORM=1` | self-condition residual add + scaleless RMSNorm | one add/norm kernel | 47 |

Total: 2,927 fewer launches for a full 48-step canvas. The expert terminal
fusion also removes one 256×2816 BF16 global intermediate (1.38 MiB) per layer.
It caches one 2816-element BF16 row in 5.5 KiB SLM, well inside the device's
128 KiB limit.

All three fusions preserve the original BF16 rounding boundaries. The expert
weighted sum is explicitly rounded to BF16 before RMS statistics, and the
self-conditioning residual is rounded to BF16 before its norm.

## W4A8 prefill path

`DIFF_INT4_W4A8_PREFILL=1` switches the quantized projections from the default
W4A16 path (BF16 activations, oneDNN decompresses s4 weights into a BF16 GEMM)
to a mixed s8×s4 GEMM: the activations are quantized to s8 per-token and oneDNN
runs its native mixed s8xs4 DPAS, which pre-Xe3p (BMG) supports directly when
the s4 matrix carries no zero points. The weights are unchanged — same raw
`tag::ba` nibble stream and group-32 BF16 scales — and the weight dequant stays
in the GEMM epilogue, so only the activation side changes.

Dispatch is applied only when it is measured to win: `M >= 128`
(`DIFF_INT4_W4A8_MIN_M`), `K >= 1024`, and `N >= K`. Narrow output projections
(o_proj, N < K) and the small-K expert down projection regress 0.5–0.96× and
stay on W4A16, as do the small per-expert decode buckets. On the wide
projections (q/k/v, fused qkv/qk) it is 1.15–1.5× on the GEMM. End-to-end the
gain is modest (prefill +0.4–3.9 %, decode/canvas +1–9 %) because the MoE expert
GEMMs dominate and remain W4A16.

Kernel-path benchmark (no model load):
`./build/arcaine_kbench diffusion-int4-w4a8 --m 256,512,1024,2048 --k 2816 --n 8192`

## Remaining INT4 MoE opportunity

The dominant unfused region is expert execution itself: a device route is
counted, copied to the host, bucketed, and dispatched as per-active-expert
oneDNN s4 decompression GEMMs around a separate GeGLU kernel.

Intel's `llm-scaler` reference demonstrates the desired Xe2 structure:

1. on-device expert histogram and prefix sum;
2. expert-sorted pair indices;
3. grouped INT4 DPAS gate/up with activation in the epilogue;
4. grouped INT4 DPAS down projection;
5. inverse permutation folded into weighted accumulation.

It cannot be copied directly. That kernel consumes IPEX K-major,
Marlin-shuffled weights with group size 128, whereas this checkpoint is native
row-major compressed-tensors packing with group size 32. The installed oneDNN
does not expose grouped matmul publicly, and its internal experimental grouped
path does not accept s4 grouped weights. A follow-on needs a focused layout
transform plus DPAS correctness/roofline harness before integration.


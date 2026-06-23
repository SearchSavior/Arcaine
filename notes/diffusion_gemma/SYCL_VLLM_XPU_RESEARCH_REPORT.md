# DiffusionGemma SYCL Optimization Research Report

Date: 2026-06-22

Scope: compare the local DiffusionGemma C++/SYCL implementation against
`references/vllm-xpu-kernels` and the production vLLM fused-MoE stack under
`vllm/`, with emphasis on MoE routing, expert execution, NVFP4/MXFP4 packing,
and benchmarkable improvement paths.

## Executive Summary

The local implementation is already structurally close to an efficient inference
engine: it uses a liveness arena, oneDNN primitive caches, fused pre/post norms,
fused attention postprocessing, expert sharding, and several env-gated expert
paths. The largest remaining gap versus `vllm-xpu-kernels` is that the local MoE
pipeline still builds most expert metadata on the host, uploads small vectors
synchronously, and then executes either many per-expert oneDNN matmuls or a
scalar grouped NVFP4 fallback. The reference keeps routing metadata and token
remapping on device, feeds a compact `rows_per_expert` layout into a grouped GEMM
scheduler, and uses vector/sub-group kernels for activation+quantization and
gather.

Highest-value next work:

1. Add a small MoE backend/capability interface before adding more kernel
   variants, modeled after production vLLM's router / prepare-finalize /
   experts / reduce split.
2. Move router softmax/top-k and per-expert row remapping to the GPU.
3. Replace local host-built `assign_token` / `assign_slot` / `compute_slot`
   metadata with a device prologue similar to vLLM's `remap_hidden_states`.
4. Extend the existing grouped-expert idea beyond Q8 by adding an env-gated
   grouped oneDNN or custom grouped GEMM path for NVFP4/BF16/int4 that consumes
   `rows_per_expert` directly, reducing per-expert launch count.
5. Fuse GeGLU + NVFP4 activation packing for the oneDNN-hybrid NVFP4 path.
6. Keep the documented DPAS custom NVFP4 work as an experimental path, but do
   not wire it into the live default path until it beats `hybrid` in the
   `-p 512,1024,2048,8192` benchmark sweep.

## Local Implementation Findings

### Router

`src/diffusion_gemma/moe.cpp` computes router scores on the GPU, then downloads
`seq * E` BF16 scores to the host and performs softmax, top-k, renormalization,
and per-expert scale application on the CPU. See
`src/diffusion_gemma/moe.cpp:103-140`.

This is simple and correct, but it creates a hard synchronization point per MoE
layer. It also feeds the expert path host vectors (`idx`, `weight`) instead of
device-resident routing metadata.

### Expert Bucketing

`src/common/gpu/expert_parallel.cpp` builds `count`, `base`, `bucket_cap`,
`assign_token`, `assign_slot`, `slot_for_tk`, and weight arrays on the host, then
uploads them with blocking `q.memcpy(...).wait()` calls. See
`src/common/gpu/expert_parallel.cpp:143-238`. The hot/tail strategy is useful:
cold experts use a fixed tail capacity and hot experts get exact-size padded
runs, limiting worst-case padding. That is a good local heuristic and should not
be discarded.

The reference design suggests the next improvement: keep the same logical
bucket model, but build it on GPU into `rows_per_expert` and
`unpermuted_row_to_permuted_row`, so the CPU does not sit between router and
experts.

### NVFP4 Paths

The local NVFP4 code has three expert paths behind
`DIFF_NVFP4_EXPERT_KERNEL`: default, `hybrid`, and `custom`
(`src/common/gpu/expert_parallel.cpp:36-53`). The hybrid path groups activation
packing and GeGLU, then still loops over active experts and calls oneDNN FP4
matmul per expert (`src/common/gpu/expert_parallel.cpp:336-413`). The custom
path uses one scalar grouped matmul kernel (`matmul_nvfp4_grouped_custom`) for
gate/up and down (`src/common/gpu/expert_parallel.cpp:250-335`), but that kernel
dequantizes and accumulates per output element rather than driving XMX/DPAS.

The local packing code uses 16-value groups with f8_e4m3 scales and f4_e2m1
payloads (`src/common/gpu/nvfp4.hpp:140-180`) and oneDNN FP4 matmul attributes
with source/weight/destination scales (`src/common/gpu/nvfp4.hpp:183-245`).
This is a strong base for oneDNN-based experiments.

### Q8 Grouped Expert Path

One local path has already moved in the direction recommended by the reference:
Q8 experts default to a grouped block-list kernel behind
`DIFF_Q8_GROUPED_EXPERT`, and optionally fuse gate/up with GeGLU behind
`DIFF_Q8_FUSED_EXPERT_GEGLU` (`src/common/gpu/expert_parallel.cpp:80-90`,
`src/common/gpu/expert_parallel.cpp:550-598`). This is an important positive
signal: the codebase already has a local pattern for grouped expert execution,
but it is still fed by host-built block metadata and has not been generalized to
the NVFP4 hybrid path.

## Reference Implementation Findings

### Fused MoE Pipeline

`vllm-xpu-kernels` structures MoE as:

1. `remap_hidden_states`: count expert rows, build a token-to-permuted-row map,
   and expand hidden states into expert-contiguous rows.
2. grouped GEMM for `w13`.
3. activation.
4. grouped GEMM for `w2`.
5. `moe_gather`: weighted top-k combine back to `[tokens, hidden]`.

The wrapper makes this pipeline explicit in
`references/vllm-xpu-kernels/vllm_xpu_kernels/fused_moe_interface.py:290-379`.

### GPU Remap

The reference first uses per-workgroup local atomics, then device atomics, to
produce `rows_per_expert` and the unpermuted-to-permuted map
(`references/vllm-xpu-kernels/csrc/moe/remap_hidden_states.cpp:9-113`). A second
kernel expands each row to its top-k expert destinations with vectorized loads
and stores (`remap_hidden_states.cpp:127-170` and following).

This directly targets a local bottleneck: local code does equivalent work on the
host and uploads multiple small arrays.

### Grouped GEMM Scheduler

The Xe2 grouped GEMM code consumes `rows_per_expert`, walks experts in a single
kernel, computes expert-local A/B/D pointers, and schedules tiles dynamically
(`references/vllm-xpu-kernels/csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2.hpp:74-175`).
Policy choices include large throughput tiles and small-M policies such as
M=8/16/32 variants (`gemm_xe2_policy.hpp:9-105`).

This matters for DiffusionGemma because expert loads are skewed and often small.
The local hot/tail strategy partially solves this for oneDNN; a grouped scheduler
could remove many tiny launches and choose small-M tiles without padding every
expert to the same fixed capacity.

### Gather

The reference gather kernel specializes on TOPK and vector width. It loads all
per-token permuted row ids and weights once, then accumulates `ElemsPerItem`
hidden elements per work-item (`references/vllm-xpu-kernels/csrc/moe/moe_gather.cpp:9-88`).
The local combine kernel is similar in shape, but it receives host-built slot
metadata. Once local remap moves to GPU, adopting the reference gather layout is
straightforward.

### Activation + FP4 Quantization

The reference has fused activation-and-MXFP4 quantization kernels that perform
activation, sub-group max reduction, power-of-two scale selection, and packed
FP4 output in one pass
(`references/vllm-xpu-kernels/csrc/quantization/fused_kernels/fused_silu_mul_mxfp4_quant.cpp:41-118`).
Its generic MXFP4 quant path also uses vectorized kernels and chooses workgroup
packing based on group count
(`references/vllm-xpu-kernels/csrc/quantization/fp4/mxfp4_quant.cpp:53-108`).

Local NVFP4 hybrid currently runs GeGLU and then packs the activation in the
same profiled block, but as two kernel submissions (`src/common/gpu/expert_parallel.cpp:388-396`).
Fusing these for GeGLU+tanh and local f8_e4m3 scale encoding is a contained,
measurable experiment.

## Production vLLM Findings

This additional pass looked at the production fused-MoE stack under `vllm/`.
The most important production lesson is architectural: vLLM makes routing,
prepare/finalize, expert compute, and weight/reduce separate replaceable
components rather than treating MoE as one monolithic code path.

### Modular MoE Contract

Production vLLM names the activation layouts explicitly:
`Standard` and `BatchedExperts`
(`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:83-92`). It also
uses an `ExpertTokensMetadata` object for per-expert token counts
(`modular_kernel.py:95-114`) and a `PrepareResultType` that can return
quantized/dispatched activations, scales, expert metadata, and optionally
dispatched top-k ids/weights (`modular_kernel.py:139-157`).

The main modular kernel then runs:

1. `_prepare`: calls `prepare_finalize.prepare(...)` or `prepare_async(...)`,
   passing `defer_input_quant=self.fused_experts.expects_unquantized_inputs`
   (`modular_kernel.py:1116-1202`).
2. `_fused_experts`: allocates workspaces from the expert implementation's
   declared shapes and calls a uniform expert `apply(...)`
   (`modular_kernel.py:1204-1283`).
3. `_finalize`: delegates combine and router-weight application to
   `prepare_finalize.finalize(...)` and the expert's
   `finalize_weight_and_reduce_impl()` (`modular_kernel.py:1285-1351`).

This is directly useful locally. Right now `expert_parallel.cpp` combines
backend selection, metadata construction, quantization placement, matmul
strategy, activation, and combine policy in one area. Before adding a DPAS path
or another grouped GEMM path, add a small local descriptor for:

- activation layout: compact rows, hot/tail rows, or batched experts
- input quantization site: prepare step or expert kernel
- combine contract: expert output already reduced, or `[tokens, topk, H]`
  needing a reduce
- supported quantization: BF16, Q8, int4, NVFP4, MXFP4
- expert map support and top-k id type

This keeps future env-gated experiments from becoming cross-product branches.

### Production Router Lessons

The default production router allocates `topk_weights`, `topk_ids`, and
`token_expert_indices` directly on device, then calls a custom op for
`topk_softmax` or `topk_sigmoid`
(`vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:69-113`).
This strongly reinforces `DIFF_ROUTER_GPU_TOPK=1`: the local host softmax/top-k
is not just an optimization opportunity; it is outside the production execution
shape.

The grouped router also has two details worth copying. First, sigmoid grouped
top-k has a fully fused path, while softmax grouped top-k still computes softmax
first and then calls a grouped top-k op
(`router/grouped_topk_router.py:41-66`). Second, vLLM has a batch-invariant mode
that switches `torch.topk(..., sorted=True)` for deterministic expert selection
(`grouped_topk_router.py:133-137`). For local GPU top-k, add a correctness/debug
gate such as:

```text
DIFF_ROUTER_BATCH_INVARIANT=1
```

That mode should choose stable tie-breaking even if it costs a little
performance. It will make CPU-vs-GPU router comparisons much easier.

### Quantization and Deferred Input Packing

Production vLLM centralizes quantization metadata in `FusedMoEQuantConfig` and
`FusedMoEQuantDesc`, describing the four MoE tensors (`a1`, `w1`, `a2`, `w2`),
group shapes, scales, global scales/alphas, zero-points, and bias support
(`vllm/vllm/model_executor/layers/fused_moe/config.py:174-240`). It also has a
large routing-method enum so kernels can reject unsupported routing semantics
before runtime (`config.py:100-171`).

The local implementation does not need the full Python type system, but it would
benefit from a compact C++ equivalent:

```text
MoeQuantDesc { dtype, group_shape, scale_kind, scale_layout }
MoeBackendCaps { supports_router_on_device, defer_input_quant, output_reduced, ... }
```

The production no-DP/EP prepare path shows the specific quantization-placement
hook to copy: if `defer_input_quant` is true, prepare returns the original input
and no scale; otherwise it calls `moe_kernel_quantize_input`
(`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py:14-37`).
For the local NVFP4 path, this suggests a new A/B gate:

```text
DIFF_NVFP4_DEFER_INPUT_PACK=1
```

This should compare packing during prepare/remap versus packing inside the
expert path. It is especially relevant if the fused GeGLU+pack kernel and a
future grouped GEMM want different activation layouts.

### Production XPU Wrapper

Production vLLM's XPU expert wrapper exposes one `XPUExperts` call shape and
flips capability booleans (`is_fp8`, `is_int4`, `is_mxfp4`, `is_mxfp8`,
`is_block_fp8`) before constructing `XpuFusedMoe`
(`vllm/vllm/model_executor/layers/fused_moe/experts/xpu_moe.py:60-180`). It
declares `expects_unquantized_inputs=True`, returns `TopKWeightAndReduceNoOP`,
and lets the XPU fused kernel produce the final reduced output
(`xpu_moe.py:72-115`). It also has separate subclasses for int4 and MXFP4
support (`xpu_moe.py:286-336`).

This differs from the lower-level `references/vllm-xpu-kernels` implementation,
but the lesson is compatible: the public expert call surface should remain
stable while the selected backend changes. Local `DIFF_NVFP4_EXPERT_KERNEL`,
`DIFF_Q8_GROUPED_EXPERT`, and future `DIFF_MOE_GROUPED_GEMM` paths should plug
into the same local expert interface instead of growing separate orchestration.

### NVFP4 Backend Oracle

Production vLLM has an NVFP4 backend selector that tries multiple kernels in a
fixed order and uses each candidate's `is_supported_config(...)` result before
selecting it (`vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:151-274`).
It also converts weights/scales into backend-specific formats through one
function, `convert_to_nvfp4_moe_kernel_format(...)`
(`oracle/nvfp4.py:278-407`), then builds one `FusedMoEQuantConfig`
(`oracle/nvfp4.py:410-430` and following).

Local code should adopt the same idea in miniature:

- Parse env vars into a requested backend.
- Check shape, dtype, activation, layout, and scale support up front.
- Log why a requested backend falls back.
- Put weight/scale layout conversion in one place, not inside every runtime
  expert branch.

This would make it safer to compare `default`, `hybrid`, `custom`, future
`grouped_gemm`, and future `dpas` paths without hidden layout mismatches.

### Weight/Reduce Placement

Production vLLM has explicit reduce policies:
`TopKWeightAndReduceNoOP` for kernels that already apply weights and reduce,
and `TopKWeightAndReduceContiguous` for expert outputs shaped like
`[tokens, topk, hidden]` that need weight application and `moe_sum`
(`vllm/vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py:44-121`).

The local combine path can use this as a design target. Treat "where router
weights are applied" and "who performs the top-k reduce" as backend properties,
not incidental behavior inside each kernel. For top-k=1 models or experiments,
also preserve a path that applies router weight on input before quantization,
as production does for the prepare path
(`prepare_finalize/no_dp_ep.py:68-76`).

### Production Benchmark Shape

Production vLLM keeps separate benchmarks for fused top-k, permute/unpermute,
and full MoE default tile choices:

- `benchmark_fused_topk.py` compares Torch top-k against vLLM fused top-k across
  expert counts and top-k values
  (`vllm/benchmarks/kernels/benchmark_fused_topk.py:12-99`).
- `benchmark_moe_permute_unpermute.py` isolates permute and unpermute kernels
  with CUDA graph replay (`benchmark_moe_permute_unpermute.py:34-180`).
- `benchmark_moe_defaults.py` compares tuned, old-default, and new-default MoE
  tile configs across batch sizes (`benchmark_moe_defaults.py:1-15`,
  `benchmark_moe_defaults.py:195-220`).

Do the same locally. End-to-end `diffusion_bench` is required, but small
microbenchmarks or profiler sections should isolate router, remap, activation
pack, down-gather, and final combine, otherwise a win in one stage can be
hidden by a regression in another.

## Additional Production Insights

These are lower-level lessons from the extra production pass. They are less
visible than the router/remap/GEMM pipeline, but they are likely to improve the
quality of local SYCL experiments.

### Reusable Remap Scratch

Production vLLM has a `MoEPermuteScratch` object that preallocates the metadata
needed for repeated permute/unpermute calls: token-expert indices,
expert offsets, permuted and inverse-permuted indices, sorted row ids, top-k
conversion buffers, and sort workspace
(`vllm/vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py:7-80`).
The permute path then validates shape/dtype and reuses slices of those buffers
instead of allocating fresh tensors on every call (`moe_permute_unpermute.py:82-250`).

Local implication: `DIFF_MOE_GPU_REMAP=1` should not only move remap to device;
it should also add a persistent scratch object keyed by max tokens, top-k,
local experts, hidden size, and dtype. Gate the reusable variant separately:

```text
DIFF_MOE_GPU_REMAP_SCRATCH=1
```

This will separate "GPU remap is faster" from "allocation churn is lower",
which are different wins.

### Shape-Tuned Config Tables

Production vLLM looks up tuned MoE tile configs by `(E, N, device, dtype,
block_shape)`, allows a user-provided tuned-config folder, and chooses the
nearest measured batch size (`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:999-1074`,
`fused_moe.py:1303-1332`). If no table exists, its defaults still vary tile
sizes by batch size, quantization mode, block shape, and tokens per expert
(`fused_moe.py:1203-1300`).

Local implication: hardcoding one `DIFF_NVFP4_EXPERT_ROUND` or one DPAS tile
policy will probably leave performance on the floor. Add a simple tuning table
for local expert kernels:

```text
DIFF_MOE_TUNED_CONFIG_DIR=/path/to/configs
```

A local config key can start with `(quant, E, hidden, inter, topk, layout)`.
Values can be `expert_round`, remap layout, tile M/N/K, subgroup size, and
whether fused GeGLU+pack is enabled. The required prompt sweep
`-p 512,1024,2048,8192` is a good first grid.

### Runtime Fallback Composition

Production vLLM has a `FallbackExperts` wrapper that composes a preferred
expert implementation with a fallback implementation, requires them to agree on
activation format and weight/reduce policy, then selects the implementation at
runtime (`vllm/vllm/model_executor/layers/fused_moe/experts/fallback.py:1-146`).

Local implication: use this pattern for partial SYCL experiments. A future DPAS
or grouped-GEMM path can handle aligned/high-occupancy cases and fall back to
`hybrid` for unsupported shapes without changing the caller. That is safer than
adding all-or-nothing env branches.

### Expert Load Recording in Router

Production vLLM's base router can map logical expert ids to physical expert ids
for load balancing and record expert load in the same GPU kernel
(`vllm/vllm/model_executor/layers/fused_moe/router/base_router.py:15-116`). Its
router template computes routing, optionally captures logical ids, applies the
mapping, and finally converts id dtype (`base_router.py:221-275`).

Local implication: even without full dynamic expert load balancing, a GPU router
can atomically record `expert_token_count` into a metrics buffer. Add:

```text
DIFF_MOE_RECORD_EXPERT_LOAD=1
```

This gives a real distribution for deciding whether `hot_tail`, compact rows,
or grouped scheduling is best per layer and prompt length.

### Sentinel Experts and Zero Output

Production kernels explicitly handle expert id `-1` as "not on this rank" and
write zeros for those output blocks
(`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:39-51`,
`fused_moe.py:155-167`). That keeps the kernel path uniform when expert maps
filter out remote experts.

Local implication: if the GPU remap path introduces an `expert_map`, keep a
sentinel `-1` convention through the remap/GEMM/combine interface. It avoids
branching the host orchestration for local versus non-local experts and makes
future EP behavior easier to reason about.

### 64-bit Offset Audit

Production vLLM has explicit comments and regression coverage for int32
overflow in stride/offset products; large MoE outputs cast token offsets to
int64 before pointer arithmetic (`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:155-179`,
`vllm/tests/kernels/moe/test_moe.py:399-436`).

Local implication: audit every SYCL kernel that computes offsets like
`row * hidden`, `slot * inter`, `expert * stride`, or `token * topk * dim`.
Use `size_t` or `uint64_t` for products before adding byte offsets. This is not
just defensive programming; the required `8192` prompt case plus top-k and
large intermediate dimensions can move surprisingly close to 32-bit limits.

### Zero-Expert Pattern

Production has a `ZeroExpertRouter` that computes zero-expert contributions as
a router-side side effect, remaps those routes to expert `0` with weight `0`,
and lets downstream MoE ignore them
(`vllm/vllm/model_executor/layers/fused_moe/router/zero_expert_router.py:12-113`).
The runner later adds the saved zero-expert output to the final result
(`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:612-716`).

Local implication: if DiffusionGemma variants ever introduce routed identity,
null, or shared pseudo-experts, do not push that complexity into expert kernels.
Handle pseudo-experts in router/finalize and keep expert compute limited to real
experts.

## Recommended Experiments

The `DIFF_*` variables in this section are proposed experiment gates unless the
report explicitly identifies them as existing. Keeping each change behind a
descriptive environment variable preserves A/B testing and matches the existing
style in `expert_parallel.cpp`.

### 0. Local MoE Backend Descriptor

Before adding another kernel branch, add a small internal descriptor behind no
behavioral change:

```text
DIFF_MOE_BACKEND_TRACE=1
```

The trace should print the selected router path, remap layout, expert kernel,
quantization placement, reduce policy, and fallback reason. This mirrors
production vLLM's backend-oracle style and will make the next A/B runs much
less ambiguous.

Also add a runtime fallback wrapper for experimental expert kernels: preferred
backend first, `hybrid` fallback second, with an explicit check that both agree
on activation layout and reduce policy.

### 1. GPU Router Top-K

Add a device router path behind:

```text
DIFF_ROUTER_GPU_TOPK=1
DIFF_ROUTER_BATCH_INVARIANT=0|1
```

Implementation sketch:

- Keep existing router matmul.
- Add a SYCL kernel that does per-row softmax/top-k/renormalize for `E=128`,
  `top_k=8`, then multiplies by `per_expert_scale`.
- Output device arrays: `topk_ids[int32]`, `topk_weights[float]`.
- Keep the existing CPU router as the default fallback.
- In batch-invariant mode, use stable tie-breaking for CPU/GPU comparability.

Why first: it removes a per-layer GPU-to-host download and makes the later GPU
remap path possible.

Reference anchor: vLLM's router kernels use group reductions for softmax and
top-k (`references/vllm-xpu-kernels/csrc/moe/topk.cpp:43-180`).

### 2. Device Remap / Rows-Per-Expert Prologue

Add a remap path behind:

```text
DIFF_MOE_GPU_REMAP=1
DIFF_MOE_GPU_REMAP_SCRATCH=0|1
```

Implementation sketch:

- Convert local `idx`/`weight` inputs to device-side `topk_ids`/`topk_weights`.
- Produce `rows_per_expert[localE]`, `unpermuted_row_to_permuted_row[A_all]`,
  and `Xe[A_local, H]` on the device.
- Reuse current arena allocations and current combine fallback initially.
- Add a persistent scratch path to separate allocation savings from kernel
  speedups.
- Keep hot/tail padding as an optional compatibility mode behind:

```text
DIFF_MOE_GPU_REMAP_LAYOUT=compact|hot_tail
```

Why second: it attacks `upload_alloc(...).wait()` and host bucketing in
`run_shard`, while preserving the downstream oneDNN matmul path.

Reference anchor: `remap_hidden_states` counts rows with local and device
atomics and writes the permuted map fully on GPU
(`references/vllm-xpu-kernels/csrc/moe/remap_hidden_states.cpp:48-113`).

### 3. Grouped Expert Matmul Interface

For NVFP4/BF16/int4, add a grouped expert matmul experiment behind:

```text
DIFF_MOE_GROUPED_GEMM=onednn
```

Initial version can be less ambitious than vLLM's CUTLASS-style scheduler:

- Consume compact expert-contiguous rows and `rows_per_expert`.
- Batch same-M or rounded-M experts together when possible.
- Fall back to existing per-expert oneDNN calls for outlier shapes.

Second version:

```text
DIFF_MOE_GROUPED_GEMM=sycl
```

This can borrow the reference's single-kernel scheduler concept for BF16/int4
first, before attempting NVFP4. Q8 already has a local grouped expert kernel, so
it should be treated as a comparison point and source of reusable local patterns.

Why third: after GPU remap, the main remaining overhead is many expert GEMM
launches and padding policy. A grouped interface lets the scheduler decide
between compact rows, hot/tail rows, and small-M kernels.

### 4. Fused GeGLU + NVFP4 Pack

Add:

```text
DIFF_NVFP4_FUSED_GEGLU_PACK=1
DIFF_NVFP4_DEFER_INPUT_PACK=0|1
```

Implementation sketch:

- Input: BF16 `gu[rows, 2 * inter]`.
- Output: packed f4_e2m1 `act_packed[rows, inter/2]` and f8_e4m3
  `act_scale[rows, inter/16]`.
- Fuse tanh-GELU, multiply by up, max-abs over each 16-value group, encode scale,
  and pack two nibbles per byte.

Why fourth: it reduces one large intermediate write/read pair in the current
hybrid path without changing the matmul backend.

Caveat: prior DPAS notes found gate/up+GeGLU fusion slower for the custom DPAS
path due to register pressure, but this experiment is different: it fuses only
the activation-to-pack transition used by the oneDNN hybrid path.

### 5. Weight Layout and Primitive Cache Audit

Continue testing:

```text
DIFF_NVFP4_WEIGHT_LAYOUT=any
DIFF_NVFP4_EXPERT_ROUND=8|16|32|64
DIFF_MOE_TUNED_CONFIG_DIR=/path/to/configs
```

The local `weight_any` reorder path is already present
(`src/common/gpu/nvfp4.hpp:221-245`). The missing piece is a systematic result
matrix across prompt sizes, expert placement, and kernels.

Use a tuned-config directory once the matrix grows beyond a couple of flags.
Production vLLM's nearest-batch config lookup is a good minimal model.

### 6. DPAS Path as Research, Not Default

Keep the DPAS investigation behind a descriptive flag, for example:

```text
DIFF_NVFP4_EXPERT_KERNEL=dpas
```

The current source parser accepts `default`, `hybrid`, and `custom`; `dpas` is
documented in `NVFP4_DPAS_KERNEL.md` as a research direction/result, not as a
currently live runtime choice in `expert_parallel.cpp`. That makes it valuable
as a learning path, but not the next default runtime path. Use it to validate
tile policy ideas from `vllm-xpu-kernels`, especially small-M policies
(`M=8/16/32`) and dynamic tile scheduling, and only wire it into the live parser
after it beats `hybrid`.

## Benchmark Protocol

Do not run repo tests; this repository currently has no test suite. For
performance work, use env-gated A/B runs and the required prompt sweep:

```bash
DIFF_ROUTER_GPU_TOPK=0 \
DIFF_MOE_GPU_REMAP=0 \
./build/diffusion_bench \
  --model models/diffusiongemma-26B-A4B-it-NVFP4 \
  -p 512,1024,2048,8192 \
  -n 256 \
  -ds 48 \
  -w 1 \
  -r 5 \
  --kernels hybrid \
  --experts shard \
  --md
```

Example A/B command for the first combined GPU-router/remap experiment:

```bash
DIFF_ROUTER_GPU_TOPK=1 \
DIFF_MOE_GPU_REMAP=1 \
DIFF_MOE_GPU_REMAP_LAYOUT=compact \
./build/diffusion_bench \
  --model models/diffusiongemma-26B-A4B-it-NVFP4 \
  -p 512,1024,2048,8192 \
  -n 256 \
  -ds 48 \
  -w 1 \
  -r 5 \
  --kernels hybrid \
  --experts shard \
  --md
```

Recommended metrics to capture:

- `ffn.*.router`
- `exp.*.setup_gather`
- `exp.*.gpu_topk`
- `exp.*.gpu_remap`
- `exp.*.gpu_remap_scratch`
- `exp.*.expert_load_histogram`
- `exp.*.pack`
- `exp.*.gateup_mm`
- `exp.*.geglu_pack`
- `exp.*.down_mm`
- `exp.*.combine`
- end-to-end forward passes/s and decode tokens/s

## Suggested Implementation Order

1. Add the local MoE backend descriptor and `DIFF_MOE_BACKEND_TRACE=1`.
2. Add a preferred/fallback expert wrapper so experiments can fall back to
   `hybrid` by shape or capability.
3. Introduce device buffers for top-k ids and weights while preserving the CPU
   router as default.
4. Implement `DIFF_ROUTER_GPU_TOPK=1`, with
   `DIFF_ROUTER_BATCH_INVARIANT=1` for validation.
5. Implement `DIFF_MOE_RECORD_EXPERT_LOAD=1` inside the GPU routing/remap path.
6. Implement `DIFF_MOE_GPU_REMAP=1` and initially feed current oneDNN expert
   matmul loops.
7. Add `DIFF_MOE_GPU_REMAP_SCRATCH=1`.
8. Switch combine to the `unpermuted_row_to_permuted_row` layout and compare
   against current `slot_for_tk`.
9. Implement `DIFF_NVFP4_FUSED_GEGLU_PACK=1` and compare
   `DIFF_NVFP4_DEFER_INPUT_PACK=0|1`.
10. Prototype `DIFF_MOE_GROUPED_GEMM=onednn`.
11. Add `DIFF_MOE_TUNED_CONFIG_DIR` when tile/rounding choices become
    shape-dependent.
12. Revisit custom grouped SYCL/DPAS only after the orchestration overheads are
   measured away.

## Bottom Line

The best opportunity is not a single heroic matmul kernel yet. It is removing
the host boundary in the MoE path and adopting the reference's device-resident
MoE dataflow: GPU top-k, GPU remap, `rows_per_expert`, grouped expert execution,
and vectorized gather. That path preserves the currently best oneDNN NVFP4
hybrid matmuls while reducing synchronization and launch overhead around them.

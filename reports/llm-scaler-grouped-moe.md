# llm-scaler: Grouped MoE GEMM & Expert Routing Analysis

Reference survey for Arcaine's Qwen3.5-MoE AWQ INT4 grouped DPAS MoE kernel
(`src/modeling/qwen3_5_moe/kernels/int4_grouped_moe.hpp`, dispatched by
`QWEN35_MOE_INT4_IMPL=onednn|dpas`).

- Repo surveyed: `reference/llm-scaler` (Intel llm-scaler, vLLM custom ESIMD/SYCL kernels)
- Date: 2026-07-30
- All paths relative to `reference/llm-scaler/vllm/custom-esimd-kernels-vllm/` unless absolute.

**Scope note:** llm-scaler's int4 is **symmetric, group_size=128 only** — there is no
AWQ/asymmetric zero-point support anywhere (grep for `zero_point|qzeros|awq|asym` across
`csrc/` → no hits). Our AWQ g32 zero-point folding has no counterpart there.

## 0. Architecture map

Three independent implementations, dispatched by batch size at the vLLM layer:

| Path | Files | Style | Used when |
|---|---|---|---|
| **Decode (batch)** int4 | `csrc/moe_batch/moe_int4.sycl` (3712 lines) + `moe_topk.h` | plain SYCL nd_range, scalar MAC, no DPAS | `num_tokens <= 128` (patch `vllm/patches/vllm_for_multi_arc.patch:13654`) |
| **Decode (batch)** fp8 w8a16 | `csrc/moe_batch/moe.sycl` (1819 lines) | ESIMD + DPAS | `num_tokens <= 128`, fp8 layers (patch:13697) |
| **Prefill** int4 | `csrc/moe_prefill/moe_prefill_int4.sycl` (1040) + `csrc/moe_batch/int4_nmajor_gemm.h` | ESIMD + DPAS, expert-major grouped GEMM | prefill; production vLLM integration uses CUTLASS grouped GEMM instead (§6) |

vLLM integration: decode → `moe_forward_cutlass_nmajor_int4_full` (patch:13677);
prefill → `_esimd_prefill_moe_apply` (patch:6992-7052), gated by `USE_ESIMD_MOE_PREFILL=1`
/ `DISABLE_CUTLASS_PREFILL` (patch:7189, 7431-7436).

## 1. moe_batch GEMM kernels: variants, tiling, DPAS, layouts

### 1a. Decode int4 "cutlass_nmajor" path (plain SYCL, no DPAS)

Dispatch ladder in `moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared`
(`moe_int4.sycl:2615-2657`):

- `n_tokens == 1` → `moe_tiny_m_up_cutlass_int4_with_shared_fp16_kernel` (1772) +
  `moe_tiny_m_down_cutlass_int4_with_shared_fp16_htile_kernel` (2309).
- `n_tokens > 1` (≤64, enforced 2566) → weight-stationary
  `moe_ws_up_cutlass_int4_with_shared_fp16_kernel` (2059) with `N_TILE ∈ {2,4,8}` chosen
  by `n_tokens<=2 / <=4 / else` (2071-2079), then
  `moe_ws_down_cutlass_int4_with_shared_fp16_kernel<IndexT, H_TILE=4|8>` (2092, chosen at
  2640-2656).

Weight layout ("CUTLASS N-major", uint8): `w13 [E, 2I, H/2]`, `w2 [E, H, I/2]` — one
packed byte per 2 K-elements, **K contiguous per output row** (TORCH_CHECKs at
2594-2596). Scales `[E, N, K/128]` fp16, **group_size hardcoded 128**
(`k_groups = hidden_size/128` at 1706, 1941, 2103, 2325; `group = k>>7` at 2353).
Nibble order: low nibble = even k (1739-1742, 2354-2361). Sign: two's-complement
`decode_s4_nibble` = `v>=8 ? v-16 : v` (1689-1692).

Kernel shapes:

- **tiny up** (1694-1770): grid = `n_tokens*top_k*I` work-groups of 64 WIs; each WG
  computes **one output element** (one (token, route, col)); WIs stride packed bytes
  (`kb += 64`), scalar FMA into fp32, tree reduction over `sycl::local_accessor`
  (1750-1760), fused SiLU·mul at 1764-1765.
- **ws up** (1932-2005): `range<2>(n_tokens*top_k, I/N_TILE)`, 1 WI per (route,
  col-tile); vectorized: loads 64 packed bytes as `simd<uint8_t,64>`, widens to uint16,
  splits lo/hi nibbles, sign-fix via `merge(x-16, x>=8)` (1976-1985), fp32
  `simd<float,64>` accumulators, per-WI `detail::sum` reduction (1997). x is loaded as
  fp16 and de-interleaved even/odd with `select<VL,2>` to match nibble order (1963-1965).
- **tiny down htile** (2309-2408): grid = `n_tokens * (H/4)` WGs of 64; `H_TILE=4`; loops
  over `route` (top_k) inside the kernel — **routing weight is folded into the
  activation** (`in_val = route_weight * in_row[k]`, 2351) before the MAC, so combine is
  free; SLM tree reduce (2384-2404). Shared experts fused in the same kernel (2368-2382).
- **ws down** (2092-2176): `range<2>(n_tokens, H/H_TILE)`, `VL=64` SIMD; same route-loop
  + `rw *` folding (2118-2146); shared experts fused (2149-2168).

There is also a `moe_multi_m_*` pair (3419, 3513) for `moe_forward_cutlass_nmajor_int4_full`
when `M>32` (dispatch at 3571-3583: `M<=32 && shared>0` → tiny path; else multi_m) — same
scalar style, `M>32` within the ≤128-token decode window.

Larger-batch int4 path `moe_forward_full_int4` (2795-3025) uses the older IPEX K-major
layout (`[E, K_packed, 2I]` int32, marlin-shuffled) with `moe_up_routed_int4[_slm]_kernel`
(358/488) and `moe_down_routed_int4[_slm]_kernel` (734/840), hybrid-dispatched by WI
count: `up_wis = n_tokens*top_k*(I/16) < 2048` → SLM variant (2875-2895, 2926-2947).
Supports both IPEX K-major and GGML N-major via `use_ggml_layout` flag (2819).

### 1b. Prefill int4 path (ESIMD + DPAS, expert-major)

`moe_prefill_int4.sycl`:

- **Up** `moe_prefill_up_forward_kernel<IT, BS=128, MAX_M=32, N=16>` (121-264): grid
  `range<2>(num_experts, I/16)`, **one work-item per (expert, N-tile)** — single-WI ESIMD
  kernels (no sub-group cooperation). M handled by looping `MAX_M=32` chunks over the
  expert's compacted token range `[t0,t1)` with `min()` clamping (156-164). Weight = IPEX
  K-major `[E, K/8, 2I]` int32, **marlin-shuffled** nibbles:
  `unshuffle[8] = {0,2,4,6,1,3,5,7}` (136), dequant `((w>>shift)&0xF) - 8` (217-219) —
  symmetric, −8 bias, no zero-points.
- b_tile: `lsc_gather<uint32_t,8>` 16 lanes × MS rows, cache_hint `cached/cached`
  (187-194). a_tile built by scalar per-nibble dequant into `simd<fp16,16>` rows, scaled
  by fp16 per-group scale (203-223). DPAS: `xmx_ns::dpas<8, 8, fp16, fp16, fp16, fp16>`
  (230) — **fp16×fp16 inputs, fp16 accumulator** (not s8 DPAS, not fp32 accum). Fused
  SiLU·mul on accumulators (242-243), scatter intermediate by pair_idx (246-258).
- **Down** `moe_prefill_down_forward_kernel<IT, BS=128, MAX_M=32, N=32>` (270-395):
  identical structure, no routing weight (applied later), scatters per-pair
  `expert_output` (375-389).
- Template instantiation confirms **group_size=128 only**:
  `moe_prefill_up_forward_kernel<fp16, 128>` / `<bf16, 128>` (531, 542, 595, 606).

`int4_nmajor_gemm.h` (included at `moe_int4.sycl:3227`, header comment "standalone for
benchmarking"): N-major variant of the same DPAS design (`moe_up_int4_nmajor_kernel`
41-179, `moe_down_int4_nmajor_kernel` 186-325), weight `[E, N, K/8]` int32 N-major so
**no marlin unshuffle needed** (comment at 131), two's-complement dequant (134-139). Only
consumed by the standalone bench op `moe_gemm_int4_nmajor` (`moe_int4.sycl:3232`,
E generic but no routing; N_TILE=16, MAX_M=16, BS=128 at 3255-3257 with comment
"was 32 — smaller M tile, less GRF pressure"). **No production call sites** — the MoE
forward functions never invoke these kernels.

### 1c. fp8 batch path (ESIMD + DPAS, route-per-token)

`moe.sycl`: `moe_up_routed_e4m3_kernel` (364-448) etc. — grid
`nd_range<2>(n_tokens*top_k, (I/16)*GS)` with `GS=16` WIs per (route, n_tile); weights
loaded via **`lsc_load_2d` / `config_2d_mem_access<uint8_t,16,16,1>`** 2D block loads
(401-415), fp8→fp16 software conversion (`fp8e4m3_to_half` 25, `fp8e4m3_block_to_vnni`
63), `dpas<8, 1, float, float, fp16, fp16>` — **fp32 accumulator, repeat=1** (407, 415).
K-split across 16 WIs, SLM reduction (`slm_init`, `slm_block_store`, barrier, tid==0 sums
16 partials, 383-443). Per-expert **scalar fp32 scale** applied after accumulation
(`gate_up_scale[eid]`, 432-434) — no per-group scales. Down kernel (801-853) is
deliberately single-WI per (route, n_tile): comment "Single thread does all K iterations
… No SLM, no barrier — eliminates sync overhead for small K" (828-829). Routing weight
and scale folded post-DPAS (842-844).

## 2. Routing handoff and device-side compaction

**Decode int4/fp8 fused paths do NOT compact.** Kernels index `topk_idx`/`topk_weight`
directly as `(token, route)` pairs (e.g. `moe_int4.sycl:1724`, 2119, 2347;
`moe.sycl:386-392` `token = route_idx / top_k`). The same expert's weights are re-read
once per routed token — acceptable because decode M is tiny and the whole design is
weight-BW-bound.

Top-k itself is on-device:

- `MoE_TopK_V2_Kernel<NUM_EXPERTS, TOPK>` (`xpu/esimd_kernels/moe_ops.h:106-219`): one WG
  of size 1 per token (`nd_range {T},{1}`, 226-229); fully unrolled softmax over 64-wide
  chunks in fp32 (126-153), then TOPK rounds of global argmax + zero-out via
  `chunk_argmax_and_zero` (159-176), fp32 top-k renormalization (178-184). Template
  specializations for (256,8), (128,8), (128,10), (512,8/10) — **256/8 is exactly
  Qwen3.5-MoE** (dispatch `moe.sycl:1262-1281`, `moe_int4.sycl:2696-2713`, 2847-2864,
  3592-3597).
- Generic fallback `moe_topk_forward_kernel_impl` (`moe_batch/moe_topk.h:12-102`): one WI
  per token, scores padded to `simd<fp16,512>`, 32-wide "heap" with `hmin`/
  `fbl(pack_mask(...))` scan (57-77).

**Device-side compaction exists in two places**, both feeding expert-major grouped GEMMs:

1. `moe_prefill_gather_forward_kernel<GS=1024>` (`moe_prefill_int4.sycl:40-106`): single
   WG, SLM counters+offsets, 4 phases — zero, histogram via SLM `atomic_ref` (63-70),
   serial prefix sum on thread 0 (74-80), scatter `expert_tokens[pos]=pair_idx` + inverse
   `pair_to_perm` via a second SLM atomic pass (94-102). Outputs `expert_offsets`,
   `expert_tokens`, `pair_to_perm`, `rows_for_experts` (wrapper 458-488). Notably
   **single work-group** — fine for prefill pair counts, no multi-WG histogram.
2. `moe_route_precompute_kernel` (`moe_int4.sycl:1475-1555`): a **4-kernel** pipeline —
   zero (1508), global-atomic histogram (1517-1528), single-task prefix (1530-1539),
   global-atomic fill (1541-1554). Registered as `moe_route_precompute_int4` (3635) and
   used by the **prototype** Python composition path
   `moe_forward_routed_cutlass_nmajor_int4` (`ops.py:1228-1232`), which feeds CUTLASS
   `cutlass_grouped_gemm_xe2` — not by the fused tiny/ws decode kernels. Python fallback
   does `torch.argsort` (`ops.py:1152-1156`) with comment "a production decode path
   should replace it with a fused C++/SYCL prologue" (1135-1136).

Pair-index encoding everywhere: `pair_idx = token*top_k + slot`; token recovered by
`pair_idx / top_k` (e.g. `int4_nmajor_gemm.h:82`, `moe_prefill_int4.sycl:164`, 810).

Host-side CPU sort helpers also exist: `build_sorted_token_ids` / `build_topk_ids`
(`moe_ops.h:238-271`) for the older ESIMD MoE ops.

## 3. Zero-point / scale handling

- **No zero-points, anywhere.** All int4 is symmetric: decode N-major uses
  two's-complement nibbles (`v>=8 ? v-16 : v`, `moe_int4.sycl:1689-1692`); prefill
  K-major marlin uses unsigned-minus-8 (`((w>>shift)&0xF) - 8`,
  `moe_prefill_int4.sycl:217-219`, down 356-357). File-level comment: "Dequant:
  float_val = (nibble_unsigned - 8) * scale" (`moe_int4.sycl:5`).
- Scales are **fp16, per (expert, output-channel, K-group), group_size=128 hardcoded**
  in every int4 kernel (`k_groups = K/128`; `k>>7` at `moe_int4.sycl:2353`; prefill
  `BS=128` instantiation at 531/595). The fp8 path uses one fp32 scale **per expert**
  (`moe.sycl:432`, 843).
- Dequant is always inlined before the MAC: `v * scale` per element in scalar kernels
  (`moe_int4.sycl:1745-1748`); `row *= s` on the fp16 a_tile before DPAS in ESIMD
  kernels (`moe_prefill_int4.sycl:221`, `int4_nmajor_gemm.h:140`).
- Accumulator precision varies: fp32 in the plain-SYCL decode kernels and fp8 DPAS path;
  **fp16 accumulators in the int4 DPAS prefill/nmajor kernels** (`simd<fp16, ACC_SZ>
  acc`, `int4_nmajor_gemm.h:84-85`, dpas fp16 C at 149).

Implication for our AWQ kernel: llm-scaler never had to solve asymmetric g32; the closest
structural analog to our zero-point-fold-into-B trick does not exist there.

## 4. Combine / scatter-back strategies (four distinct ones)

1. **Fused in down kernel (decode int4):** down GEMM loops routes per token, multiplies
   `route_weight * activation` before the MAC (`moe_int4.sycl:2351`, 2121/2146). Zero
   extra passes, zero atomics — possible only because the kernel is organized token-major
   (each WG owns one token's outputs).
2. **Separate accumulate kernel (prefill int4):** down kernel writes per-pair
   `expert_output` unweighted; `moe_prefill_accumulate_forward_kernel`
   (`moe_prefill_int4.sycl:398-421`) does one WI per token:
   `for s in top_k: acc += w * output[tok*top_k+s]` in fp32, 64-wide blocks. Variant
   `..._permuted_forward_kernel` (430-454) folds the inverse permutation `pair_to_perm`
   in, avoiding a `torch.index_select` (comment 423-428).
3. **Separate accumulate (fp8):** `moe_accumulate_kernel` (`moe.sycl:1163-1189`) sums
   `rows_per_token` rows per token (routing weight already applied in the down kernel,
   842-844).
4. **Pseudo-atomic RMW (standalone nmajor bench only):** `moe_down_int4_nmajor_kernel`
   does `lsc_gather` + add + `scatter` with comment "Atomic add for accumulate across
   experts" (`int4_nmajor_gemm.h:310-319`) — **not actually atomic**; safe there only
   because the bench is effectively single-expert-per-token. Do not copy this pattern.

CUTLASS composition path uses `moe_route_gather_int4_kernel` (`moe_int4.sycl:1629-1654`):
one WI per (token, hidden), linear scan over all routes checking
`sorted_rows[route]==token` — O(num_routes) per output element; fine as prototype,
clearly not production-grade. Production prefill gather is XPU-K
`torch.ops._moe_C.moe_gather` (patch:7050-7051).

## 5. Multi-expert batching / load balancing

- Decode int4: **none.** Route-per-token grids; every (token, route) reads its expert's
  full weight rows independently. "Load balance" is implicit: grids are sized
  `n_tokens*top_k*(I or H /tile)` so occupancy scales with total routes (e.g.
  `moe_int4.sycl:1712-1716`, 1946). The `up_wis < 2048` heuristic (2875-2876) switches to
  SLM-reduction variants purely to raise occupancy at tiny M, not to balance experts.
- Prefill: expert-major grouped GEMM with grid `range<2>(num_experts, N/N_TILE)`; each WI
  loops its expert's compacted rows in 32-row chunks (`int4_nmajor_gemm.h:75`,
  `moe_prefill_int4.sycl:156`). Empty experts exit immediately (`if (t0 == t1) return;`,
  69 / prefill 151). **Load imbalance across experts is unmitigated** — a hot expert with
  10× rows takes 10× longer on its WIs while idle-expert WIs retire; there is no
  splitting of a hot expert's M-range across WIs (no "stream-K"-style fixup).
- Shared experts are fused into the same kernels (extra rows beyond top_k:
  `shared_row = token*rows_per_token + top_k + sid`, `moe.sycl:1017`,
  `moe_int4.sycl:664`), with a precomputed sigmoid gate (`moe_down_gate_precompute_kernel`,
  `moe.sycl:965-996`; SLM-dedup variant for bs=1 at `moe_int4.sycl:1097-1174`).

## 6. ESIMD vs SYCL-DPAS tradeoffs (as evidenced in-tree)

- The repo **never uses `joint_matrix` / `SubgroupMatrixMultiplyAccumulateINTEL`** (our
  approach). Its only DPAS usage is ESIMD `xmx::dpas`: `dpas<8,8,fp16,...>` for int4
  prefill/nmajor (dequant-to-fp16 first), `dpas<8,1,float,...,fp16>` for fp8 decode.
  I.e. they always dequantize int4→fp16 in registers and use **fp16 DPAS**, never s8/u8
  DPAS — despite shipping packed int4 weights. Grep confirms no `dpas` in
  `moe_int4.sycl` at all; the decode int4 path is pure scalar/SIMD FMA.
- ESIMD is used where 2D block loads (`lsc_load_2d`, `config_2d_mem_access`), explicit
  cache hints, GRF-sized tiles, and SLM byte addressing matter (`moe.sycl:401-415`,
  `slm_init` 383); plain SYCL where simplicity/occupancy matter (`moe_int4.sycl` decode
  kernels, SLM via `local_accessor` + `item.barrier`).
- Perf guidance recorded in comments: `MAX_M = 16 // was 32 — smaller M tile, less GRF
  pressure` (`moe_int4.sycl:3256`); "No SLM, no barrier — eliminates sync overhead for
  small K" (`moe.sycl:828-829`); hybrid SLM dispatch when `up_wis < 2048`
  (`moe_int4.sycl:2876-2878`).
- **Production prefill doesn't use their own DPAS GEMMs at all**: `_esimd_prefill_moe_apply`
  (patch:6992-7052) composes ESIMD topk + XPU-K `remap_hidden_states` + **CUTLASS
  `cutlass_grouped_gemm_interface`** (int4 N-major) + ESIMD silu_mul + XPU-K `moe_gather`,
  with comments "ESIMD for topk and silu_mul (faster at all M)", "XPU-K for
  scatter/remap and gather (faster at all M, especially small M)", "TopK — ESIMD (2-3x
  faster than XPU-K at all M)", "Gather — XPU-K (10x faster than ESIMD at small M)"
  (patch:6995-6997, 7010, 7013, 7047). The all-ESIMD `moe_prefill_full_int4` pipeline
  (`moe_prefill_int4.sycl:920-988`) exists but the patch routes around it — an implicit
  verdict that CUTLASS Xe2 grouped GEMM beats their hand-written DPAS GEMM at prefill M.
- Layout evolution tells the same story: they migrated from IPEX K-major + marlin shuffle
  to CUTLASS N-major uint8 (comment `int4_nmajor_gemm.h:11-14`; bench
  `tests/bench_moe_int4_layouts.py` compares `ipex_k_major` vs `cutlass_auto` vs
  `cutlass_tiny_full` vs `cutlass_grouped_gemm_xe2`, main at 179-287). N-major removes
  the unshuffle and makes each output row's K-bytes contiguous — matching our layout
  choice.

## 7. Other relevant techniques

- **2D block loads**: `xesimd::lsc_load_2d<uint8_t,16,16,1>` with `config_2d_mem_access`
  for weight tiles (fp8 path, `moe.sycl:401-415`) — the ESIMD equivalent of Xe2 block-2D;
  the int4 DPAS kernels instead use plain `block_load<uint32_t,8>`
  (`moe_prefill_int4.sycl:203-206`).
- **Cache hints**: `cached/cached` on activation gathers and weight loads
  (`int4_nmajor_gemm.h:108`, `moe_prefill_int4.sycl:190`); `uncached/uncached` on the
  output RMW gather (`int4_nmajor_gemm.h:315`).
- **SLM usage**: only for reductions and tiny metadata — tree reduction over fp32
  partials (`moe_int4.sycl:1750-1760`, 2384-2404), ESIMD `slm_block_store`/`slm_init`
  K-split reduction (`moe.sycl:383-430`), SLM histogram/prefix in the compaction kernel
  (`moe_prefill_int4.sycl:48-102`), SLM gate-value dedup (`moe_int4.sycl:1122-1153`).
  **No SLM staging of weights or activations for GEMM tiling anywhere** — all GEMM
  dataflow is register/GRF-resident (ESIMD) or straight from L2 (scalar kernels). No
  software prefetch found (`prefetch` grep: no kernel hits).
- **Nibble handling**: two canonical patterns — scalar `decode_s4_nibble` (1689-1692) and
  vectorized widen-to-u16 + mask + `merge(x-16, x>=8)` (1976-1985, 2135-2139). X
  de-interleave via `select<VL,2>` to pair even/odd k with lo/hi nibbles (1963-1965,
  2127-2129).
- **Buffer caching**: `static thread_local torch::Tensor` intermediates reused across
  calls (`moe_int4.sycl:1376-1380`, `moe.sycl` `sg_*`/`s2_*` buffers) to avoid allocation
  in the hot decode path.
- **Kernel-count discipline for decode**: the whole decode layer is 2 launches (up+shared
  fused, down+shared+combine fused) after a 1-launch topk (`moe_int4.sycl:2615-2657`);
  bs=1 further fuses shared-expert down + accumulate into `moe_down_finalize_*`
  (2992-3021, comment "BSZ=1: fused down_finalize (lowest launch overhead)").

## Direct takeaways for our kernel (grounded in the above)

1. Our **expert-major compacted-pairs design matches their prefill/nmajor grouped GEMM**,
   which they consider the right structure at M beyond tiny — but note their expert-major
   grid gives one WI per (expert, N-tile) with serial M-loop and **no hot-expert
   splitting**; our device-side compaction plus SPIR-V DPAS path is strictly more
   sophisticated than anything in their int4 decode path.
2. Their decode path proves the opposite extreme: at M≤8ish, token-major
   weight-stationary kernels with **route-weight folded into activations** eliminate the
   combine pass entirely (§4.1) — worth considering for Arcaine's small-M regime (we
   currently run a separate combine kernel).
3. Nobody there does **asymmetric g32 or s8 DPAS on packed weights** — our int4-DPAS with
   zp-folded B operand is novel relative to this repo; their choice (dequant→fp16 DPAS)
   reflects fp16-DPAS being the path of least resistance in ESIMD, not a measured verdict
   against s8.
4. Practical details worth borrowing: `min()`-clamped tail handling instead of
   predication (`int4_nmajor_gemm.h:76-82`), pair_idx = token*top_k+slot encoding with
   `/top_k` recovery, SLM single-WG compaction (their 4-kernel global-atomic variant at
   `moe_int4.sycl:1475-1555` is the multi-WG alternative), and the `up_wis < 2048`
   occupancy heuristic for choosing reduction strategy.
5. Anti-patterns to avoid: the fake "atomic add" RMW combine
   (`int4_nmajor_gemm.h:310-319`) and the O(routes)-per-element gather
   (`moe_int4.sycl:1644-1649`) — both are bring-up code, not production.

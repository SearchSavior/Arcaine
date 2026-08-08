# Qwen3.5 Gated DeltaNet — full codepath, kernels, and Arcaine recommendations

Date: 2026-08-08
Scope: the Qwen3.5 dense (non-MoE) linear-attention path — **Gated DeltaNet** — as
implemented in the reference `vllm-xpu-kernels` repo, compared against the
current Arcaine `qwen3_5_moe` implementation, to drive performance work in
Arcaine. Read-only analysis; no build/run.

Report layout:
1. Repo/provenance map + dependency tree
2. Dispatch topology (which kernel runs when)
3. Kernel-by-kernel deep-dive with tile semantics
4. The DPAS atom (from sycl-tla) and GEMM tile machinery
5. oneDNN applicability
6. Arcaine current state (baseline)
7. Recommendations

---

## 1. Provenance and repo map

| Repo | Role |
| --- | --- |
| `reference/vllm-xpu-kernels` (HEAD `de8eae1`) | Source of the kernels reviewed |
| `reference/sycl-tla` | Intel CUTLASS/cute for Xe — powers the DPAS MMA atom + block-2D copies |
| `reference/oneDNN` | oneDNN 3.13 in container (`/opt/onednn/libdnnl.so.3.13`); gemmstone ngen backend |
| `src/modeling/qwen3_5_moe/` (Arcaine) | Target: `gated_delta_chunk{,_hybrid,_xmx}.hpp` |

The GDN (Gated DeltaNet) suite lives entirely under
`reference/vllm-xpu-kernels/csrc/xpu/gdn_attn/`:

```
gdn_attn/
  gdn_attn_utils.h                  constants (chunk=64, eps=1e-6, ActMode)
  causal_conv1d.hpp                 (1146 l) reference SIMT conv1d (decode/spec)
  gated_delta_rule.hpp              (833 l) reference SIMT linear-attn (decode/spec)
  gdn_attn_interface.cpp            (758 l) torch dispatch + host checks
  xe_2/
    gemm.hpp                        (490 l) cute/DPAS tiled-GEMM helpers
    chunk_gated_delta_rule_xe2.cpp/.h  host entry
    chunk_gated_delta_rule_kernels_xe2.hpp  (1634 l) the 5 chunk kernels (prefill)
    chunk_causal_conv1d_xe2.hpp     (916 l) chunked conv + zba-reorder + state update
    chunk_causal_conv1d_tiled_xe2.hpp (877 l) tiled (high-seq) conv
    l2norm.cpp/.h/l2norm_kernel.hpp (198 l) q/k L2 norm
```

### Dependency tree

```
torch.ops._xpu_C.causal_conv1d  ──┬── (spec)   gdn::causal_conv1d (reference SIMT)
   & .gated_delta_rule           ──┴── (decode) gdn::gated_delta_rule (reference)
                                  └── (prefill) chunk_causal_conv1d_{tiled,xe2} → chunk_gated_delta_rule_xe2
                                                       └─ l2norm (if not fused)

chunk_gated_delta_rule_xe2 (host, chunk_gated_delta_rule_xe2.cpp:38)
  └── gdn::chunk_gated_delta_rule_impl_xe2 (kernels_xe2.hpp:1506)
        ├── ChunkPrepareKernel   → chunk_prepare_kernel   (gate + inclusive scan)
        ├── ChunkComputeAKernel  → chunk_compute_A_kernel (K·Kᵀ decayed Gram)
        ├── ChunkInverseOptKernel→ chunk_inverse_opt_kernel (triangular solve; BMG)
        │   └─(fallback) ChunkInverseKernel → chunk_inverse_kernel
        ├── ChunkComputeWUKernel → chunk_compute_wu_kernel (u=A·Vᵀβ, w=A·Kᵀg)
        └── ChunkFwdOKernel      → chunk_fwd_o_kernel     (state recurrence + output)
              └─ all GEMMs call gemm.hpp: gemm_TTS / _STS / _TSS / _k_multi / _fused_2A
                    └─ cute::gemm(mma) where mma = MMA_Atom<XE_DPAS_TT<8,float,bf16>>
                          └─ (sycl-tla) dpas.bf.bf.8.M inline asm  → Xe2 DPAS/XMX
```

---

## 2. Dispatch topology (from `gdn_attn_interface.cpp`)

The interface cleanly separates three execution regimes — this determines *which*
kernels dominate:

- **Spec-decode** (`num_spec_decodes>0`): reference kernels only
  (`gdn::causal_conv1d`, `gdn::gated_delta_rule`). `:269-316`, `:569-589`.
- **Non-spec decode** (`num_prefills==0 & decodes>0`): reference kernels too
  (`:616-636` via `NATIVE_CONV_LAUNCHER` `:466`, `gated_delta_rule_xe2` not called).
- **Prefill** (`num_prefills>0`, Xe2 build): chunked path
  (`chunk_gated_delta_rule_xe2` `:599`, conv `chunk_*_xe2` `:367-471`).

So the **XD2 chunked DPAS suite is the prefill workhorse**; decode uses the
SIMT reference kernels (which are serial per token — matches Arcaine's finding
that decode has no DPAS relevance). All tensor shapes are cross-checked
exhaustively at the interface (narrow, contiguity, dtype) — high confidence in
the contracts.

Key shape invariants (enforced): `num_v_heads % num_k_heads == 0`,
`head_v_dim` and `head_k_dim` divisible by the block; for the chunked path
1:1 or integral GQA. `chunk_size=64`.

---

## 3. Kernel-by-kernel deep-dive (Xe2 chunked prefill)

### 3.1 `chunk_prepare_kernel` (kernels_xe2.hpp:52-131) — gate + prefix scan
- Computes per-token gate `a_h = softplus(a + dt_bias)·(−exp(A_log))` then an
  **inclusive prefix sum** in place, reverse-stride across lanes
  (`:118-125`). Each sub-group accumulates `local_num = chunk/16` elements,
  then `inclusive_scan_over_group` stitches sub-groups. The `a` buffer is f32
  `[num_v_heads, total_seq]` (`seq` innermost). This matches downstream reads in
  `fwd_o` which index `a[...+e]` with `e`=chunk position.
- **OOB guard added** at `:80` (`v_head_id >= num_v_heads`) for ragged
  subgroup-count launch.
- This is the "g" (gate/decay cumsum) producer for all subsequent kernels.

### 3.2 `chunk_compute_A_kernel` (kernels_xe2.hpp:133-279) — the chunk Gram
- Computes `A = mask(K·Kᵀ · exp(g_m−g_n) · β_n)`:
  - `gemm_TTS(K, K, ...)` → the raw K·Kᵀ in registers (`:248`)
  - per-element scale by `exp(g[m]−g[n])·β[m]`... `:257-262` reads `β[m]` for row
    and `exp(g[m]−g[n])` for the decay
  - **diag forced to 1.0** (`:263-265`), **strict upper zeroed** (`m<n`, `:266`)
- Grid: WG per (chunk); loops `v_head_id` serially, staging the decay `g` to SLM
  per head (`:204-216`). One `(token, k_head)` WG... actually one WG per chunk
  (`item.get_group(1)=chunk_id`), `v_head` serial.
- Produce the **unit-lower-triangular** decayed Gram, stored as f32.

### 3.3 `chunk_inverse_kernel` / `chunk_inverse_opt_kernel` (kernels_xe2.hpp:281-389 / :391-689)
- `inverse`: classic in-place serial solve. Loads unit-lower `A`, sets diagonal
  to 1 (`:349-352`), then for column `n` computes `I[m,n] = −Σ_{k} A[m,k]·I[k,n]`
  for `m>n` (`:355-371`). O(C³) triadic with an SLM ping-pong (`A_ptr_load`/
  `A_ptr_save`), parallel over `(m,n)` lanes. Writes back unit-lower `(I−A)⁻¹`.
- `inverse_opt`: same math but with the `16x16x16` MMA policy (`chunk_gemm_policy_inverse`,
  `:39-42`) to reduce the serial cost; selected only on BMG (`:1351`). This is the
  biggest prefill-adjustable bottleneck (triangular solve is a sequential
  dependency per column).

### 3.4 `chunk_compute_wu_kernel` (kernels_xe2.hpp:691-882)
- `u[dv] = Σ A·Vᵀ·β`  →  `gemm_TTS_k_multi(A, Vᵀ, β_slm)` (`:834-835`)
- `w[dk] = Σ A·Kᵀ·g`   →  `gemm_TTS_k_multi(A, Kᵀ, g_slm)` (`:871-872`), only when
  the chunk has prior state (`:840`).
- Uses `gemm_TTS_k_multi` so the `β` and `g` per-K scaling folds into the A
  fragment in registers (the `K_multi` path, gemm.hpp:358-373).

### 3.5 `chunk_fwd_o_kernel` (kernels_xe2.hpp:884-1231) — the state recurrence
The heart. Per chunk, per `v_head`, per `dv`-tile:
- **O2 = Q·Kᵀ masked+decayed** (`:1068-1087`): QK MMA, then per-element
  `*exp(g[m]−g[n])`, zero `m>n`, → `make_identity_tensor` copy to A temp (O2).
- **Fused WS+QS** (`gemm_TTS_fused_2A`, `:1108-1127`): single S[dv] load reused
  for `W·S` (→U) and `Q·S` (→O1). This is the gemm.hpp dual-A optimization.
- **U_new = U_old − W·S** epilogue (`:1128-1144`): register-sub in-place.
- **O = O1·exp(g) + O2·U** (`:1163`): accumulates WS output + QK×U.
- **State update** `S = exp(g_last)·S + Σ (K·exp(g_last−g))ᵀ·v_new`
  (`:1183-1212`) via `gemm_TTS_k_multi(Uᵀ, Kᵀ, g_multi)`.
- SLM arena = `3*chunk_size` floats (`:916-921`) for `g`, `g_multi=exp(g_last−g)`,
  `g_exp=exp(g)`.
- Loops **serially over batch_id** inside the WG (`:949`), but **each batch's
  sequence is chunked and cross-chunk parallel via the dv-tiles** — the SSM
  recurrence forces intra-batch serial order.

Distinctly clever bits:
- **`gemm_TTS_k_multi`** folds the decay into the A fragment register-scale per
  K-tile (`gemm.hpp:366-373`) — one pass, no separate multiply kernel.
- **`gemm_TTS_fused_2A`** (`gemm.hpp:382-489`) computes `W·S` and `Q·S` from one
  S borrow: B loaded once, both DPAS.
- WAR barriers documented at every SLM refill (`:981-985`).
- `native::exp` vs `exp` inconsistency at `:980-992` vs `compute_A:262` (noted
  as a numerics risk in my earlier review).

---

## 4. The DPAS atom and GEMM tile machinery

### 4.1 The atomic MMA (`sycl-tla`, `include/cute/arch/mma_xe.hpp:44-115`)

`XE_DPAS_TT<M=8, TD=f, TA=bf, TB=bf>` → `dpas.bf.bf.8.M (M1,16)` inline asm:

- **Shape (M, N, K) = (8, 16, 16)** per instruction.
  - `K = 256 / max(bits(A),bits(B)) = 256/16 = 16` (`mma_xe.hpp:50`).
  - B is VNNI-transformed: `16 × 16` (K×N), 2 bf16 packed per dword
    (`mma_traits_xe.hpp`, BLayout `K/16 groups of 16×16`).
- Datatype combos enumerate `dpas_type` f/bf/hf/tf32 + s8/s4/u4 (`:135-169`);
  the Qwen path uses `f,bf,bf,f` (bf16×bf16→f32).
- Register footprints (`mma_xe.hpp:52-60`): A `(M*K+15)/16 = 8` dwords, B `K=16`
  (8 dwords VNNI), C/D `M=8` floats.

### 4.2 `TiledMMA` composition (`TiledMMAHelper`, `chunk_gemm_policy_*:19-42`)
- WG/block tiles: `64×64×32` for compute_A (`_2x1/_2x2/_4x2` variants),
  `16×16×16` for inverse, `64×64×32` for wu/fwd_o.
- `64×64×32` with an `8×16×16` atom = `8 × 4` = 32 subgroups (SG_M=8, SG_N=4)
  → 512 threads/WG (matches `MaxThreadsPerSM=512`, launcher `:1281-1282`).
- `grf_size<256>` property (`:1289`) required to hold the fragments.

### 4.3 Block-2D copies + prefetch (`gemm.hpp`)
- `copy_a/copy_b` = `get_block_2d_copy_A/B` → Xe2 LSC block-2D load/store to SLM.
- Software pipeline: prefetch distance 3 (`:106`), `barrier_arrive/wait`
  producer-consumer (`:120-135`), `reorder()` BLOCK→VNNI fragment shuffle before
  `cute::gemm`.
- `cute::gemm(mma, tCrA, tCrB, tCrC)` = the DPAS accumulate loop.

These are the exact primitives Arcaine could reuse — but note Arcaine currently
hand-rolls the SPIR-V DPAS intrinsic (`__spirv_SubgroupMatrixMultiplyAccumulateINTEL`),
bypassing cute/block-2D entirely (see section 6).

---

## 5. oneDNN applicability (`reference/oneDNN`, 3.13 in container)

From the earlier w4a8 deep-dive: the non-grouped dense matmul on this GPU runs
through `gemm_t` → gemmstone (ngen DPAS). Relevant to GDN:

- **Batched 3D matmul** (the Gram `K@K^T`, `Q@K^T` across `[B,64,128]`) — exactly
  what Arcaine's hybrid already does via one `dnnl::matmul` with batch dim
  (hybrid `gram_exec`, `:117-128`). oneDNN batched matmul dispatches to
  gemmstone with batch, reusing the same ngen engine.
- **BF16 accumulation** and grouped scales are available.
- **Not a fit**: the triangular solve (`chunk_inverse`), the serial SSM
  recurrence, and the per-element decay masks are not expressible as oneDNN
  primitives — they stay custom. oneDNN covers the big regular GEMMs
  (K·Kᵀ, Q·Kᵀ, A·Vᵀ, A·Kᵀ, W·S, Q·S, U·Kᵀ) but not the elementwise-gated
  triangular parts.
- The `gemmstone` tile engine (K·Kᵀ = 64³ batched over B) reaches ~110 TFLOPS
  dense int4/bf16 per the qwen35 roofline note; bf16 matmul ~at peak.

**Net**: oneDNN is already the right tool for the 4 big block GEMMs
(K·Kᵀ/Q·Kᵀ/Gram, A·Vᵀ/A·Kᵀ, W·S/Q·S/U·Kᵀ) — the vllm path instead JIT-specializes
them with cute/dpas for full fusion of the decay masks. Arcaine's hybrid already
split this correctly; the remaining gap is the state-pass GEMMs (see 7).

---

## 6. Arcaine current baseline (for comparison)

`src/modeling/qwen3_5_moe/kernels/gated_delta_chunk*.hpp`:
- **`gated_delta_chunk.hpp`** (scalar SIMT, 253 l): one 256-lane WG/head, loops
  chunks serially, all math in SLM fp32. No DPAS. This is the "device" fallback/
  reference and is compute-slow (the 2026-08-01 memnotes: prefill dominated by
  the host scalar path at ~14 ms/tok before the hybrid).
- **`gated_delta_chunk_xmx.hpp`** (367 l): DPAS via
  `__spirv_SubgroupMatrixMultiplyAccumulateINTEL` with `kQ8DpasFP16` — 8×16×16,
  dv-half per WG. Serial per head/chunk though (per the memnote: xmx serialized).
- **`gated_delta_chunk_hybrid.hpp`** (478 l, **current default**): splits the
  pipeline — oneDNN **batched** K·Kᵀ/Q·Kᵀ over all (head,chunk) pairs
  (`gram_exec`), a custom `gram_t` (decay mask + T-solve + writeback), and a
  persistent 2-WG/head `state_pass` running the 4 state GEMMs with the same DPAS
  fp16 idiom as xmx, with S resident in fp32 SLM.

So Arcaine already independently converged on the *same* split as vllm-xpu's
chunked path (Gram→solve→state), but with **hand-rolled SPIR-V DPAS** instead of
the cute/block-2D MMA infrastructure.

---

## 7. Recommendations to enhance Arcaine GDN prefill performance

Grounded in the vllm-xpu implementation (all file refs to `kernels_xe2.hpp`,
`gemm.hpp`, sycl-tla `mma_xe.hpp`).

### A. Close the prefill-parallelism gap in `state_pass` (biggest win)
The vllm `fwd_o` parallelizes across **dv-tiles within the per-(batch,chunk)
recurrence** and even enables cross-chunk state-pass overlap with careful SLM
layout. Arcaine `state_pass` already uses 2 WGs/head (dv-halves) — but the chunk
loop is still serial per head. **Action**: keep S fp32-resident in SLM across
all chunks (already done in hybrid), and verify the phase-to-phase barriers are
minimal. Cross-check `fwd_o`'s `head_v_dim / chunk_size` dv-tiling (`:1111`,
`:1183`) — Arcaine splits dv in half; vllm tiles dv at `chunk_size=64` — Arcaine
matches (128/2 = 64). Confirmed aligned.

### B. Adopt the fused dual-A GEMM for the WS+QS pair
vllm's `gemm_TTS_fused_2A` (`gemm.hpp:382-489`) shares **one** S[dv] load across
`W·S` (→U) and `Q·S` (→O1). Arcaine's hybrid computes `v_tmp` with `(sw*K)@S`
and core with `(Q*eg)@S` + `QK@v_new` as **separate** DPAS loops
(`hybrid state_pass :308-387`). **Action**: fuse `W·S` and `Q·S` into one pass
over a shared S fragment — halves S SLM bandwidth, the dominant cost in the
state recurrence. This is a direct port of vllm's fused_2A.

### C. Use `gemm_TTS_k_multi` for the decay-scaled GEMMs (delete manual epilogues)
vllm folds the `exp(g)`/`β` K-scaling into the A fragment *inside* the k-loop
(`gemm.hpp:358-373`) instead of a separate elementwise pass. Arcaine/hybrid
applies `sEg`, `sDec`, `sSW` as pre-scaled SLM constants then separate DPAS —
inefficient vs folding. **Action**: fold the per-k scaling into the A register
fragment as vllm's `_k_multi` variant does.

### D. Solve the triangular invert bottleneck
`fwd_o` is bottlenecked by the O(C)=64 serial solve. vllm added
`chunk_inverse_opt_kernel` (`:391`, MMA `16x16x16`) to cut it. Arcaine's
`gram_t` `:183-191` uses the same O(C²) row-parallel forward-substitution as
vllm's mine. Options (in increasing effort):
- D1: increase C from 64? No — 64 is forced by `d_k=d_v=128` and the DPAS K=16
  alignment. Keep C=64.
- D2: Precompute `(I−A)⁻¹` only ONCE and reuse the decayed Gram structure —
  not possible (A depends on β/decay per chunk), but note the *solve* is
  independent of the GEMM and could overlap with the `compute_wu`/`fwd_o` GEMMs
  on different SMs. **Action: split `gram_t` into two cooperative kernels**
  (masked-Gram-GEMM and triangular-solve) so they can run concurrently — this is
  the natural mapping onto the 2×G31 hardware.
- D3: Port vllm's `inverse_opt` (only compute chunk_size inverse, not the full
  tile) — the MMA version amortizes the triadic loop.

### E. Decode is a real target — the bottleneck is quantified, and there's a proven kernel for it
Measured (mbench, dense Qwen3.6-27B-AWQ-INT4, 1×B70, `-d 0,512,1024,2048,8192`):
```
tg128 @ d0    25.02 t/s
tg128 @ d512  21.40 t/s
tg128 @ d1024 18.70 t/s
tg128 @ d2048 14.93 t/s
tg128 @ d8192 6.76 t/s
```
Config that drives this: **GQA 6:1** (24 query heads, 4 KV heads, head_dim=256),
64 layers = **16 full-attn / 48 linear**. Decode per token:
- 48 linear layers → O(1) in KV length (fused-ESIMD `kernels.hpp:384-567`, state
  carry). Not the scaling problem.
- 16 full-attn layers → O(past) each, scanning KV[0..past].

**Root cause of the 3.7× collapse to d8192:** the production full-attn decode
dispatches to the **general** `qwen35_xmx_attention` (`operators.hpp:166`), NOT
the decode-optimized GQA kernel. With GQA 6:1, the general kernel launches one
WG per *(1 query × 256 output)* tile — so each of the 24 query heads re-reads the
*same* KV head's cache independently. That is **6× redundant KV traffic per
decode step** (24/4). At `past=8192`, head_dim 256, 16 layers, this is the
dominant decode cost and it scales linearly with `past` — exactly the measured
curve.

The dedicated kernel already exists and is unused in the model path:
`qwen35_xmx_attention_decode_gqa` (`kernels.hpp:916-1060`) — one WG per
*(KV head, partition)* collectively computing that KV head's 6 queries in the
8-row M-tile, **loading K/V once per KV head and reusing across the GQA group**
(`:913-915`). It is referenced only by the benchmark (`attention_bench.cpp:117`),
never by the model forward.

**Recommendation (high impact, low effort):** dispatch decode to
`qwen35_xmx_attention_decode_gqa` in `operators.hpp` for the full-attn layers
when `seq<=1` (or `seq` small), falling back to the general kernel at prefill M.
Expected: cuts the full-attn KV-scan traffic ~6×, which is the long-KV decode
bottleneck. This directly targets the 25→6.76 degradation. Gate behind an env
var (`QWEN35_DECODE_XMX_GQA=1`) per AGENTS.md and A/B at
`-d 512,1024,2048,8192`.

Secondary (same class): verify the general kernel at M=1 doesn't already take a
fast path — it does not (no `seq==1` branch, confirmed in `kernels.hpp:767-911`).
If the decode-GQA swap proves insufficient, the next lever is paged/tiled KV so
the 8192-key scan hits resident L2 (B70 16GB; KV for 4 heads×256 is 4 MB/key-row
block, bounded by batch), plus a split-decode that only rescans the delta.

### F. Consider adopting sycl-tla `MMA_Atom<XE_DPAS_TT>` + block-2D
Arcaine hand-rolls `__spirv_SubgroupMatrixMultiplyAccumulateINTEL`. vllm uses
the cute/`XE_DPAS_TT` machinery (`mma_xe.hpp:80-115`) which gives block-2D
prefetch, VNNI reorder, and fragment management for free. **Action (strategic)**:
if Arcaine is willing to depend on sycl-tla headers (already vendored as a
reference), porting the state-pass to cute would (a) unify with the reference,
(b) unlock LSC block-2D prefetching (prefetch_dist=3) for the S/K operands, (c)
reduce hand-rolting risk. If not, keep the SPIR-V DPAS but at least implement
feature B+C manually.

### G. Perf-gating (mandated by AGENTS.md)
Any of the above must be toggled by an env var for A/B. Suggest:
- `QWEN35_GDN_FUSE_WSQS=<0|1>` — the fused dual-A (feature B).
- `QWEN35_GDN_KFOLD=<0|1>` — register K-scaling (feature C).
- `QWEN35_GDN_INV_MODE=<solve|mm|split>` — D2/D3.
Benchmark with `arcaine_kbench qwen35-gdn --device 1 --md`, shapes
p 512,1024,2048,4096, and numeric check `core_cos`/`state_max_rel` (from the
2026-08-01 hybrid closeout).

---

## 8. Correctness items to verify (carried from review)

1. `native::exp` (fwd_o `:980-992`) vs `sycl::exp` (compute_A `:262`) asymmetry —
   unify or document tolerance; long-seq drift risk.
2. Divisibility guards absent for `head_v_dim % chunk_size` and
   `head_k_dim % chunk_size` in fwd_o loops (`:1111`,`:1183`) — Qwen3.5 (128) is
   0-safe, but a different head_dim silently breaks. Add dispatch-time checks.
3. `num_v_heads % num_k_heads` (kv_ratio) assumed integral with no guard.
4. `chunk_prepare_kernel`'s in-place prefix scan is layout-dependent
   (`a[head, seq]` seq-innermost) and is the most fragile piece — numeric parity
   check against the reference `gated_delta_rule.hpp` on a ragged batch.

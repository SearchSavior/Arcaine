# llm-scaler FP8 MoE/GEMM Kernel Study

Reference for Arcaine gemma4 MoE FP8 on BMG G31.
Source: `reference/llm-scaler/vllm/custom-esimd-kernels-vllm/` (note: `reference/` is .gitignored, so `glob`/`index` skip it — use explicit paths).

## TWO FP8 MoE GEMM paths exist (both built; both wired in setup.py)

- **Path A — FUSED full-MoE** `csrc/moe_batch/moe.sycl` (1819L). setup.py:154, setup_moe_only.py:27. Production fused: up+gate fused, SiLU fused, down with routing-weight-folded scale, separate finalize. DPAS<8,1>. THE MoE forward.
- **Path B — MODULAR grouped GEMM** `csrc/xpu/esimd_kernels/fp8_moe_gemm.h` (438L) via `esimd_kernel_moe.sycl:140` → `moe_gemm_fp8_e5m2_dispatch[_pert]`. setup.py:63. Standalone grouped GEMM primitive (Python `esimd_moe_gemm_fp8`). DPAS<8,8>. Surrounded by separate topk/scatter/silu/gather in moe_ops.h.

## Path A: fused moe.sycl

- `fp8e5m2_block_to_vnni` @100: dequant fp8e5m2→fp16 (`fp8e5m2_to_half`) then pack 8 k-pairs: `convert<u32>(even_u16) | (convert<u32>(odd_u16)<<16)`. E5M2 free-dequant.
- `moe_up_routed_e5m2_kernel` @546: grid `nd_range<2>({n_tokens*top_k, n_tiles*GS},{1,GS})`, GS=16 threads/WG. 1 WG/(route,N-tile). GS threads split K. **Loads gate+up weights BOTH using same a_tile** (fused gate/up, one input load). `dpas<8,1,float,float,fp16,fp16>` per 16×16 K-tile. SLM reduce (`slm_init<GS*32*4>`) → tid==0 applies per-expert scale `gate_up_scale[eid]` @614, then SiLU `gs/(1+exp(-gs))*us` @618-619, store intermediates[routed_row]. FUSED up+gate+scale+SiLU.
- `moe_down_routed_e5m2_kernel` @912: grid `range<2>(n_tokens*top_k, n_tiles)`, 1 WI/(route,N-tile), NO SLM. `dpas<8,1,...>`. **Folds routing weight + per-expert scale in ONE mul**: `acc *= w * ds` @953 (w=routing_weights[route], ds=down_scale[eid]). Writes per-route (not accumulated).
- `moe_down_finalize_e5m2_kernel` @1331: grid `range<2>(n_tokens, hidden_size)`, 1 WI/out-elem. Sums top_k routed outputs @1343, then per shared expert: sigmoid gate (recomputed), shared-down GEMV (**NO DPAS** — element-wise `fp8e5m2_to_half<64>*fp16` @1357, BW-bound), accumulate. = routed-reduce + shared-expert finalize.
- DPAS<8,1> = SystolicDepth 8, RepeatCount 1: A=16fp16, B=256fp16(16×16 VNNI), C=16f. One 16×16 out-tile/call.

## Path B: modular fp8_moe_gemm.h (MoE_FP8_E5M2_Kernel_V3)

- Layouts: W[E*N,K] flat u8 E5M2; scale [E,N]f32 per-N OR [E]f32 per-tensor; in[T,K]fp16; out[T,N]fp16; expert_idx[E+1]u32 offsets.
- **Hybrid grid**: WG=(expert, n_wg_id ∈ N-tile-group, m_chunk_id ∈ M-chunk). M-chunks across WGs → W read once/N-tile (re-reads hit L3, W/expert << 18 MB). N-tiles merged/WG (n_per_wg) cap WGs.
- MT_MAX (×8 rows) small (1/2) → avoid wasted loads low-token experts. K: 4 sub×16, LOAD=64. Transposed uint32 load → E5M2 pair→VNNI → `dpas<8,8,float,float,fp16,fp16>`/M-tile.
- Dispatch (moe_gemm_fp8_e5m2_dispatch): n_per_wg cap WGs ≤4096-8192 (per-N) / wg_cap 51200|38400 (per-tensor, K-aware: small K → higher, fits L1). MT_MAX mt<=1→1, <=2→2, else→1+m_chunks.

## MoE pipeline (moe_ops.h) around Path B

- **TopK V2** `MoE_TopK_V2_Kernel`: 1 WG/token. Softmax in f32 (avoid fp16 precision loss). **Bit-cast argmax** to bypass `-ffast-math` float==: `bit_cast_view<int32_t>` compare to max-bits @89-90, pick lowest idx via `h_min`. TOPK rounds, zero winner each round by integer-index mask. h_sum/h_max/h_min recursive-halve helpers. NUM_EXPERTS%64==0,<=512; TOPK<=16.
- **Scatter fused** 3-phase GPU (replaces CPU `build_sorted_token_ids`): (1) `Scatter_Init` atomic_add expert counts → per-slot offset (IPEX pattern) @453; (2) `Scatter_Prefix` 1-thread prefix-sum → expert_start[E+1]+max_tokens @461; (3) `Scatter_Copy` copy hidden(128-chunk)+weights, build topk_ids reverse map @490.
- **SiLU_Mul** @576: gate/up split, `silu=x/(1+exp(-x))*up`, 64-chunk. **GELU_Tanh_Mul** @683: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))` via `(exp(2z)-1)/(exp(2z)+1)`.
- **Gather** @627: weighted reduce `final[t]=Σk moe_out[topk_ids[t,k]]*w[topk_ids]`, 128-chunk, scalar topk loop, fp32 acc.
- moe_topk.h: alt TopK using heap (hmax/hmin/pack_mask/fbl) — shared by FP8+INT4 moe.sycl.

## Dense FP8 GEMM (fp8_GEMM_pert.h, 4075L; live entry esimd_kernel_gemm.sycl:38 → GEMM_fp8_pert_dispatch @4030)

Iteration V2..V13; only V7, V9, WS, mpar, batched_gemv LIVE (rest under `#ifdef GEMM_EXPERIMENTAL` = dead).
- M==1 → batched_gemv (K-split SLM, BW). **N<=16 && M>=2 → mpar** (1 WG/row, K_SPLIT threads). Fixes **2x cliff @ M=9** V7/WS N-parallel underutil tiny N. M>64 → WS (V7/V9 cap M_TILES=8=64; beyond silently uninit → NaN). K%64==0 && E4M3 → **V9**. K%64==0 && N>=1024 → **V7**. M<=3 → gemv; M<=8 → ws<128,8>; else → ws<128,16>.
- **V9** flagship E4M3 DPAS: 1 WG/16-N tile, K_THREADS split K + SLM reduce. Input merged 2D `lsc_load_2d<fp16,16,H,1>` H=8/16/32 for M_TILES=1/2/4+. Transposed `lsc_load_2d<uint32_t,4,16,1,TRANSPOSE=true>` → u32=4 adj FP8 bytes/N-row → fused `fp8_e4m3_pair_to_vnni` (dequant BOTH bytes in-place via bit_cast_view compound assign, result IS VNNI) → b_tile 256fp16 → DPAS/M-tile. `lsc_prefetch_2d` weight prefetch (M_TILES<=2). K_THREADS=`max(1,min(4,640/n_wgs))`, div K*64, 3→2. M_TILES=ceil(M/8) cap 8.
- **WS** fallback (BW, NO DPAS): grid{N,m_tiles}, 1D K-loop block_load u8 W→dequant→float FMA w/ fp16 in, scale@end reduce. Tail: overlap-read last VL, zero overlap prefix.
- Perf gotcha: oneDNN JIT fuses dequant+VNNI via register-regioning ESIMD can't emit → 64 narrow mov/sub-tile in V7 (55% body). V9 transposed-u32-load sidesteps.

## FP8 dequant (used everywhere)

- **E5M2 FREE**: bias=15 == FP16 bias, 2 mant → top 2 of FP16's 10. Place E5M2 byte in high 8 bits of FP16 → exact. `fp8_e5m2_pair_to_vnni`: `(b_lo|(b_hi<<16))<<8`. Path A+B both.
- **E4M3 needs +8 bias**. oneDNN branchless: `shl(8)→asr(1)→and(0xBFFF)→mul(256.0)` (4 ops). Subnormal flush `.merge(sign,exp==0)`. E4M3 branchless small subnormal err (ok for filtered); NaN 0x7F → ~240.
- `fp8_mode`: 0=E4M3, 1=E5M2 runtime-select (from `weight.scalar_type()==Float8_e5m2`). Keep dequant GRF bounded: split into 256-wide halves (page.attn).

## Supporting kernels (2nd study)

- **int4_nmajor_gemm.h**: N-major grouped INT4 MoE. `dpas<8,8,fp16,fp16,fp16,fp16>` **fp16 accumulator** (not fp32). Nibble dequant branch-free two's-comp `n>=8?n-16:n`. **Per-group fp16 scale folded INTO A-tile before DPAS** @140 (vs FP8 per-expert scale after reduce). Down-proj cross-expert accumulate via `lsc_gather<u16>`+scatter atomic RMW @311. No SLM/prefetch/2D — gather-based.
- **page.attn.fp8.h** FP8 KV paged attn: **BMG RepeatCount=4+fp16 dpas is BROKEN (NaN) → use RepeatCount=8 + zero-pad last 4 A-rows** @294 (CRITICAL BMG gotcha). k_scale folded into `matMulQuantCoeff=0.0625*k_scale` applied in softmax pre-exp (zero hot-loop cost) @80,343; v_scale folded into final softmax divide @640. VNNI pack `lo|hi<<16` via strided `select<16,4>` DW-writes. Atomics for global max (inc arrival poll, fcmpxchg, fmax).
- **fp8_GEMV_v2.h** decode M=1: NO DPAS, fp32 element-wise MAC, **deferred single scale mul** after reduce (97% BW). K_SPLIT SLM partial reduce. `select_vl_ks`: default vl=512,ks=1; K<512→vl=128; small N+large K→ks=8/4; shrink vl until divides K/ks. Weakness: K-unfriendly (K=1056) → vl=32 slow. Host redirects K-unfriendly to **fp8_GEMV_bmg** (VL_BIG + power-of-2 VL_TAIL block_load, target 640 threads). `DISABLE_BMG_GEMV=1` env var for AB-test (AGENTS.md convention).
- **scaled_resadd_norm_gemv_fp8.h** gemma4 xfuse+qkv entry: fused resadd+RMSNorm+FP8 GEMV. **Two SLM regions** sharing one slm_init: [0..K_SPLIT-1] sum_sq→inv_rms broadcast, [K_SPLIT..2K_SPLIT-1] GEMV partials→reduce×scale. K_SPLIT==1 variant stages `res_chunks[MAX_CHUNKS]` in GRF so RMSNorm reuses w/o reload. Dispatch N>=2048 & K%(256*8)==0 → ks=8 etc.

## Cross-cutting BMG G31 takeaways (reuse in Arcaine)

- Use `esimd::xmx::dpas` NOT joint_matrix. Forms: `dpas<8,1,...>` (Path A fused, 16×16 tile) or `dpas<8,8,...>` (modular, M-tiling). fp32 acc for FP8, fp16 acc ok for INT4.
- **BMG RepeatCount=4+fp16 NaN → always RC=8 + zero-pad A** (page.attn). Reusable in any fp16-operand DPAS.
- Fold quant scales for free: per-expert/per-tensor FP8 scale → single mul after K-reduce (Path A @614/953 folds routing weight too in ONE mul; v2 deferred scale); per-group INT4 → multiply into A-tile before DPAS (int4); attention k_scale→softmax coeff, v_scale→final divide.
- Occupancy lever = K_SPLIT + SLM partial reduce, target ~640 threads. K-unfriendly K → VL_BIG+power-of-2 VL_TAIL block_load not shrink to vl=32.
- Fuse aggressively: gate+up (shared a_tile load), SiLU after scale, resadd+norm+gemv, down+routing-weight. Each fusion removes a kernel launch + hidden reload.
- Cap DPAS M-tiles 8 (64 rows) + bounds-check/fallback beyond (else NaN propagate). Merge N-tiles/WG cap count, K-aware. Small MT_MAX MoE (low-token experts). VNNI pack strided `select<16,2>`(lo→even,hi→odd) beats convert+shift+or.
- Bit-cast argmax for -ffast-math safety (TopK). f32 softmax/norm to avoid fp16 precision loss.
- AB-test gate new kernel paths via env var (DISABLE_BMG_GEMV pattern per AGENTS.md).

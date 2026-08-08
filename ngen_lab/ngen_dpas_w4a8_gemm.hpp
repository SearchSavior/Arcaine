// nGEN kernel-body author for the full-GEMM W4A8 matmul (s8 activations x s4
// weights -> fp32 C with per-K-tile fp rescale), built on the verified
// s8x4 16x16x32 tile (ngen_dpas_8x8_matmul_s4_16x16x32).
//
//   C[M,N] (f32) = sum_t act_s[m,t] * w_s[t,n] * (Aq[M,K] s8 @ W[N,K] s4 tile t)
//
// where t indexes 32-deep K tiles (dpas sdepth 8), act_s is the per-(row,
// K-group) s8 activation scale EXPANDED per 32-tile on the host (act gran 128
// upstream -> as32[m,t] = act_s[m, t/4]), and w_s is the per-(32-group, column)
// weight scale (f32 here; bf16 in production, converted at repack time).
//
// Thread/grid model (follows the lab tile kernels): SIMD16, one work-group of
// 16 lanes per thread; lane m = C row m0+m. Each thread owns kTiles 16x16 C
// tiles along N (contiguous n-tiles). Grid = {N/16/kTiles, M/16, 1}.
//
// The SHAPE IS BAKED INTO THE CODE (kN/kK/kKt below; production has a fixed
// set of 5 shapes and specializes at init): all address strides are
// immediates, the K loop is unrolled by kUnroll tiles with immediate load
// offsets, and pointer bumps amortize over the unroll.
//
// Layouts:
//   A  (a):   s8 pre-swizzled by the producer (quant pre-pass in production,
//             packA in the harness) into DPAS src1 block order:
//             [m_tile][k_tile][j][lane m] dwords, where dword (j,m) =
//             {A[m_tile*16+m][k_tile*32+4j..+3]}. Each (m_tile,k_tile) block
//             is 8 GRFs = 512B contiguous -> 4 fully-coalesced 128B block
//             loads (scattered A loads cost ~5x dpas issue time; see memory
//             ngen_lab_w4a8).
//   B  (b):   s4 DPAS src2 stream [n_tile][k_tile][h][r][j] dwords, where
//             dword (h,r,j) = 8 K-nibbles {W[n][k0+8j+e], e=0..7} for column
//             n = n_tile*16 + 8h + r. Per (n,k_tile) that is 16 CONSECUTIVE
//             bytes of the (N,K/2) nibble-packed weight (2 s4/byte along K).
//   ws (ws):  f32 (K/32, N) row-major; tile t row = 16 consecutive f32 at
//             n0 — one block(D32,16) load.
//   as (as):  f32 (M, K/32) row-major (host-expanded per-32-tile activation
//             scales); per-lane scattered D32 load, advanced +4B per tile.
//             NOTE: the kernel preloads one unroll-group ahead, so the buffer
//             must have 16*kUnroll extra pad floats after the last row
//             (in-bounds, values unused).
//   C  (c):   f32 (M,N) row-major, scattered D32 stores.
//
// Per-K-tile epilogue (probe-mandated per-group rescale; see Arcaine memory
// qwen35_w4a8_probe). Issue-count optimizations (measured: the naive
// 5-instr/GRF epilogue dominates the whole kernel):
//   - dpas src0 is a permanently-zero GRF block (kZ), so the s32 tile accs
//     never need init or re-zeroing;
//   - cvt / as-mul / C-add run at SIMD32 over column pairs (adjacent C GRFs
//     share the per-lane act_s vector, which is duplicated into 2 GRFs);
//   - the per-column w_s multiply must stay SIMD16 (scalar per column).
//   No ternary mad: src0/src1 are forced scalar on Xe2 (docs/findings.md 2.4).

#pragma once

#include "ngen.hpp"

namespace ngen_lab {
namespace w4a8gemm {

// Shape baked into codegen (set before kernel generation, lab idiom).
static int kN = 0, kK = 0, kKt = 0;   // N, K, K/32
// Tiles per thread along N, and K-loop unroll in 32-deep tiles.
static int kTiles = 1, kUnroll = 4;
static int kPrefetch = 0;
// P2 kernel variant (kTiles==1, kMode==0, kUnroll%4==0 only): the s32 tile
// accs are double-buffered so the epilogue of tile u-1 TRAILS dpas(u) --
// breaking the in-order issue serialization that drains the DPAS pipe --
// and the act_s multiply is hoisted to the 128-group boundary (as32 is
// constant across 4 consecutive tiles) via a fp32 Ctmp accumulator, cutting
// the per-tile epilogue from 5 to 4 fp lane-ops per column pair.
static bool kP2 = false;

// Codegen experiment modes (perf ceiling analysis; results are numerically
// WRONG in modes != 0, harness must skip the check):
//   0 = full kernel (per-tile rescale)
//   1 = no per-tile epilogue: s32 accumulates the whole K, no ws/as loads,
//       single (wrong) rescale after the loop
//   2 = mode 1 + no in-loop loads at all (pure dpas issue rate)
//   3 = loads only (A/B/ws/as), no dpas, no epilogue (load-path ceiling)
//   4 = P2-only perf ablation: full P2 body but the epilogue reads the
//       permanent zero block instead of the dpas output (no dpas->cvt RAW;
//       numerically wrong, harness skips the check)
//   5 = P2 with production-style mad epilogue (gemmstone outerProductRepackC):
//       cvt -> mul ws -> mad(C, C, acc, as) fuses the as-multiply and the
//       C-accumulate; no Ctmp, no group flush. Numerically exact.
static int kMode = 0;
// Cache hint applied to A/B stream loads (codegen experiment).
static ngen::CacheSettingsLSC kCacheAB = ngen::CacheSettingsLSC::Default;
// Workgroup cooperation: WGY threads along M share the B/ws streams (same
// n-tiles), WGZ threads along N share the A stream, exploiting L1 sharing
// between co-scheduled threads on the same Xe-core. Optional per-K-loop
// barrier keeps co-scheduled threads in lockstep for L1 temporal reuse.
static int kWgY = 1, kWgZ = 1;
static bool kBarrier = false;
// SWZ experiment: launch order transposed (kernel reads group IDs swapped).
static bool kSwz = false;

// ---- register map (parametric in T = kTiles) -------------------------------
constexpr int kA = 16;                 // A: 8 GRFs [16..23]
inline int bB() { return 24; }         // B: 4T GRFs (2 per N-half per tile)
inline int bWS() { return 24 + 4 * kTiles; }   // ws: T GRFs (16 f32 each)
inline int bASv() { return 24 + 5 * kTiles; }  // act_s lane vec, duplicated: 2
inline int bS() { return 24 + 5 * kTiles + 2; }      // s32 acc: 16T GRFs
inline int bC() { return 24 + 5 * kTiles + 2 + 16 * kTiles; }  // f32 C: 16T
// T=2: A 16-23, B 24-31, WS 32-33, AS 34-35, S 36-67, C 68-99.
// P2 map (T=1 only): S0 36-51, S1 52-67, Ctmp 68-83, C 84-99.
constexpr int kP2S0 = 36, kP2S1 = 52, kP2Ctmp = 68, kP2C = 84;
constexpr int kZ = 100;                // 8 zeroed s32 GRFs (dpas src0)
// fixed address/temp region: 108..127 (exactly fills the 128-GRF budget).
constexpr int kBaseAS = 108, kBaseC = 109;
constexpr int kAddrB = 110;            // +t, T scalar uq bases
constexpr int kAddrWS = 112;           // +t, T scalar uq bases
constexpr int kIt = 114;
constexpr int kStep = 115;             // uq, kUnroll*512 (A bump per iter)
constexpr int kAddrA = 116;
constexpr int kIdx = 117, kRows = 118, kTmp = 119;
constexpr int kPtrAS = 120;            // 2 GRFs
constexpr int kPtrC = 122;             // 2 GRFs

template <typename Generator>
void gemmKernelBody(Generator &g, ngen::Subregister argA,
        ngen::Subregister argB, ngen::Subregister argWS,
        ngen::Subregister argAS, ngen::Subregister argC,
        ngen::Subregister gidX, ngen::Subregister gidY,
        ngen::GRF lidY, ngen::GRF lidZ) {
    using namespace ngen;
    GRF rBaseAS(kBaseAS), rBaseC(kBaseC);
    GRF rIt(kIt), rStep(kStep), rAddrA(kAddrA);
    GRF rIdx(kIdx), rRows(kRows), rTmp(kTmp);
    GRF rPtrAS(kPtrAS), rPtrC(kPtrC);

    // ---- setup -------------------------------------------------------------
    g.mov(1, rBaseAS.uq(0), argAS);
    g.mov(1, rBaseC.uq(0), argC);
    g.mov(1, rIt.ud(0), uint32_t(kKt / kUnroll));
    g.mov(1, rStep.uq(0), uint64_t(kUnroll * 512));

    // Lane row indices: rows = m_tile*16 + (0..15), m_tile = gidY*kWgY +
    // lidY. m_tile is stashed in rIdx (free once the lane vector has been
    // expanded into rRows).
    g.mov(8, rIdx.uw(0)(1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
    g.mov(8, rIdx.uw(8)(1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
    g.mov(16, rRows.uw(0)(2), rIdx.uw(0)(1));
    g.mov(16, rRows.uw(1)(2), uint16_t(0));
    if (kWgY > 1) {
        g.mul(1, rIdx.ud(0), gidY, uint32_t(kWgY));
        g.add(1, rIdx.ud(0), rIdx.ud(0), lidY.uw(0));
    } else {
        g.mov(1, rIdx.ud(0), gidY);
    }
    g.shl(1, rTmp.ud(0), rIdx.ud(0), 4);
    g.add(16, rRows.ud(0)(1), rRows.ud(0)(1), rTmp.ud(0)(0));

    // A scalar base (swizzled layout): a + gidY * (K*16) bytes; per m-tile
    // the swizzled stream is Kt blocks of 512B. gidY*(K*16) fits u32
    // (255 * 17408*16 = 71M). NOTE: mul immediates are effectively 16-bit on
    // Xe2 (K*16 = 2^16 at K=4096 silently broke -> device lost), so multiply
    // by K and shift.
    g.mov(1, rAddrA.uq(0), argA);
    g.mul(1, rTmp.ud(0), rIdx.ud(0), uint32_t(kK));
    g.shl(1, rTmp.ud(0), rTmp.ud(0), 4);
    g.mov(1, rTmp.ud(1), 0u);
    g.add(1, rAddrA.uq(0), rAddrA.uq(0), rTmp.uq(0));

    // AS lane addresses: as + rows*(Kt*4).
    g.mul(16, rTmp.ud(0)(1), rRows.ud(0)(1), uint32_t(kKt * 4));
    g.mov(16, rPtrAS.ud(0)(2), rTmp.ud(0)(1));
    g.mov(16, rPtrAS.ud(1)(2), 0u);
    g.add(8, rPtrAS.uq(0)(1), rPtrAS.uq(0)(1), rBaseAS.uq(0)(0));
    g.add(8, GRF(kPtrAS + 1).uq(0)(1), GRF(kPtrAS + 1).uq(0)(1),
            rBaseAS.uq(0)(0));

    // ngrp = gidX*kWgZ + lidZ, stashed in rBaseAS (free once the AS lane
    // pointers are computed).
    GRF rNgrp(kBaseAS);
    if (kWgZ > 1) {
        g.mul(1, rNgrp.ud(0), gidX, uint32_t(kWgZ));
        g.add(1, rNgrp.ud(0), rNgrp.ud(0), lidZ.uw(0));
    } else {
        g.mov(1, rNgrp.ud(0), gidX);
    }

    // C lane addresses: c + rows*(N*4) + n_tile0*T*64.
    g.mul(16, rTmp.ud(0)(1), rRows.ud(0)(1), uint32_t(kN * 4));
    g.mov(16, rPtrC.ud(0)(2), rTmp.ud(0)(1));
    g.mov(16, rPtrC.ud(1)(2), 0u);
    g.mul(1, rTmp.ud(0), rNgrp.ud(0), uint32_t(kTiles * 64));
    g.mov(1, rTmp.ud(1), 0u);
    g.add(16, rPtrC.ud(0)(2), rPtrC.ud(0)(2), rTmp.ud(0)(0));
    g.add(8, rPtrC.uq(0)(1), rPtrC.uq(0)(1), rBaseC.uq(0)(0));
    g.add(8, GRF(kPtrC + 1).uq(0)(1), GRF(kPtrC + 1).uq(0)(1), rBaseC.uq(0)(0));

    // Per-tile scalar bases: B at b + (gidX*T+t)*(K*8), ws at ws +
    // (gidX*T+t)*64 (first row; +N*4 per K tile via immediate offsets).
    for (int t = 0; t < kTiles; t++) {
        GRF rAddrB(kAddrB + t), rAddrWS(kAddrWS + t);
        g.mov(1, rAddrB.uq(0), argB);
        g.mov(1, rAddrWS.uq(0), argWS);
        g.mul(1, rTmp.ud(0), rNgrp.ud(0), uint32_t(kTiles));
        if (t > 0) g.add(1, rTmp.ud(0), rTmp.ud(0), uint32_t(t));
        g.mul(1, rTmp.ud(0), rTmp.ud(0), uint32_t(kK));
        g.shl(1, rTmp.ud(0), rTmp.ud(0), 3);  // x K*8 bytes per n-tile of B
        g.mov(1, rTmp.ud(1), 0u);
        g.add(1, rAddrB.uq(0), rAddrB.uq(0), rTmp.uq(0));
        // ws base: (ngrp*T + t) * 64
        g.mul(1, rTmp.ud(0), rNgrp.ud(0), uint32_t(kTiles * 64));
        if (t > 0) g.add(1, rTmp.ud(0), rTmp.ud(0), uint32_t(64 * t));
        g.mov(1, rTmp.ud(1), 0u);
        g.add(1, rAddrWS.uq(0), rAddrWS.uq(0), rTmp.uq(0));
    }

    // Zero the fp32 C accumulators (and P2's Ctmp) plus the shared
    // dpas-src0 zero block.
    const int cBase = kP2 ? kP2C : bC();
    for (int i = 0; i < 16 * kTiles; i += 2)
        g.mov(32, GRF(cBase + i).f(), 0.0f);
    if (kP2)
        for (int i = 0; i < 16; i += 2)
            g.mov(32, GRF(kP2Ctmp + i).f(), 0.0f);
    for (int i = 0; i < 8; i += 2)
        g.mov(32, GRF(kZ + i).d(), 0);

    // ws ping-pong buffer: tile v's w_scale row lives in buffer v%2, so the
    // load for tile u+1 (emitted right after dpas(u)) does not collide with
    // the epilogue of tile u. Buffer 1 uses the free GRFs 124..124+T-1.
    // Even kUnroll only: group parity must not flip between groups.
    constexpr int kWS2 = 124;
    auto wsBuf = [&](int v, int t) {
        return GRF(((v & 1) ? kWS2 : bWS()) + t);
    };
    // Loads for K-tile v (v == kUnroll targets the next unroll-group's tile 0
    // via pre-bump base pointers + one-group offsets; the trailing over-read
    // after the last group is never consumed and the harness pads the
    // allocations).
    auto emitLoads = [&](int v) {
        const int aOff = v * 512, bOff = v * 256, wsOff = v * kN * 4;
        // Buffer of the consuming epilogue: local tile v%kUnroll, parity
        // (v%kUnroll)%2. v == kUnroll is next group's tile 0 -> buffer 0.
        // REQUIRES even kUnroll (else tile kUnroll-1 also uses buffer 0).
        const int wb = (v == kUnroll) ? 0 : (v & 1);
        // A tile: 4 contiguous 128B block loads (pre-swizzled src1 order).
        for (int j = 0; j < 4; j++)
            g.load(1, GRF(kA + 2 * j),
                    block(DataSizeLSC::D32, 32) | kCacheAB, g.A64,
                    rAddrA + aOff + 128 * j);
        for (int t = 0; t < kTiles; t++) {
            GRF rAddrB(kAddrB + t), rAddrWS(kAddrWS + t);
            // B: 2 N-halves x 128B (s4).
            for (int h = 0; h < 2; h++)
                g.load(1, GRF(bB() + 4 * t + 2 * h),
                        block(DataSizeLSC::D32, 32) | kCacheAB, g.A64,
                        rAddrB + bOff + h * 128);
            // w_scale row for this K tile: 16 f32. P2 splits ws loads out
            // of emitLoads (they must issue AFTER the trailing epilogue's
            // reads of the same ping-pong buffer).
            if ((kMode == 0 || kMode == 3) && !kP2)
                g.load(1, GRF((wb ? kWS2 : bWS()) + t),
                        block(DataSizeLSC::D32, 16), g.A64, rAddrWS + wsOff);
        }
    };

    // Prefetch tile u of the NEXT unroll group into L1 (load with null dst;
    // oneDNN's prefetch idiom). Fire-and-forget: no scoreboard, so these add
    // memory-level parallelism without register cost. Over-prefetches past
    // the end on the final groups; harness pads the allocations.
    auto emitPrefetch = [&](int u) {
        const int aOff = (kUnroll + u) * 512, bOff = (kUnroll + u) * 256;
        if (kPrefetch & 1)
            for (int j = 0; j < 4; j++)
                g.load(1, g.null, block(DataSizeLSC::D32, 32) | kCacheAB,
                        g.A64, rAddrA + aOff + 128 * j);
        if (kPrefetch & 2)
            for (int t = 0; t < kTiles; t++) {
                GRF rAddrB(kAddrB + t);
                for (int h = 0; h < 2; h++)
                    g.load(1, g.null, block(DataSizeLSC::D32, 32) | kCacheAB,
                            g.A64, rAddrB + bOff + h * 128);
            }
        // NB: no ws prefetch — 64B null-dst block prefetch hangs (Xe2).
    };

    // Preload activation-scale lane vector for K tile 0 (duplicated into 2
    // GRFs so the SIMD32 epilogue can use it as a full-width operand).
    if (kMode == 0 || kMode == 3 || kMode == 4 || kMode == 5) {
        g.load(16, GRF(bASv()), scattered(DataSizeLSC::D32, 1), g.A64, rPtrAS);
        g.mov(16, GRF(bASv() + 1).f(), GRF(bASv()).f());
    }

    // P2 per-tile epilogue for tile v: Ctmp += w_s * cvt(S[v&1]); at the
    // 128-group boundary (v%4==3) C += act_s*Ctmp, Ctmp = 0, and the next
    // group's act_s lane vector is preloaded (expanded layout, so any tile
    // of the group reads the same value; the final group's preload reads
    // the padded tail, unused).
    auto emitEpilogueP2 = [&](int v) {
        // Mode 4: source the epilogue from Ctmp (fp-written, never read by
        // dpas) to remove the dpas->cvt RAW while keeping the instruction
        // stream identical. NB: kZ is only 8 GRFs -- using it here clobbered
        // rIt/address registers above it and hung the loop.
        const int sBase = kMode == 4 ? kP2Ctmp : ((v & 1) ? kP2S1 : kP2S0);
        GRF rWS = wsBuf(v, 0);
        if (kMode == 5) {
            // Direct-C accumulate (no Ctmp, no g128 flush). as is constant
            // across the g128 group so applying it per tile is exact. NB:
            // mad(32) with 2-GRF sources hits fixup_ternary_rgn
            // (hs==1 && vs==width) and produced garbage here; mul+add instead.
            for (int n = 0; n < 16; n += 2) {
                GRF rS(sBase + n), rCf(kP2C + n);
                g.mov(32, rS.f(), rS.d());
                g.mul(16, rS.f(), rS.f(), rWS.f(n)(0));
                g.mul(16, GRF(sBase + n + 1).f(), GRF(sBase + n + 1).f(),
                        rWS.f(n + 1)(0));
                g.mul(32, rS.f(), rS.f(), GRF(bASv()).f());
                g.add(32, rCf.f(), rCf.f(), rS.f());
            }
        } else {
        for (int n = 0; n < 16; n += 2) {
            GRF rS(sBase + n), rCt(kP2Ctmp + n);
            g.mov(32, rS.f(), rS.d());              // cvt, cols n,n+1
            g.mul(16, rS.f(), rS.f(), rWS.f(n)(0)); // w_s[n]
            g.mul(16, GRF(sBase + n + 1).f(), GRF(sBase + n + 1).f(),
                    rWS.f(n + 1)(0));
            g.add(32, rCt.f(), rCt.f(), rS.f());
        }
        if ((v & 3) == 3 && kMode != 4) {
            for (int i = 0; i < 16; i += 2) {
                g.mul(32, GRF(kP2Ctmp + i).f(), GRF(kP2Ctmp + i).f(),
                        GRF(bASv()).f());
                g.add(32, GRF(kP2C + i).f(), GRF(kP2C + i).f(),
                        GRF(kP2Ctmp + i).f());
                g.mov(32, GRF(kP2Ctmp + i).f(), 0.0f);
            }
        }
        }
        if ((v & 3) == 3) {
            g.load(16, GRF(bASv()), scattered(DataSizeLSC::D32, 1), g.A64,
                    rPtrAS + 4 * (v + 4));
            g.mov(16, GRF(bASv() + 1).f(), GRF(bASv()).f());
        }
    };

    // ---- K loop (unrolled by kUnroll 32-deep tiles) --------------------------
    // Rotated body: dpas(u) -> loads(u+1) -> epilogue(u). The u+1 loads'
    // memory latency hides under the epilogue's fp work; their WAR on A/B
    // releases as soon as dpas(u) has read its sources.
    if (kMode != 2 && kUnroll > 1) emitLoads(0);
    if (kP2) // ws tile 0 -> buffer 0 (P2 keeps ws loads out of emitLoads)
        g.load(1, GRF(bWS()), block(DataSizeLSC::D32, 16), g.A64,
                GRF(kAddrWS));
    Label lTop;
    g.mark(lTop);
    if (kBarrier) g.barrier(rTmp, GRF(0)); // lockstep for L1 temporal reuse
    if (kP2) {
        // P2 rotated body: dpas(u) -> loads(u+1, A/B only) -> epilogue(u-1)
        // -> ws load(u+1). dpas(u) writes S[u&1] while epilogue(u-1) reads
        // S[(u-1)&1], so the DPAS pipe no longer drains behind the full
        // epilogue issue. The ws load for tile u+1 targets buffer (u+1)&1
        // == (u-1)&1 and must issue after epilogue(u-1)'s reads (WAR).
        GRF rAddrWS0(kAddrWS);
        for (int u = 0; u < kUnroll; u++) {
            for (int h = 0; h < 2; h++) // src0 = zero -> fresh s32/tile
                g.dpas(16, 8, 8,
                        GRF(((u & 1) ? kP2S1 : kP2S0) + 8 * h).d(),
                        GRF(kZ).d(), GRF(kA).b(), GRF(bB() + 2 * h).s4());
            emitLoads(u + 1);
            if (kPrefetch) emitPrefetch(u);
            if (u > 0) emitEpilogueP2(u - 1);
            const int v = u + 1;
            const int wb = (v == kUnroll) ? 0 : (v & 1);
            g.load(1, GRF(wb ? kWS2 : bWS()),
                    block(DataSizeLSC::D32, 16), g.A64,
                    rAddrWS0 + v * kN * 4);
        }
        emitEpilogueP2(kUnroll - 1); // drain the trailing epilogue
    } else
    for (int u = 0; u < kUnroll; u++) {
        if (kUnroll == 1 && kMode != 2)
            emitLoads(0); // serial body: no cross-tile pipelining
        if (kMode <= 2)
            for (int t = 0; t < kTiles; t++)
                for (int h = 0; h < 2; h++) // src0 = zero -> fresh s32/tile
                    g.dpas(16, 8, 8, GRF(bS() + t * 16 + 8 * h).d(),
                            GRF(kZ).d(), GRF(kA).b(),
                            GRF(bB() + 4 * t + 2 * h).s4());
        if (kMode != 2 && kUnroll > 1) emitLoads(u + 1);
        if (kPrefetch && kMode != 2) emitPrefetch(u);
        if (kMode == 0) {
            // Per-tile rescale: Cfp += act_s * w_s[n] * cvt(s32).
            for (int t = 0; t < kTiles; t++) {
                GRF rWS = wsBuf(u, t);
                for (int n = 0; n < 16; n += 2) {
                    GRF rS(bS() + t * 16 + n), rCf(bC() + t * 16 + n);
                    g.mov(32, rS.f(), rS.d());                // cvt, cols n,n+1
                    g.mul(16, rS.f(), rS.f(), rWS.f(n)(0));   // w_s[n]
                    g.mul(16, GRF(bS() + t * 16 + n + 1).f(),
                            GRF(bS() + t * 16 + n + 1).f(), rWS.f(n + 1)(0));
                    g.mul(32, rS.f(), rS.f(), GRF(bASv()).f());  // act_s (dup)
                    g.add(32, rCf.f(), rCf.f(), rS.f());
                }
            }
            // Preload the scale vector kUnroll tiles ahead (buffer padded).
            g.load(16, GRF(bASv()), scattered(DataSizeLSC::D32, 1), g.A64,
                    rPtrAS + 4 * (u + 1));
            g.mov(16, GRF(bASv() + 1).f(), GRF(bASv()).f());
        } else if (kMode == 3) {
            g.load(16, GRF(bASv()), scattered(DataSizeLSC::D32, 1), g.A64,
                    rPtrAS + 4 * (u + 1));
        }
    }
    // Advance A/B/ws/as base pointers by one unroll group.
    g.add(1, rAddrA.uq(0), rAddrA.uq(0), rStep.uq(0)(0));
    for (int t = 0; t < kTiles; t++) {
        GRF rAddrB(kAddrB + t), rAddrWS(kAddrWS + t);
        g.add(1, rAddrB.uq(0), rAddrB.uq(0), uint64_t(kUnroll * 256));
        g.add(1, rAddrWS.uq(0), rAddrWS.uq(0), uint64_t(kUnroll * kN * 4));
    }
    if (kMode == 0 || kMode == 3) {
        g.add(8, rPtrAS.uq(0)(1), rPtrAS.uq(0)(1), uint64_t(kUnroll * 4));
        g.add(8, GRF(kPtrAS + 1).uq(0)(1), GRF(kPtrAS + 1).uq(0)(1),
                uint64_t(kUnroll * 4));
    }
    g.add(1, rIt.d(0), rIt.d(0), -1);
    g.cmp(1 | g.gt | g.f0[0], rIt.d(0), 0);
    g.jmpi(1 | g.f0[0], lTop);

    if (kMode != 0) {
        // Numerically-wrong single rescale so the store isn't optimized away
        // and the C-store cost stays on the measured path.
        for (int t = 0; t < kTiles; t++)
            for (int n = 0; n < 16; n++) {
                GRF rS(bS() + t * 16 + n), rCf(bC() + t * 16 + n);
                g.mov(16, rS.f(), rS.d());
                g.add(16, rCf.f(), rCf.f(), rS.f());
            }
    }

    // ---- store C -------------------------------------------------------------
    for (int t = 0; t < kTiles; t++)
        for (int n = 0; n < 16; n++)
            g.store(16, scattered(DataSizeLSC::D32, 1), g.A64,
                    rPtrC + 4 * (t * 16 + n), GRF(cBase + t * 16 + n));
}

} // namespace w4a8gemm
} // namespace ngen_lab

// nGEN kernel-body author for the production-shaped W4A8 matmul (s8 activations
// x s4 weights -> bf16 C with per-32-K-tile weight rescale), built on the
// verified s8x4 16x16x32 tile (ngen_dpas_8x8_matmul_s4_16x16x32) and the
// full-GEMM prototype (previous revision of this file).
//
//   C[m,n] = bf16( act_s[m] * sum_t ws[t,n] * s32[m,n,t] )
//
// where t indexes 32-deep K tiles (dpas sdepth 8), s32 = the exact s32
// tile {A_s8[m,:] @ W_s4[n,:]} over tile t, ws is the per-(32-group, column)
// weight scale ((G,N) row-major, weight_scale layout; f32 by default, bf16
// under kWSBf16), and act_s is the PER-TOKEN activation scale (f32, M values).
//
// Production design (vs the previous g128 prototype):
//   - act_s is constant across all K tiles, so it is hoisted out of the K loop
//     and folded into the bf16 store at the end. No as32 buffer, no per-tile
//     act_s loads, no g128 group-boundary flush, no P2/Ctmp machinery.
//   - Per-tile epilogue is the minimum the per-group weight scale requires:
//     cvt s32->f32, mul by ws, add into Cacc (3 fp ops per column pair).
//   - C is bf16 row-major, matching the production consumers. The asymmetric-zp
//     correction is NOT part of this kernel (handled by the caller's rowsum +
//     corr pipeline).
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
//   ws (ws):  (G,N) row-major; tile t row = 16 consecutive values at n0 —
//             one block(D32,16) load (f32), or block(D16,16)+cvt (kWSBf16).
//   act_s(as): f32 (M) row-major per-token scales; one block(D32,16) load per
//             m-tile at setup.
//   C  (c):   bf16 (M,N) row-major, scattered D16 stores.
//
// Per-K-tile epilogue (probe-mandated per-group rescale; see memory
// qwen35_w4a8_probe):
//   - dpas src0 is a permanently-zero GRF block (kZ), so the s32 tile accs
//     never need init or re-zeroing;
//   - cvt runs at SIMD32 over column pairs (adjacent C GRFs);
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
// ws dtype: 0 = f32 (default), 1 = bf16 loaded + cvt in-register (production
// memory-saving option; measures the epilogue cost of the extra cvt).
static bool kWSBf16 = false;
// Load-subset diagnostics (perf ceiling analysis, results numerically WRONG):
// bitmask over which streams emitLoads issues: 1=A, 2=B, 4=ws, 7=all.
static int kLdSub = 7;
// Wide-load experiment: 1 = B as one block(D32,64) 256B load per n-tile
// (2x128B -> 1x256B); 2 = A as 2x block(D32,64) 256B (4x128B -> 2x256B).
// block(D32,128) (512B) transposes wrong; 256B may not.
static int kLdWide = 0;
// ws re-layout (harness-generated): ws_relay[nblock][Kt][T*16] f32 with the WG's
// T n-tile rows adjacent -> ONE block(D32, 16T) ws load per K-tile instead of T
// separate 64B loads. Cuts 3 load instructions per tile at T=4 (LSU-issue test).
static bool kWsRelay = false;

// Codegen experiment modes (perf ceiling analysis; results are numerically
// WRONG in modes != 0, harness must skip the check):
//   0 = full kernel (per-tile rescale + bf16 store)
//   1 = no per-tile epilogue: s32 accumulates the whole K, no ws loads,
//       single (wrong) rescale after the loop
//   2 = mode 1 + no in-loop loads at all (pure dpas issue rate)
//   3 = loads only (A/B/ws/act_s), no dpas, no epilogue (load-path ceiling)
static int kMode = 0;
// Cache hint applied to A/B stream loads (codegen experiment).
static ngen::CacheSettingsLSC kCacheAB = ngen::CacheSettingsLSC::Default;
// Cache hint for the A stream only (defaults to kCacheAB). A is re-read by
// every n-group (1088x at gate_up), so it wants L3 caching (cross-n-group
// reuse) with a streaming L1 (no L1 pollution); B/ws are shared across WGY
// threads and want L1 caching. NGEN_LAB_W4A8_CACHEA=1 -> L1UC_L3C, 2 -> L1C_L3C.
static ngen::CacheSettingsLSC kCacheA = ngen::CacheSettingsLSC::Default;
// Workgroup cooperation: WGY threads along M share the B/ws streams (same
// n-tiles), WGZ threads along N share the A stream, exploiting L1 sharing
// between co-scheduled threads on the same Xe-core. Optional per-K-loop
// barrier keeps co-scheduled threads in lockstep for L1 temporal reuse.
static int kWgY = 1, kWgZ = 1;
static bool kBarrier = false;
// SWZ experiment: launch order transposed (kernel reads group IDs swapped).
static bool kSwz = false;

// ---- register map (parametric in T = kTiles) -------------------------------
// Data region: A 16..23 (8 GRFs), B (4T) after it, ws (T), act_s (1), s32 S
// (16T), f32 C (16T). The dpas-src0 zero block (kZ, 8 GRFs) and the fixed
// address/temp region sit AFTER the data region so T>=3 fits in the large-GRF
// (kGRF=256) budget. T=2: A 16-23, B 24-31, WS 32-33, AS 34, S 35-66, C 67-98,
// Z 99-106, fixed 107-126. T=4: data 16-171, Z 172-179, fixed 180-199.
//
// ARF NOTE: the s32 dpas result cannot live in the accumulator registers on
// Xe2 (L0) — dpas dst=acc faults (verified: every acc index, src0 form, GRF
// count, accWrCtrl). The acc pipe is fp-dpas-only; int kernels keep S in GRFs.
constexpr int kA = 16;
inline int bB()  { return 24; }                     // 4T GRFs (2 per N-half)
inline int bWS() { return 24 + 4 * kTiles; }        // T GRFs (16 f32 each)
inline int bAS() { return 24 + 5 * kTiles; }        // 1 GRF (per-lane f32)
inline int bS()  { return 24 + 5 * kTiles + 1; }    // s32 acc: 16T GRFs
inline int bC()  { return 24 + 5 * kTiles + 1 + 16 * kTiles; }  // f32 C: 16T
inline int kZ()   { return bC() + 16 * kTiles; }    // 8 zeroed s32 GRFs (dpas src0)
inline int kF()   { return kZ() + 8; }              // fixed address/temp base
// Spacing is T-dependent: kAddrB/kAddrWS/kWS2 each hold T scalar bases.
inline int kBaseAS() { return kF(); }
inline int kBaseC()  { return kF() + 1; }
inline int kAddrB()  { return kF() + 2; }           // +t, T scalar uq bases
inline int kAddrWS() { return kF() + 2 + kTiles; }  // +t, T scalar uq bases
inline int kIt()     { return kF() + 2 + 2 * kTiles; }
inline int kStep()   { return kF() + 3 + 2 * kTiles; }  // uq, kUnroll*512 (A bump)
inline int kAddrA()  { return kF() + 4 + 2 * kTiles; }
inline int kIdx()    { return kF() + 5 + 2 * kTiles; }
inline int kRows()   { return kF() + 6 + 2 * kTiles; }
inline int kTmp()    { return kF() + 7 + 2 * kTiles; }
inline int kPtrC()   { return kF() + 8 + 2 * kTiles; }  // 2 GRFs (per-lane uq C)
inline int kWS2()    { return kF() + 10 + 2 * kTiles; } // ws ping-pong buffer 1 (+t)
inline int kTmpBf()  { return kF() + 10 + 3 * kTiles; } // bf16 cvt / load scratch
inline int kAddrAS() { return kF() + 11 + 3 * kTiles; } // scalar uq act_s base

template <typename Generator>
void gemmKernelBody(Generator &g, ngen::Subregister argA,
        ngen::Subregister argB, ngen::Subregister argWS,
        ngen::Subregister argAS, ngen::Subregister argC,
        ngen::Subregister gidX, ngen::Subregister gidY,
        ngen::GRF lidY, ngen::GRF lidZ) {
    using namespace ngen;
    GRF rBaseAS(kBaseAS()), rBaseC(kBaseC());
    GRF rIt(kIt()), rStep(kStep()), rAddrA(kAddrA());
    GRF rIdx(kIdx()), rRows(kRows()), rTmp(kTmp());
    GRF rPtrC(kPtrC());
    GRF rAddrAS(kAddrAS());

    // ---- setup -------------------------------------------------------------
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

    // Per-token act_s for this m-tile: one block(D32,16) load at as + m_tile*64
    // (16 rows of f32, one per lane). Loaded once; applied only at the store.
    g.mov(1, rAddrAS.uq(0), argAS);
    g.mul(1, rTmp.ud(0), rIdx.ud(0), uint32_t(64));
    g.mov(1, rTmp.ud(1), 0u);
    g.add(1, rTmp.uq(0), rAddrAS.uq(0), rTmp.uq(0));
    g.load(1, GRF(bAS()), block(DataSizeLSC::D32, 16), g.A64, rTmp);

    // ngrp = gidX*kWgZ + lidZ, stashed in rBaseAS (free now).
    GRF rNgrp(kBaseAS());
    if (kWgZ > 1) {
        g.mul(1, rNgrp.ud(0), gidX, uint32_t(kWgZ));
        g.add(1, rNgrp.ud(0), rNgrp.ud(0), lidZ.uw(0));
    } else {
        g.mov(1, rNgrp.ud(0), gidX);
    }

    // C lane addresses (bf16): c + rows*(N*2) + n_tile0*T*32. The row stride
    // kN*2 = 69632 at N=34816 exceeds the 16-bit mul immediate on Xe2 (the
    // "K*16 = 2^16 silently broke -> device lost" trap), so multiply by kN
    // then shift by 1 (×2 for bf16) instead.
    g.mul(16, rTmp.ud(0)(1), rRows.ud(0)(1), uint32_t(kN));
    g.shl(16, rTmp.ud(0)(1), rTmp.ud(0)(1), 1);
    g.mov(16, rPtrC.ud(0)(2), rTmp.ud(0)(1));
    g.mov(16, rPtrC.ud(1)(2), 0u);
    g.mul(1, rTmp.ud(0), rNgrp.ud(0), uint32_t(kTiles * 32));
    g.mov(1, rTmp.ud(1), 0u);
    g.add(16, rPtrC.ud(0)(2), rPtrC.ud(0)(2), rTmp.ud(0)(0));
    g.add(8, rPtrC.uq(0)(1), rPtrC.uq(0)(1), rBaseC.uq(0)(0));
    g.add(8, GRF(kPtrC() + 1).uq(0)(1), GRF(kPtrC() + 1).uq(0)(1), rBaseC.uq(0)(0));

    // Per-tile scalar bases: B at b + (gidX*T+t)*(K*8), ws at ws +
    // (gidX*T+t)*64 (first row; +N*4 per K tile via immediate offsets).
    for (int t = 0; t < kTiles; t++) {
        GRF rAddrB(kAddrB() + t), rAddrWS(kAddrWS() + t);
        g.mov(1, rAddrB.uq(0), argB);
        g.mov(1, rAddrWS.uq(0), argWS);
        g.mul(1, rTmp.ud(0), rNgrp.ud(0), uint32_t(kTiles));
        if (t > 0) g.add(1, rTmp.ud(0), rTmp.ud(0), uint32_t(t));
        g.mul(1, rTmp.ud(0), rTmp.ud(0), uint32_t(kK));
        g.shl(1, rTmp.ud(0), rTmp.ud(0), 3);  // x K*8 bytes per n-tile of B
        g.mov(1, rTmp.ud(1), 0u);
        g.add(1, rAddrB.uq(0), rAddrB.uq(0), rTmp.uq(0));
        // ws base: relay layout -> ws + ngrp*(Kt*T*rowBytes), all T n-tile rows
        // in one contiguous block (no per-tile offset). Plain layout keeps the
        // per-tile (ngrp*T + t)*rowBytes.
        const uint32_t wsRowBytes = kWSBf16 ? 32 : 64;
        g.mul(1, rTmp.ud(0), rNgrp.ud(0),
                uint32_t(kWsRelay ? (kKt * kTiles * wsRowBytes)
                                  : (kTiles * wsRowBytes)));
        if (!kWsRelay && t > 0)
            g.add(1, rTmp.ud(0), rTmp.ud(0), uint32_t(wsRowBytes * t));
        g.mov(1, rTmp.ud(1), 0u);
        g.add(1, rAddrWS.uq(0), rAddrWS.uq(0), rTmp.uq(0));
    }

    // Zero the fp32 C accumulators plus the shared dpas-src0 zero block.
    const int cBase = bC();
    for (int i = 0; i < 16 * kTiles; i += 2)
        g.mov(32, GRF(cBase + i).f(), 0.0f);
    for (int i = 0; i < 8; i += 2)
        g.mov(32, GRF(kZ() + i).d(), 0);

    // ws ping-pong buffer: tile v's w_scale row lives in buffer v%2, so the
    // load for tile u+1 (emitted right after dpas(u)) does not collide with
    // the epilogue of tile u. Buffer 1 lives at kWS2. Even kUnroll only:
    // group parity must not flip between groups.
    auto wsBuf = [&](int v, int t) {
        return GRF(((v & 1) ? kWS2() : bWS()) + t);
    };
    // Loads for K-tile v (v == kUnroll targets the next unroll-group's tile 0
    // via pre-bump base pointers + one-group offsets; the trailing over-read
    // after the last group is never consumed and the harness pads the
    // allocations).
    auto emitLoads = [&](int v) {
        const int aOff = v * 512, bOff = v * 256;
        const int wsOff = kWsRelay
                ? v * kTiles * (kWSBf16 ? 32 : 64)
                : v * kN * (kWSBf16 ? 2 : 4);
        // Buffer of the consuming epilogue: local tile v%kUnroll, parity
        // (v%kUnroll)%2. v == kUnroll is next group's tile 0 -> buffer 0.
        // REQUIRES even kUnroll (else tile kUnroll-1 also uses buffer 0).
        const int wb = (v == kUnroll) ? 0 : (v & 1);
        // A tile: 4 contiguous 128B block loads (pre-swizzled src1 order).
        // B tile: 2 N-halves x 128B (s4). 4x/2x loads verified correct; a
        // single 512B/256B block load is faster but transposes wrong (FAIL).
        if (kLdSub & 1) {
        if (kLdWide & 2)
            for (int j = 0; j < 2; j++)
                g.load(1, GRF(kA + 4 * j),
                        block(DataSizeLSC::D32, 64) | kCacheA, g.A64,
                        rAddrA + aOff + 256 * j);
        else
        for (int j = 0; j < 4; j++)
            g.load(1, GRF(kA + 2 * j),
                    block(DataSizeLSC::D32, 32) | kCacheA, g.A64,
                    rAddrA + aOff + 128 * j);
        }
        for (int t = 0; t < kTiles; t++) {
            GRF rAddrB(kAddrB() + t), rAddrWS(kAddrWS() + t);
            if (kLdSub & 2) {
            if (kLdWide & 1)
                g.load(1, GRF(bB() + 4 * t),
                        block(DataSizeLSC::D32, 64) | kCacheAB, g.A64,
                        rAddrB + bOff);
            else
            for (int h = 0; h < 2; h++)
                g.load(1, GRF(bB() + 4 * t + 2 * h),
                        block(DataSizeLSC::D32, 32) | kCacheAB, g.A64,
                        rAddrB + bOff + h * 128);
            }
            // w_scale row for this K tile: 16 values. wsOff = v*kN*4 can
            // exceed the LSC immediate-offset range for large N (gateup
            // N=34816), so materialize the address first.
            if ((kMode == 0 || kMode == 3) && (kLdSub & 4)) {
                g.add(1, rTmp.uq(0), rAddrWS.uq(0), uint64_t(wsOff));
                if (kWSBf16) {
                    g.load(1, GRF(kTmpBf()),
                            block(DataSizeLSC::D16, 16), g.A64, rTmp);
                    g.mov(16, GRF(wsBuf(v, t)).f(), GRF(kTmpBf()).bf());
                } else if (kWsRelay) {
                    // 2x128B blocks (the 256B single block is measurably
                    // slower than 4x64B; 128B is the HW sweet spot).
                    for (int j = 0; j < 2; j++)
                        g.load(1, GRF(((v & 1) ? kWS2() : bWS()) + 2 * j),
                                block(DataSizeLSC::D32, 32), g.A64,
                                rTmp + 128 * j);
                } else {
                    g.load(1, GRF(wsBuf(v, t)),
                            block(DataSizeLSC::D32, 16), g.A64, rTmp);
                }
            }
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
                GRF rAddrB(kAddrB() + t);
                for (int h = 0; h < 2; h++)
                    g.load(1, g.null, block(DataSizeLSC::D32, 32) | kCacheAB,
                            g.A64, rAddrB + bOff + h * 128);
            }
        // NB: no ws prefetch — 64B null-dst block prefetch hangs (Xe2).
    };

    // ---- K loop (unrolled by kUnroll 32-deep tiles) --------------------------
    // Rotated body: dpas(u) -> loads(u+1) -> epilogue(u). The u+1 loads'
    // memory latency hides under the epilogue's fp work; their WAR on A/B
    // releases as soon as dpas(u) has read its sources.
    if (kMode != 2 && kUnroll > 1) emitLoads(0);
    Label lTop;
    g.mark(lTop);
    if (kBarrier) g.barrier(rTmp, GRF(0)); // lockstep for L1 temporal reuse
    for (int u = 0; u < kUnroll; u++) {
        if (kUnroll == 1 && kMode != 2)
            emitLoads(0); // serial body: no cross-tile pipelining
        if (kMode <= 2)
            for (int t = 0; t < kTiles; t++)
                for (int h = 0; h < 2; h++) // src0 = zero -> fresh s32/tile
                    g.dpas(16, 8, 8, GRF(bS() + t * 16 + 8 * h).d(),
                            GRF(kZ()).d(), GRF(kA).b(),
                            GRF(bB() + 4 * t + 2 * h).s4());
        if (kMode != 2 && kUnroll > 1) emitLoads(u + 1);
        if (kPrefetch && kMode != 2) emitPrefetch(u);
        if (kMode == 0) {
            // Per-tile rescale: Cacc += ws[n] * cvt(s32). No act_s here — it is
            // constant across K and applied once at the store.
            for (int t = 0; t < kTiles; t++) {
                GRF rWS = kWsRelay ? GRF(((u & 1) ? kWS2() : bWS()) + t)
                                   : wsBuf(u, t);
                for (int n = 0; n < 16; n += 2) {
                    GRF rS(bS() + t * 16 + n), rCf(bC() + t * 16 + n);
                    g.mov(32, rS.f(), rS.d());               // cvt, cols n,n+1
                    g.mul(16, rS.f(), rS.f(), rWS.f(n)(0));  // ws[n]
                    g.mul(16, GRF(bS() + t * 16 + n + 1).f(),
                            GRF(bS() + t * 16 + n + 1).f(), rWS.f(n + 1)(0));
                    g.add(32, rCf.f(), rCf.f(), rS.f());
                }
            }
        }
    }
    // Advance A/B/ws base pointers by one unroll group.
    g.add(1, rAddrA.uq(0), rAddrA.uq(0), rStep.uq(0)(0));
    for (int t = 0; t < kTiles; t++) {
        GRF rAddrB(kAddrB() + t), rAddrWS(kAddrWS() + t);
        g.add(1, rAddrB.uq(0), rAddrB.uq(0), uint64_t(kUnroll * 256));
        g.add(1, rAddrWS.uq(0), rAddrWS.uq(0),
                uint64_t(kWsRelay
                        ? kUnroll * kTiles * (kWSBf16 ? 32 : 64)
                        : kUnroll * kN * (kWSBf16 ? 2 : 4)));
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

    // ---- store C (bf16, per-token act_s folded in) --------------------------
    // bf16 RNE rounding via integer ops (one value per u32 element, matching
    // the D16U32 scattered-store layout oneDNN uses on Xe2): round-half-up
    // (0x7FFF bias) differs from true RNE only on exact ties (never for random
    // data; the check allows 1 ulp).
    for (int t = 0; t < kTiles; t++)
        for (int n = 0; n < 16; n++) {
            GRF rC(cBase + t * 16 + n);
            g.mul(16, rC.f(), rC.f(), GRF(bAS()).f());     // act_s[lane]
            g.add(16, GRF(kTmpBf()).ud(), rC.ud(), 0x7FFFu); // RNE half-ulp bias
            g.shr(16, GRF(kTmpBf()).ud(), GRF(kTmpBf()).ud(), 16);
            g.store(16, scattered(DataSizeLSC::D16U32, 1), g.A64,
                    rPtrC + 2 * (t * 16 + n), GRF(kTmpBf()));
        }
}

} // namespace w4a8gemm
} // namespace ngen_lab

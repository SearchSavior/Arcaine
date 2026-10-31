// nGEN kernel-body author for the M-MAJOR W4A8 matmul (s8 activations x s4
// weights -> bf16 C with per-32-K-tile weight rescale + per-token act scale at
// the store). This is the oneDNN-gemmstone-style rewrite: the per-thread tile
// is M-major (kMt m-tiles of 16 rows x one 16-col n-tile) instead of the
// N-major (T n-tiles) layout. Per K-tile a thread loads its kMt A tiles
// (cached, short-latency) and ONE B tile (the long-latency weight stream),
// vs T B tiles in the N-major kernel. kWgY threads share B/ws (same n-tile,
// different m-groups); kWgZ threads share A (same m-tile, different n-tiles) —
// both L1 cooperation paths cut the L2 re-read traffic. Register headroom
// pays for a DEEP software pipeline of the loads (kPipeDepth D tiles ahead
// into D+1 buffers — the k128-style load scheduling oneDNN uses).
//
//   C[m,n] = bf16( act_s[m] * sum_t ws[t,n] * s32[m,n,t] )
//
// Thread/grid model: SIMD16, lane = C row m0+m (16 rows). Each thread owns
// kMt m-tiles (kMt*16 rows) x ONE n-tile (16 cols). Grid =
// {N/(16*kWgZ), M/(16*kMt*kWgY), 1}. WG = (16, kWgY, kWgZ) threads.
//
// Layouts (same as the N-major kernel; see ngen_dpas_w4a8_gemm.hpp):
//   A  (a): s8 pre-swizzled [m_tile][k_tile][j][lane m], 512B per (mt,kt);
//          m-tile mt_abs's stream at a + mt_abs*K*16.
//   B  (b): s4 ngen src2 stream [n_tile][k_tile][h][r][j], 256B per (nt,kt).
//   ws (ws): (K/32, N) f32 row-major, 16 f32 per (tile, n-tile) row.
//   act_s(as): f32 (M) per-token.
//   C  (c): bf16 (M,N) row-major, D16U32 scattered stores.
//
// The SHAPE IS BAKED (kN/kK/kKt). K loop: the A/B/ws base pointers advance one
// tile per iteration, so every load's immediate offset stays small
// (<= ahead*stride); the loads for tile u+ahead issue right after dpas(u) into
// buffer (u+ahead)%(D+1); dpas(u) reads buffer u%(D+1). kPipeDepth=0 uses one
// buffer with loads(u+1) (the N-major behavior). Per-tile epilogue: cvt
// s32->f32 (SIMD32 over column pairs), mul by ws (SIMD16, scalar per column),
// add into Cacc. act_s applied once at the bf16 store. dpas src0 = permanent
// zero block. ARF dst blocked for s32 on Xe2 — S lives in GRFs.

#pragma once

#include "ngen.hpp"

namespace ngen_lab {
namespace w4a8m {

static int kN = 0, kK = 0, kKt = 0;   // N, K, K/32
static int kMt = 2;                   // m-tiles per thread (16 rows each)
static int kPipeDepth = 0;            // 0 = per-tile loads; D = D-ahead pipeline
static int kWgY = 1, kWgZ = 1;        // WG cooperation threads
static int kMode = 0;                 // 0 = full; 1/2/3 = ablations (WRONG)
static bool kLdWS = true;             // DIAG: ws-load toggle (1 = load)
static ngen::CacheSettingsLSC kCacheAB = ngen::CacheSettingsLSC::Default;

// ---- register map (parametric in Mt, D = kPipeDepth) ----------------------
// A: (D+1) buffers x kMt m-tiles x 8 GRFs. Buffer b, m-tile mt:
//     GRF(aA0() + b*(8*kMt) + mt*8).
inline int aA0() { return 16; }
inline int nBuf() { return kPipeDepth + 1; }
// B: (D+1) buffers x 4 GRFs (one n-tile, 2 halves). Buffer b: GRF(bB0()+b*4).
inline int bB0() { return aA0() + 8 * kMt * nBuf(); }
// ws: 2 ping-pong GRFs. act_s: kMt GRFs (one 16-f32 vector per m-tile).
inline int bWS() { return 16 + 8 * kMt * nBuf() + 4 * nBuf() + 56; }  // DIAG
inline int bWS2() { return bWS() + 1; }
inline int bAS() { return bWS() + 2; }
// S: kMt*2 dpas results x 8 GRFs. S[mt][h] = GRF(bS() + (mt*2+h)*8).
inline int bS() { return bAS() + kMt; }
// C: kMt x 16 cols x 1 GRF (f32). C[mt][col] = GRF(bC() + mt*16 + col).
inline int bC() { return bS() + 16 * kMt; }
inline int kZ() { return bC() + 16 * kMt; }   // 8 zeroed s32 GRFs (dpas src0)
inline int kF() { return kZ() + 8; }          // fixed address/temp region
// Per-m-tile A bases (kMt uq GRFs), per-m-tile C address vectors (2kMt GRFs).
inline int kBaseAmt() { return kF(); }
inline int kPtrCmt() { return kF() + kMt; }          // 2*kMt GRFs
inline int kAddrB() { return kF() + 3 * kMt; }
inline int kAddrWS() { return kF() + 3 * kMt + 1; }
inline int kBaseC() { return kF() + 3 * kMt + 2; }
inline int kIdx() { return kF() + 3 * kMt + 3; }
inline int kRows() { return kF() + 3 * kMt + 4; }
inline int kTmp() { return kF() + 3 * kMt + 5; }
inline int kTmpBf() { return kF() + 3 * kMt + 6; }
inline int kAddrAS() { return kF() + 3 * kMt + 7; }
// Mt=2, D=3 (4 buffers): A 16-79, B 80-95, WS 96-97, AS 98-99, S 100-131,
// C 132-163, Z 164-171, fixed 172-187. Highest register = kF()+3*Mt+8:
//   Mt=2: 26 + 20*nBuf + 6;  nBuf=9 (D=8) -> 212 (fits 256).
//   Mt=1: 16 + 12*nBuf + 14; nBuf=18 (D=17) -> 246 (fits 256).

template <typename Generator>
void gemmKernelBody(Generator &g, ngen::Subregister argA,
        ngen::Subregister argB, ngen::Subregister argWS,
        ngen::Subregister argAS, ngen::Subregister argC,
        ngen::Subregister gidX, ngen::Subregister gidY,
        ngen::GRF lidY, ngen::GRF lidZ) {
    using namespace ngen;
    GRF rIdx(kIdx()), rRows(kRows()), rTmp(kTmp());
    const int nbuf = nBuf();
    const int ahead = kPipeDepth == 0 ? 1 : kPipeDepth;

    // ---- setup -------------------------------------------------------------
    // Lane rows: 0..15 + mgroup*16 (mgroup = gidY*kWgY + lidY).
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

    // A bases per m-tile: a + (mgroup*kMt + mt)*K*16. Compute mgroup*kMt*K*16
    // in rTmp (u64), then per-mt add mt*K*16 (fits u32 via mul+shl).
    g.mul(1, rTmp.ud(0), rIdx.ud(0), uint32_t(kMt * kK));
    g.shl(1, rTmp.ud(0), rTmp.ud(0), 4);
    g.mov(1, rTmp.ud(1), 0u);
    for (int mt = 0; mt < kMt; mt++) {
        GRF rBase(kBaseAmt() + mt);
        g.mov(1, rBase.uq(0), argA);
        g.add(1, rBase.uq(0), rBase.uq(0), rTmp.uq(0));
        if (mt > 0) {
            g.mov(1, rTmp.ud(2), uint32_t(mt * kK));
            g.shl(1, rTmp.ud(2), rTmp.ud(2), 4);   // *16 bytes
            g.mov(1, rTmp.ud(3), 0u);
            g.add(1, rBase.uq(0), rBase.uq(0), rTmp.uq(2));
        }
    }

    // act_s per m-tile: block(D32,16) at as + (mgroup*kMt+mt)*64. The mgroup
    // offset is folded into rAddrAS once; per-mt only adds mt*64.
    GRF rAddrAS(kAddrAS());
    g.mov(1, rAddrAS.uq(0), argAS);
    g.mul(1, rTmp.ud(0), rIdx.ud(0), uint32_t(kMt * 64));
    g.mov(1, rTmp.ud(1), 0u);
    g.add(1, rAddrAS.uq(0), rAddrAS.uq(0), rTmp.uq(0));
    for (int mt = 0; mt < kMt; mt++) {
        g.add(1, rTmp.uq(0), rAddrAS.uq(0), uint64_t(mt * 64));
        g.load(1, GRF(bAS() + mt), block(DataSizeLSC::D32, 16), g.A64, rTmp);
    }

    // n-group: ngrp = gidX*kWgZ + lidZ (rTmp is free after the AS loads).
    GRF rNgrp(kTmp());
    if (kWgZ > 1) {
        g.mul(1, rNgrp.ud(0), gidX, uint32_t(kWgZ));
        g.add(1, rNgrp.ud(0), rNgrp.ud(0), lidZ.uw(0));
    } else {
        g.mov(1, rNgrp.ud(0), gidX);
    }

    // B base: b + ngrp*K*8. ws base: ws + ngrp*64. C base: argC.
    GRF rAddrB(kAddrB()), rAddrWS(kAddrWS()), rBaseC(kBaseC());
    g.mov(1, rAddrB.uq(0), argB);
    g.mov(1, rAddrWS.uq(0), argWS);
    g.mov(1, rBaseC.uq(0), argC);
    g.mul(1, rTmp.ud(0), rNgrp.ud(0), uint32_t(kK));
    g.shl(1, rTmp.ud(0), rTmp.ud(0), 3);
    g.mov(1, rTmp.ud(1), 0u);
    g.add(1, rAddrB.uq(0), rAddrB.uq(0), rTmp.uq(0));
    g.mul(1, rTmp.ud(0), rNgrp.ud(0), uint32_t(64));
    g.mov(1, rTmp.ud(1), 0u);
    g.add(1, rAddrWS.uq(0), rAddrWS.uq(0), rTmp.uq(0));

    // C address vectors per m-tile (2 GRFs of 16 uq): c + row*N*2 + ngrp*32,
    // row = (mgroup*kMt+mt)*16 + lane (rRows already has +mgroup*16; add mt*16).
    for (int mt = 0; mt < kMt; mt++) {
        // The per-lane uq vector for this m-tile is a 2-GRF region
        // (kPtrCmt()+2*mt); low/high words interleave via ud(0)(2)/ud(1)(2)
        // exactly like the N-major kernel's rPtrC.
        GRF rCv(kPtrCmt() + 2 * mt);
        if (mt > 0)
            g.add(16, rTmp.ud(0)(1), rRows.ud(0)(1), uint32_t(mt * 16));
        else
            g.mov(16, rTmp.ud(0)(1), rRows.ud(0)(1));
        g.mul(16, rTmp.ud(0)(1), rTmp.ud(0)(1), uint32_t(kN));
        g.shl(16, rTmp.ud(0)(1), rTmp.ud(0)(1), 1);
        g.mov(16, rCv.ud(0)(2), rTmp.ud(0)(1));    // low words
        g.mov(16, rCv.ud(1)(2), 0u);               // high words
        g.mul(1, rTmp.ud(0), rNgrp.ud(0), uint32_t(32));
        g.mov(1, rTmp.ud(1), 0u);
        g.add(16, rCv.ud(0)(2), rCv.ud(0)(2), rTmp.ud(0)(0));
        g.add(8, rCv.uq(0)(1), rCv.uq(0)(1), rBaseC.uq(0)(0));
        g.add(8, GRF(kPtrCmt() + 2 * mt + 1).uq(0)(1),
                GRF(kPtrCmt() + 2 * mt + 1).uq(0)(1), rBaseC.uq(0)(0));
    }

    // Zero the f32 C accumulators and the dpas-src0 zero block.
    for (int i = 0; i < 16 * kMt; i += 2)
        g.mov(32, GRF(bC() + i).f(), 0.0f);
    for (int i = 0; i < 8; i += 2)
        g.mov(32, GRF(kZ() + i).d(), 0);

    // ---- loads for absolute tile v (buffer b = v%nbuf) ---------------------
    // The base pointers are advanced one tile per K iteration, so the load
    // offsets are fixed at (v - currentTile)*stride = ahead*stride.
    auto emitLoads = [&](int v, int buf, int tileOff) {
        const int aOff = tileOff * 512, bOff = tileOff * 256,
                  wsOff = tileOff * kN * 4;
        // A: per m-tile 4x128B.
        for (int mt = 0; mt < kMt; mt++) {
            GRF rBase(kBaseAmt() + mt);
            for (int j = 0; j < 4; j++)
                g.load(1, GRF(aA0() + buf * 8 * kMt + mt * 8 + 2 * j),
                        block(DataSizeLSC::D32, 32) | kCacheAB, g.A64,
                        rBase + aOff + 128 * j);
        }
        // B: one 256B wide load.
        g.load(1, GRF(bB0() + buf * 4),
                block(DataSizeLSC::D32, 64) | kCacheAB, g.A64,
                GRF(kAddrB()) + bOff);
        // ws: 16 f32 (materialized: wsOff can exceed the LSC immediate range).
        if (kMode == 0) {
            g.add(1, rTmp.uq(0), GRF(kAddrWS()).uq(0), uint64_t(wsOff));
            g.load(1, GRF(bWS()),
                    block(DataSizeLSC::D32, 16), g.A64, GRF(kAddrWS()));
        }
    };
    // Advance A/B/ws bases by one tile.
    auto advanceBases = [&]() {
        for (int mt = 0; mt < kMt; mt++)
            g.add(1, GRF(kBaseAmt() + mt).uq(0),
                    GRF(kBaseAmt() + mt).uq(0), uint64_t(512));
        g.add(1, GRF(kAddrB()).uq(0), GRF(kAddrB()).uq(0), uint64_t(256));
        g.add(1, GRF(kAddrWS()).uq(0), GRF(kAddrWS()).uq(0),
                uint64_t(kN * 4));
    };

    // ---- dpas + epilogue for one K-tile ------------------------------------
    auto emitTile = [&](int u, int buf) {
        // dpas: kMt m-tiles x 2 halves, fresh s32 (src0 = zero block).
        if (kMode <= 2)
            for (int mt = 0; mt < kMt; mt++)
                for (int h = 0; h < 2; h++)
                    g.dpas(16, 8, 8, GRF(bS() + (mt * 2 + h) * 8).d(),
                            GRF(kZ()).d(),
                            GRF(aA0() + buf * 8 * kMt + mt * 8).b(),
                            GRF(bB0() + buf * 4 + 2 * h).s4());
        // Per-tile rescale: C += ws[col] * cvt(s32). S[mt][h] holds columns
        // 8h..8h+7 (8 GRFs). act_s is applied once at the store.
        if (kMode == 0) {
            GRF rWS(bWS());
            for (int mt = 0; mt < kMt; mt++)
                for (int h = 0; h < 2; h++)
                    for (int c = 0; c < 4; c++) {
                        const int col = 8 * h + 2 * c;
                        GRF rS(bS() + (mt * 2 + h) * 8 + 2 * c);
                        GRF rCf(bC() + mt * 16 + col);
                        g.mov(32, rS.f(), rS.d());
                        g.mul(16, rS.f(), rS.f(), rWS.f(col)(0));
                        g.mul(16, GRF(bS() + (mt * 2 + h) * 8 + 2 * c + 1).f(),
                                GRF(bS() + (mt * 2 + h) * 8 + 2 * c + 1).f(),
                                rWS.f(col + 1)(0));
                        g.add(32, rCf.f(), rCf.f(), rS.f());
                    }
        }
    };

    // ---- K loop ------------------------------------------------------------
    // Prologue: load the first `ahead` tiles (base is at tile 0).
    for (int v = 0; v < ahead && v < kKt; v++)
        emitLoads(v, v % nbuf, v);          // prologue: tiles 0..ahead-1
    for (int u = 0; u < kKt; u++) {
        emitTile(u, u % nbuf);
        if (u + ahead < kKt) emitLoads(u + ahead, (u + ahead) % nbuf, ahead);
        advanceBases();
    }

    // ---- store C (bf16, per-token act_s folded in) --------------------------
    for (int mt = 0; mt < kMt; mt++)
        for (int n = 0; n < 16; n++) {
            GRF rC(bC() + mt * 16 + n);
            g.mul(16, rC.f(), rC.f(), GRF(bAS() + mt).f());
            g.add(16, GRF(kTmpBf()).ud(), rC.ud(), 0x7FFFu);  // RNE half-ulp
            g.shr(16, GRF(kTmpBf()).ud(), GRF(kTmpBf()).ud(), 16);
            g.store(16, scattered(DataSizeLSC::D16U32, 1), g.A64,
                    GRF(kPtrCmt() + 2 * mt) + 2 * n, GRF(kTmpBf()));
        }
}

} // namespace w4a8m
} // namespace ngen_lab

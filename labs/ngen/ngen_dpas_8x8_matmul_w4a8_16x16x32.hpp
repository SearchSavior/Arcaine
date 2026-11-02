// nGEN kernel-body authors for the fixed-shape w4a8 16x16x32 matmul.
// Compiled twice: against the binary generator (LevelZeroCodeGenerator<Xe2>)
// in the .cpp harness, and against the text generator (AsmCodeGenerator,
// NGEN_ASM) in ngen_lab_asm.cpp. Only the generator-agnostic instruction
// subset is used; all interface values are passed as parameters.

#pragma once

#include "ngen.hpp"

namespace ngen_lab {
namespace w4a8 {

// Fixed matmul shape.
constexpr int kM = 16, kN = 16, kK = 32;
constexpr size_t kABytes = kM * kK / 2;        // 256 B, row-major s4 (nibbles)
constexpr size_t kBBytes = kK * kN;            // 512 B natural / packed
constexpr size_t kCBytes = kM * kN * 4;        // 1 KiB, row-major s32

// Tiles per thread (selected at runtime via NGEN_LAB_S4_TILES).
static int kTiles = 1;

// ---- register map (parametric in tiles-per-thread T) ----------------------
//   A: 4T GRFs starting at kA
//   B: 8 GRFs starting at kA + 4T (shared across tiles; 4 per N-half, s8)
//   C: 16T GRFs starting at kA + 4T + 4
//   addresses/temps: fixed at the top of the GRF space
constexpr int kA = 16;
constexpr int kAddrA = 96, kAddrB = 97, kAddrC = 98;
constexpr int kTmp = 99, kIt = 100, kIdx = 101, kOff = 102;

// Address-vector base: A uses (kPtrBase + 4*t), C uses (kPtrBase + 4*T + 4*t)
// for tile t (each vector is 2 GRFs). Fits for T <= 4 (8 vectors = 16 GRFs).
constexpr int kPtrBase = 108;

inline int aBase() { return kA; }                 // A GRFs: kA .. kA+4T-1
inline int bBase() { return kA + 4 * kTiles; }    // B GRFs: +8 (4/half)
inline int cBase() { return kA + 4 * kTiles + 8; } // C GRFs: +16T
inline int ptrA(int t) { return kPtrBase + 4 * t; }
inline int ptrC(int t) { return kPtrBase + 4 * kTiles + 4 * t; }

// Builds the per-tile scalar bases and the A/C lane-address vectors for all
// kTiles tiles. Same idiom as the BF16 variant: 64-bit-safe adds, same-type
// widening, explicit GRF(base+1).
template <typename Generator>
void setupTileAddresses(Generator &g, ngen::Subregister argA,
        ngen::Subregister argB, ngen::Subregister argC,
        ngen::Subregister gid) {
    using namespace ngen;
    GRF rAddrA(kAddrA), rAddrB(kAddrB), rAddrC(kAddrC), rTmp(kTmp);
    GRF rIdx(kIdx), rOff(kOff);

    g.mov(1, rAddrA.uq(0), argA);
    g.mov(1, rAddrB.uq(0), argB);
    g.mov(1, rAddrC.uq(0), argC);

    // Thread's first tile base: group id * (kTiles tiles) * tile bytes.
    g.mul(1, rTmp.ud(0), gid, uint32_t(kTiles * kABytes));
    g.mov(1, rTmp.ud(1), 0u);
    g.add(1, rAddrA.uq(0), rAddrA.uq(0), rTmp.uq(0));
    g.mul(1, rTmp.ud(0), gid, uint32_t(kTiles * kCBytes));
    g.mov(1, rTmp.ud(1), 0u);
    g.add(1, rAddrC.uq(0), rAddrC.uq(0), rTmp.uq(0));

    // Lane byte offsets 64*m (C is 4 bytes/row element x 16 = 64B/row).
    g.mov(8, rIdx.uw(0)(1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
    g.mov(8, rIdx.uw(8)(1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
    g.mov(16, rOff.uw(0)(2), rIdx.uw(0)(1));
    g.mov(16, rOff.uw(1)(2), uint16_t(0));
    g.shl(16, rOff.ud(0)(1), rOff.ud(0)(1), 6);

    // A rows are 16B each (s4); C rows are 64B each.
    // rOffA and rOff are the SAME GRF, so capture rOffC = 64m BEFORE the
    // in-place shifts (which would otherwise corrupt the C stride).
    GRF rOffA(kOff), rOffC(kOff + 1);
    g.mov(16, rOffC.ud(0)(1), rOff.ud(0)(1)); // C lane offset: 64*m
    g.shr(16, rOffA.ud(0)(1), rOff.ud(0)(1), 2); // A lane offset: 16*m (64m/4)

    for (int t = 0; t < kTiles; t++) {
        GRF rpA(ptrA(t)), rpC(ptrC(t));
        // A vector: lane m = tile base + 32m. Tile t base = rAddrA + t*512.
        g.mov(8, rpA.ud(0)(2), rOffA.ud(0)(1));
        g.mov(8, rpA.ud(1)(2), 0u);
        g.mov(8, GRF(ptrA(t) + 1).ud(0)(2), rOffA.ud(8)(1));
        g.mov(8, GRF(ptrA(t) + 1).ud(1)(2), 0u);
        // C vector: lane m = tile base + 64m. Tile t base = rAddrC + t*1024.
        g.mov(8, rpC.ud(0)(2), rOffC.ud(0)(1));
        g.mov(8, rpC.ud(1)(2), 0u);
        g.mov(8, GRF(ptrC(t) + 1).ud(0)(2), rOffC.ud(8)(1));
        g.mov(8, GRF(ptrC(t) + 1).ud(1)(2), 0u);

        g.add(8, rpA.uq(0)(1), rpA.uq(0)(1), rAddrA.uq(0)(0));
        g.add(8, GRF(ptrA(t) + 1).uq(0)(1), GRF(ptrA(t) + 1).uq(0)(1),
                rAddrA.uq(0)(0));
        g.add(8, rpC.uq(0)(1), rpC.uq(0)(1), rAddrC.uq(0)(0));
        g.add(8, GRF(ptrC(t) + 1).uq(0)(1), GRF(ptrC(t) + 1).uq(0)(1),
                rAddrC.uq(0)(0));

        // Advance the scalar bases to the next tile (A: 512B, C: 1024B).
        g.add(1, rAddrA.uq(0), rAddrA.uq(0), uint32_t(kABytes));
        g.add(1, rAddrC.uq(0), rAddrC.uq(0), uint32_t(kCBytes));
    }
}

// Loads A for all kTiles tiles (each 8 GRFs).
template <typename Generator>
void loadATiles(Generator &g) {
    using namespace ngen;
    for (int t = 0; t < kTiles; t++) {
        GRF rPtrA(ptrA(t));
        for (int j = 0; j < 4; j++)
            g.load(16, GRF(aBase() + t * 4 + j), scattered(DataSizeLSC::D32, 1),
                    g.A64, rPtrA + 4 * j);
    }
}

// Stores C for all kTiles tiles (each 16 GRFs s32).
template <typename Generator>
void storeCTiles(Generator &g) {
    using namespace ngen;
    for (int t = 0; t < kTiles; t++) {
        GRF rPtrC(ptrC(t));
        for (int n = 0; n < 16; n++)
            g.store(16, scattered(DataSizeLSC::D32, 1), g.A64, rPtrC + 4 * n,
                    GRF(cBase() + t * 16 + n));
    }
}

// Kernel body: C += A*B, iters times, over kTiles independent tiles.
template <typename Generator>
void dpasKernelBody(Generator &g, ngen::Subregister argA,
        ngen::Subregister argB, ngen::Subregister argC,
        ngen::Subregister argIters, ngen::Subregister gid) {
    using namespace ngen;
    GRF rAddrB(kAddrB), rIt(kIt);

    setupTileAddresses(g, argA, argB, argC, gid);

    // Load packed B once (shared by all tiles): 2 N-halves x 4 GRFs (s8).
    for (int h = 0; h < 2; h++)
        for (int j = 0; j < 4; j++)
            g.load(1, GRF(bBase() + 4 * h + j), block(DataSizeLSC::D32, 16), g.A64,
                    rAddrB + (h * 256 + j * 64));

    loadATiles(g);

    // Zero the accumulators.
    for (int i = 0; i < 16 * kTiles; i++)
        g.mov(16, GRF(cBase() + i).d(), 0);

    // Accumulation loop. Per iteration, all tiles' dpas issue; the 2*kTiles
    // chains are independent (different C regs), hiding dpas latency.
    g.mov(1, rIt.ud(0), argIters);
    Label lTop;
    g.mark(lTop);
    for (int t = 0; t < kTiles; t++)
        for (int h = 0; h < 2; h++) // N halves -> dst 8 GRFs apart
            g.dpas(16, 8, 8, GRF(cBase() + t * 16 + 8 * h).d(),
                    GRF(cBase() + t * 16 + 8 * h).d(),
                    GRF(aBase() + t * 4).s4(), GRF(bBase() + 4 * h).b());
    g.add(1, rIt.d(0), rIt.d(0), -1);
    g.cmp(1 | g.gt | g.f0[0], rIt.d(0), 0);
    g.jmpi(1 | g.f0[0], lTop);

    storeCTiles(g);
}

} // namespace w4a8
} // namespace ngen_lab

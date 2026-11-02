// ngen_lab: shared nGEN kernel-body authors for a fixed-shape BF16 16x16x32 matmul.
//
//   C[16x16] (fp32) += A[16x32] (bf16, row-major) * B[32x16] (bf16)
//
// This header is compiled twice:
//   - against the binary generator (ngen::LevelZeroCodeGenerator<HW::Xe2>) in ngen_dpas_8x8_matmul_bf16_16x16x32.cpp
//   - against the text generator  (ngen::AsmCodeGenerator, NGEN_ASM)  in asmdump TU
// It therefore uses only the generator-agnostic instruction subset and takes all
// interface-dependent values (argument registers, group id) as parameters.
//
// DPAS register layout (Xe2, 64B GRFs, SIMD16, lanes = M):
//   A operand (src1, per-lane):   GRF j, lane m  = {A[m][2j], A[m][2j+1]} (lo,hi)
//                                 -> loaded with scattered D32x8 messages (lane m
//                                    address = row m; lane m data dword = column,
//                                    and scattered data is stored column-per-GRF
//                                    in global order -> GRF j = dword j of every
//                                    row = exact k-pair layout). This matches the
//                                    oneDNN DSL, which never uses transposed
//                                    block loads here (only scattered or 2D).
//   B operand (src2, broadcast):  piece (kchunk c, nhalf h), dword r*8+j
//                                 = {B[16c+2j][8h+r], B[16c+2j+1][8h+r]}
//                                 -> produced on the host by the B packer;
//                                    loaded as contiguous 64B chunks.
//   C accumulator (dst/src0):     GRF n, lane m = C[m][n] (fp32)
//                                 -> stored with two scattered D32x8 messages
//                                    (lane m address = row m half); the scattered
//                                    data layout (GRF j = element j across lanes)
//                                    matches the column-per-GRF C layout, and the
//                                    dense row-major buffer layout for the host.
// One dpas(16 lanes, sdepth 8, rcount 8) computes C[16 lanes x 8 cols] += K=16;
// the full 16x16x32 tile needs 4 dpas instructions (2 K chunks x 2 N halves).
//
// Integer arithmetic never mixes widths in one instruction (ngen_emulation.hpp
// emov ground truth): uw->ud and ud->uq widening is done with same-type strided
// movs (low subword copy + zeroed high subword).

#pragma once

#include <cstddef>
#include <cstdint>

namespace ngen_lab {

// Fixed matmul shape.
inline constexpr int kM = 16;
inline constexpr int kN = 16;
inline constexpr int kK = 32;
inline constexpr size_t kABytes = kM * kK * 2; // 1 KiB, row-major bf16
inline constexpr size_t kBBytes = kK * kN * 2; // 1 KiB natural / packed
inline constexpr size_t kCBytes = kM * kN * 4; // 1 KiB, row-major fp32

// ---- register maps (base GRF indices) --------------------------------------
namespace regs {
// Common (both kernels): scalar addresses, temps, lane-offset/address vectors.
inline constexpr int kAddrA = 96;   // scalar uq: per-tile A base
inline constexpr int kAddrB = 97;   // scalar uq: B base
inline constexpr int kAddrC = 98;   // scalar uq: per-tile C base
inline constexpr int kTmp = 99;     // scalar temp
inline constexpr int kIt = 100;     // loop counter
inline constexpr int kIdx = 101;    // lane indices (uw)
inline constexpr int kOff = 102;    // lane byte offsets 64*m (ud)
inline constexpr int kPtrA = 104;   // 2 GRFs: A row address vector (uq)
inline constexpr int kPtrC = 106;   // 2 GRFs: C row address vector (uq)
// DPAS kernel
inline constexpr int kDpasA = 16;   // 16 GRFs
inline constexpr int kDpasB = 32;   // 16 GRFs (4 packed pieces)
inline constexpr int kDpasC = 48;   // 16 GRFs
// Scalar kernel
inline constexpr int kScARaw = 16;  // 16 GRFs, bf16 pairs as loaded
inline constexpr int kScAUnp = 32;  // 32 GRFs, unpacked fp32
inline constexpr int kScAcc = 64;   // 16 GRFs, fp32 accumulators
inline constexpr int kScBk = 80;    // 1 GRF, current B row (8 dwords, exec-1 view)
inline constexpr int kScT = 81;     // 1 GRF, scalar temp
inline constexpr int kScTmp = 82;   // 1 GRF, 16-lane product temp
} // namespace regs

// ---- shared setup helpers --------------------------------------------------
// Builds everything the tile kernels need from the payload values:
//   - scalar per-tile base addresses (gid * tileBytes added with a proper
//     64-bit add: offset zero-extended to uq first)
//   - A/C per-lane row address vectors (lane m -> base + 64*m), built with
//     the proven same-type widening idiom (uw->ud->uq via strided movs).
template <typename Generator>
void setupTileAddresses(Generator &g, ngen::Subregister argA,
        ngen::Subregister argB, ngen::Subregister argC, ngen::Subregister gid) {
    using namespace ngen;
    GRF rAddrA(regs::kAddrA), rAddrB(regs::kAddrB), rAddrC(regs::kAddrC);
    GRF rTmp(regs::kTmp);
    GRF rIdx(regs::kIdx), rOff(regs::kOff);
    GRF rPtrA(regs::kPtrA), rPtrC(regs::kPtrC);

    g.mov(1, rAddrA.uq(0), argA);
    g.mov(1, rAddrB.uq(0), argB);
    g.mov(1, rAddrC.uq(0), argC);

    // Per-tile offsets (A and C tiles are both 1 KiB): 64-bit safe add.
    g.mul(1, rTmp.ud(0), gid, uint32_t(kABytes));
    g.mov(1, rTmp.ud(1), 0u);
    g.add(1, rAddrA.uq(0), rAddrA.uq(0), rTmp.uq(0));
    g.add(1, rAddrC.uq(0), rAddrC.uq(0), rTmp.uq(0));

    // Lane byte offsets 64*m: uv -> uw -> (strided movs) -> ud -> shl 6.
    g.mov(8, rIdx.uw(0)(1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
    g.mov(8, rIdx.uw(8)(1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
    g.mov(16, rOff.uw(0)(2), rIdx.uw(0)(1)); // uw->ud: low words
    g.mov(16, rOff.uw(1)(2), uint16_t(0));   //         zero high words
    g.shl(16, rOff.ud(0)(1), rOff.ud(0)(1), 6);

    // Widen to uq (strided movs) and add the scalar bases (broadcast).
    // NOTE: GRF(104)[1] would be a subregister of the same GRF, not GRF(105);
    // build the high-half GRFs explicitly (GRF(104+1) etc.).
    const int aBase = rPtrA.getBase(), cBase = rPtrC.getBase();
    g.mov(8, GRF(aBase).ud(0)(2), rOff.ud(0)(1));   // lanes 0-7 low
    g.mov(8, GRF(aBase).ud(1)(2), 0u);              //         high
    g.mov(8, GRF(aBase + 1).ud(0)(2), rOff.ud(8)(1)); // lanes 8-15 low
    g.mov(8, GRF(aBase + 1).ud(1)(2), 0u);          //         high
    g.mov(8, GRF(cBase).ud(0)(2), rOff.ud(0)(1));
    g.mov(8, GRF(cBase).ud(1)(2), 0u);
    g.mov(8, GRF(cBase + 1).ud(0)(2), rOff.ud(8)(1));
    g.mov(8, GRF(cBase + 1).ud(1)(2), 0u);
    g.add(8, GRF(aBase).uq(0)(1), GRF(aBase).uq(0)(1), rAddrA.uq(0)(0));
    g.add(8, GRF(aBase + 1).uq(0)(1), GRF(aBase + 1).uq(0)(1), rAddrA.uq(0)(0));
    g.add(8, GRF(cBase).uq(0)(1), GRF(cBase).uq(0)(1), rAddrC.uq(0)(0));
    g.add(8, GRF(cBase + 1).uq(0)(1), GRF(cBase + 1).uq(0)(1), rAddrC.uq(0)(0));
}

// Loads the A tile into DPAS src1 layout with the proven vcount==1 scattered
// pattern (the same semantics verified in l0_min stage 3): per lane, one
// element from that lane's address.
//   - GRF dstBase+j must hold dword column j for all 16 rows (the DPAS k-pair
//     layout: lane m's k-pair j += A[m][2j], A[m][2j+1]).
// So send j loads GRF dstBase+j from address vector rPtrA + 4*j (column j).
// 16 sends, one per A GRF (dstBase..dstBase+15).
template <typename Generator>
void loadATile(Generator &g, int dstBase) {
    using namespace ngen;
    GRF rPtrA(regs::kPtrA);
    for (int j = 0; j < 16; j++)
        g.load(16, GRF(dstBase + j), scattered(DataSizeLSC::D32, 1), g.A64,
                rPtrA + 4 * j);
}

// Stores the C tile row-major with the proven vcount==1 scattered pattern.
// GRF srcBase+n = C column n for all 16 rows; storing column n goes to lane m's
// row address + 4n. 16 stores, one per C GRF (srcBase..srcBase+15).
template <typename Generator>
void storeCTile(Generator &g, int srcBase) {
    using namespace ngen;
    GRF rPtrC(regs::kPtrC);
    for (int n = 0; n < 16; n++)
        g.store(16, scattered(DataSizeLSC::D32, 1), g.A64, rPtrC + 4 * n,
                GRF(srcBase + n));
}

// ---- kernel bodies ---------------------------------------------------------
//   argA/argB/argC : uq subregisters holding device pointers from the payload.
//   argIters       : ud subregister, number of accumulate iterations.
//   gid            : ud subregister with the work-group id in dimension 0.
template <typename Generator>
void dpasKernelBody(Generator &g, ngen::Subregister argA, ngen::Subregister argB,
        ngen::Subregister argC, ngen::Subregister argIters,
        ngen::Subregister gid) {
    using namespace ngen;
    GRF rAddrB(regs::kAddrB), rIt(regs::kIt);

    setupTileAddresses(g, argA, argB, argC, gid);

    // Load packed B weight: 4 broadcast pieces of 256B each. Each piece fills 4
    // GRFs; load it as 4 transposed-block D32x16 loads (each 64B -> 1 GRF in
    // natural dword order). This is the Xe2 LSC new-dataport block path
    // (legacy block_hword/block_oword throw on Xe2).
    for (int p = 0; p < 4; p++)
        for (int j = 0; j < 4; j++)
            g.load(1, GRF(regs::kDpasB + 4 * p + j),
                    block(DataSizeLSC::D32, 16), g.A64,
                    rAddrB + p * 256 + j * 64);

    loadATile(g, regs::kDpasA);

    // Zero the accumulator.
    for (int i = 0; i < 16; i++)
        g.mov(16, GRF(regs::kDpasC + i).f(), 0.f);

    // Accumulation loop: C += A * B, `iters` times.
    g.mov(1, rIt.ud(0), argIters);
    Label lTop;
    g.mark(lTop);
    for (int h = 0; h < 2; h++)     // N halves -> dst 8 GRFs apart
        for (int c = 0; c < 2; c++) // K chunks -> A 8 GRFs, B 4 GRFs apart
            g.dpas(16, 8, 8, GRF(regs::kDpasC + 8 * h).f(),
                    GRF(regs::kDpasC + 8 * h).f(),
                    GRF(regs::kDpasA + 8 * c).bf(),
                    GRF(regs::kDpasB + 4 * (2 * c + h)).bf());
    g.add(1, rIt.d(0), rIt.d(0), -1);
    g.cmp(1 | g.gt | g.f0[0], rIt.d(0), 0);
    g.jmpi(1 | g.f0[0], lTop);

    storeCTile(g, regs::kDpasC);
}

// Scalar variant: identical math expressed with fp32 mad instructions only.
// Lane m owns C row m (16 fp32 accumulators). A is unpacked bf16->fp32 once
// (bf16 -> fp32 is a 16-bit shift / mask). B rows are streamed one per K step
// with an exec-1 transposed-block load and broadcast to all lanes per element.
template <typename Generator>
void scalarKernelBody(Generator &g, ngen::Subregister argA,
        ngen::Subregister argB, ngen::Subregister argC,
        ngen::Subregister argIters, ngen::Subregister gid) {
    using namespace ngen;
    GRF rAddrB(regs::kAddrB), rIt(regs::kIt);
    GRF rBk(regs::kScBk), rT(regs::kScT), rTmp(regs::kScTmp);

    setupTileAddresses(g, argA, argB, argC, gid);

    // Load A tile (row-major bf16) and unpack to fp32 in registers.
    loadATile(g, regs::kScARaw);
    for (int j = 0; j < 16; j++) {
        g.shl(16, GRF(regs::kScAUnp + 2 * j).ud(), GRF(regs::kScARaw + j).ud(),
                16);
        g.and_(16, GRF(regs::kScAUnp + 2 * j + 1).ud(),
                GRF(regs::kScARaw + j).ud(), 0xFFFF0000u);
    }

    for (int i = 0; i < 16; i++)
        g.mov(16, GRF(regs::kScAcc + i).f(), 0.f);

    g.mov(1, rIt.ud(0), argIters);
    Label lTop;
    g.mark(lTop);
    for (int k = 0; k < kK; k++) {
        // Load B row k (16 bf16 = 8 dwords = 32B) as a contiguous block into 1 GRF.
        g.load(1, rBk, block(DataSizeLSC::D32, 8), g.A64, rAddrB + k * 32);
        for (int n = 0; n < kN; n++) {
            if ((n & 1) == 0)
                g.shl(1, rT.ud(0), rBk.ud(n / 2), 16);
            else
                g.and_(1, rT.ud(0), rBk.ud(n / 2), 0xFFFF0000u);
            auto bBcast = rT.f(0);
            bBcast.setRegion(0, 1, 0); // broadcast scalar to all 16 lanes
            auto aRow = GRF(regs::kScAUnp + k).f();
            aRow.setRegion(1, 16, 1); // lane m reads row m's A[m][k]
            auto tmp = rTmp.f();
            tmp.setRegion(1, 16, 1);
            // Ternary mad's only vector operand is src2 (the addend); src0/src1
            // are forced scalar by the HW (oneDNN fixup_ternary_rgn). Use
            // mul+add (binary ops, full vector support) instead: acc += A*B.
            g.mul(16, tmp, aRow, bBcast);
            auto acc = GRF(regs::kScAcc + n).f();
            acc.setRegion(1, 16, 1);
            g.add(16, acc, acc, tmp);
        }
    }
    g.add(1, rIt.d(0), rIt.d(0), -1);
    g.cmp(1 | g.gt | g.f0[0], rIt.d(0), 0);
    g.jmpi(1 | g.f0[0], lTop);

    storeCTile(g, regs::kScAcc);
}

} // namespace ngen_lab

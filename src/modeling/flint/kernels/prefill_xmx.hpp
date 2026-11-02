#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include "../weights.hpp"

// XMX (matrix engine) INT8 prefill GEMM.
//
// Same math as prefill::gemm_q4<true>(), but the integer dot product runs on
// DPAS instead of the vector pipe:
//
//   C[t][r] = sum_g ( s[r,g]*ai(t,r,g) + b[r,g]*xi(t,g) ) * xs[t,g]
//   ai(t,r,g) = sum_{k in g} xq[t,k] * code[r,k]     (int32, from DPAS)
//   xi(t,g)   = sum_{k in g} xq[t,k]                 (int32, precomputed)
//
// code[r,k] is the raw 0..15 MLX affine nibble (value = s*code + b), i.e. the
// exact algebra gemm_q4<true>() implements.  Two facts make that layout usable
// with zero unpacking:
//   * an s8 x u4 DPAS atom is M=RC(1..8) x N=16 x K=32 and its B (weight)
//     fragment for a 32-wide k slab is exactly the packed 16 rows x 16 bytes,
//     dword index = wd*16 + n_in (see the "DPAS B-fragment order" block
//     below).  repack_q4_dpas() puts every tensor in that order once at load
//     time, so the GEMM fetches a fragment with one contiguous 256-byte
//     block_load instead of 64 scalar dword gathers.  u4 (not s4) is the
//     right operand type: the signed reading would be short by 8*s*xi per
//     group.
//   * the A (activation) fragment is horizontally packed, byte index = m*32+k,
//     so each lane's 16 bytes are 16 consecutive k of one token: one 16B
//     gather straight out of the row-major int8 quantized activations.
//
// One quantization group is 64 k-elements = 2 DPAS; the int32 DPAS results of
// both halves are folded into a float accumulator with the group scale/bias.
namespace flint {
namespace prefill_xmx {

inline constexpr int kSg = 16;   // DPAS execution size (N of the atom), lanes
inline constexpr int kRC = 8;    // DPAS repeat count = tokens per atom
inline constexpr int kRows = 16; // output rows per atom
// An ESIMD work-item is one hardware thread running the whole explicit-vector
// program (not one SIMD lane), so one thread owns one 16-token x 16-row tile
// and the work-group is just the tile grid: 8x8 threads = 64 threads, which is
// this driver's per-work-group thread cap.  Launching kSg threads per tile
// instead runs the tile 16 times and measures ~16x too little throughput.
inline constexpr int kSubM = 8;  // tile columns (token blocks) per work-group
inline constexpr int kSubN = 8;  // tile rows (row blocks) per work-group
inline constexpr int kTPB = 2;   // 8-token blocks per thread
inline constexpr int kBM = kSubM * kRC * kTPB;  // tokens per work-group tile
inline constexpr int kBN = kSubN * kRows;       // rows per work-group tile
inline constexpr int kThreads = kSubM * kSubN;

inline constexpr int kAFrag = kRC * 32;        // int8 A fragment (32 k x 8 tok)
inline constexpr int kBFrag = 32 * kRows / 2;  // u4 B fragment (32 k x 16 row)
inline constexpr int kRes = kRC * kRows;       // int32 accumulator per atom

// ---------------------------------------------------------------------------
// Weight layout: DPAS B-fragment order.
//
// For a tensor with N rows of K nibbles (K/2 packed bytes per row), the dwords
// of a row are permuted so that the 256-byte B fragment of every (row block, k
// slab) is one contiguous block:
//
//   rowblock = n / 16        n_in = n % 16
//   j        = dword within the row (byte offset 4*j), j = 0 .. K/8-1
//   slab     = j / 4         (16 bytes = 32 nibbles = one DPAS K-half)
//   wd       = j % 4
//   new      = ((rowblock * (K/32)) + slab) * 64 + wd * 16 + n_in
//
// This is a byte-identical permutation (same total size), so the GEMM turns
// the 64-dword fragment gather into a single block_load<int, 64>.
inline constexpr int kSlabDwords = 64;  // dwords of one B fragment
inline constexpr int kSlabSlabs = 16;   // slabs staged per work-group

// Repack one 4-bit tensor (all experts) from row-major into the order above.
// `tmp` must hold stride_w * experts bytes: the permutation moves bytes across
// rows, so it is staged through it.  Caller checks K%64, N%16, alignment and
// drains the queue once after the last tensor.
inline void repack_q4_dpas(sycl::queue& q, const QTensor& t, uint8_t* tmp) {
    const int K = t.k;
    const int n_experts = t.experts > 0 ? t.experts : 1;
    const int slabs = K / 32;
    const int rbs = t.n / 16;
    const size_t slice = t.experts > 0 ? t.stride_w : (size_t)t.n * (K / 2);
    const int chunks = (slabs + kSlabSlabs - 1) / kSlabSlabs;
    constexpr int kWG = 256;
    // Padded SLM row stride (65 dwords): keeps the transposed read
    // conflict-free across the banks.
    constexpr int kStride = kSlabSlabs * 4 + 1;
    for (int e = 0; e < n_experts; ++e) {
        uint8_t* dst = tmp;
        const uint8_t* src = t.w + (size_t)e * slice;
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> slm(kSlabSlabs * kStride, h);
            h.parallel_for(
                sycl::nd_range<1>((size_t)rbs * chunks * kWG, kWG),
                [=](sycl::nd_item<1> it) {
                    const int g = (int)it.get_group(0);
                    const int rb = g / chunks;
                    const int s0 = (g % chunks) * kSlabSlabs;
                    const int ns = sycl::min(kSlabSlabs, slabs - s0);
                    const int tid = (int)it.get_local_id(0);
                    const auto s32 = reinterpret_cast<const sycl::uint4*>(src);
                    auto d32 = reinterpret_cast<sycl::uint4*>(dst);
                    // Coalesced 16-byte stage-in: `ns` slabs of each of the 16
                    // rows of this row block are contiguous in the source row.
                    const sycl::uint4* rowbase = s32 + (size_t)rb * 16 * (K / 32);
                    const int nq_in = 16 * ns;
                    for (int i = tid; i < nq_in; i += kWG) {
                        const int row = i / ns, q4i = i % ns;
                        const sycl::uint4 v =
                            rowbase[(size_t)row * (K / 32) + s0 + q4i];
                        // SLM row stride is kStride dwords (65): 16-byte
                        // writes only stay aligned when q4i*4 fits, so the
                        // staging rows are written as 4 dwords.
                        uint32_t* r = &slm[row * kStride + q4i * 4];
                        r[0] = v.x(); r[1] = v.y(); r[2] = v.z(); r[3] = v.w();
                    }
                    sycl::group_barrier(it.get_group());
                    // Coalesced 16-byte write-out: slab-major, wd*16 + n_in.
                    const int nq_out = ns * 16;
                    for (int i = tid; i < nq_out; i += kWG) {
                        const int sl = i / 16, q4i = i % 16;
                        const int wd = q4i / 4, n0 = (q4i % 4) * 4;
                        sycl::uint4 v;
                        v.x() = slm[(n0 + 0) * kStride + sl * 4 + wd];
                        v.y() = slm[(n0 + 1) * kStride + sl * 4 + wd];
                        v.z() = slm[(n0 + 2) * kStride + sl * 4 + wd];
                        v.w() = slm[(n0 + 3) * kStride + sl * 4 + wd];
                        d32[(size_t)(rb * slabs + s0 + sl) * (kSlabDwords / 4) +
                            q4i] = v;
                    }
                });
        });
        // In-place: copy the repacked slice back over the original.
        q.memcpy(const_cast<uint8_t*>(src), tmp, slice);
    }
}

// xi[row][g] = sum of the 64 int8 activations of group g.  The GEMM reads this
// once per (token, group); it is never recomputed per tile.
inline void group_sums(sycl::queue& q, const int8_t* xq, int* xi, int rows,
                       int K) {
    const int G = K / 64;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>((size_t)rows * G * 32, 32),
                       [=](sycl::nd_item<1> it) {
                           const int grp = (int)it.get_group(0);
                           const int row = grp / G;
                           const int g = grp % G;
                           const auto sg = it.get_sub_group();
                           const int lane = (int)sg.get_local_id()[0];
                           int acc = 0;
                           for (int i = g * 64 + lane; i < g * 64 + 64; i += 32)
                               acc += (int)xq[(size_t)row * K + i];
                           acc = sycl::reduce_over_group(sg, acc,
                                                         sycl::plus<int>());
                           if (lane == 0) xi[(size_t)row * G + g] = acc;
                       });
    });
}

// Returns false (nothing enqueued) when the shape or an alignment does not fit
// the XMX tile; callers fall back to prefill::gemm_q4<true>().
inline bool gemm_q4_xmx(sycl::queue& q, const QTensor& wm, const int8_t* Aq,
                        const float* xs, const int* xi, const int* row_index,
                        bf16* C, int M, int N, const bf16* residual = nullptr,
                        const int* out_index = nullptr, int out_stride = 0,
                        int expert = 0) {
    if (out_stride == 0) out_stride = N;
    const int K = wm.k;
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (K % 64 != 0 || N % kRows != 0 || (out_stride & 3) != 0) return false;
    if (!Aq || !xs || !xi || !wm.valid() || !wm.s || !wm.b) return false;
    if ((reinterpret_cast<uintptr_t>(Aq) & 15) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(C) & 3) != 0) return false;
    const size_t ewoff = (size_t)expert * wm.stride_w;
    const size_t esoff = (size_t)expert * wm.stride_s;
    if ((reinterpret_cast<uintptr_t>(wm.w + ewoff) & 15) != 0) return false;

    const int G = K / 64;
    const int tiles_m = (M + kBM - 1) / kBM;
    const int tiles_n = (N + kBN - 1) / kBN;

    const int32_t* wbase = reinterpret_cast<const int32_t*>(wm.w + ewoff);
    const int64_t* Aq64 = reinterpret_cast<const int64_t*>(Aq);
    int64_t* C64 = reinterpret_cast<int64_t*>(C);
    const int16_t* s16 = reinterpret_cast<const int16_t*>(wm.s + esoff);
    const int16_t* b16 = reinterpret_cast<const int16_t*>(wm.b + esoff);

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)tiles_m * tiles_n * kThreads, kThreads),
            sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_size<16>},
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                namespace esimd = sycl::ext::intel::esimd;
                using dpas_t = sycl::ext::intel::esimd::xmx::dpas_argument_type;

                const int tid = (int)it.get_local_id(0);
                const int sgid = tid;
                const int sub_m = sgid / kSubN;  // token block
                const int sub_n = sgid % kSubN;  // row block
                const int wg = (int)it.get_group(0);
                const int tm = wg / tiles_n;
                const int tn = wg % tiles_n;
                const int tbase = tm * kBM + sub_m * kRC * kTPB;
                const int r0 = tn * kBN + sub_n * kRows;
                if (r0 >= N || tbase >= M) return;
                const int r0c = r0;

                const uint32_t mlast = (uint32_t)(M - 1);

                // Scales/bias depend on the row only, so they are one element
                // per lane with lane j <-> row j: the row layout the fold uses
                // (an int32-token accumulator view is select<16,1>(16*t) ->
                // element j = row j of token t, also one element per lane).
                const esimd::simd<uint32_t, kRows> eR =
                    esimd::simd<uint32_t, kRows>(0, 1);
                const esimd::simd<uint32_t, kRows> soff =
                    ((uint32_t)r0c + eR) * (uint32_t)(G * sizeof(bf16));

                // A (activations) / output: element i of a 32-wide int64
                // operand is 8 bytes of token tbase + tb*8 + i/4 (2 per lane).
                esimd::simd<uint32_t, kAFrag / 8> aoff[kTPB];
                esimd::simd<uint32_t, kRes / 4> coff[kTPB];
                esimd::simd_mask<kAFrag / 8> live[kTPB];
                // Per-token activation scale/sum pointers: loop invariant, only
                // the group offset (+g*4 bytes) changes inside the K loop.
                const float* xsp[kTPB][kRC];
                const int* xip[kTPB][kRC];
                const esimd::simd<uint32_t, kAFrag / 8> e8 =
                    esimd::simd<uint32_t, kAFrag / 8>(0, 1);
#pragma unroll
                for (int tb = 0; tb < kTPB; ++tb) {
                    const esimd::simd<uint32_t, kAFrag / 8> rawtok =
                        tbase + tb * kRC + e8 / 4u;
                    const esimd::simd<uint32_t, kAFrag / 8> tok =
                        esimd::min(rawtok, mlast);
                    const esimd::simd<uint32_t, kAFrag / 8> crow =
                        (uint32_t)r0c + 8u * ((e8 / 2u) & 1u) + 4u * (e8 % 2u);
                    // Element i is an 8-byte chunk: live only if its token and
                    // its row are inside the problem.
                    live[tb] = (rawtok < (uint32_t)M) & (crow < (uint32_t)N);
                    esimd::simd<uint32_t, kAFrag / 8> arow;
                    esimd::simd<uint32_t, kAFrag / 8> orow;
                    if (row_index) {
                        esimd::simd<int, kAFrag / 8> r =
                            esimd::gather<int, kAFrag / 8>(row_index, tok * 4u);
                        arow = esimd::convert<uint32_t>(r);
                    } else {
                        arow = tok;
                    }
                    if (out_index) {
                        esimd::simd<int, kAFrag / 8> r =
                            esimd::gather<int, kAFrag / 8>(out_index, tok * 4u);
                        orow = esimd::convert<uint32_t>(r);
                    } else {
                        orow = tok;
                    }
                    // A fragment byte index = m*32 + k: element i of the 32-wide
                    // operand is bytes 16*l..16*l+15 of token l/2, i.e. 2 dwords
                    // at k = 16*(l&1) and 16*(l&1)+8.
                    aoff[tb] = arow * (uint32_t)K + 16u * ((e8 / 2u) & 1u) +
                               8u * (e8 % 2u);
                    // lane l stores rows r0 + 8*(l&1) .. +7 of token l/2, 8 bytes
                    // per int64 element (element i is the upper half when i odd).
                    coff[tb] = (orow * (uint32_t)out_stride + crow) * 2u;
                    // One scalar (uniform load) per token instead of a
                    // 128-element gather with 8 copies per lane.
#pragma unroll
                    for (int t = 0; t < kRC; ++t) {
                        const int raw = tbase + tb * kRC + t;
                        const uint32_t xtok =
                            (uint32_t)(raw < (int)mlast ? raw : (int)mlast);
                        const uint32_t xrow =
                            row_index ? (uint32_t)row_index[xtok] : xtok;
                        xsp[tb][t] = xs + (size_t)xrow * (size_t)G;
                        xip[tb][t] = xi + (size_t)xrow * (size_t)G;
                    }
                }

                // One 16-row (one element per lane) float accumulator per
                // (token block, token): lane j holds row j of that token.
                esimd::simd<float, kRows> facc[kTPB][kRC];
#pragma unroll
                for (int tb = 0; tb < kTPB; ++tb)
#pragma unroll
                    for (int t = 0; t < kRC; ++t) facc[tb][t] = 0.0f;

                for (int g = 0; g < G; ++g) {
                    esimd::simd<int, kRes> ai[kTPB];
#pragma unroll
                    for (int tb = 0; tb < kTPB; ++tb) ai[tb] = 0;
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        // B (weights) does not depend on the token block, so it
                        // is loaded once and feeds every DPAS of this half: one
                        // contiguous 256-byte block load out of the repacked
                        // weights (fragment = slab (g*2+half) of row block r0c).
                        const uint32_t boff_dwords =
                            (((uint32_t)r0c / (uint32_t)kRows) *
                                 (uint32_t)(K / 32) +
                             (uint32_t)(g * 2 + half)) *
                            (uint32_t)kSlabDwords;
                        esimd::simd<int, kBFrag / 4> braw =
                            esimd::block_load<int, kBFrag / 4>(wbase,
                                                               boff_dwords * sizeof(int));
                        const esimd::simd<uint8_t, kBFrag> b8 =
                            braw.template bit_cast_view<uint8_t>();
#pragma unroll
                        for (int tb = 0; tb < kTPB; ++tb) {
                            esimd::simd<int64_t, kAFrag / 8> araw =
                                esimd::gather<int64_t, kAFrag / 8>(
                                    Aq64, aoff[tb] +
                                              (uint32_t)(g * 64 + half * 32));
                            esimd::simd<int8_t, kAFrag> a8 =
                                araw.template bit_cast_view<int8_t>();
                            ai[tb] = esimd::xmx::dpas<
                                8, kRC, int, int, uint8_t, int8_t, dpas_t::u4,
                                dpas_t::s8, kRes, kBFrag, kAFrag>(ai[tb], b8, a8);
                        }
                    }
                    // acc[t][r] += (s[r]*ai[t][r] + b[r]*xi[t]) * xs[t]: s/b
                    // are a single uniform 16-row vector, xs/xi are scalars, so
                    // every per-group op below is plain vector arithmetic.
                    esimd::simd<int16_t, kRows> sraw =
                        esimd::gather<int16_t, kRows>(s16,
                                                      soff + (uint32_t)(g * 2));
                    esimd::simd<int16_t, kRows> braw =
                        esimd::gather<int16_t, kRows>(b16,
                                                      soff + (uint32_t)(g * 2));
                    const esimd::simd<bf16, kRows> sbf =
                        sraw.template bit_cast_view<bf16>();
                    const esimd::simd<bf16, kRows> bbf =
                        braw.template bit_cast_view<bf16>();
                    const esimd::simd<float, kRows> sv =
                        esimd::simd<float, kRows>(sbf);
                    const esimd::simd<float, kRows> bv =
                        esimd::simd<float, kRows>(bbf);
#pragma unroll
                    for (int tb = 0; tb < kTPB; ++tb) {
#pragma unroll
                        for (int t = 0; t < kRC; ++t) {
                            const float xsv = xsp[tb][t][g];
                            const float xiv = (float)xip[tb][t][g];
                            const esimd::simd<int, kRows> a16 =
                                ai[tb].template select<kRows, 1>(t * kRows);
                            const esimd::simd<float, kRows> cf =
                                esimd::convert<float>(a16);
                            facc[tb][t] =
                                facc[tb][t] + (cf * sv + xiv * bv) * xsv;
                        }
                    }
                }

#pragma unroll
                for (int tb = 0; tb < kTPB; ++tb) {
                    // Re-assemble the per-token 16-row accumulators into the
                    // DPAS result layout (element e = 16*token + row).
                    esimd::simd<float, kRes> acc;
#pragma unroll
                    for (int t = 0; t < kRC; ++t)
                        acc.template select<kRows, 1>(t * kRows) = facc[tb][t];
                    esimd::simd<bf16, kRes> out = acc;
                    if (residual) {
                        esimd::simd<int64_t, kRes / 4> rraw =
                            esimd::gather<int64_t, kRes / 4>(
                                reinterpret_cast<const int64_t*>(residual),
                                coff[tb]);
                        esimd::simd<bf16, kRes> rbf =
                            rraw.template bit_cast_view<bf16>();
                        // gemm_q4() rounds the dot to bf16, then adds.
                        out = esimd::simd<bf16, kRes>(
                            esimd::simd<float, kRes>(out) +
                            esimd::simd<float, kRes>(rbf));
                    }
                    esimd::simd<int64_t, kRes / 4> cpack =
                        out.template bit_cast_view<int64_t>();
                    esimd::scatter<int64_t, kRes / 4>(C64, coff[tb], cpack,
                                                      live[tb]);
                }
            });
    });
    return true;
}

}  // namespace prefill_xmx
}  // namespace flint

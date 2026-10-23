#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include "weights.hpp"

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
//     fragment for a 32-wide k slab is exactly the packed row bytes:
//     dword index = (k/8)*16 + n, low nibble = even k.  u4 (not s4) is the
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
// 4 sub-groups (64 work-items): ESIMD kernels are capped at 64 work-items per
// work-group on this driver, so a 16-sub-group tile is not launchable.
inline constexpr int kSubM = 2;  // sub-groups along tokens
inline constexpr int kSubN = 2;  // sub-groups along rows
inline constexpr int kTPB = 2;   // 8-token blocks per sub-group
inline constexpr int kBM = kSubM * kRC * kTPB;  // tokens per work-group tile
inline constexpr int kBN = kSubN * kRows;       // rows per work-group tile
inline constexpr int kThreads = kSubM * kSubN * kSg;

inline constexpr int kAFrag = kRC * 32;        // int8 A fragment (32 k x 8 tok)
inline constexpr int kBFrag = 32 * kRows / 2;  // u4 B fragment (32 k x 16 row)
inline constexpr int kRes = kRC * kRows;       // int32 accumulator per atom

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
    if ((reinterpret_cast<uintptr_t>(wm.w + ewoff) & 3) != 0) return false;

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
                const int sgid = tid / kSg;
                const int sub_m = sgid / kSubN;  // token block
                const int sub_n = sgid % kSubN;  // row block
                const int wg = (int)it.get_group(0);
                const int tm = wg / tiles_n;
                const int tn = wg % tiles_n;
                const int tbase = tm * kBM + sub_m * kRC * kTPB;
                const int r0 = tn * kBN + sub_n * kRows;

                const uint32_t mlast = (uint32_t)(M - 1);
                const uint32_t nlast = (uint32_t)(N - 1);

                // Scales/bias rows: element i -> row i%16 (8 rows/lane, which
                // matches the accumulator: lane l owns rows 8*(l&1)..+7).
                const esimd::simd<uint32_t, kRes> e128 =
                    esimd::simd<uint32_t, kRes>(0, 1);
                const esimd::simd<uint32_t, kRes> soff =
                    esimd::min(r0 + e128 % (uint32_t)kRows, nlast) *
                    (uint32_t)(G * sizeof(bf16));

                // B (weights): element i -> row i%16, k-slab i/16; 4 rows/lane.
                const esimd::simd<uint32_t, kBFrag / 4> eB =
                    esimd::simd<uint32_t, kBFrag / 4>(0, 1);
                const esimd::simd<uint32_t, kBFrag / 4> boff =
                    esimd::min(r0 + eB % (uint32_t)kRows, nlast) *
                        (uint32_t)(K / 2) +
                    (eB / (uint32_t)kRows) * 4u;

                // A (activations) / output: element i of a 32-wide int64
                // operand is 8 bytes of token tbase + tb*8 + i/4 (2 per lane).
                esimd::simd<uint32_t, kAFrag / 8> aoff[kTPB];
                esimd::simd<uint32_t, kRes / 4> coff[kTPB];
                esimd::simd_mask<kAFrag / 8> live[kTPB];
                esimd::simd<uint32_t, kRes> xoff[kTPB];
                const esimd::simd<uint32_t, kAFrag / 8> e8 =
                    esimd::simd<uint32_t, kAFrag / 8>(0, 1);
#pragma unroll
                for (int tb = 0; tb < kTPB; ++tb) {
                    const esimd::simd<uint32_t, kAFrag / 8> rawtok =
                        tbase + tb * kRC + e8 / 4u;
                    const esimd::simd<uint32_t, kAFrag / 8> tok =
                        esimd::min(rawtok, mlast);
                    const esimd::simd<uint32_t, kAFrag / 8> crow =
                        (uint32_t)r0 + 8u * ((e8 / 2u) & 1u) + 4u * (e8 % 2u);
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
                    const esimd::simd<uint32_t, kRes> xtok =
                        esimd::min(tbase + tb * kRC + e128 / (uint32_t)kRows,
                                   mlast);
                    esimd::simd<uint32_t, kRes> xrow;
                    if (row_index) {
                        esimd::simd<int, kRes> r =
                            esimd::gather<int, kRes>(row_index, xtok * 4u);
                        xrow = esimd::convert<uint32_t>(r);
                    } else {
                        xrow = xtok;
                    }
                    xoff[tb] = xrow * (uint32_t)(G * sizeof(float));
                }

                esimd::simd<float, kRes> facc[kTPB];
#pragma unroll
                for (int tb = 0; tb < kTPB; ++tb) facc[tb] = 0.0f;

                for (int g = 0; g < G; ++g) {
                    esimd::simd<int, kRes> ai[kTPB];
#pragma unroll
                    for (int tb = 0; tb < kTPB; ++tb) ai[tb] = 0;
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
#pragma unroll
                        for (int tb = 0; tb < kTPB; ++tb) {
                            esimd::simd<int64_t, kAFrag / 8> araw =
                                esimd::gather<int64_t, kAFrag / 8>(
                                    Aq64, aoff[tb] +
                                              (uint32_t)(g * 64 + half * 32));
                            esimd::simd<int, kBFrag / 4> braw =
                                esimd::gather<int, kBFrag / 4>(
                                    wbase,
                                    boff + (uint32_t)(g * 32 + half * 16));
                            esimd::simd<int8_t, kAFrag> a8 =
                                araw.template bit_cast_view<int8_t>();
                            esimd::simd<uint8_t, kBFrag> b8 =
                                braw.template bit_cast_view<uint8_t>();
                            ai[tb] = esimd::xmx::dpas<
                                8, kRC, int, int, uint8_t, int8_t, dpas_t::u4,
                                dpas_t::s8, kRes, kBFrag, kAFrag>(ai[tb], b8, a8);
                        }
                    }
                    // acc += (s*ai + b*xi) * xs, one saturation group at a time.
                    esimd::simd<int16_t, kRes> sraw =
                        esimd::gather<int16_t, kRes>(s16,
                                                     soff + (uint32_t)(g * 2));
                    esimd::simd<int16_t, kRes> braw =
                        esimd::gather<int16_t, kRes>(b16,
                                                     soff + (uint32_t)(g * 2));
                    esimd::simd<bf16, kRes> sbf =
                        sraw.template bit_cast_view<bf16>();
                    esimd::simd<bf16, kRes> bbf =
                        braw.template bit_cast_view<bf16>();
                    const esimd::simd<float, kRes> sv =
                        esimd::simd<float, kRes>(sbf);
                    const esimd::simd<float, kRes> bv =
                        esimd::simd<float, kRes>(bbf);
#pragma unroll
                    for (int tb = 0; tb < kTPB; ++tb) {
                        const esimd::simd<float, kRes> xsv =
                            esimd::gather<float, kRes>(
                                xs, xoff[tb] + (uint32_t)(g * 4));
                        const esimd::simd<float, kRes> xiv =
                            esimd::convert<float>(esimd::gather<int, kRes>(
                                xi, xoff[tb] + (uint32_t)(g * 4)));
                        esimd::simd<float, kRes> t =
                            esimd::convert<float>(ai[tb]) * sv;
                        t = t + xiv * bv;
                        facc[tb] = facc[tb] + t * xsv;
                    }
                }

#pragma unroll
                for (int tb = 0; tb < kTPB; ++tb) {
                    esimd::simd<bf16, kRes> out = facc[tb];
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

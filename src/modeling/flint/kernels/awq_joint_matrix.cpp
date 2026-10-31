#include "awq_joint_matrix.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace flint::awq_joint_matrix {
namespace {
namespace matrix = sycl::ext::oneapi::experimental::matrix;

// Both subgroup widths use the 8x16x16 atom, lowered to native DPAS.
// The installed IGC lacks SLM helpers for the SG32 32x64 composite shape.
// A wider subgroup does not imply DPASW.
template <int SG>
sycl::event launch(sycl::queue& q, const QTensor& t, const bf16* A,
            const int* row_index, bf16* C, int M, int N,
            const bf16* residual, const int* out_index, int out_stride, int expert) {
    constexpr int TM = 8;
    constexpr int TN = 16;
    constexpr int BM = TM * 2, BN = TN * 2, BK = 64, TK = 16;
    constexpr int threads = SG * 4;
    const int K = t.k;
    const size_t tiles_n = (size_t(N) + BN - 1) / BN;
    const size_t tiles_m = (size_t(M) + BM - 1) / BM;
    const auto* weights = t.w + size_t(expert) * t.stride_w;
    const auto* scales = t.awq_s + size_t(expert) * t.stride_s;
    const auto* zeros = t.awq_z + size_t(expert) * t.stride_b;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<bf16, 1> a(BM * BK, h), b(BK * BN, h);
        sycl::local_accessor<float, 1> c(BM * BN, h);
        h.parallel_for(sycl::nd_range<1>(tiles_m * tiles_n * threads, threads),
                       sycl::ext::oneapi::experimental::properties{
                           sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int tid = int(it.get_local_linear_id());
            const size_t tile = it.get_group_linear_id();
            const int m0 = int(tile / tiles_n) * BM;
            const int n0 = int(tile % tiles_n) * BN;
            const auto sg = it.get_sub_group();
            const int sm = int(sg.get_group_linear_id()) / 2 * TM;
            const int sn = int(sg.get_group_linear_id()) % 2 * TN;
            matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::a,
                                 TM, TK, matrix::layout::row_major> ma;
            matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::b,
                                 TK, TN, matrix::layout::ext_intel_packed> mb;
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator,
                                 TM, TN> acc;
            matrix::joint_matrix_fill(sg, acc, 0.f);
            for (int k0 = 0; k0 < K; k0 += BK) {
                for (int i = tid; i < BM * BK; i += threads) {
                    const int m = m0 + i / BK;
                    bf16 value(0.f);
                    if (m < M) {
                        const int source = row_index ? row_index[m] : m;
                        value = A[size_t(source) * K + k0 + i % BK];
                    }
                    a[i] = value;
                }
                // Each packed byte supplies a BF16 pair in VNNI order.
                // Parameters are group-major [K/64,N], independently strided
                // for each expert, matching the AWQ vector kernel.
                for (int i = tid; i < BK / 2 * BN; i += threads) {
                    const int n = n0 + i % BN;
                    const int k = k0 + i / BN * 2;
                    bf16 lo(0.f), hi(0.f);
                    if (n < N) {
                        const uint8_t packed = weights[size_t(n) * (K / 2) + k / 2];
                        const size_t param = size_t(k0 / BK) * N + n;
                        const float s = float(scales[param]);
                        const int z = int(zeros[param]);
                        lo = bf16(float(int(packed & 15) - z) * s);
                        hi = bf16(float(int(packed >> 4) - z) * s);
                    }
                    b[i * 2] = lo;
                    b[i * 2 + 1] = hi;
                }
                sycl::group_barrier(it.get_group());
                for (int kk = 0; kk < BK; kk += TK) {
                    matrix::joint_matrix_load(sg, ma,
                        a.template get_multi_ptr<sycl::access::decorated::no>() +
                            sm * BK + kk, BK);
                    matrix::joint_matrix_load(sg, mb,
                        b.template get_multi_ptr<sycl::access::decorated::no>() +
                            kk / 2 * BN * 2 + sn * 2, BN * 2);
                    matrix::joint_matrix_mad(sg, acc, ma, mb, acc);
                }
                sycl::group_barrier(it.get_group());
            }
            matrix::joint_matrix_store(sg, acc,
                c.template get_multi_ptr<sycl::access::decorated::no>() + sm * BN + sn,
                BN, matrix::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int i = tid; i < BM * BN; i += threads) {
                const int m = m0 + i / BN, n = n0 + i % BN;
                if (m < M && n < N) {
                    const int destination = out_index ? out_index[m] : m;
                    const size_t at = size_t(destination) * out_stride + n;
                    const bf16 value(c[i]);
                    C[at] = residual ? bf16(float(residual[at]) + float(value)) : value;
                }
            }
        });
    });
}

// Keep the kernel properties on the functor so both the subgroup size and
// 256-GRF allocation reach the device compiler without deprecated launch APIs.
template <int SG, typename F>
struct MatrixKernel {
    F body;
    void operator()(sycl::nd_item<1> it) const { body(it); }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::grf_size<256>,
            sycl::ext::oneapi::experimental::sub_group_size<SG>};
    }
};

template <int SG, typename F>
auto matrix_kernel(F body) { return MatrixKernel<SG, F>{body}; }

// Four subgroups; each owns RM x RN native atoms and reuses A/B fragments
// across its independent FP32 accumulators. No output staging in SLM.
template <int SG, int RM, int RN>
sycl::event launch_tiled(sycl::queue& q, const QTensor& t, const bf16* A,
                  const int* row_index, bf16* C, int M, int N,
                  const bf16* residual, const int* out_index, int out_stride, int expert) {
    constexpr int TM = 8, TN = 16;
    constexpr int BM = 2 * RM * TM, BN = 2 * RN * TN, BK = 64, TK = 16;
    constexpr int threads = SG * 4;
    const int K = t.k;
    const size_t tiles_n = (size_t(N) + BN - 1) / BN;
    const size_t tiles_m = (size_t(M) + BM - 1) / BM;
    const auto* weights = t.w + size_t(expert) * t.stride_w;
    const auto* scales = t.awq_s + size_t(expert) * t.stride_s;
    const auto* zeros = t.awq_z + size_t(expert) * t.stride_b;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<bf16, 1> a(BM * BK, h), b(BK * BN, h);
        h.parallel_for(sycl::nd_range<1>(tiles_m * tiles_n * threads, threads),
                       matrix_kernel<SG>([=](sycl::nd_item<1> it) {
            const int tid = int(it.get_local_linear_id());
            const size_t tile = it.get_group_linear_id();
            const int m0 = int(tile / tiles_n) * BM;
            const int n0 = int(tile % tiles_n) * BN;
            const auto sg = it.get_sub_group();
            const int sm = int(sg.get_group_linear_id()) / 2 * RM * TM;
            const int sn = int(sg.get_group_linear_id()) % 2 * RN * TN;
            matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::a,
                                 TM, TK, matrix::layout::row_major> ma[RM];
            matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::b,
                                 TK, TN, matrix::layout::ext_intel_packed> mb[RN];
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator,
                                 TM, TN> acc[RM][RN];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int ni = 0; ni < RN; ++ni)
                    matrix::joint_matrix_fill(sg, acc[mi][ni], 0.f);

            for (int k0 = 0; k0 < K; k0 += BK) {
                for (int i = tid; i < BM * BK; i += threads) {
                    const int m = m0 + i / BK;
                    bf16 value(0.f);
                    if (m < M) {
                        const int source = row_index ? row_index[m] : m;
                        value = A[size_t(source) * K + k0 + i % BK];
                    }
                    a[i] = value;
                }
                // A work-item loads 16 contiguous packed bytes, rather than
                // issuing a separate strided global gather for every byte.
                // Adjacent work-items own adjacent n rows: their VNNI-pair
                // writes are consecutive dwords in SLM.
                for (int i = tid; i < BN * 2; i += threads) {
                    const int nl = i % BN;
                    const int n = n0 + nl;
                    const int pair0 = i / BN * 16;
                    sycl::vec<uint8_t, 16> packed(0);
                    float scale = 0.f;
                    int zero = 0;
                    if (n < N) {
                        packed.load(0, weights + size_t(n) * (K / 2) + k0 / 2 + pair0);
                        const size_t param = size_t(k0 / BK) * N + n;
                        scale = float(scales[param]);
                        zero = int(zeros[param]);
                    }
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const int code = int(packed[j]);
                        const int dst = ((pair0 + j) * BN + nl) * 2;
                        b[dst] = bf16(float((code & 15) - zero) * scale);
                        b[dst + 1] = bf16(float((code >> 4) - zero) * scale);
                    }
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int kk = 0; kk < BK; kk += TK) {
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi)
                        matrix::joint_matrix_load(sg, ma[mi],
                            a.template get_multi_ptr<sycl::access::decorated::no>() +
                                (sm + mi * TM) * BK + kk, BK);
#pragma unroll
                    for (int ni = 0; ni < RN; ++ni)
                        matrix::joint_matrix_load(sg, mb[ni],
                            b.template get_multi_ptr<sycl::access::decorated::no>() +
                                kk / 2 * BN * 2 + (sn + ni * TN) * 2, BN * 2);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                        for (int ni = 0; ni < RN; ++ni)
                            matrix::joint_matrix_mad(sg, acc[mi][ni], ma[mi], mb[ni], acc[mi][ni]);
                }
                sycl::group_barrier(it.get_group());
            }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int ni = 0; ni < RN; ++ni)
                    sycl::ext::intel::experimental::matrix::joint_matrix_apply(
                        sg, acc[mi][ni], [&](float& x, size_t row, size_t col) {
                            const size_t m = size_t(m0 + sm + mi * TM) + row;
                            const size_t n = size_t(n0 + sn + ni * TN) + col;
                            if (m < size_t(M) && n < size_t(N)) {
                                const int dst = out_index ? out_index[m] : int(m);
                                const size_t at = size_t(dst) * out_stride + n;
                                const bf16 value(x);
                                C[at] = residual ? bf16(float(residual[at]) + float(value)) : value;
                            }
                        });
        }));
    });
}

// Packed SLM staging. GlobalA specializes aligned, full, contiguous input
// tiles to joint_matrix_load from global memory; gathered/tail tiles stage A.
template <int SG, int RM, int RN, bool GlobalA>
sycl::event launch_packed(sycl::queue& q, const QTensor& t, const bf16* A,
                         const int* row_index, bf16* C, int M, int N,
                         const bf16* residual, const int* out_index, int out_stride, int expert) {
    constexpr int TM = 8, TN = 16, BK = 64, TK = 16;
    constexpr int BM = 2 * RM * TM, BN = 2 * RN * TN, threads = SG * 4;
    const int K = t.k;
    const size_t tiles_n = (size_t(N) + BN - 1) / BN;
    const size_t tiles_m = (size_t(M) + BM - 1) / BM;
    const auto* weights = t.w + size_t(expert) * t.stride_w;
    const auto* scales = t.awq_s + size_t(expert) * t.stride_s;
    const auto* zeros = t.awq_z + size_t(expert) * t.stride_b;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint64_t, 1> aw(GlobalA ? 1 : BM * BK / 4, h);
        sycl::local_accessor<uint32_t, 1> bw(BK * BN / 2, h);
        h.parallel_for(sycl::nd_range<1>(tiles_m * tiles_n * threads, threads),
                       matrix_kernel<SG>([=](sycl::nd_item<1> it) {
            const int tid = int(it.get_local_linear_id());
            const size_t tile = it.get_group_linear_id();
            const int m0 = int(tile / tiles_n) * BM, n0 = int(tile % tiles_n) * BN;
            const auto sg = it.get_sub_group();
            const int sm = int(sg.get_group_linear_id()) / 2 * RM * TM;
            const int sn = int(sg.get_group_linear_id()) % 2 * RN * TN;
            matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::a,
                                 TM, TK, matrix::layout::row_major> ma[RM];
            matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::b,
                                 TK, TN, matrix::layout::ext_intel_packed> mb[RN];
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator,
                                 TM, TN> acc[RM][RN];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int ni = 0; ni < RN; ++ni)
                    matrix::joint_matrix_fill(sg, acc[mi][ni], 0.f);
            for (int k0 = 0; k0 < K; k0 += BK) {
                if constexpr (!GlobalA) {
                    // Four BF16 values per store, with no added alignment
                    // requirement on the original activation pointer.
                    for (int i = tid; i < BM * BK / 4; i += threads) {
                        const int m = m0 + i / (BK / 4);
                        uint64_t bits = 0;
                        if (m < M) {
                            const int source = row_index ? row_index[m] : m;
                            __builtin_memcpy(&bits, A + size_t(source) * K +
                                                   k0 + (i % (BK / 4)) * 4, sizeof(bits));
                        }
                        aw[i] = bits;
                    }
                }
                for (int i = tid; i < BN * 2; i += threads) {
                    const int nl = i % BN, n = n0 + nl, pair0 = i / BN * 16;
                    sycl::vec<uint8_t, 16> packed(0);
                    float scale = 0.f;
                    int zero = 0;
                    if (GlobalA || n < N) {
                        packed.load(0, weights + size_t(n) * (K / 2) + k0 / 2 + pair0);
                        const size_t param = size_t(k0 / BK) * N + n;
                        scale = float(scales[param]);
                        zero = int(zeros[param]);
                    }
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const int code = int(packed[j]);
                        const bf16 lo(float((code & 15) - zero) * scale);
                        const bf16 hi(float((code >> 4) - zero) * scale);
                        bw[(pair0 + j) * BN + nl] = uint32_t(sycl::bit_cast<uint16_t>(lo)) |
                            (uint32_t(sycl::bit_cast<uint16_t>(hi)) << 16);
                    }
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int kk = 0; kk < BK; kk += TK) {
                    if constexpr (GlobalA) {
                        const auto ap = sycl::address_space_cast<
                            sycl::access::address_space::global_space, sycl::access::decorated::no>(A);
#pragma unroll
                        for (int mi = 0; mi < RM; ++mi)
                            matrix::joint_matrix_load(sg, ma[mi], ap +
                                size_t(m0 + sm + mi * TM) * K + k0 + kk, K);
                    } else {
                        const auto ap = sycl::address_space_cast<
                            sycl::access::address_space::local_space, sycl::access::decorated::no>(
                                reinterpret_cast<bf16*>(aw.template get_multi_ptr<sycl::access::decorated::no>().get()));
#pragma unroll
                        for (int mi = 0; mi < RM; ++mi)
                            matrix::joint_matrix_load(sg, ma[mi], ap +
                                (sm + mi * TM) * BK + kk, BK);
                    }
                    const auto bp = sycl::address_space_cast<
                        sycl::access::address_space::local_space, sycl::access::decorated::no>(
                            reinterpret_cast<bf16*>(bw.template get_multi_ptr<sycl::access::decorated::no>().get()));
#pragma unroll
                    for (int ni = 0; ni < RN; ++ni)
                        matrix::joint_matrix_load(sg, mb[ni], bp +
                            kk / 2 * BN * 2 + (sn + ni * TN) * 2, BN * 2);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                        for (int ni = 0; ni < RN; ++ni)
                            matrix::joint_matrix_mad(sg, acc[mi][ni], ma[mi], mb[ni], acc[mi][ni]);
                }
                if (k0 + BK < K) sycl::group_barrier(it.get_group());
            }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int ni = 0; ni < RN; ++ni)
                    sycl::ext::intel::experimental::matrix::joint_matrix_apply(
                        sg, acc[mi][ni], [&](float& x, size_t row, size_t col) {
                            const size_t m = size_t(m0 + sm + mi * TM) + row;
                            const size_t n = size_t(n0 + sn + ni * TN) + col;
                            if (GlobalA || (m < size_t(M) && n < size_t(N))) {
                                const int dst = out_index ? out_index[m] : int(m);
                                const size_t at = size_t(dst) * out_stride + n;
                                const bf16 value(x);
                                C[at] = residual ? bf16(float(residual[at]) + float(value)) : value;
                            }
                        });
        }));
    });
}

template <int SG, int RM>
sycl::event submit_tiled(Staging st, bool global_a, sycl::queue& q, const QTensor& t, const bf16* A,
                        const int* rows, bf16* C, int M, int N, const bf16* residual,
                        const int* outputs, int stride, int expert) {
    if (st == Staging::scalar)
        return launch_tiled<SG, RM, 2>(q, t, A, rows, C, M, N, residual, outputs, stride, expert);
    if (global_a)
        return launch_packed<SG, RM, 2, true>(q, t, A, rows, C, M, N, residual, outputs, stride, expert);
    return launch_packed<SG, RM, 2, false>(q, t, A, rows, C, M, N, residual, outputs, stride, expert);
}

template <int SG>
sycl::event submit(Tile tiling, Staging st, bool global_a, sycl::queue& q, const QTensor& t, const bf16* A,
                   const int* rows, bf16* C, int M, int N, const bf16* residual,
                   const int* outputs, int stride, int expert) {
    switch (tiling) {
        case Tile::m16n32:
            return launch<SG>(q, t, A, rows, C, M, N, residual, outputs, stride, expert);
        case Tile::m32n64:
            return submit_tiled<SG, 2>(st, global_a, q, t, A, rows, C, M, N, residual, outputs, stride, expert);
        case Tile::m64n64:
            return submit_tiled<SG, 4>(st, global_a, q, t, A, rows, C, M, N, residual, outputs, stride, expert);
    }
    throw std::invalid_argument("flint: invalid joint_matrix tile");
}

} // namespace

Mode mode() {
    const char* value = std::getenv("ARCAINE_FLINT_AWQ_JOINT_MATRIX");
    if (!value || !std::strcmp(value, "0")) return Mode::disabled;
    if (!std::strcmp(value, "dpas")) return Mode::dpas;
    if (!std::strcmp(value, "dpas-sg32")) return Mode::dpas_sg32;
    if (!std::strcmp(value, "dpasw"))
        throw std::invalid_argument("flint: DPASW is a DG2/XeHP instruction, unavailable on Xe2; "
                                    "use dpas or dpas-sg32");
    throw std::invalid_argument("flint: ARCAINE_FLINT_AWQ_JOINT_MATRIX must be 0, dpas or dpas-sg32");
}

Tile tile() {
    const char* value = std::getenv("ARCAINE_FLINT_AWQ_JOINT_MATRIX_TILE");
    if (!value) return mode() == Mode::dpas_sg32 ? Tile::m64n64 : Tile::m32n64;
    if (!std::strcmp(value, "16x32")) return Tile::m16n32;
    if (!std::strcmp(value, "32x64")) return Tile::m32n64;
    if (!std::strcmp(value, "64x64")) return Tile::m64n64;
    throw std::invalid_argument("flint: ARCAINE_FLINT_AWQ_JOINT_MATRIX_TILE must be 16x32, 32x64 or 64x64");
}

const char* tile_name() {
    switch (tile()) {
        case Tile::m16n32: return "16x32x64";
        case Tile::m32n64: return "32x64x64";
        case Tile::m64n64: return "64x64x64";
    }
    return "";
}

Staging staging() {
    const char* value = std::getenv("ARCAINE_FLINT_AWQ_JOINT_MATRIX_STAGE");
    if (!value || !std::strcmp(value, "packed")) return Staging::packed;
    if (!std::strcmp(value, "scalar")) return Staging::scalar;
    throw std::invalid_argument("flint: ARCAINE_FLINT_AWQ_JOINT_MATRIX_STAGE must be scalar or packed");
}

const char* staging_name() {
    if (tile() == Tile::m16n32) return "original";
    switch (staging()) {
        case Staging::scalar: return "scalar";
        case Staging::packed: return "packed";
    }
    return "";
}

bool global_a_enabled() {
    const char* value = std::getenv("ARCAINE_FLINT_AWQ_JOINT_MATRIX_GLOBAL_A");
    if (!value || !std::strcmp(value, "1")) return true;
    if (!std::strcmp(value, "0")) return false;
    throw std::invalid_argument("flint: ARCAINE_FLINT_AWQ_JOINT_MATRIX_GLOBAL_A must be 0 or 1");
}

bool supported(const sycl::device& device, Mode selected) {
    if (selected == Mode::disabled || !device.is_gpu() ||
        device.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
        !device.has(sycl::aspect::ext_intel_matrix) || !device.has(sycl::aspect::fp16))
        return false;
    const size_t sg = selected == Mode::dpas ? 16 : 32;
    const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sizes.begin(), sizes.end(), sg) == sizes.end() ||
        device.get_info<sycl::info::device::max_work_group_size>() < sg * 4 ||
        device.get_info<sycl::info::device::local_mem_size>() <
            8192u) return false;
    const auto combinations = device.get_info<
        sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    for (const auto& c : combinations)
        if (c.atype == matrix::matrix_type::bf16 && c.btype == matrix::matrix_type::bf16 &&
            c.ctype == matrix::matrix_type::fp32 && c.dtype == matrix::matrix_type::fp32 &&
            (c.msize == 8 || (c.max_msize >= 8 && c.msize == 0)) &&
            c.nsize == 16 && c.ksize == 16) return true;
    return false;
}

bool gemm(sycl::queue& q, const QTensor& t, const bf16* A,
          const int* row_index, bf16* C, int M, int N,
          const bf16* residual, const int* out_index, int out_stride, int expert,
          sycl::event* completion) {
    const Mode selected = mode();
    if (selected == Mode::disabled) return false;
    const Tile tiling = tile();
    const Staging st = staging();
    const int tile_m = tiling == Tile::m32n64 ? 32 : 64;
    const bool global_a = global_a_enabled() && st == Staging::packed &&
        tiling != Tile::m16n32 && !row_index && (reinterpret_cast<uintptr_t>(A) & 63) == 0 &&
        M > 0 && N > 0 && M % tile_m == 0 && N % 64 == 0;
    if (t.format != WeightFormat::AWQ || !t.w || !t.awq_s || !t.awq_z ||
        !A || !C || A == C || M <= 0 || N <= 0 || N != t.n || t.k <= 0 || t.k % 64 ||
        M > std::numeric_limits<int>::max() - 128 ||
        N > std::numeric_limits<int>::max() - 128 ||
        expert < 0 || t.experts < 0 ||
        (t.experts == 0 ? expert != 0 : expert >= t.experts)) return false;
    if (!out_stride) out_stride = N;
    if (out_stride < N || !q.has_property<sycl::property::queue::in_order>()) return false;
    const size_t wb = size_t(N) * (t.k / 2), params = size_t(N) * (t.k / 64);
    const size_t limit = std::numeric_limits<size_t>::max();
    if (expert && (t.stride_w < wb || t.stride_s < params || t.stride_b < params ||
                   t.stride_w > (limit - wb) / size_t(expert) ||
                   t.stride_s > (limit / sizeof(sycl::half) - params) / size_t(expert) ||
                   t.stride_b > (limit - params) / size_t(expert))) return false;
    // Capability queries are invariant across calls on one device. Keep them
    // out of the timed submission path, including small expert GEMMs.
    thread_local std::optional<sycl::device> cached_device;
    thread_local bool capabilities[2] = {};
    thread_local size_t local_bytes = 0;
    const auto device = q.get_device();
    if (!cached_device || *cached_device != device) {
        capabilities[0] = supported(device, Mode::dpas);
        capabilities[1] = supported(device, Mode::dpas_sg32);
        local_bytes = device.get_info<sycl::info::device::local_mem_size>();
        cached_device = device;
    }
    if (!capabilities[selected == Mode::dpas ? 0 : 1]) return false;
    size_t slm = tiling == Tile::m16n32 ? 8192 :
                       tiling == Tile::m32n64 ? 12288 :
                       16384;
    if (global_a) slm = 8192;
    if (local_bytes < slm) return false;
    auto event = selected == Mode::dpas
        ? submit<16>(tiling, st, global_a, q, t, A, row_index, C, M, N, residual, out_index, out_stride, expert)
        : submit<32>(tiling, st, global_a, q, t, A, row_index, C, M, N, residual, out_index, out_stride, expert);
    if (completion) *completion = std::move(event);
    return true;
}
} // namespace flint::awq_joint_matrix

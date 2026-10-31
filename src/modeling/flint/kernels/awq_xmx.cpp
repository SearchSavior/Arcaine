#include "awq_xmx.hpp"
#include "awq_joint_matrix.hpp"

#if defined(SYCL_INTEL_TARGET) && defined(CUTLASS_ENABLE_SYCL)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/xe_epilogue.hpp>
#include <cutlass/epilogue/fusion/xe_callbacks.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

namespace flint::awq_xmx {
namespace detail {

using namespace cute;

struct alignas(2) F16Scale {
    uint16_t storage;

    CUTLASS_HOST_DEVICE
    friend float operator*(int value, F16Scale scale) {
        return float(value) * float(cutlass::half_t::bitcast(scale.storage));
    }
};

using Element = cutlass::bfloat16_t;
using TileShape = Shape<_16, _64, _64>;
using Subgroups = Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>;
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>, Subgroups>::TiledMMA;
using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;
using StrideB = cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>;
using StrideCD = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
using StrideSZ = Stride<_1, int64_t, int64_t>;
using Mainloop = cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopIntelXeXMX16MixedPrecision<3>, TileShape,
    Element, StrideA,
    cute::tuple<uint4_t, F16Scale, StrideSZ, uint8_t, StrideSZ>, StrideB,
    TiledMma, XE_2D_U16x16x32_LD_N, void, void, cute::identity,
    XE_2D_U4x32x16_LD_T, void, void, cute::identity>;
using EpiloguePolicy = cutlass::epilogue::IntelXeXMX16;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<float, float, float, float>;
using Callbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpiloguePolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma{}))>;
using Epilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpiloguePolicy, TileShape, float, StrideCD, Element, StrideCD, Callbacks,
    XE_2D_U32x8x16_LD_N, void, void, XE_2D_U16x8x16_ST_N, void, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>>;

static_assert(sizeof(F16Scale) == sizeof(sycl::half));
static_assert(alignof(F16Scale) == alignof(sycl::half));
static_assert(std::is_trivially_copyable_v<F16Scale>);
static_assert(std::is_same_v<decltype(int{} * F16Scale{}), float>);
static_assert(sizeof(Element) == sizeof(bf16));
static_assert(alignof(Element) == alignof(bf16));
static_assert(Mainloop::SG_N == 32 && Mainloop::MaxThreadsPerBlock == 32);
static_assert(std::is_same_v<typename Mainloop::ElementMMA, Element>);
static_assert(std::is_same_v<typename Mainloop::ElementAccumulator, float>);

constexpr size_t kMaxScratchBytes = size_t(256) << 20;
constexpr int kMaxSurface = 1 << 24;

struct Scratch {
    sycl::context context;
    sycl::device device;
    uint8_t* data = nullptr;
    size_t capacity = 0;
    sycl::event ready;
    bool pending = false;

    explicit Scratch(sycl::queue& q) : context(q.get_context()), device(q.get_device()) {}

    ~Scratch() {
        try {
            if (pending) ready.wait();
            if (data) sycl::free(data, context);
        } catch (...) {
        }
    }

    void reserve(size_t bytes, sycl::queue& q) {
        if (bytes <= capacity) return;
        if (pending) ready.wait_and_throw();
        pending = false;
        if (data) sycl::free(data, context);
        data = nullptr;
        capacity = 0;
        size_t next = 4096;
        while (next < bytes) next *= 2;
        data = static_cast<uint8_t*>(sycl::aligned_alloc_device(64, next, q));
        if (!data) throw std::bad_alloc{};
        capacity = next;
    }
};

Scratch& get_scratch(size_t bytes, sycl::queue& q) {
    thread_local std::unique_ptr<Scratch> cache;
    if (cache && (cache->context != q.get_context() || cache->device != q.get_device()))
        cache.reset();
    if (!cache) cache = std::make_unique<Scratch>(q);
    cache->reserve(bytes, q);
    return *cache;
}

bool aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 63) == 0;
}

}

bool gemm_cute(sycl::queue& q, const QTensor& t, const bf16* A,
          const int* row_index, bf16* C, int M, int N,
          const bf16* residual, const int* out_index, int out_stride, int expert) {
    using namespace detail;
    const char* enabled = std::getenv("ARCAINE_FLINT_AWQ_XMX");
    if (!enabled || enabled[0] != '1' || enabled[1] != '\0') return false;
    if (!q.has_property<sycl::property::queue::in_order>()) return false;
    const int K = t.k;
    if (t.format != WeightFormat::AWQ || !t.w || !t.awq_s || !t.awq_z || !A || !C)
        return false;
    if (M <= 0 || N <= 0 || N != t.n || K <= 0 || K % 64 || N % 16)
        return false;
    if (M > kMaxSurface || K > kMaxSurface / 2 || N > kMaxSurface / 4)
        return false;
    if (out_stride == 0) out_stride = N;
    if (out_stride < N || expert < 0 || t.experts < 0) return false;
    if ((t.experts == 0 && expert != 0) || (t.experts > 0 && expert >= t.experts))
        return false;

    const size_t weight_bytes = size_t(N) * (K / 2);
    const size_t scale_elements = size_t(N) * (K / 64);
    if ((t.stride_w && t.stride_w != weight_bytes) ||
        (t.stride_s && t.stride_s != scale_elements)) return false;
    if (expert && (!t.stride_w || !t.stride_s)) return false;
    if (size_t(expert) > std::numeric_limits<size_t>::max() / weight_bytes ||
        size_t(expert) > std::numeric_limits<size_t>::max() / scale_elements)
        return false;
    const uint8_t* weights = t.w + size_t(expert) * t.stride_w;
    const sycl::half* scales = t.awq_s + size_t(expert) * t.stride_s;
    const uint8_t* zeros = t.awq_z + size_t(expert) * t.stride_s;
    if (!aligned(weights) || !aligned(scales) || !aligned(zeros)) return false;

    const auto device = q.get_device();
    if (!device.is_gpu() || device.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
        !device.has(sycl::aspect::fp16) || !device.has(sycl::aspect::ext_intel_matrix) ||
        !device.has(sycl::aspect::usm_device_allocations) ||
        device.get_info<sycl::info::device::max_work_group_size>() < 32) return false;
    const auto subgroups = device.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(subgroups.begin(), subgroups.end(), size_t(16)) == subgroups.end())
        return false;

    const int raw_m = (M + 15) / 16 * 16;
    const size_t max_extent = std::numeric_limits<int>::max();
    if (size_t(raw_m) * K > max_extent || size_t(raw_m) * N > max_extent ||
        size_t(N) * K > max_extent) return false;
    const bool stage_a = row_index || raw_m != M || !aligned(A);
    const bool stage_c = residual || out_index || out_stride != N || raw_m != M ||
                         !aligned(C) || A == C;
    const size_t a_bytes = stage_a ? size_t(raw_m) * K * sizeof(bf16) : 0;
    const size_t c_bytes = stage_c ? size_t(raw_m) * N * sizeof(bf16) : 0;
    if (a_bytes > kMaxScratchBytes || c_bytes > kMaxScratchBytes - a_bytes) return false;

    const StrideA stride_a{int64_t(K), _1{}, int64_t(raw_m) * K};
    const StrideB stride_b{int64_t(K), _1{}, int64_t(N) * K};
    const StrideCD stride_cd{int64_t(N), _1{}, int64_t(raw_m) * N};
    const StrideSZ stride_sz{_1{}, int64_t(N), int64_t(scale_elements)};
    cutlass::KernelHardwareInfo hw_info{};
    hw_info.sm_count = device.get_info<sycl::info::device::max_compute_units>();
    Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {raw_m, N, K, 1},
        {reinterpret_cast<const Element*>(A), stride_a,
         reinterpret_cast<const uint4_t*>(weights), stride_b,
         reinterpret_cast<const F16Scale*>(scales), stride_sz, zeros, stride_sz, 64},
        {{1.f, 0.f}, nullptr, stride_cd, reinterpret_cast<Element*>(C), stride_cd},
        hw_info};
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess ||
        Gemm::get_workspace_size(args) != 0) return false;

    Scratch* scratch = nullptr;
    bf16* compact_a = nullptr;
    bf16* compact_c = nullptr;
    if (a_bytes + c_bytes) {
        scratch = &get_scratch(a_bytes + c_bytes, q);
        if (stage_a) {
            compact_a = reinterpret_cast<bf16*>(scratch->data);
            args.mainloop.ptr_A = reinterpret_cast<const Element*>(compact_a);
        }
        if (stage_c) {
            compact_c = reinterpret_cast<bf16*>(scratch->data + a_bytes);
            args.epilogue.ptr_D = reinterpret_cast<Element*>(compact_c);
        }
    }

    Gemm operation;
    cutlass::cudaStream_t stream = &q;
    if (operation.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
        return false;

    try {
        if (scratch && scratch->pending)
            q.ext_oneapi_submit_barrier({scratch->ready});
        if (stage_a) {
            q.parallel_for(sycl::range<1>(size_t(raw_m) * K), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const int row = int(i / K);
                if (row < M) {
                    const int source = row_index ? row_index[row] : row;
                    compact_a[i] = A[size_t(source) * K + i % K];
                } else {
                    compact_a[i] = bf16(0.f);
                }
            });
        }
        if (operation.run(stream) != cutlass::Status::kSuccess)
            throw std::runtime_error("flint: AWQ XMX GEMM launch failed");
        if (stage_c) {
            scratch->ready = q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const int row = int(i / N);
                const int destination = out_index ? out_index[row] : row;
                const size_t output = size_t(destination) * out_stride + i % N;
                const bf16 value = compact_c[i];
                C[output] = residual ? bf16(float(residual[output]) + float(value)) : value;
            });
            scratch->pending = true;
        } else if (scratch) {
            scratch->ready = q.ext_oneapi_submit_barrier();
            scratch->pending = true;
        }
    } catch (...) {
        if (scratch) {
            try {
                q.wait();
                if (scratch->pending) scratch->ready.wait();
                scratch->pending = false;
            } catch (...) {
                scratch->data = nullptr;
                scratch->capacity = 0;
            }
        }
        throw;
    }
    return true;
}

}

#else

namespace flint::awq_xmx {

bool gemm_cute(sycl::queue&, const QTensor&, const bf16*, const int*, bf16*, int, int,
          const bf16*, const int*, int, int) {
    return false;
}

}

#endif

namespace flint::awq_xmx {

bool gemm(sycl::queue& q, const QTensor& t, const bf16* A,
          const int* row_index, bf16* C, int M, int N,
          const bf16* residual, const int* out_index, int out_stride, int expert) {
    if (awq_joint_matrix::mode() != awq_joint_matrix::Mode::disabled)
        return awq_joint_matrix::gemm(q, t, A, row_index, C, M, N,
                                      residual, out_index, out_stride, expert);
    return gemm_cute(q, t, A, row_index, C, M, N, residual, out_index, out_stride, expert);
}

} // namespace flint::awq_xmx

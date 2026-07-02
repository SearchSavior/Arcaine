#pragma once
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <cstdlib>
#include <cstdio>
#include <string>
#include "buffer.hpp"
#include "engine.hpp"

#ifndef DIFF_DPAS_INTRINSIC_DECL
#define DIFF_DPAS_INTRINSIC_DECL
using diff_dpas_v8s = short __attribute__((ext_vector_type(8)));
using diff_dpas_v8i = int   __attribute__((ext_vector_type(8)));
using diff_dpas_v8f = float __attribute__((ext_vector_type(8)));

SYCL_EXTERNAL inline diff_dpas_v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, diff_dpas_v8s A, diff_dpas_v8i B, diff_dpas_v8f C, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return diff_dpas_v8f{}; }
#endif
#endif

static constexpr int kNvfp4DpasBF16 = 0x3000;

struct Nvfp4Linear {
    int in_features = 0;
    int out_features = 0;
    float input_global_scale = 1.0f;
    float weight_global_scale = 1.0f;
    // Packed low-nibble-first f4_e2m1 weights, logical shape (out_features, in_features).
    GpuBuffer<uint8_t> weight_packed;
    // f8_e4m3 scales transposed for oneDNN, logical shape (in_features / 16, out_features).
    GpuBuffer<uint8_t> weight_scale;
    // One-element f32 scale consumed by oneDNN as DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST.
    GpuBuffer<float> dst_scale;
    // Optional oneDNN-chosen weight layout, enabled by DIFF_NVFP4_WEIGHT_LAYOUT=any.
    mutable GpuBuffer<uint8_t> weight_any;
    mutable size_t weight_any_bytes = 0;
    mutable int weight_any_gpu = -1;
    // Optional Xe2 DPAS coalesced layout:
    // [n/16][k/16][16 lanes][8 packed bytes].
    mutable GpuBuffer<uint8_t> weight_coal;
    mutable int weight_coal_gpu = -1;

    bool empty() const { return weight_packed.empty(); }
};

enum class Nvfp4WeightLayout { Raw = 0, Any = 1 };

inline Nvfp4WeightLayout nvfp4_weight_layout() {
    static Nvfp4WeightLayout layout = [] {
        const char* env = std::getenv("DIFF_NVFP4_WEIGHT_LAYOUT");
        if (env && (std::string(env) == "any" || std::string(env) == "xe" ||
                    std::string(env) == "reorder"))
            return Nvfp4WeightLayout::Any;
        return Nvfp4WeightLayout::Raw;
    }();
    return layout;
}

inline bool nvfp4_verbose() {
    static bool enabled = std::getenv("DIFF_NVFP4_VERBOSE") != nullptr;
    return enabled;
}

struct Nvfp4MatmulKey {
    int gpu, M, K, N, layout;
    bool operator==(const Nvfp4MatmulKey& o) const {
        return gpu == o.gpu && M == o.M && K == o.K && N == o.N && layout == o.layout;
    }
};

struct Nvfp4MatmulKeyHash {
    size_t operator()(const Nvfp4MatmulKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.layout) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Nvfp4MatmulEntry {
    dnnl::matmul primitive;
    dnnl::memory::desc weights_md;
};

inline float nvfp4_e4m3_to_float(uint8_t bits) {
    if (bits == 0) return 0.0f;
    float sign = (bits & 0x80) ? -1.0f : 1.0f;
    int exp = (bits >> 3) & 0x0f;
    int mant = bits & 0x07;
    float v = exp == 0
        ? (mant / 8.0f) * sycl::exp2(-6.0f)
        : (1.0f + mant / 8.0f) * sycl::exp2((float)exp - 7.0f);
    return sign * v;
}

inline uint8_t nvfp4_encode_e4m3_positive(float x) {
    if (!(x > 0.0f)) return 0;
    if (x >= 448.0f) return 0x7e;

    constexpr float kSubStep = 0.001953125f;      // 2^-9
    constexpr float kNormalMin = 0.015625f;       // 2^-6
    constexpr float kSubNormalCut = 0.0146484375f; // midpoint between 7*2^-9 and 2^-6

    if (x < kSubNormalCut) {
        int mant = (int)sycl::floor(x / kSubStep + 0.5f);
        if (mant <= 0) return 0;
        if (mant > 7) mant = 7;
        return (uint8_t)mant;
    }

    float efloat = sycl::floor(sycl::log2(sycl::fmax(x, kNormalMin)));
    int actual_exp = (int)efloat;
    if (actual_exp < -6) actual_exp = -6;
    int exp = actual_exp + 7;
    if (exp > 15) return 0x7e;

    float step = sycl::exp2((float)actual_exp - 3.0f);
    int mant = (int)sycl::floor(x / step - 8.0f + 0.5f);
    if (mant < 0) mant = 0;
    if (mant > 7) {
        mant = 0;
        ++exp;
    }
    if (exp > 15) return 0x7e;
    if (exp == 15 && mant > 6) return 0x7e;
    return (uint8_t)((exp << 3) | mant);
}

inline uint8_t nvfp4_encode_e2m1(float x) {
    uint8_t sign = 0;
    if (x < 0.0f) { sign = 0x8; x = -x; }

    uint8_t code = 0;
    if (x < 0.25f) code = 0;          // 0
    else if (x < 0.75f) code = 1;     // 0.5
    else if (x < 1.25f) code = 2;     // 1
    else if (x < 1.75f) code = 3;     // 1.5
    else if (x < 2.5f) code = 4;      // 2
    else if (x < 3.5f) code = 5;      // 3
    else if (x < 5.0f) code = 6;      // 4
    else code = 7;                    // 6
    return sign | code;
}

inline void pack_bf16_to_nvfp4(
    sycl::queue& q,
    const bf16* src,
    uint8_t* packed,
    uint8_t* scales,
    int M,
    int K,
    float input_global_scale)
{
    if (K % 16 != 0) throw std::runtime_error("NVFP4 activation K must be divisible by 16");
    int G = K / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(M, G), [=](sycl::id<2> id) {
            int m = (int)id[0];
            int g = (int)id[1];
            int k0 = g * 16;

            float max_abs = 0.0f;
            for (int i = 0; i < 16; ++i) {
                float v = bf16_to_float(src[(size_t)m * K + k0 + i]);
                max_abs = sycl::fmax(max_abs, sycl::fabs(v));
            }

            uint8_t scale_bits = 0;
            float scale = 0.0f;
            if (max_abs > 0.0f) {
                float raw_scale = max_abs * input_global_scale / 6.0f;
                scale_bits = nvfp4_encode_e4m3_positive(raw_scale);
                scale = nvfp4_e4m3_to_float(scale_bits);
            }
            scales[(size_t)m * G + g] = scale_bits;

            for (int i = 0; i < 16; i += 2) {
                uint8_t lo = 0, hi = 0;
                if (scale > 0.0f) {
                    float v0 = bf16_to_float(src[(size_t)m * K + k0 + i])
                             * input_global_scale / scale;
                    float v1 = bf16_to_float(src[(size_t)m * K + k0 + i + 1])
                             * input_global_scale / scale;
                    lo = nvfp4_encode_e2m1(v0);
                    hi = nvfp4_encode_e2m1(v1);
                }
                packed[(size_t)m * (K / 2) + (k0 + i) / 2] = lo | (hi << 4);
            }
        });
    });
}

inline Nvfp4MatmulEntry& nvfp4_matmul_entry(GpuEngine& ctx, int M, int K, int N) {
    static std::unordered_map<Nvfp4MatmulKey, Nvfp4MatmulEntry, Nvfp4MatmulKeyHash> cache;
    Nvfp4WeightLayout layout = nvfp4_weight_layout();
    Nvfp4MatmulKey key{ctx.index, M, K, N, (int)layout};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_SRC, 3, {1, 16}, dt::f8_e4m3);
        attr.set_scales(DNNL_ARG_WEIGHTS, 3, {16, 1}, dt::f8_e4m3);
        attr.set_scales(DNNL_ARG_DST, 0, {}, dt::f32);
        auto weights_md = (layout == Nvfp4WeightLayout::Any)
            ? dnnl::memory::desc({K, N}, dt::f4_e2m1, tag::any)
            : dnnl::memory::desc({K, N}, dt::f4_e2m1, tag::ba);
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({M, K}, dt::f4_e2m1, tag::ab),
            weights_md,
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            attr);
        auto concrete_weights = pd.weights_desc();
        if (nvfp4_verbose()) {
            auto dims = concrete_weights.get_dims();
            auto strides = concrete_weights.get_strides();
            std::fprintf(stderr,
                "[nvfp4] gpu=%d M=%d K=%d N=%d layout=%s impl=%s weight_bytes=%zu dims=(%lld,%lld) strides=(%lld,%lld)\n",
                ctx.index, M, K, N,
                layout == Nvfp4WeightLayout::Any ? "any" : "raw",
                pd.impl_info_str(), (size_t)concrete_weights.get_size(),
                (long long)dims[0], (long long)dims[1],
                (long long)strides[0], (long long)strides[1]);
        }
        auto result = cache.emplace(key, Nvfp4MatmulEntry{dnnl::matmul(pd), concrete_weights});
        it = result.first;
    }
    return it->second;
}

inline const uint8_t* nvfp4_weight_data(const Nvfp4Linear& W,
                                        const dnnl::memory::desc& weights_md,
                                        int K, int N, GpuEngine& ctx) {
    if (nvfp4_weight_layout() == Nvfp4WeightLayout::Raw)
        return W.weight_packed.data();

    size_t bytes = weights_md.get_size();
    if (W.weight_any.empty() || W.weight_any_bytes != bytes || W.weight_any_gpu != ctx.index) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        W.weight_any = GpuBuffer<uint8_t>(bytes, ctx.queue);
        W.weight_any_bytes = bytes;
        W.weight_any_gpu = ctx.index;
        auto raw_md = dnnl::memory::desc({K, N}, dt::f4_e2m1, tag::ba);
        auto raw_mem = dnnl::sycl_interop::make_memory(
            raw_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            W.weight_packed.data());
        auto any_mem = dnnl::sycl_interop::make_memory(
            weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            W.weight_any.data());
        dnnl::reorder(raw_mem, any_mem).execute(ctx.stream, raw_mem, any_mem);
        ctx.stream.wait();
    }
    return W.weight_any.data();
}

inline void matmul_nvfp4_packed(
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int M,
    int K,
    const Nvfp4Linear& W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.in_features != K)
        throw std::runtime_error("matmul_nvfp4_packed: K does not match weight shape");
    if (K % 16 != 0)
        throw std::runtime_error("matmul_nvfp4_packed: K must be divisible by 16");
    if (W.dst_scale.empty())
        throw std::runtime_error("matmul_nvfp4_packed: missing persistent destination scale");

    int N = W.out_features;
    int G = K / 16;

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto src_md = dnnl::memory::desc({M, K}, dt::f4_e2m1, tag::ab);
    auto dst_md = dnnl::memory::desc({M, N}, dt::bf16, tag::ab);
    auto src_scales_md = dnnl::memory::desc({M, G}, dt::f8_e4m3, tag::ab);
    auto weight_scales_md = dnnl::memory::desc({G, N}, dt::f8_e4m3, tag::ab);
    auto dst_scale_md = dnnl::memory::desc({1}, dt::f32, tag::a);

    auto& entry = nvfp4_matmul_entry(ctx, M, K, N);
    const uint8_t* weight_data = nvfp4_weight_data(W, entry.weights_md, K, N, ctx);

    entry.primitive.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(A_packed))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            entry.weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(weight_data))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(A_scale))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            weight_scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            W.weight_scale.data())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_scale_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            W.dst_scale.data())},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

// ---------------------------------------------------------------------------
// Batched NVFP4 GEMM (Direction B, Phase 1).
//
// oneDNN batched f4_e2m1 matmul. Validated by Probe 1 (nvfp4_batched_probe):
//   - src scales     [B,M,G] f8_e4m3, mask=7, groups{1,1,16}
//   - weight scales  [B,G,N] f8_e4m3, mask=7, groups{1,16,1}   (batched; the
//     shared [G,N] layout is REJECTED by oneDNN at execute — do not use it)
//   - dst scale      [1]     f32,    mask=0 (shared across the batch)
// Weights are fed in the raw per-expert layout (each [N,K] f4, K-contiguous,
// matching the single kernel's tag::ba on {K,N}). Concatenating E experts
// end-to-end gives a [B,K,N] tensor that is K-contiguous within each batch =>
// tag::acb (B slow, N mid, K fast). No reorder / no Any-layout path (the bench
// builds the batched buffer directly from the per-expert packed weights).
// ---------------------------------------------------------------------------

struct Nvfp4BatchedMatmulKey {
    int gpu, B, M, K, N;
    bool operator==(const Nvfp4BatchedMatmulKey& o) const {
        return gpu == o.gpu && B == o.B && M == o.M && K == o.K && N == o.N;
    }
};

struct Nvfp4BatchedMatmulKeyHash {
    size_t operator()(const Nvfp4BatchedMatmulKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        auto mix = [&](int v) {
            h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        mix(k.B); mix(k.M); mix(k.K); mix(k.N);
        return h;
    }
};

struct Nvfp4BatchedMatmulEntry {
    dnnl::matmul primitive;
    dnnl::memory::desc weights_md;
};

inline Nvfp4BatchedMatmulEntry& nvfp4_matmul_batched_entry(
    GpuEngine& ctx, int B, int M, int K, int N)
{
    static std::unordered_map<Nvfp4BatchedMatmulKey, Nvfp4BatchedMatmulEntry,
                              Nvfp4BatchedMatmulKeyHash> cache;
    Nvfp4BatchedMatmulKey key{ctx.index, B, M, K, N};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_SRC,     7, {1, 1, 16}, dt::f8_e4m3); // [B,M,G]
        attr.set_scales(DNNL_ARG_WEIGHTS, 7, {1, 16, 1}, dt::f8_e4m3); // [B,G,N]
        attr.set_scales(DNNL_ARG_DST,     0, {},          dt::f32);    // [1]
        auto weights_md = dnnl::memory::desc({B, K, N}, dt::f4_e2m1, tag::acb);
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({B, M, K}, dt::f4_e2m1, tag::abc),
            weights_md,
            dnnl::memory::desc({B, M, N}, dt::bf16,     tag::abc),
            attr);
        if (nvfp4_verbose()) {
            auto cwd = pd.weights_desc();
            auto wd = cwd.get_dims();
            auto ws = cwd.get_strides();
            std::fprintf(stderr,
                "[nvfp4-batched] gpu=%d B=%d M=%d K=%d N=%d impl=%s "
                "w_dims=(%lld,%lld,%lld) w_strides=(%lld,%lld,%lld)\n",
                ctx.index, B, M, K, N, pd.impl_info_str(),
                (long long)wd[0], (long long)wd[1], (long long)wd[2],
                (long long)ws[0], (long long)ws[1], (long long)ws[2]);
        }
        auto result = cache.emplace(key, Nvfp4BatchedMatmulEntry{dnnl::matmul(pd), pd.weights_desc()});
        it = result.first;
    }
    return it->second;
}

inline void matmul_nvfp4_packed_batched(
    const uint8_t* A_packed,    // [B, M, K] f4_e2m1 (tag::abc, K contiguous)
    const uint8_t* A_scale,     // [B, M, G] f8_e4m3 (tag::abc, G contiguous)
    int B, int M, int K,
    const uint8_t* W_packed_batch, // [B, K, N] f4_e2m1 (tag::acb, K contiguous)
    const uint8_t* W_scale_batch,  // [B, G, N] f8_e4m3 (tag::abc, N contiguous)
    const float*    dst_scale,      // [1]        f32 (shared)
    int N, bf16* C,                  // [B, M, N] bf16 (tag::abc)
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (K % 16 != 0)
        throw std::runtime_error("matmul_nvfp4_packed_batched: K must be divisible by 16");
    int G = K / 16;

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto src_md         = dnnl::memory::desc({B, M, K}, dt::f4_e2m1, tag::abc);
    auto dst_md         = dnnl::memory::desc({B, M, N}, dt::bf16,     tag::abc);
    auto src_scales_md  = dnnl::memory::desc({B, M, G}, dt::f8_e4m3, tag::abc);
    auto w_scales_md    = dnnl::memory::desc({B, G, N}, dt::f8_e4m3, tag::abc);
    auto dst_scale_md   = dnnl::memory::desc({1},       dt::f32,     tag::a);

    auto& entry = nvfp4_matmul_batched_entry(ctx, B, M, K, N);
    entry.primitive.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(A_packed))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            entry.weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(W_packed_batch))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(A_scale))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            w_scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(W_scale_batch))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_scale_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<float*>(dst_scale))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

inline void matmul_nvfp4(
    const bf16* A,
    int M,
    int K,
    const Nvfp4Linear& W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.in_features != K)
        throw std::runtime_error("matmul_nvfp4: K does not match weight shape");
    if (K % 16 != 0)
        throw std::runtime_error("matmul_nvfp4: K must be divisible by 16");

    int G = K / 16;
    auto& q = ctx.queue;

    GpuBuffer<uint8_t> A_packed((size_t)M * K / 2, q);
    GpuBuffer<uint8_t> A_scale((size_t)M * G, q);
    pack_bf16_to_nvfp4(q, A, A_packed.data(), A_scale.data(), M, K,
                       W.input_global_scale);
    matmul_nvfp4_packed(A_packed.data(), A_scale.data(), M, K, W, C, ctx);

    // A_packed/A_scale are temporary workspaces owned by this call. Keep the
    // execution synchronous until callers pass reusable workspaces explicitly.
    ctx.stream.wait();
}

inline float nvfp4_e2m1_to_float(uint8_t bits) {
    float mag = 0.0f;
    switch (bits & 0x07) {
        case 0: mag = 0.0f; break;
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        default: mag = 6.0f; break;
    }
    return (bits & 0x08) ? -mag : mag;
}

inline float nvfp4_e4m3_fast(uint8_t b) {
    uint32_t exp = (b >> 3) & 0x0f;
    uint32_t mant = b & 0x07;
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    if (exp == 0) return sign * (float)mant * (1.0f / 512.0f);
    uint32_t bits = ((uint32_t)(b & 0x80) << 24) |
                    ((exp - 7 + 127) << 23) |
                    (mant << 20);
    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

inline float nvfp4_e2m1_fast(uint8_t bits) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (bits & 8) ? -mag[bits & 7] : mag[bits & 7];
}

// Vectorized (ESIMD simd) arithmetic e2m1 dequant — no LUT, no scalar loop.
// Validated bit-exact vs nvfp4_e2m1_fast by src/nvfp4_vec_dequant_probe.cpp.
// nib holds 4-bit e2m1 codes (0..15); returns the dequanted float values.
namespace esimd_vec {
namespace esimd_local = sycl::ext::intel::esimd;
template <int N>
inline esimd_local::simd<float, N> e2m1(esimd_local::simd<uint16_t, N> nib) {
    auto s = nib >> 3;
    auto e = (nib >> 1) & esimd_local::simd<uint16_t, N>(3);
    auto m = nib & esimd_local::simd<uint16_t, N>(1);
    auto one = esimd_local::simd<uint16_t, N>(1);
    auto one_shl_e = one << e;                       // 1,2,4,8
    auto two_pow = esimd_local::convert<float>(one_shl_e) * 0.5f;  // 0.5,1,2,4
    auto m_f = esimd_local::convert<float>(m);
    auto normal = two_pow + two_pow * 0.5f * m_f;     // two_pow*(1+0.5*m)
    auto subnorm = 0.5f * m_f;
    auto absval = normal;
    absval.merge(subnorm, e == esimd_local::simd<uint16_t, N>(0));
    auto val = absval;
    val.merge(-absval, s != esimd_local::simd<uint16_t, N>(0));
    return val;
}

// Vectorized (ESIMD simd) arithmetic e4m3 dequant — bit-exact vs nvfp4_e4m3_fast.
// b holds 8-bit e4m3 codes (0..255); returns the dequanted float values.
// Normal: 2^(exp-7)*(1+mant/8); subnormal (exp==0): mant*2^-9 (= (mant/8)*2^-6).
// Uses float(1<<exp)/128 instead of exp2() to stay bit-exact and avoid negative-shift UB.
template <int N>
inline esimd_local::simd<float, N> e4m3(esimd_local::simd<uint16_t, N> b) {
    auto s = b >> 7;                                         // sign (bit 7)
    auto exp = (b >> 3) & esimd_local::simd<uint16_t, N>(0x0f);
    auto mant = b & esimd_local::simd<uint16_t, N>(0x07);
    auto one = esimd_local::simd<uint16_t, N>(1);
    auto one_shl_e = one << exp;                             // 1..32768 (fits uint16)
    auto two_pow = esimd_local::convert<float>(one_shl_e) * (1.0f / 128.0f);  // 2^-6..2^8
    auto mant_f = esimd_local::convert<float>(mant) * (1.0f / 8.0f);
    auto normal = two_pow * (1.0f + mant_f);                // 2^(e-7)*(1+mant/8)
    auto subnorm = esimd_local::convert<float>(mant) * (1.0f / 512.0f);       // mant*2^-9
    auto absval = normal;
    absval.merge(subnorm, exp == esimd_local::simd<uint16_t, N>(0));
    auto val = absval;
    val.merge(-absval, s != esimd_local::simd<uint16_t, N>(0));
    return val;
}
}  // namespace esimd_vec

inline int nvfp4_dpas_ksplit_factor(int m_tiles, int k_tiles, int n_tiles) {
    static int target = [] {
        const char* env = std::getenv("DIFF_NVFP4_DPAS_OCC");
        int v = env ? std::atoi(env) : 2048;
        return v > 0 ? v : 2048;
    }();
    int base_groups = m_tiles * n_tiles;
    int ks = (target + base_groups - 1) / (base_groups > 0 ? base_groups : 1);
    if (ks < 1) ks = 1;
    if (ks > k_tiles) ks = k_tiles;
    if (ks > 32) ks = 32;
    return ks;
}

inline const uint16_t* nvfp4_dequant_lut(GpuEngine& ctx) {
    static std::vector<GpuBuffer<uint16_t>>* luts =
        new std::vector<GpuBuffer<uint16_t>>(GpuEngine::count());
    auto& buf = (*luts)[ctx.index];
    if (buf.empty()) {
        std::vector<uint16_t> h(256 * 16);
        for (int sb = 0; sb < 256; ++sb)
            for (int nb = 0; nb < 16; ++nb)
                h[(size_t)sb * 16 + nb] =
                    float_to_bf16(nvfp4_e2m1_fast((uint8_t)nb) *
                                  nvfp4_e4m3_fast((uint8_t)sb));
        buf = GpuBuffer<uint16_t>(256 * 16, ctx.queue);
        buf.upload(h.data(), h.size());
    }
    return buf.data();
}

inline const uint8_t* nvfp4_coalesced_weight(const Nvfp4Linear& W,
                                             int K,
                                             int N,
                                             GpuEngine& ctx) {
    if (!W.weight_coal.empty() && W.weight_coal_gpu == ctx.index)
        return W.weight_coal.data();
    int halfK = K / 2;
    int ktiles = K / 16;
    W.weight_coal = GpuBuffer<uint8_t>((size_t)N * halfK, ctx.queue);
    W.weight_coal_gpu = ctx.index;
    const uint8_t* src = W.weight_packed.data();
    uint8_t* dst = W.weight_coal.data();
    ctx.queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)N, (size_t)halfK), [=](sycl::id<2> id) {
            int n = (int)id[0];
            int b = (int)id[1];
            int kt = b / 8;
            int j = b % 8;
            dst[(size_t)(n / 16) * ktiles * 128 +
                (size_t)kt * 128 +
                (n % 16) * 8 + j] = src[(size_t)n * halfK + b];
        });
    });
    ctx.queue.wait();
    return dst;
}

inline diff_dpas_v8i nvfp4_dequant_b_coal(const uint8_t* wcoal,
                                          const uint16_t* lut,
                                          const uint8_t* wscale,
                                          int n,
                                          int lane,
                                          int kt,
                                          int K,
                                          int N) {
    const uint16_t* lrow = lut + (size_t)wscale[(size_t)kt * N + n] * 16;
    const uint8_t* row = wcoal +
        (size_t)(n / 16) * (K / 16) * 128 +
        (size_t)kt * 128 + lane * 8;
    diff_dpas_v8i b;
    for (int j = 0; j < 8; ++j) {
        uint8_t by = row[j];
        b[j] = (int)((uint32_t)lrow[by & 0x0f] |
                     ((uint32_t)lrow[by >> 4] << 16));
    }
    return b;
}

inline void pack_bf16_to_nvfp4_grouped(
    sycl::queue& q,
    const bf16* src,
    int K,
    const int32_t* row_slot,
    const int32_t* row_expert,
    int rows,
    const float* input_global_scale,
    uint8_t* packed,
    uint8_t* scales)
{
    if (K % 16 != 0) throw std::runtime_error("grouped NVFP4 activation K must be divisible by 16");
    int G = K / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)rows, (size_t)G), [=](sycl::id<2> id) {
            int r = (int)id[0];
            int g = (int)id[1];
            int slot = row_slot[r];
            int expert = row_expert[r];
            int k0 = g * 16;
            const bf16* src_row = src + (size_t)slot * K;
            uint8_t* packed_row = packed + (size_t)slot * (K / 2);
            uint8_t* scale_row = scales + (size_t)slot * G;
            float input_scale = input_global_scale[expert];

            float max_abs = 0.0f;
            for (int i = 0; i < 16; ++i) {
                float v = bf16_to_float(src_row[k0 + i]);
                max_abs = sycl::fmax(max_abs, sycl::fabs(v));
            }

            uint8_t scale_bits = 0;
            float scale = 0.0f;
            if (max_abs > 0.0f) {
                float raw_scale = max_abs * input_scale / 6.0f;
                scale_bits = nvfp4_encode_e4m3_positive(raw_scale);
                scale = nvfp4_e4m3_to_float(scale_bits);
            }
            scale_row[g] = scale_bits;

            for (int i = 0; i < 16; i += 2) {
                uint8_t lo = 0, hi = 0;
                if (scale > 0.0f) {
                    float v0 = bf16_to_float(src_row[k0 + i]) * input_scale / scale;
                    float v1 = bf16_to_float(src_row[k0 + i + 1]) * input_scale / scale;
                    lo = nvfp4_encode_e2m1(v0);
                    hi = nvfp4_encode_e2m1(v1);
                }
                packed_row[(k0 + i) / 2] = lo | (hi << 4);
            }
        });
    });
}

inline void matmul_nvfp4_grouped_custom(
    sycl::queue& q,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* row_slot,
    const int32_t* row_expert,
    int rows,
    const uint8_t* const* W_packed_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0) throw std::runtime_error("grouped NVFP4 matmul K must be divisible by 16");
    int G = K / 16;
    int halfK = K / 2;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)rows, (size_t)N), [=](sycl::id<2> id) {
            int r = (int)id[0];
            int n = (int)id[1];
            int slot = row_slot[r];
            int expert = row_expert[r];
            const uint8_t* a_row = A_packed + (size_t)slot * halfK;
            const uint8_t* as_row = A_scale + (size_t)slot * G;
            const uint8_t* w = W_packed_by_expert[expert];
            const uint8_t* ws = W_scale_by_expert[expert];
            float inv_dst = 1.0f / dst_scale_by_expert[expert][0];
            float acc = 0.0f;

            for (int g = 0; g < G; ++g) {
                float a_scale = nvfp4_e4m3_to_float(as_row[g]);
                float w_scale = nvfp4_e4m3_to_float(ws[(size_t)g * N + n]);
                float scale = a_scale * w_scale;
                if (scale == 0.0f) continue;
                int byte0 = g * 8;
                for (int b = 0; b < 8; ++b) {
                    uint8_t av = a_row[byte0 + b];
                    uint8_t wv = w[(size_t)n * halfK + byte0 + b];
                    float a0 = nvfp4_e2m1_to_float(av & 0x0f);
                    float a1 = nvfp4_e2m1_to_float((av >> 4) & 0x0f);
                    float w0 = nvfp4_e2m1_to_float(wv & 0x0f);
                    float w1 = nvfp4_e2m1_to_float((wv >> 4) & 0x0f);
                    acc += (a0 * w0 + a1 * w1) * scale;
                }
            }
            C[(size_t)slot * N + n] = float_to_bf16(acc * inv_dst);
        });
    });
}

inline void pack_bf16_to_nvfp4_grouped_rows(
    sycl::queue& q,
    const bf16* src,
    int K,
    const int32_t* row_expert,
    int max_rows,
    const float* input_global_scale,
    uint8_t* packed,
    uint8_t* scales)
{
    if (K % 16 != 0) throw std::runtime_error("counted grouped NVFP4 activation K must be divisible by 16");
    int G = K / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)max_rows, (size_t)G), [=](sycl::id<2> id) {
            int row = (int)id[0];
            int g = (int)id[1];
            int expert = row_expert[row];
            if (expert < 0) return;

            int k0 = g * 16;
            const bf16* src_row = src + (size_t)row * K;
            uint8_t* packed_row = packed + (size_t)row * (K / 2);
            uint8_t* scale_row = scales + (size_t)row * G;
            float input_scale = input_global_scale[expert];

            float max_abs = 0.0f;
            for (int i = 0; i < 16; ++i) {
                float v = bf16_to_float(src_row[k0 + i]);
                max_abs = sycl::fmax(max_abs, sycl::fabs(v));
            }

            uint8_t scale_bits = 0;
            float scale = 0.0f;
            if (max_abs > 0.0f) {
                float raw_scale = max_abs * input_scale / 6.0f;
                scale_bits = nvfp4_encode_e4m3_positive(raw_scale);
                scale = nvfp4_e4m3_to_float(scale_bits);
            }
            scale_row[g] = scale_bits;

            for (int i = 0; i < 16; i += 2) {
                uint8_t lo = 0, hi = 0;
                if (scale > 0.0f) {
                    float v0 = bf16_to_float(src_row[k0 + i]) * input_scale / scale;
                    float v1 = bf16_to_float(src_row[k0 + i + 1]) * input_scale / scale;
                    lo = nvfp4_encode_e2m1(v0);
                    hi = nvfp4_encode_e2m1(v1);
                }
                packed_row[(k0 + i) / 2] = lo | (hi << 4);
            }
        });
    });
}

inline void matmul_nvfp4_grouped_rows_custom(
    sycl::queue& q,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* row_expert,
    int max_rows,
    const uint8_t* const* W_packed_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0) throw std::runtime_error("counted grouped NVFP4 matmul K must be divisible by 16");
    int G = K / 16;
    int halfK = K / 2;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)max_rows, (size_t)N), [=](sycl::id<2> id) {
            int row = (int)id[0];
            int n = (int)id[1];
            int expert = row_expert[row];
            if (expert < 0) return;

            const uint8_t* a_row = A_packed + (size_t)row * halfK;
            const uint8_t* as_row = A_scale + (size_t)row * G;
            const uint8_t* w = W_packed_by_expert[expert];
            const uint8_t* ws = W_scale_by_expert[expert];
            float inv_dst = 1.0f / dst_scale_by_expert[expert][0];
            float acc = 0.0f;

            for (int g = 0; g < G; ++g) {
                float a_scale = nvfp4_e4m3_to_float(as_row[g]);
                float w_scale = nvfp4_e4m3_to_float(ws[(size_t)g * N + n]);
                float scale = a_scale * w_scale;
                if (scale == 0.0f) continue;
                int byte0 = g * 8;
                for (int b = 0; b < 8; ++b) {
                    uint8_t av = a_row[byte0 + b];
                    uint8_t wv = w[(size_t)n * halfK + byte0 + b];
                    float a0 = nvfp4_e2m1_to_float(av & 0x0f);
                    float a1 = nvfp4_e2m1_to_float((av >> 4) & 0x0f);
                    float w0 = nvfp4_e2m1_to_float(wv & 0x0f);
                    float w1 = nvfp4_e2m1_to_float((wv >> 4) & 0x0f);
                    acc += (a0 * w0 + a1 * w1) * scale;
                }
            }
            C[(size_t)row * N + n] = float_to_bf16(acc * inv_dst);
        });
    });
}

inline void matmul_nvfp4_grouped_rows_xe2(
    GpuEngine& ctx,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* row_expert,
    int max_rows,
    const uint8_t* const* W_coal_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0 || N % 16 != 0)
        throw std::runtime_error("xe2 grouped NVFP4 matmul requires K%16 and N%16");
    if (max_rows <= 0) return;

    auto& q = ctx.queue;
    int halfK = K / 2;
    int ktiles = K / 16;
    int mtiles = (max_rows + 7) / 8;
    int ntiles = N / 16;
    int KS = nvfp4_dpas_ksplit_factor(mtiles, ktiles, ntiles);
    const uint16_t* lut = nvfp4_dequant_lut(ctx);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm((size_t)KS * 8 * 16, h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)mtiles * KS, (size_t)N),
                              sycl::range<2>((size_t)KS, (size_t)16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int m0 = (int)it.get_group(0) * 8;
                int s = (int)it.get_local_id(0);
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                int expert = (m0 < max_rows) ? row_expert[m0] : -1;

                diff_dpas_v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
                if (expert >= 0) {
                    const uint8_t* wcoal = W_coal_by_expert[expert];
                    const uint8_t* wscale = W_scale_by_expert[expert];
                    for (int kt = s; kt < ktiles; kt += KS) {
                        diff_dpas_v8i b =
                            nvfp4_dequant_b_coal(wcoal, lut, wscale, n, lane, kt, K, N);
                        int k0 = kt * 16;
                        int kk = k0 + lane;
                        diff_dpas_v8s a;
                        for (int m = 0; m < 8; ++m) {
                            int row = m0 + m;
                            uint16_t av = 0;
                            if (row < max_rows && row_expert[row] == expert) {
                                uint8_t byte = A_packed[(size_t)row * halfK + kk / 2];
                                uint8_t nib = (kk & 1) ? (byte >> 4) : (byte & 0x0f);
                                av = float_to_bf16(
                                    nvfp4_e2m1_fast(nib) *
                                    nvfp4_e4m3_fast(A_scale[(size_t)row * ktiles + kt]));
                            }
                            a[m] = (short)av;
                        }
                        c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, a, b, c, kNvfp4DpasBF16);
                    }
                }

                for (int m = 0; m < 8; ++m)
                    slm[(size_t)(s * 8 + m) * 16 + lane] = c[m];
                it.barrier(sycl::access::fence_space::local_space);

                if (s == 0 && expert >= 0) {
                    float inv_dst = 1.0f / dst_scale_by_expert[expert][0];
                    for (int m = 0; m < 8; ++m) {
                        int row = m0 + m;
                        if (row >= max_rows || row_expert[row] != expert) continue;
                        float sum = 0.0f;
                        for (int ss = 0; ss < KS; ++ss)
                            sum += slm[(size_t)(ss * 8 + m) * 16 + lane];
                        C[(size_t)row * N + n] = float_to_bf16(sum * inv_dst);
                    }
                }
            });
    });
}

// xe2v2: per-expert DPAS grouped NVFP4 GEMM, aligned to the xe2 idiom.
// Replaces the k-split + SLM-reduction + global-LUT dequant of the v1 xe2
// kernel with a single-subgroup, register-blocked-M design (cf. the q8
// grouped-expert kernel): one work-group per (expert, N/16 tile), BLK M-tiles
// resident in registers, B dequanted once per k-step and reused across BLK
// rows, arithmetic e2m1*e4m3 dequant (no lookup table, no global traffic).
// Per-expert work-groups make the row->expert mapping exact (no tile straddles
// into another expert's weights), so correctness does not rely on per-expert row
// padding. Gated by DIFF_NVFP4_GROUPED_GEMM=xe2v2 for A/B vs hybrid/oneDNN.
inline void matmul_nvfp4_grouped_rows_xe2_v2(
    GpuEngine& ctx,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* expert_offsets,
    const int32_t* rows_per_expert,
    int localE,
    const uint8_t* const* W_packed_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0 || N % 16 != 0)
        throw std::runtime_error("xe2v2 grouped NVFP4 matmul requires K%16 and N%16");
    if (localE <= 0) return;
    constexpr int BLK = 4;
    auto& q = ctx.queue;
    int halfK = K / 2;
    int ktiles = K / 16;
    int ntiles = N / 16;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)localE, (size_t)ntiles * 16),
                              sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                int offset = expert_offsets[e];
                int count = rows_per_expert[e];
                if (count <= 0) return;
                int mt = (count + 7) / 8;

                const uint8_t* w = W_packed_by_expert[e];
                const uint8_t* ws = W_scale_by_expert[e];
                const uint8_t* wrow_base = w + (size_t)n * halfK;
                float inv_dst = 1.0f / dst_scale_by_expert[e][0];

                for (int tile0 = 0; tile0 < mt; tile0 += BLK) {
                    diff_dpas_v8f c[BLK];
                    for (int t = 0; t < BLK; ++t) c[t] = diff_dpas_v8f{0, 0, 0, 0, 0, 0, 0, 0};
                    for (int kt = 0; kt < ktiles; ++kt) {
                        int k0 = kt * 16;
                        float wscale = nvfp4_e4m3_fast(ws[(size_t)kt * N + n]);
                        const uint8_t* wrow = wrow_base + k0 / 2;
                        diff_dpas_v8i b;
                        for (int j = 0; j < 8; ++j) {
                            uint8_t by = wrow[j];
                            uint16_t lo = float_to_bf16(nvfp4_e2m1_fast(by & 0x0f) * wscale);
                            uint16_t hi = float_to_bf16(nvfp4_e2m1_fast(by >> 4) * wscale);
                            b[j] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
                        }
                        int kk = k0 + lane;
                        for (int t = 0; t < BLK; ++t) {
                            int tile = tile0 + t;
                            if (tile >= mt) break;  // uniform across the subgroup
                            int m0 = offset + tile * 8;
                            diff_dpas_v8s a;
                            for (int m = 0; m < 8; ++m) {
                                int row = m0 + m;
                                uint16_t av = 0;
                                if ((row - offset) < count) {
                                    uint8_t byte = A_packed[(size_t)row * halfK + kk / 2];
                                    uint8_t nib = (kk & 1) ? (byte >> 4) : (byte & 0x0f);
                                    av = float_to_bf16(
                                        nvfp4_e2m1_fast(nib) *
                                        nvfp4_e4m3_fast(A_scale[(size_t)row * ktiles + kt]));
                                }
                                a[m] = (short)av;
                            }
                            c[t] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                                16, a, b, c[t], kNvfp4DpasBF16);
                        }
                    }
                    for (int t = 0; t < BLK; ++t) {
                        int tile = tile0 + t;
                        if (tile >= mt) break;
                        int m0 = offset + tile * 8;
                        for (int m = 0; m < 8; ++m) {
                            int row = m0 + m;
                            if ((row - offset) < count)
                                C[(size_t)row * N + n] = float_to_bf16(c[t][m] * inv_dst);
                        }
                    }
                }
            });
    });
}

// xe2v3: block-loaded B via ESIMD lsc_load_2d + ESIMD xmx::dpas (FULL ESIMD — no
// SPIRV __spirv_SubgroupMatrixMultiplyAccumulateINTEL, whose spir64 codegen would
// clash with ESIMD's genx64 codegen and fail at link/runtime). One work-group per
// (expert, n-tile), 16 lanes. dpas is PER-LANE REPLICATED (each lane holds the
// full 8x16 result tile), so the 16 lanes parallelise across m-tiles: lane l owns
// m-tile (tile0+l), all sharing the same replicated block-loaded B. B is fetched
// from the coalesced layout [n/16][k/16][16][8] with one lsc_load_2d per (n_tile,
// k_tile). Arithmetic e2m1*e4m3 dequant (no global LUT).
//   B VNNI layout (validated by src/nvfp4_dpas_probe.cpp, layout L2):
//     B[(kp*N + n)*2 + kparity], kp=k/2, kparity 0=even(low nibble)/1=odd(high).
//   A row-major [M,K]: A[m*K+k];  C row-major [M,N]: C[m*N+n].
inline void matmul_nvfp4_grouped_rows_xe2_v3(
    GpuEngine& ctx,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* expert_offsets,
    const int32_t* rows_per_expert,
    int localE,
    const uint8_t* const* W_coal_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0 || N % 16 != 0)
        throw std::runtime_error("xe2v3 grouped NVFP4 matmul requires K%16 and N%16");
    if (localE <= 0) return;
    namespace esimd = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    namespace esimd_x = sycl::ext::intel::experimental::esimd;
    using bf16_t = sycl::ext::oneapi::bfloat16;
    constexpr int TM = 8;    // dpas M (rows) per tile  = RepeatCount
    constexpr int TN = 16;  // dpas N (cols) per tile  = ExecutionSize
    constexpr int TK = 16;  // dpas K (contraction)    = SystolicDepth * OpsPerChannel
    constexpr int LANES = 16;
    constexpr int BW = 8;   // n-row width in bytes (TK/2 nibble-pairs)
    // Block-load geometry: lsc_load_2d has a minimum pitch of 16 bytes, so an
    // 8-byte-wide row cannot be loaded at pitch 8 (the HW rounds pitch up to 16,
    // reading every other n-row). Load 16-byte rows x 8 rows instead (pitch 16 ==
    // contiguous for the 128-byte slab). Each register row r holds TWO 8-byte
    // n-rows: col 0..7 -> n=2r, col 8..15 -> n=2r+1. (Isolate probe: a=b=c=0.)
    constexpr int LBW = 16;
    constexpr int LBH = 8;
    auto& q = ctx.queue;
    int halfK = K / 2;
    int ktiles = K / TK;
    int ntiles = N / TN;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)localE, (size_t)ntiles * LANES),
                              sycl::range<2>(1, LANES)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int nt = (int)it.get_group(1);
                int offset = expert_offsets[e];
                int count = rows_per_expert[e];
                if (count <= 0) return;
                int mt = (count + TM - 1) / TM;

                const uint8_t* wcoal = W_coal_by_expert[e];
                const uint8_t* ws = W_scale_by_expert[e];
                // base of this n_tile's coalesced region: [n_tile][k_tile][16][8],
                // each (n_tile, k_tile) slab is 128 contiguous bytes (16 rows x 8).
                const uint8_t* wcoal_nt = wcoal + (size_t)nt * ktiles * 128;
                float inv_dst = 1.0f / dst_scale_by_expert[e][0];

                for (int tile0 = 0; tile0 < mt; tile0 += LANES) {
                    int m_tile = tile0 + lane;        // each lane owns one m-tile
                    bool active = (m_tile < mt);
                    esimd::simd<float, TM * TN> c(0.0f);
                    for (int kt = 0; kt < ktiles; ++kt) {
                        // --- B: block-load (n_tile, k_tile) slab (replicated) ---
                        // LBW=16, LBH=8: register row r -> n=2r (col0..7), n=2r+1
                        // (col8..15); each byte holds k=2kp (low) & k=2kp+1 (high).
                        auto v = esimd_x::lsc_load_2d<uint8_t, LBW, LBH, /*NBlocks*/1,
                                                       /*Transposed*/false,
                                                       /*Transformed*/false>(
                            wcoal_nt + (size_t)kt * 128,
                            LBW - 1, LBH - 1, LBW - 1, 0, 0);
                        // Replicated load: every lane holds the full slab. Dequant to
                        // VNNI layout L2: b[(kp*N+n)*2 + kparity], kparity 0=even
                        // (low nibble), 1=odd (high nibble). Validated by dpas probe.
                        esimd::simd<bf16_t, TK * TN> b;
                        #pragma unroll
                        for (int r = 0; r < LBH; ++r) {
                            #pragma unroll
                            for (int c = 0; c < LBW; ++c) {
                                int n = (c < BW) ? 2 * r : 2 * r + 1;
                                int kp = c & (BW - 1);
                                uint8_t byte = v[r * LBW + c];
                                float wscale =
                                    nvfp4_e4m3_fast(ws[(size_t)kt * N + nt * TN + n]);
                                float lo = nvfp4_e2m1_fast(byte & 0x0f) * wscale;
                                float hi = nvfp4_e2m1_fast(byte >> 4) * wscale;
                                b[(kp * TN + n) * 2 + 0] = bf16_t(lo);
                                b[(kp * TN + n) * 2 + 1] = bf16_t(hi);
                            }
                        }
                        // --- A: this lane's m-tile, row-major [TM, TK] ---
                        esimd::simd<bf16_t, TM * TK> a;
                        #pragma unroll
                        for (int m = 0; m < TM; ++m) {
                            int row = offset + m_tile * TM + m;
                            bool ok = active && ((row - offset) < count);
                            if (ok) {
                                float ascale =
                                    nvfp4_e4m3_fast(A_scale[(size_t)row * ktiles + kt]);
                                #pragma unroll
                                for (int k = 0; k < TK; ++k) {
                                uint8_t byte = A_packed[(size_t)row * halfK + (kt * TK + k) / 2];
                                uint8_t nib = (k & 1) ? ((byte >> 4) & 0x0f)
                                                      : (byte & 0x0f);
                                    a[m * TK + k] =
                                        bf16_t(nvfp4_e2m1_fast(nib) * ascale);
                                }
                            } else {
                                #pragma unroll
                                for (int k = 0; k < TK; ++k) a[m * TK + k] = bf16_t(0.0f);
                            }
                        }
                        // --- C += A @ B (dpas: Result = C + A*B; B is VNNI) ---
                        c = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(c, b, a);
                    }
                    // --- store this lane's 8x16 result tile ---
                    if (active) {
                        #pragma unroll
                        for (int m = 0; m < TM; ++m) {
                            int row = offset + m_tile * TM + m;
                            if ((row - offset) >= count) continue;
                            #pragma unroll
                            for (int n = 0; n < TN; ++n)
                                C[(size_t)row * N + nt * TN + n] =
                                    float_to_bf16(c[m * TN + n] * inv_dst);
                        }
                    }
                }
            });
    });
}

// xe2v4: vectorized-dequant rewrite of v3. Same launch geometry and lane->m-tile
// mapping (one WG per (expert, n-tile), 16 lanes, B block-loaded+replicated, lanes
// split the m-tiles), but the e2m1/e4m3 dequant is vectorized over ESIMD simd ops
// instead of per-byte scalar loops — the dominant cost in v3. B is transposed
// (n-outer -> kp-outer) via 8 strided selects to match dpas VNNI layout L2 (the
// only layout the dpas probe accepted). The block-load register linear order
// equals the flat slab order: v[i]=byte(n=i/8, kp=i%8), so the transpose is a
// 16x8 -> 8x16 matrix transpose (kp/n field swap), not a gather.
inline void matmul_nvfp4_grouped_rows_xe2_v4(
    GpuEngine& ctx,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* expert_offsets,
    const int32_t* rows_per_expert,
    int localE,
    const uint8_t* const* W_coal_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0 || N % 16 != 0)
        throw std::runtime_error("xe2v4 grouped NVFP4 matmul requires K%16 and N%16");
    if (localE <= 0) return;
    namespace esimd = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    namespace esimd_x = sycl::ext::intel::experimental::esimd;
    using bf16_t = sycl::ext::oneapi::bfloat16;
    constexpr int TM = 8;    // dpas M (rows) per tile  = RepeatCount
    constexpr int TN = 16;  // dpas N (cols) per tile  = ExecutionSize
    constexpr int TK = 16;  // dpas K (contraction)    = SystolicDepth * OpsPerChannel
    constexpr int LANES = 16;
    constexpr int LBW = 16;  // block-load width (bytes); pitch==width => contiguous 128B
    constexpr int LBH = 8;   // block-load height (rows)
    auto& q = ctx.queue;
    int halfK = K / 2;
    int ktiles = K / TK;
    int ntiles = N / TN;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)localE, (size_t)ntiles * LANES),
                              sycl::range<2>(1, LANES)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int nt = (int)it.get_group(1);
                int offset = expert_offsets[e];
                int count = rows_per_expert[e];
                if (count <= 0) return;
                int mt = (count + TM - 1) / TM;

                const uint8_t* wcoal = W_coal_by_expert[e];
                const uint8_t* ws = W_scale_by_expert[e];
                const uint8_t* wcoal_nt = wcoal + (size_t)nt * ktiles * 128;
                float inv_dst = 1.0f / dst_scale_by_expert[e][0];

                for (int tile0 = 0; tile0 < mt; tile0 += LANES) {
                    int m_tile = tile0 + lane;        // each lane owns one m-tile
                    bool active = (m_tile < mt);
                    esimd::simd<float, TM * TN> c(0.0f);
                    for (int kt = 0; kt < ktiles; ++kt) {
                        // --- B: block-load the (n_tile,k_tile) slab (replicated) ---
                        auto v = esimd_x::lsc_load_2d<uint8_t, LBW, LBH, /*NBlocks*/1,
                                                       /*Transposed*/false,
                                                       /*Transformed*/false>(
                            wcoal_nt + (size_t)kt * 128,
                            LBW - 1, LBH - 1, LBW - 1, 0, 0);
                        // Transpose n-outer -> kp-outer: bt[kp*16+n] = v[n*8+kp].
                        esimd::simd<uint8_t, 128> bt;
                        #pragma unroll
                        for (int kp = 0; kp < 8; ++kp)
                            bt.select<16, 1>(kp * 16) = v.select<16, 8>(kp);
                        // Dequant the 16 e4m3 weight scales for this n-tile
                        // (contiguous in memory) via a single vectorized load+dequant.
                        auto wscale_raw = esimd::block_load<uint8_t, 16>(
                            ws + (size_t)kt * N + (size_t)nt * TN);
                        auto wscale_deq = esimd_vec::e4m3<16>(
                            esimd::convert<uint16_t>(wscale_raw));
                        // Replicate to 128: scale[kp*16+n] = wscale_deq[n].
                        esimd::simd<float, 128> scale;
                        #pragma unroll
                        for (int kp = 0; kp < 8; ++kp)
                            scale.select<16, 1>(kp * 16) = wscale_deq;
                        // Dequant lo/hi nibbles, scale, convert to bf16, interleave
                        // into VNNI L2: b[(kp*16+n)*2+kparity]. Process lo then hi so
                        // both float tiles are not live simultaneously.
                        esimd::simd<bf16_t, TK * TN> b;
                        {
                            auto lo_nib = esimd::convert<uint16_t>(
                                bt & esimd::simd<uint8_t, 128>(0x0f));
                            auto lo_f = esimd_vec::e2m1<128>(lo_nib) * scale;
                            b.select<128, 2>(0) = esimd::convert<bf16_t>(lo_f);
                        }
                        {
                            auto hi_nib = esimd::convert<uint16_t>(
                                bt >> 4);
                            auto hi_f = esimd_vec::e2m1<128>(hi_nib) * scale;
                            b.select<128, 2>(1) = esimd::convert<bf16_t>(hi_f);
                        }
                        // --- A: this lane's m-tile, 8 rows x 8 bytes (row-major) ---
                        esimd::simd<bf16_t, TM * TK> a;
                        if (active) {
                            esimd::simd<uint8_t, 64> abytes;
                            esimd::simd<uint8_t, TM> ascale_raw(0);
                            #pragma unroll
                            for (int m = 0; m < TM; ++m) {
                                int row = offset + m_tile * TM + m;
                                bool ok = (row - offset) < count;
                                int safe_row = ok ? row : offset;
                                const uint8_t* rp = A_packed +
                                    (size_t)safe_row * halfK + (size_t)kt * (TK / 2);
                                abytes.select<8, 1>(m * 8) =
                                    esimd::block_load<uint8_t, 8>(rp);
                                if (ok) ascale_raw[m] =
                                    A_scale[(size_t)safe_row * ktiles + kt];
                            }
                            // Vectorized e4m3 dequant of the 8 row scales.
                        auto ascale_deq = esimd_vec::e4m3<TM>(
                            esimd::convert<uint16_t>(ascale_raw));
                        // Replicate per-row scale to 64: ascale_vec[m*8+j]=ascale_deq[m].
                        esimd::simd<float, 64> ascale_vec;
                        #pragma unroll
                        for (int m = 0; m < TM; ++m)
                            ascale_vec.select<8, 1>(m * 8) =
                                esimd::simd<float, 8>(ascale_deq[m]);
                        {
                            auto lo_nib = esimd::convert<uint16_t>(
                                abytes & esimd::simd<uint8_t, 64>(0x0f));
                            auto lo_f = esimd_vec::e2m1<64>(lo_nib) * ascale_vec;
                            a.select<64, 2>(0) = esimd::convert<bf16_t>(lo_f);
                        }
                        {
                            auto hi_nib = esimd::convert<uint16_t>(
                                abytes >> 4);
                            auto hi_f = esimd_vec::e2m1<64>(hi_nib) * ascale_vec;
                            a.select<64, 2>(1) = esimd::convert<bf16_t>(hi_f);
                        }
                        } else {
                            a = esimd::simd<bf16_t, TM * TK>(bf16_t(0.0f));
                        }
                        // --- C += A @ B (dpas: Result = C + A*B; B is VNNI) ---
                        c = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(c, b, a);
                    }
                    // --- store this lane's 8x16 result tile ---
                    if (active) {
                        #pragma unroll
                        for (int m = 0; m < TM; ++m) {
                            int row = offset + m_tile * TM + m;
                            if ((row - offset) >= count) continue;
                            #pragma unroll
                            for (int n = 0; n < TN; ++n)
                                C[(size_t)row * N + nt * TN + n] =
                                    float_to_bf16(c[m * TN + n] * inv_dst);
                        }
                    }
                }
            });
    });
}

// v5: same full-ESIMD (block_load + xmx::dpas) kernel as v4, but lanes own
// N-tiles instead of M-tiles. v4 wasted 50-94% of dpas calls because ESIMD
// subgroups are lockstep: all 16 lanes ran the full kt-loop even when only
// 1-8 M-tiles were active (MoE, small mt). By making each lane own one
// N-tile (ntiles=88 >= 16 -> all 16 lanes busy), dpas calls drop 14.7x at
// p=512. Roles swap vs v4: A is now SHARED across lanes (one M-tile, loaded
// redundantly/coalesced) and B is PER-LANE (each lane's own N-tile, loaded
// from the coalesced weight layout via a contiguous 128B block_load). The
// dequant arithmetic is identical to v4; only the load site + loop nesting
// differ. The M-tile dimension becomes an inner sequential loop (was the
// lane-parallel tile0 loop in v4).
inline void matmul_nvfp4_grouped_rows_xe2_v5(
    GpuEngine& ctx,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* expert_offsets,
    const int32_t* rows_per_expert,
    int localE,
    const uint8_t* const* W_coal_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0 || N % 16 != 0)
        throw std::runtime_error("xe2v5 grouped NVFP4 matmul requires K%16 and N%16");
    if (localE <= 0) return;
    namespace esimd = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    using bf16_t = sycl::ext::oneapi::bfloat16;
    constexpr int TM = 8;    // dpas M (rows) per tile  = RepeatCount
    constexpr int TN = 16;  // dpas N (cols) per tile  = ExecutionSize
    constexpr int TK = 16;  // dpas K (contraction)    = SystolicDepth * OpsPerChannel
    constexpr int LANES = 16;
    auto& q = ctx.queue;
    int halfK = K / 2;
    int ktiles = K / TK;
    int ntiles = N / TN;
    int nchunks = (ntiles + LANES - 1) / LANES;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)localE, (size_t)nchunks * LANES),
                              sycl::range<2>(1, LANES)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int nt_chunk = (int)it.get_group(1);
                int nt = nt_chunk * LANES + lane;       // this lane's N-tile
                int offset = expert_offsets[e];
                int count = rows_per_expert[e];
                if (count <= 0) return;
                int mt = (count + TM - 1) / TM;
                if (nt >= ntiles) return;               // tail lanes of last chunk

                const uint8_t* wcoal = W_coal_by_expert[e];
                const uint8_t* ws = W_scale_by_expert[e];
                const uint8_t* wcoal_nt = wcoal + (size_t)nt * ktiles * 128;
                const uint8_t* ws_nt = ws + (size_t)nt * TN;
                float inv_dst = 1.0f / dst_scale_by_expert[e][0];

                for (int m_tile = 0; m_tile < mt; ++m_tile) {
                    esimd::simd<float, TM * TN> c(0.0f);
                    for (int kt = 0; kt < ktiles; ++kt) {
                        // --- B: this lane's own N-tile slab (128B contiguous) ---
                        auto v = esimd::block_load<uint8_t, 128>(
                            wcoal_nt + (size_t)kt * 128);
                        // Transpose n-outer -> kp-outer: bt[kp*16+n] = v[n*8+kp].
                        esimd::simd<uint8_t, 128> bt;
                        #pragma unroll
                        for (int kp = 0; kp < 8; ++kp)
                            bt.select<16, 1>(kp * 16) = v.select<16, 8>(kp);
                        // Dequant the 16 e4m3 weight scales for this N-tile.
                        auto wscale_raw = esimd::block_load<uint8_t, 16>(
                            ws_nt + (size_t)kt * N);
                        auto wscale_deq = esimd_vec::e4m3<16>(
                            esimd::convert<uint16_t>(wscale_raw));
                        esimd::simd<float, 128> scale;
                        #pragma unroll
                        for (int kp = 0; kp < 8; ++kp)
                            scale.select<16, 1>(kp * 16) = wscale_deq;
                        esimd::simd<bf16_t, TK * TN> b;
                        {
                            auto lo_nib = esimd::convert<uint16_t>(
                                bt & esimd::simd<uint8_t, 128>(0x0f));
                            auto lo_f = esimd_vec::e2m1<128>(lo_nib) * scale;
                            b.select<128, 2>(0) = esimd::convert<bf16_t>(lo_f);
                        }
                        {
                            auto hi_nib = esimd::convert<uint16_t>(
                                bt >> 4);
                            auto hi_f = esimd_vec::e2m1<128>(hi_nib) * scale;
                            b.select<128, 2>(1) = esimd::convert<bf16_t>(hi_f);
                        }
                        // --- A: shared M-tile, 8 rows x 8 bytes (row-major) ---
                        esimd::simd<uint8_t, 64> abytes;
                        esimd::simd<uint8_t, TM> ascale_raw(0);
                        #pragma unroll
                        for (int m = 0; m < TM; ++m) {
                            int row = offset + m_tile * TM + m;
                            bool ok = (row - offset) < count;
                            int safe_row = ok ? row : offset;
                            const uint8_t* rp = A_packed +
                                (size_t)safe_row * halfK + (size_t)kt * (TK / 2);
                            abytes.select<8, 1>(m * 8) =
                                esimd::block_load<uint8_t, 8>(rp);
                            if (ok) ascale_raw[m] =
                                A_scale[(size_t)safe_row * ktiles + kt];
                        }
                        auto ascale_deq = esimd_vec::e4m3<TM>(
                            esimd::convert<uint16_t>(ascale_raw));
                        esimd::simd<float, 64> ascale_vec;
                        #pragma unroll
                        for (int m = 0; m < TM; ++m)
                            ascale_vec.select<8, 1>(m * 8) =
                                esimd::simd<float, 8>(ascale_deq[m]);
                        esimd::simd<bf16_t, TM * TK> a;
                        {
                            auto lo_nib = esimd::convert<uint16_t>(
                                abytes & esimd::simd<uint8_t, 64>(0x0f));
                            auto lo_f = esimd_vec::e2m1<64>(lo_nib) * ascale_vec;
                            a.select<64, 2>(0) = esimd::convert<bf16_t>(lo_f);
                        }
                        {
                            auto hi_nib = esimd::convert<uint16_t>(
                                abytes >> 4);
                            auto hi_f = esimd_vec::e2m1<64>(hi_nib) * ascale_vec;
                            a.select<64, 2>(1) = esimd::convert<bf16_t>(hi_f);
                        }
                        // --- C += A @ B (dpas: per-lane, own A & B) ---
                        c = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(c, b, a);
                    }
                    // --- store this lane's 8x16 result tile ---
                    #pragma unroll
                    for (int m = 0; m < TM; ++m) {
                        int row = offset + m_tile * TM + m;
                        if ((row - offset) >= count) continue;
                        #pragma unroll
                        for (int n = 0; n < TN; ++n)
                            C[(size_t)row * N + nt * TN + n] =
                                float_to_bf16(c[m * TN + n] * inv_dst);
                    }
                }
            });
    });
}

// v6: parallelize M across work-groups (best of both regimes).
// v5 serialized the M-tiles in an inner loop, which dominated latency when mt
// was large (e.g. p=8192 -> mt=8 -> ~8x slower than onednn-loop). v6 lifts m_tile
// into the work-group grid: dim0 = localE*max_mt, so each work-group does exactly
// one (expert, m_tile, n_chunk). The 16 lanes still own N-tiles (small-batch win
// preserved: at mt=1 the grid collapses to v5's shape). B for a given (expert,
// n_chunk) is loaded by mt work-groups but hits L2; A is unique per (expert, m_tile).
inline void matmul_nvfp4_grouped_rows_xe2_v6(
    GpuEngine& ctx,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* expert_offsets,
    const int32_t* rows_per_expert,
    int localE,
    int max_mt,
    const uint8_t* const* W_coal_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0 || N % 16 != 0)
        throw std::runtime_error("xe2v6 grouped NVFP4 matmul requires K%16 and N%16");
    if (localE <= 0 || max_mt <= 0) return;
    namespace esimd = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    using bf16_t = sycl::ext::oneapi::bfloat16;
    constexpr int TM = 8;    // dpas M (rows) per tile  = RepeatCount
    constexpr int TN = 16;  // dpas N (cols) per tile  = ExecutionSize
    constexpr int TK = 16;  // dpas K (contraction)    = SystolicDepth * OpsPerChannel
    constexpr int LANES = 16;
    auto& q = ctx.queue;
    int halfK = K / 2;
    int ktiles = K / TK;
    int ntiles = N / TN;
    int nchunks = (ntiles + LANES - 1) / LANES;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)localE * max_mt, (size_t)nchunks * LANES),
                              sycl::range<2>(1, LANES)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int em = (int)it.get_group(0);
                int e = em / max_mt;
                int m_tile = em % max_mt;
                int lane = (int)it.get_local_id(1);
                int nt_chunk = (int)it.get_group(1);
                int nt = nt_chunk * LANES + lane;       // this lane's N-tile
                int offset = expert_offsets[e];
                int count = rows_per_expert[e];
                if (count <= 0) return;
                int mt = (count + TM - 1) / TM;
                if (m_tile >= mt) return;                // expert has fewer M-tiles than max_mt
                if (nt >= ntiles) return;                // tail lanes of last chunk

                const uint8_t* wcoal = W_coal_by_expert[e];
                const uint8_t* ws = W_scale_by_expert[e];
                const uint8_t* wcoal_nt = wcoal + (size_t)nt * ktiles * 128;
                const uint8_t* ws_nt = ws + (size_t)nt * TN;
                float inv_dst = 1.0f / dst_scale_by_expert[e][0];

                esimd::simd<float, TM * TN> c(0.0f);
                for (int kt = 0; kt < ktiles; ++kt) {
                    // --- B: this lane's own N-tile slab (128B contiguous) ---
                    auto v = esimd::block_load<uint8_t, 128>(
                        wcoal_nt + (size_t)kt * 128);
                    // Transpose n-outer -> kp-outer: bt[kp*16+n] = v[n*8+kp].
                    esimd::simd<uint8_t, 128> bt;
                    #pragma unroll
                    for (int kp = 0; kp < 8; ++kp)
                        bt.select<16, 1>(kp * 16) = v.select<16, 8>(kp);
                    // Dequant the 16 e4m3 weight scales for this N-tile.
                    auto wscale_raw = esimd::block_load<uint8_t, 16>(
                        ws_nt + (size_t)kt * N);
                    auto wscale_deq = esimd_vec::e4m3<16>(
                        esimd::convert<uint16_t>(wscale_raw));
                    esimd::simd<float, 128> scale;
                    #pragma unroll
                    for (int kp = 0; kp < 8; ++kp)
                        scale.select<16, 1>(kp * 16) = wscale_deq;
                    esimd::simd<bf16_t, TK * TN> b;
                    {
                        auto lo_nib = esimd::convert<uint16_t>(
                            bt & esimd::simd<uint8_t, 128>(0x0f));
                        auto lo_f = esimd_vec::e2m1<128>(lo_nib) * scale;
                        b.select<128, 2>(0) = esimd::convert<bf16_t>(lo_f);
                    }
                    {
                        auto hi_nib = esimd::convert<uint16_t>(
                            bt >> 4);
                        auto hi_f = esimd_vec::e2m1<128>(hi_nib) * scale;
                        b.select<128, 2>(1) = esimd::convert<bf16_t>(hi_f);
                    }
                    // --- A: this M-tile, 8 rows x 8 bytes (row-major) ---
                    esimd::simd<uint8_t, 64> abytes;
                    esimd::simd<uint8_t, TM> ascale_raw(0);
                    #pragma unroll
                    for (int m = 0; m < TM; ++m) {
                        int row = offset + m_tile * TM + m;
                        bool ok = (row - offset) < count;
                        int safe_row = ok ? row : offset;
                        const uint8_t* rp = A_packed +
                            (size_t)safe_row * halfK + (size_t)kt * (TK / 2);
                        abytes.select<8, 1>(m * 8) =
                            esimd::block_load<uint8_t, 8>(rp);
                        if (ok) ascale_raw[m] =
                            A_scale[(size_t)safe_row * ktiles + kt];
                    }
                    auto ascale_deq = esimd_vec::e4m3<TM>(
                        esimd::convert<uint16_t>(ascale_raw));
                    esimd::simd<float, 64> ascale_vec;
                    #pragma unroll
                    for (int m = 0; m < TM; ++m)
                        ascale_vec.select<8, 1>(m * 8) =
                            esimd::simd<float, 8>(ascale_deq[m]);
                    esimd::simd<bf16_t, TM * TK> a;
                    {
                        auto lo_nib = esimd::convert<uint16_t>(
                            abytes & esimd::simd<uint8_t, 64>(0x0f));
                        auto lo_f = esimd_vec::e2m1<64>(lo_nib) * ascale_vec;
                        a.select<64, 2>(0) = esimd::convert<bf16_t>(lo_f);
                    }
                    {
                        auto hi_nib = esimd::convert<uint16_t>(
                            abytes >> 4);
                        auto hi_f = esimd_vec::e2m1<64>(hi_nib) * ascale_vec;
                        a.select<64, 2>(1) = esimd::convert<bf16_t>(hi_f);
                    }
                    // --- C += A @ B (dpas: per-lane, own A & B) ---
                    c = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(c, b, a);
                }
                // --- store this lane's 8x16 result tile ---
                #pragma unroll
                for (int m = 0; m < TM; ++m) {
                    int row = offset + m_tile * TM + m;
                    if ((row - offset) >= count) continue;
                    #pragma unroll
                    for (int n = 0; n < TN; ++n)
                        C[(size_t)row * N + nt * TN + n] =
                            float_to_bf16(c[m * TN + n] * inv_dst);
                }
            });
    });
}

// v7: SLM-cached B (hardware-aligned fix for the large-batch latency gap).
// v5/v6 reload B from global every m_tile (mt times) and every kt; at large mt this
// dominates latency (global block_load stalls, ~4% of 608 GB/s peak -> NOT BW bound but
// stall/occupancy bound). v7 dequants B once per kt-chunk into SLM (128 KB available, unused
// by v5/v6), then all mt m_tiles read B from SLM (~5-10 cyc) instead of global.
// gateup full B = 360 KB > 128 KB -> kt-chunk KC=8 (64 KB chunk); down full B = 90 KB fits.
// mt (<=8) live float accumulators persist across kt-chunks. Structure = v5 (lanes->N-tiles,
// M-tile serial inner loop); v6's M-parallelism reverted (it gave no speedup, down p=8192
// byte-identical v5/v6 -> latency not parallelism is the lever).
inline void matmul_nvfp4_grouped_rows_xe2_v7(
    GpuEngine& ctx,
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    int K,
    const int32_t* expert_offsets,
    const int32_t* rows_per_expert,
    int localE,
    const uint8_t* const* W_coal_by_expert,
    const uint8_t* const* W_scale_by_expert,
    const float* const* dst_scale_by_expert,
    int N,
    bf16* C)
{
    if (K % 16 != 0 || N % 16 != 0)
        throw std::runtime_error("xe2v7 grouped NVFP4 matmul requires K%16 and N%16");
    if (localE <= 0) return;
    namespace esimd = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    using bf16_t = sycl::ext::oneapi::bfloat16;
    constexpr int TM = 8;    // dpas M (rows) per tile  = RepeatCount
    constexpr int TN = 16;   // dpas N (cols) per tile  = ExecutionSize
    constexpr int TK = 16;   // dpas K (contraction)    = SystolicDepth * OpsPerChannel
    constexpr int LANES = 16;
    constexpr int KC = 8;    // kt-chunk size (SLM budget: KC*128*16 = 16KB? -> see below)
    auto& q = ctx.queue;
    int halfK = K / 2;
    int ktiles = K / TK;
    int ntiles = N / TN;
    int nchunks = (ntiles + LANES - 1) / LANES;

    // SLM layout: per lane per kt, the dequantized B is simd<bf16_t,TK*TN> = 256 bf16 = 512 B.
    // Slot offset(ki,lane) = ki*(LANES*512) + lane*512. Full chunk = KC*LANES*512 =
    // 8*16*512 = 65536 B = 64 KB (fits 128 KB). One code path for gateup/down (KC=8).
    constexpr int SLM_SIZE = KC * LANES * 512;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)localE, (size_t)nchunks * LANES),
                              sycl::range<2>(1, LANES)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                esimd::slm_init<SLM_SIZE>();
                int e = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int nt_chunk = (int)it.get_group(1);
                int nt = nt_chunk * LANES + lane;       // this lane's N-tile
                int offset = expert_offsets[e];
                int count = rows_per_expert[e];
                if (count <= 0) return;
                int mt = (count + TM - 1) / TM;
                if (nt >= ntiles) return;               // tail lanes of last chunk

                const uint8_t* wcoal = W_coal_by_expert[e];
                const uint8_t* ws = W_scale_by_expert[e];
                const uint8_t* wcoal_nt = wcoal + (size_t)nt * ktiles * 128;
                const uint8_t* ws_nt = ws + (size_t)nt * TN;
                float inv_dst = 1.0f / dst_scale_by_expert[e][0];

                // mt live accumulators (persist across kt-chunks).
                esimd::simd<float, TM * TN> acc[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] = esimd::simd<float, TM * TN>(0.0f);

                int kchunks = (ktiles + KC - 1) / KC;
                for (int kc = 0; kc < kchunks; ++kc) {
                    int kt_base = kc * KC;
                    int kt_end = kt_base + KC;
                    if (kt_end > ktiles) kt_end = ktiles;
                    int kc_len = kt_end - kt_base;

                    // --- Load + dequant this chunk's B into SLM (once per chunk) ---
                    #pragma unroll
                    for (int ki = 0; ki < KC; ++ki) {
                        int kt = kt_base + ki;
                        uint32_t slm_off = (uint32_t)ki * LANES * 512 + (uint32_t)lane * 512;
                        if (kt < kt_end) {
                            auto v = esimd::block_load<uint8_t, 128>(
                                wcoal_nt + (size_t)kt * 128);
                            // Transpose n-outer -> kp-outer: bt[kp*16+n] = v[n*8+kp].
                            esimd::simd<uint8_t, 128> bt;
                            #pragma unroll
                            for (int kp = 0; kp < 8; ++kp)
                                bt.select<16, 1>(kp * 16) = v.select<16, 8>(kp);
                            auto wscale_raw = esimd::block_load<uint8_t, 16>(
                                ws_nt + (size_t)kt * N);
                            auto wscale_deq = esimd_vec::e4m3<16>(
                                esimd::convert<uint16_t>(wscale_raw));
                            esimd::simd<float, 128> scale;
                            #pragma unroll
                            for (int kp = 0; kp < 8; ++kp)
                                scale.select<16, 1>(kp * 16) = wscale_deq;
                            esimd::simd<bf16_t, TK * TN> b;
                            {
                                auto lo_nib = esimd::convert<uint16_t>(
                                    bt & esimd::simd<uint8_t, 128>(0x0f));
                                auto lo_f = esimd_vec::e2m1<128>(lo_nib) * scale;
                                b.select<128, 2>(0) = esimd::convert<bf16_t>(lo_f);
                            }
                            {
                                auto hi_nib = esimd::convert<uint16_t>(
                                    bt >> 4);
                                auto hi_f = esimd_vec::e2m1<128>(hi_nib) * scale;
                                b.select<128, 2>(1) = esimd::convert<bf16_t>(hi_f);
                            }
                            esimd::slm_block_store(slm_off, b);
                        }
                    }

                    // --- For every m_tile, dpas using A(global) + B(SLM) ---
                    for (int m_tile = 0; m_tile < mt; ++m_tile) {
                        esimd::simd<float, TM * TN>& c = acc[m_tile];
                        #pragma unroll
                        for (int ki = 0; ki < KC; ++ki) {
                            int kt = kt_base + ki;
                            if (kt >= kt_end) break;
                            uint32_t slm_off = (uint32_t)ki * LANES * 512 + (uint32_t)lane * 512;
                            auto b = esimd::slm_block_load<bf16_t, TK * TN>(slm_off);
                            // --- A: this M-tile, 8 rows x 8 bytes (row-major) ---
                            esimd::simd<uint8_t, 64> abytes;
                            esimd::simd<uint8_t, TM> ascale_raw(0);
                            #pragma unroll
                            for (int m = 0; m < TM; ++m) {
                                int row = offset + m_tile * TM + m;
                                bool ok = (row - offset) < count;
                                int safe_row = ok ? row : offset;
                                const uint8_t* rp = A_packed +
                                    (size_t)safe_row * halfK + (size_t)kt * (TK / 2);
                                abytes.select<8, 1>(m * 8) =
                                    esimd::block_load<uint8_t, 8>(rp);
                                if (ok) ascale_raw[m] =
                                    A_scale[(size_t)safe_row * ktiles + kt];
                            }
                            auto ascale_deq = esimd_vec::e4m3<TM>(
                                esimd::convert<uint16_t>(ascale_raw));
                            esimd::simd<float, 64> ascale_vec;
                            #pragma unroll
                            for (int m = 0; m < TM; ++m)
                                ascale_vec.select<8, 1>(m * 8) =
                                    esimd::simd<float, 8>(ascale_deq[m]);
                            esimd::simd<bf16_t, TM * TK> a;
                            {
                                auto lo_nib = esimd::convert<uint16_t>(
                                    abytes & esimd::simd<uint8_t, 64>(0x0f));
                                auto lo_f = esimd_vec::e2m1<64>(lo_nib) * ascale_vec;
                                a.select<64, 2>(0) = esimd::convert<bf16_t>(lo_f);
                            }
                            {
                                auto hi_nib = esimd::convert<uint16_t>(
                                    abytes >> 4);
                                auto hi_f = esimd_vec::e2m1<64>(hi_nib) * ascale_vec;
                                a.select<64, 2>(1) = esimd::convert<bf16_t>(hi_f);
                            }
                            c = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(c, b, a);
                        }
                    }
                }

                // --- store all m_tiles' results ---
                #pragma unroll
                for (int m_tile = 0; m_tile < 8; ++m_tile) {
                    if (m_tile >= mt) break;
                    esimd::simd<float, TM * TN>& c = acc[m_tile];
                    #pragma unroll
                    for (int m = 0; m < TM; ++m) {
                        int row = offset + m_tile * TM + m;
                        if ((row - offset) >= count) continue;
                        #pragma unroll
                        for (int n = 0; n < TN; ++n)
                            C[(size_t)row * N + nt * TN + n] =
                                float_to_bf16(c[m * TN + n] * inv_dst);
                    }
                }
            });
    });
}

#pragma once
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
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

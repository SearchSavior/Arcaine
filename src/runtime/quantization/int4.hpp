#pragma once
// compressed-tensors "pack-quantized" int4 W4A16 weight decompression.
//
// Symmetric int4 weights (group_size along K, signed two's-complement nibbles)
// are kept packed in device memory and decompressed on the fly by oneDNN's
// matmul weight-decompression path: src stays BF16, weights are s4, and a
// per-group BF16 scale is applied along K.  This mirrors nvfp4.hpp but is
// simpler — W4A16 means there is no activation packing and no global scales.
//
// Packed layout: the safetensors `weight_packed` (I32, shape (N, K/8)) is a
// little-endian byte stream (N, K/2) with two 4-bit values per byte, low-nibble
// first — exactly oneDNN's s4 `tag::ba` for logical dims {K, N}.  The loader
// rebases compressed-tensors' unsigned zero-point-8 nibbles to two's-complement
// s4 (XOR 0x88), so no zero-point argument is needed here.  `weight_scale` is
// transposed at load to (K/group_size, N) BF16.
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "buffer.hpp"
#include "engine.hpp"

struct Int4Linear {
    int in_features = 0;
    int out_features = 0;
    int group_size = 64;
    // s4 weights, logical shape (out_features, in_features); raw packed bytes
    // (low-nibble first, 2 s4/byte) consumable directly as oneDNN s4 tag::ba.
    GpuBuffer<uint8_t> weight_packed;
    // BF16 per-group scales transposed for oneDNN, logical shape
    // (in_features / group_size, out_features).
    GpuBuffer<bf16> weight_scale;

    // oneDNN impl-preferred weight layout, materialized once on first use under
    // DIFF_INT4_WEIGHT_LAYOUT=any. The raw tag::ba layout forces oneDNN's
    // generic decompress kernel; the reordered blocked layout unlocks the fast
    // one. Cached per-GPU so a sharded weight reorders on its owning device.
    mutable GpuBuffer<uint8_t> weight_any;
    mutable size_t weight_any_bytes = 0;
    mutable int weight_any_gpu = -1;

    bool empty() const { return weight_packed.empty(); }
};

enum class Int4WeightLayout { Raw = 0, Any = 1 };

inline Int4WeightLayout int4_weight_layout() {
    static Int4WeightLayout layout = [] {
        const char* env = std::getenv("DIFF_INT4_WEIGHT_LAYOUT");
        // Measured: the reordered "any" layout is *slower* than raw tag::ba for
        // the small per-expert decode GEMMs (oneDNN picks a worse kernel), so
        // raw is the default. Set DIFF_INT4_WEIGHT_LAYOUT=any to A/B on shapes
        // where it might win (e.g. large prefill GEMMs).
        if (env && std::string(env) == "any") return Int4WeightLayout::Any;
        return Int4WeightLayout::Raw;
    }();
    return layout;
}

struct Int4MatmulKey {
    int gpu, M, K, N, group;
    bool operator==(const Int4MatmulKey& o) const {
        return gpu == o.gpu && M == o.M && K == o.K && N == o.N && group == o.group;
    }
};

struct Int4MatmulKeyHash {
    size_t operator()(const Int4MatmulKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.group) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Int4MatmulEntry {
    dnnl::matmul primitive;
    dnnl::memory::desc weights_md;
};

inline Int4MatmulEntry& int4_matmul_entry(GpuEngine& ctx, int M, int K, int N, int group) {
    static std::unordered_map<Int4MatmulKey, Int4MatmulEntry, Int4MatmulKeyHash> cache;
    Int4MatmulKey key{ctx.index, M, K, N, group};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::primitive_attr attr;
        // Per-group weight scales along K (group along dim 0, per-channel on N).
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {group, 1}, dt::bf16);
        // Enable integer-weight decompression with a BF16 compute math mode.
        attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, /*apply_to_int=*/true);
        auto weights_md = (int4_weight_layout() == Int4WeightLayout::Any)
            ? dnnl::memory::desc({K, N}, dt::s4, tag::any)
            : dnnl::memory::desc({K, N}, dt::s4, tag::ba);
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            weights_md,
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            attr);
        auto result = cache.emplace(key,
            Int4MatmulEntry{dnnl::matmul(pd), pd.weights_desc()});
        it = result.first;
    }
    return it->second;
}

// Returns a weight pointer in the layout `weights_md` expects. For Raw this is
// the packed buffer as-is; for Any it lazily reorders the raw tag::ba weights
// into the impl-preferred blocked layout and caches it on the matmul's device.
inline const uint8_t* int4_weight_data(const Int4Linear& W,
                                       const dnnl::memory::desc& weights_md,
                                       int K, int N, GpuEngine& ctx) {
    if (int4_weight_layout() == Int4WeightLayout::Raw)
        return W.weight_packed.data();

    size_t bytes = weights_md.get_size();
    if (W.weight_any.empty() || W.weight_any_bytes != bytes || W.weight_any_gpu != ctx.index) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        W.weight_any = GpuBuffer<uint8_t>(bytes, ctx.queue);
        W.weight_any_bytes = bytes;
        W.weight_any_gpu = ctx.index;
        auto raw_md = dnnl::memory::desc({K, N}, dt::s4, tag::ba);
        auto raw_mem = dnnl::sycl_interop::make_memory(
            raw_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(W.weight_packed.data()));
        auto any_mem = dnnl::sycl_interop::make_memory(
            weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            W.weight_any.data());
        dnnl::reorder(raw_mem, any_mem).execute(ctx.stream, raw_mem, any_mem);
        ctx.stream.wait();
    }
    return W.weight_any.data();
}

// C (M,N) = A (M,K) @ dequant(W)^T, where W is logical (N,K) s4 with per-group
// BF16 scales.  A and C are BF16.  Async on ctx's stream.
inline void matmul_int4(
    const bf16* A,
    int M,
    int K,
    const Int4Linear& W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.in_features != K)
        throw std::runtime_error("matmul_int4: K does not match weight shape");
    if (K % W.group_size != 0)
        throw std::runtime_error("matmul_int4: K must be divisible by group_size");

    int N = W.out_features;
    int G = K / W.group_size;

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto src_md = dnnl::memory::desc({M, K}, dt::bf16, tag::ab);
    auto dst_md = dnnl::memory::desc({M, N}, dt::bf16, tag::ab);
    auto wscale_md = dnnl::memory::desc({G, N}, dt::bf16, tag::ab);

    auto& entry = int4_matmul_entry(ctx, M, K, N, W.group_size);
    const uint8_t* weight_data = int4_weight_data(W, entry.weights_md, K, N, ctx);

    entry.primitive.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            entry.weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(weight_data))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            wscale_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(W.weight_scale.data()))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

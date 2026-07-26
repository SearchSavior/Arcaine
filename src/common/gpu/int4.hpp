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
#include "ops.hpp"

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

    // Optional asymmetric zero-point correction (compressed-tensors AWQ):
    // scale*zp transposed to (groups, out_features) BF16. Empty for symmetric
    // (diffusion) checkpoints — matmul_int4 skips the correction entirely.
    GpuBuffer<bf16> weight_zp_corr;

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

// SA(M, G) = sum_{k in group g} A(m, k), with G = K / group. A is (M,K) row-major.
inline void group_sum_bf16(sycl::queue& q, const bf16* A, int M, int K,
                           int group, bf16* SA) {
    int G = K / group;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)M * G), [=](sycl::id<1> id) {
            size_t idx = id[0];
            int m = (int)(idx / (size_t)G);
            int g = (int)(idx % (size_t)G);
            const bf16* row = A + (size_t)m * K;
            float acc = 0.0f;
            for (int j = 0; j < group; ++j)
                acc += bf16_to_float(row[(size_t)g * group + j]);
            SA[(size_t)m * G + g] = float_to_bf16(acc);
        });
    });
}

// C[i] = C[i] - X[i], element-wise over n BF16 values.
inline void axpy_bf16_inplace_sub(sycl::queue& q, bf16* C, const bf16* X, size_t n) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            C[id[0]] = float_to_bf16(
                bf16_to_float(C[id[0]]) - bf16_to_float(X[id[0]]));
        });
    });
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

    // Asymmetric INT4 (compressed-tensors AWQ) zero-point correction.
    //
    // The weight is w = scale*(q - zp) = scale*q - scale*zp. The oneDNN s4 GEMM
    // above computed only Part1 = scale*q: its decompression has no zero-point
    // argument and the XOR-0x88 rebased nibbles are symmetric around 0, so the
    // scale*zp term is missing. We must add it back here:
    //     C = Part1 - corr,   corr = group_sum(A) @ (scale*zp)
    //   where group_sum(A)[m,g] = sum_{k in group g} A[m,k]  (G = K/group_size).
    //
    // LOAD-BEARING — not optional, not gated. Without this subtraction every
    // output channel is biased by a per-group constant scale*zp * sum(A over
    // the group); those biases accumulate across all groups and every layer,
    // collapsing generation to a degenerate repeating token. Verified on
    // cyankiwi_gemma-4-12B-it-qat-AWQ-INT4: with the block below, "capital of
    // France" -> "Paris"; without it, -> "предметы предметы предметы ...".
    // The correction is exact in f64 (ref sim) and ~0.1% rel err in bf16.
    //
    // Only asymmetric checkpoints populate weight_zp_corr (loader.cpp
    // upload_int4_linear_awq); symmetric int4 checkpoints (diffusion_gemma)
    // leave it empty and skip this branch untouched, so dense/symmetric paths
    // are unaffected.
    //
    // Layout (the second load-bearing detail): weight_zp_corr is (G, N)
    // row-major holding c^T where c[n,g] = scale*zp, so the no-transpose GEMM
    // gives corr = SA @ c^T = sum_g SA[m,g]*c[n,g]. MUST use matmul_bf16_nn
    // (weights tag::ab). matmul_bf16 forces tag::ba and would read this (G,N)
    // buffer *transposed*, producing garbage even with the right values.
    if (!W.weight_zp_corr.empty()) {
        int G = K / W.group_size;
        GpuBuffer<bf16> SA((size_t)M * G, ctx.queue);
        group_sum_bf16(ctx.queue, A, M, K, W.group_size, SA.data());
        GpuBuffer<bf16> corr((size_t)M * N, ctx.queue);
        matmul_bf16_nn(SA.data(), M, G, W.weight_zp_corr.data(), N, corr.data(), ctx);
        axpy_bf16_inplace_sub(ctx.queue, C, corr.data(), (size_t)M * N);
    }
}

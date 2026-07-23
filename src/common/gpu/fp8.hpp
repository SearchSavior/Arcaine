#pragma once

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <unordered_map>

#include "buffer.hpp"
#include "engine.hpp"

// compressed-tensors float-quantized linear:
//   dequant_weight[n, k] = float(weight_e4m3[n, k]) * weight_scale[n]
// The checkpoint stays resident as E4M3. oneDNN's Intel GPU weight-
// decompression path converts it in the GEMM kernel, avoiding the ~10.8 GiB
// expansion that a persistent BF16 copy would require for Qwen3.5-27B.
struct Fp8Linear {
    int in_features = 0;
    int out_features = 0;
    GpuBuffer<uint8_t> weight;       // [N,K], F8_E4M3 bytes
    GpuBuffer<bf16> weight_scale;    // [N,1], BF16 per output channel

    bool empty() const { return weight.empty(); }
};

struct Fp8MatmulKey {
    int gpu = 0;
    int M = 0;
    int K = 0;
    int N = 0;
    bool operator==(const Fp8MatmulKey& other) const {
        return gpu == other.gpu && M == other.M && K == other.K && N == other.N;
    }
};

struct Fp8MatmulKeyHash {
    size_t operator()(const Fp8MatmulKey& key) const {
        size_t hash = std::hash<int>{}(key.gpu);
        auto mix = [&](int value) {
            hash ^= std::hash<int>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        };
        mix(key.M);
        mix(key.K);
        mix(key.N);
        return hash;
    }
};

struct Fp8MatmulEntry {
    dnnl::matmul primitive;
    dnnl::memory::desc weights_md;
};

inline Fp8MatmulEntry& fp8_matmul_entry(GpuEngine& ctx, int M, int K, int N) {
    static std::unordered_map<Fp8MatmulKey, Fp8MatmulEntry, Fp8MatmulKeyHash> cache;
    Fp8MatmulKey key{ctx.index, M, K, N};
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    dnnl::primitive_attr attr;
    // Logical weights are [K,N]. A scale group spans all K values and one N
    // value, so runtime scales are the checkpoint's contiguous [N,1] tensor.
    attr.set_scales(DNNL_ARG_WEIGHTS, /*mask=*/3, {K, 1}, dt::bf16);
    auto weights_md = dnnl::memory::desc({K, N}, dt::f8_e4m3, tag::ba);
    dnnl::matmul::primitive_desc pd(
        ctx.engine,
        dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
        weights_md,
        dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
        attr);

    auto inserted = cache.emplace(
        key, Fp8MatmulEntry{dnnl::matmul(pd), pd.weights_desc()});
    return inserted.first->second;
}

inline void matmul_fp8(
    const bf16* A,
    int M,
    int K,
    const Fp8Linear& W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0)) {
    if (W.in_features != K)
        throw std::runtime_error("matmul_fp8: K does not match weight shape");
    if (W.weight.empty() || W.weight_scale.empty())
        throw std::runtime_error("matmul_fp8: missing weight or channel scales");

    int N = W.out_features;
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto src_md = dnnl::memory::desc({M, K}, dt::bf16, tag::ab);
    auto dst_md = dnnl::memory::desc({M, N}, dt::bf16, tag::ab);
    auto scales_md = dnnl::memory::desc({N, 1}, dt::bf16, tag::ab);
    auto& entry = fp8_matmul_entry(ctx, M, K, N);

    entry.primitive.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            entry.weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(W.weight.data()))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
            dnnl::sycl_interop::make_memory(
                scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
                W.weight_scale.data())},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)},
    });
}

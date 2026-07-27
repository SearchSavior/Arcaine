#pragma once
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <unordered_map>
#include <tuple>
#include <functional>
#include "engine.hpp"
#include "buffer.hpp"

// ---------------------------------------------------------------------------
// Primitive caches — keyed by (gpu_index, shape) so each GPU gets its own
// oneDNN primitives built against the right engine.
// ---------------------------------------------------------------------------
struct MatmulKey {
    int gpu, M, K, N;
    bool operator==(const MatmulKey& o) const {
        return gpu==o.gpu && M==o.M && K==o.K && N==o.N;
    }
};
struct MatmulKeyHash {
    size_t operator()(const MatmulKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};

struct BatchedMatmulKey {
    int gpu, B, M, K, N;
    bool transpose_W;
    bool operator==(const BatchedMatmulKey& o) const {
        return gpu==o.gpu && B==o.B && M==o.M && K==o.K && N==o.N
            && transpose_W==o.transpose_W;
    }
};
struct BatchedMatmulKeyHash {
    size_t operator()(const BatchedMatmulKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.B)          + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.M)          + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.K)          + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.N)          + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<bool>{}(k.transpose_W)+ 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};


struct BatchedMatmulStridedKey {
    int gpu, B, M, K, N;
    dnnl_dim_t as0, as1, as2;
    dnnl_dim_t ws0, ws1, ws2;
    dnnl_dim_t cs0, cs1, cs2;
    bool operator==(const BatchedMatmulStridedKey& o) const {
        return gpu==o.gpu && B==o.B && M==o.M && K==o.K && N==o.N
            && as0==o.as0 && as1==o.as1 && as2==o.as2
            && ws0==o.ws0 && ws1==o.ws1 && ws2==o.ws2
            && cs0==o.cs0 && cs1==o.cs1 && cs2==o.cs2;
    }
};
struct BatchedMatmulStridedKeyHash {
    size_t operator()(const BatchedMatmulStridedKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        auto mix = [&](auto v) {
            h ^= std::hash<long long>{}((long long)v) + 0x9e3779b9 + (h<<6) + (h>>2);
        };
        mix(k.B); mix(k.M); mix(k.K); mix(k.N);
        mix(k.as0); mix(k.as1); mix(k.as2);
        mix(k.ws0); mix(k.ws1); mix(k.ws2);
        mix(k.cs0); mix(k.cs1); mix(k.cs2);
        return h;
    }
};

struct SoftmaxKey {
    int gpu, rows, cols;
    bool operator==(const SoftmaxKey& o) const {
        return gpu==o.gpu && rows==o.rows && cols==o.cols;
    }
};
struct SoftmaxKeyHash {
    size_t operator()(const SoftmaxKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.rows) + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.cols) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};

// ---------------------------------------------------------------------------
// BF16 matmul: C(M,N) = A(M,K) @ B(N,K)^T
// B stored (N,K) row-major; oneDNN transposes via format_tag::ba.
// ctx defaults to GPU 0 — existing callers (vision/audio embedder) need no change.
// Async — caller owns synchronization.
// ---------------------------------------------------------------------------
inline void matmul_bf16(
    const bf16* A, int M, int K,
    const bf16* B, int N,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0)
) {
    static std::unordered_map<MatmulKey, dnnl::matmul, MatmulKeyHash> cache;

    MatmulKey key{ctx.index, M, K, N};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt  = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            dnnl::memory::desc({K, N}, dt::bf16, tag::ba),
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab));
        cache[key] = dnnl::matmul(pd);
        it = cache.find(key);
    }

    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    it->second.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({K, N}, dt::bf16, tag::ba),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<bf16*>(B))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

// ---------------------------------------------------------------------------
// BF16 matmul (no-transpose variant): C(M,N) = A(M,K) @ B(K,N)
// ---------------------------------------------------------------------------
inline void matmul_bf16_nn(
    const bf16* A, int M, int K,
    const bf16* B, int N,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0)
) {
    static std::unordered_map<MatmulKey, dnnl::matmul, MatmulKeyHash> cache_nn;

    MatmulKey key{ctx.index, M, K, N};
    auto it = cache_nn.find(key);
    if (it == cache_nn.end()) {
        using dt  = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            dnnl::memory::desc({K, N}, dt::bf16, tag::ab),
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab));
        cache_nn[key] = dnnl::matmul(pd);
        it = cache_nn.find(key);
    }

    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    it->second.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({K, N}, dt::bf16, tag::ab),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<bf16*>(B))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

// ---------------------------------------------------------------------------
// Batched BF16 matmul: C(B,M,N) = A(B,M,K) @ W_eff(B,K,N)
//   transpose_W=false: W is (B,K,N)
//   transpose_W=true:  W is (B,N,K), used as (B,K,N)^T
// ---------------------------------------------------------------------------
inline void matmul_bf16_batched(
    const bf16* A, int B, int M, int K,
    const bf16* W, int N, bool transpose_W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0)
) {
    static std::unordered_map<BatchedMatmulKey, dnnl::matmul,
                              BatchedMatmulKeyHash> cache_b;

    BatchedMatmulKey key{ctx.index, B, M, K, N, transpose_W};
    auto it = cache_b.find(key);
    if (it == cache_b.end()) {
        using dt = dnnl::memory::data_type;

        dnnl::memory::dims sw_logical = transpose_W
            ? dnnl::memory::dims{(dnnl_dim_t)N*K, 1, (dnnl_dim_t)K}
            : dnnl::memory::dims{(dnnl_dim_t)K*N, (dnnl_dim_t)N, 1};

        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({B, M, K}, dt::bf16,
                dnnl::memory::dims{(dnnl_dim_t)M*K, (dnnl_dim_t)K, 1}),
            dnnl::memory::desc({B, K, N}, dt::bf16, sw_logical),
            dnnl::memory::desc({B, M, N}, dt::bf16,
                dnnl::memory::dims{(dnnl_dim_t)M*N, (dnnl_dim_t)N, 1}));
        cache_b[key] = dnnl::matmul(pd);
        it = cache_b.find(key);
    }

    using dt = dnnl::memory::data_type;
    dnnl::memory::dims sw_logical = transpose_W
        ? dnnl::memory::dims{(dnnl_dim_t)N*K, 1, (dnnl_dim_t)K}
        : dnnl::memory::dims{(dnnl_dim_t)K*N, (dnnl_dim_t)N, 1};

    it->second.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({B, M, K}, dt::bf16,
                dnnl::memory::dims{(dnnl_dim_t)M*K, (dnnl_dim_t)K, 1}),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({B, K, N}, dt::bf16, sw_logical),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<bf16*>(W))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({B, M, N}, dt::bf16,
                dnnl::memory::dims{(dnnl_dim_t)M*N, (dnnl_dim_t)N, 1}),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}


// ---------------------------------------------------------------------------
// Batched BF16 matmul with explicit row-major strides:
//   C(B,M,N) = A(B,M,K) @ W(B,K,N)
// Used by decode kernels that keep KV cache in architecture-friendly views
// without copying the active window into compact temporary buffers.
// ---------------------------------------------------------------------------
inline void matmul_bf16_batched_strided(
    const bf16* A, int B, int M, int K,
    dnnl_dim_t as0, dnnl_dim_t as1, dnnl_dim_t as2,
    const bf16* W, int N,
    dnnl_dim_t ws0, dnnl_dim_t ws1, dnnl_dim_t ws2,
    bf16* C,
    dnnl_dim_t cs0, dnnl_dim_t cs1, dnnl_dim_t cs2,
    GpuEngine& ctx = GpuEngine::get(0)
) {
    static std::unordered_map<BatchedMatmulStridedKey, dnnl::matmul,
                              BatchedMatmulStridedKeyHash> cache_bs;

    BatchedMatmulStridedKey key{ctx.index, B, M, K, N,
                                as0, as1, as2, ws0, ws1, ws2, cs0, cs1, cs2};
    auto it = cache_bs.find(key);
    if (it == cache_bs.end()) {
        using dt = dnnl::memory::data_type;
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({B, M, K}, dt::bf16,
                dnnl::memory::dims{as0, as1, as2}),
            dnnl::memory::desc({B, K, N}, dt::bf16,
                dnnl::memory::dims{ws0, ws1, ws2}),
            dnnl::memory::desc({B, M, N}, dt::bf16,
                dnnl::memory::dims{cs0, cs1, cs2}));
        cache_bs[key] = dnnl::matmul(pd);
        it = cache_bs.find(key);
    }

    using dt = dnnl::memory::data_type;
    it->second.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({B, M, K}, dt::bf16,
                dnnl::memory::dims{as0, as1, as2}),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({B, K, N}, dt::bf16,
                dnnl::memory::dims{ws0, ws1, ws2}),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<bf16*>(W))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({B, M, N}, dt::bf16,
                dnnl::memory::dims{cs0, cs1, cs2}),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

// ---------------------------------------------------------------------------
// FP32 in-place softmax over last axis: (rows, cols). Async.
// ---------------------------------------------------------------------------
inline void softmax_f32(
    float* x, int rows, int cols,
    GpuEngine& ctx = GpuEngine::get(0)
) {
    static std::unordered_map<SoftmaxKey, dnnl::softmax_forward,
                              SoftmaxKeyHash> cache_sm;

    SoftmaxKey key{ctx.index, rows, cols};
    auto it = cache_sm.find(key);
    if (it == cache_sm.end()) {
        using dt  = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        auto md = dnnl::memory::desc({rows, cols}, dt::f32, tag::ab);
        dnnl::softmax_forward::primitive_desc pd(
            ctx.engine, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate, md, md, /*axis=*/1);
        cache_sm[key] = dnnl::softmax_forward(pd);
        it = cache_sm.find(key);
    }

    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto mem = dnnl::sycl_interop::make_memory(
        dnnl::memory::desc({rows, cols}, dt::f32, tag::ab),
        ctx.engine, dnnl::sycl_interop::memory_kind::usm, x);
    it->second.execute(ctx.stream, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
}

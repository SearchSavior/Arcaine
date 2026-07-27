#pragma once
// GGUF Q8_0 W8A16 weight decompression.
//
// GGUF stores each row as blocks of 32 values: fp16 scale + 32 signed int8
// quants.  The loader splits that into an s8 weight matrix in raw oneDNN ba
// layout and per-block F32 scales transposed to (K/32, N).
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

using q8_v8s = diff_dpas_v8s;
using q8_v8i = diff_dpas_v8i;
using q8_v8f = diff_dpas_v8f;

static constexpr int kQ8DpasBF16 = 0x3000;

struct Q8Linear {
    int in_features = 0;
    int out_features = 0;
    int group_size = 32;
    GpuBuffer<int8_t> weight_qs;
    GpuBuffer<float> weight_scale;
    // Optional row-major scales, used when the same table is consumed without
    // transposition (embedding lookup and soft self-conditioning).
    GpuBuffer<float> weight_scale_rows;

    bool empty() const { return weight_qs.empty(); }
};

struct Q8BatchedLinear {
    int batch = 0;
    int in_features = 0;
    int out_features = 0;
    int group_size = 32;
    GpuBuffer<int8_t> weight_qs;    // (B, out_features, in_features)
    GpuBuffer<float> weight_scale;  // (B, in_features/32, out_features)

    bool empty() const { return weight_qs.empty(); }
};

struct Q8MatmulKey {
    int gpu, M, K, N;
    bool operator==(const Q8MatmulKey& o) const {
        return gpu == o.gpu && M == o.M && K == o.K && N == o.N;
    }
};

struct Q8BatchedMatmulKey {
    int gpu, B, M, K, N;
    bool operator==(const Q8BatchedMatmulKey& o) const {
        return gpu == o.gpu && B == o.B && M == o.M && K == o.K && N == o.N;
    }
};

struct Q8BatchedMatmulKeyHash {
    size_t operator()(const Q8BatchedMatmulKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.B) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Q8MatmulKeyHash {
    size_t operator()(const Q8MatmulKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

enum class Q8Backend { Custom, Onednn };

inline Q8Backend q8_backend() {
    static Q8Backend backend = [] {
        // oneDNN is 5-15x faster than the custom DPAS kernel for every dense
        // W8A16 TN matmul (LM head, attention projections, dense MLP): its tiled
        // GEMM reuses each weight tile across the M dimension, whereas the custom
        // kernel re-reads the whole weight matrix once per 8-row tile and is
        // bandwidth-bound. The custom path is kept only for the soft_next NN
        // matmul (see matmul_q8_0_nn) and the grouped MoE-expert kernel, which
        // oneDNN handles poorly. Opt back into custom with DIFF_Q8_BACKEND=custom.
        const char* env = std::getenv("DIFF_Q8_BACKEND");
        if (env && (!std::strcmp(env, "custom") || !std::strcmp(env, "dpas")))
            return Q8Backend::Custom;
        return Q8Backend::Onednn;
    }();
    return backend;
}

inline bool q8_soft_next_use_onednn() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_Q8_SOFT_NEXT_BACKEND");
        return env && (!std::strcmp(env, "onednn") || !std::strcmp(env, "dnnl") ||
                       !std::strcmp(env, "1") || !std::strcmp(env, "on"));
    }();
    return enabled;
}

inline void matmul_q8_0_custom_kernel(
    sycl::queue& q,
    const bf16* A,
    int M,
    int K,
    const int8_t* weights,
    const float* scales,
    int N,
    bf16* C)
{
    if ((M % 8) != 0 || (N % 16) != 0) {
        int G = K / 32;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>((size_t)M, (size_t)N), [=](sycl::id<2> id) {
                int m = (int)id[0];
                int n = (int)id[1];
                const bf16* a = A + (size_t)m * K;
                const int8_t* w = weights + (size_t)n * K;
                float acc = 0.0f;
                for (int g = 0; g < G; ++g) {
                    float dot = 0.0f;
                    int off = g * 32;
                    for (int i = 0; i < 32; ++i)
                        dot += bf16_to_float(a[off + i]) * (float)w[off + i];
                    acc += dot * scales[(size_t)g * N + n];
                }
                C[(size_t)m * N + n] = float_to_bf16(acc);
            });
        });
        return;
    }

    int Mt = M / 8;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)Mt, (size_t)N),
                                         sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int m0 = (int)it.get_group(0) * 8;
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                q8_v8f c = {0,0,0,0,0,0,0,0};
                for (int k0 = 0; k0 < K; k0 += 16) {
                    float scale = scales[(size_t)(k0 / 32) * N + n];
                    const int8_t* wrow = weights + (size_t)n * K + k0;
                    q8_v8i b;
                    for (int j = 0; j < 8; ++j) {
                        uint16_t lo = float_to_bf16((float)wrow[2 * j] * scale);
                        uint16_t hi = float_to_bf16((float)wrow[2 * j + 1] * scale);
                        b[j] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
                    }
                    q8_v8s a;
                    int kk = k0 + lane;
                    for (int m = 0; m < 8; ++m)
                        a[m] = (short)A[(size_t)(m0 + m) * K + kk];
                    c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, a, b, c, kQ8DpasBF16);
                }
                for (int m = 0; m < 8; ++m)
                    C[(size_t)(m0 + m) * N + n] = float_to_bf16(c[m]);
            });
    });
}

inline void matmul_q8_0_nn_custom_kernel(
    sycl::queue& q,
    const bf16* A,
    int M,
    int K,
    const int8_t* weights,
    const float* row_scales,
    int N,
    bf16* C)
{
    if ((M % 8) != 0 || (N % 16) != 0) {
        int G = N / 32;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>((size_t)M, (size_t)N), [=](sycl::id<2> id) {
                int m = (int)id[0];
                int n = (int)id[1];
                int g = n / 32;
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float w = (float)weights[(size_t)k * N + n] *
                              row_scales[(size_t)k * G + g];
                    acc += bf16_to_float(A[(size_t)m * K + k]) * w;
                }
                C[(size_t)m * N + n] = float_to_bf16(acc);
            });
        });
        return;
    }

    // Register-block the M dimension: each work-group dequantizes one weight
    // tile `b` per k-step and reuses it across BLK 8-row DPAS blocks. This
    // contraction runs over the vocab dimension (K = V), so the V x H weight
    // table dominates traffic; without blocking it is re-read once per 8-row
    // tile (M/8 times). Blocking cuts that to ceil(M/8/BLK) full passes.
    constexpr int BLK = 4;
    int Mt = M / 8;
    int Mg = (Mt + BLK - 1) / BLK;
    int G = N / 32;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)Mg, (size_t)N),
                                         sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int blk0 = (int)it.get_group(0) * BLK;
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                int ng = n / 32;
                q8_v8f c[BLK];
                for (int t = 0; t < BLK; ++t) c[t] = q8_v8f{0,0,0,0,0,0,0,0};
                for (int k0 = 0; k0 < K; k0 += 16) {
                    q8_v8i b;
                    for (int j = 0; j < 8; ++j) {
                        int k = k0 + 2 * j;
                        float s0 = row_scales[(size_t)k * G + ng];
                        float s1 = row_scales[(size_t)(k + 1) * G + ng];
                        uint16_t lo = float_to_bf16((float)weights[(size_t)k * N + n] * s0);
                        uint16_t hi = float_to_bf16((float)weights[(size_t)(k + 1) * N + n] * s1);
                        b[j] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
                    }
                    int kk = k0 + lane;
                    for (int t = 0; t < BLK; ++t) {
                        int blk = blk0 + t;
                        if (blk >= Mt) break;   // uniform across the subgroup
                        int m0 = blk * 8;
                        q8_v8s a;
                        for (int m = 0; m < 8; ++m)
                            a[m] = (short)A[(size_t)(m0 + m) * K + kk];
                        c[t] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, a, b, c[t], kQ8DpasBF16);
                    }
                }
                for (int t = 0; t < BLK; ++t) {
                    int blk = blk0 + t;
                    if (blk >= Mt) break;
                    int m0 = blk * 8;
                    for (int m = 0; m < 8; ++m)
                        C[(size_t)(m0 + m) * N + n] = float_to_bf16(c[t][m]);
                }
            });
    });
}

inline void matmul_q8_0_batched_custom_kernel(
    sycl::queue& q,
    const bf16* A,
    int B,
    int M,
    int K,
    const int8_t* weights,
    const float* scales,
    int N,
    bf16* C)
{
    if ((M % 8) != 0 || (N % 16) != 0) {
        int G = K / 32;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<3>((size_t)B, (size_t)M, (size_t)N),
                           [=](sycl::id<3> id) {
                int b = (int)id[0];
                int m = (int)id[1];
                int n = (int)id[2];
                const bf16* a = A + ((size_t)b * M + m) * K;
                const int8_t* w = weights + ((size_t)b * N + n) * K;
                const float* s = scales + (size_t)b * G * N;
                float acc = 0.0f;
                for (int g = 0; g < G; ++g) {
                    float dot = 0.0f;
                    int off = g * 32;
                    for (int i = 0; i < 32; ++i)
                        dot += bf16_to_float(a[off + i]) * (float)w[off + i];
                    acc += dot * s[(size_t)g * N + n];
                }
                C[((size_t)b * M + m) * N + n] = float_to_bf16(acc);
            });
        });
        return;
    }

    int Mt = M / 8;
    int G = K / 32;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>(
                sycl::range<3>((size_t)B, (size_t)Mt, (size_t)N),
                sycl::range<3>(1, 1, 16)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                int bidx = (int)it.get_group(0);
                int m0 = (int)it.get_group(1) * 8;
                int lane = (int)it.get_local_id(2);
                int n = (int)it.get_group(2) * 16 + lane;
                const bf16* Ab = A + (size_t)bidx * M * K;
                const int8_t* Wb = weights + (size_t)bidx * N * K;
                const float* Sb = scales + (size_t)bidx * G * N;
                q8_v8f c = {0,0,0,0,0,0,0,0};
                for (int k0 = 0; k0 < K; k0 += 16) {
                    float scale = Sb[(size_t)(k0 / 32) * N + n];
                    const int8_t* wrow = Wb + (size_t)n * K + k0;
                    q8_v8i bv;
                    for (int j = 0; j < 8; ++j) {
                        uint16_t lo = float_to_bf16((float)wrow[2 * j] * scale);
                        uint16_t hi = float_to_bf16((float)wrow[2 * j + 1] * scale);
                        bv[j] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
                    }
                    q8_v8s av;
                    int kk = k0 + lane;
                    for (int m = 0; m < 8; ++m)
                        av[m] = (short)Ab[(size_t)(m0 + m) * K + kk];
                    c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, av, bv, c, kQ8DpasBF16);
                }
                for (int m = 0; m < 8; ++m)
                    C[((size_t)bidx * M + m0 + m) * N + n] = float_to_bf16(c[m]);
        });
    });
}

inline dnnl::matmul& q8_matmul_primitive(GpuEngine& ctx, int M, int K, int N) {
    static std::unordered_map<Q8MatmulKey, dnnl::matmul, Q8MatmulKeyHash> cache;
    Q8MatmulKey key{ctx.index, M, K, N};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {32, 1}, dt::f32);
        attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, /*apply_to_int=*/true);
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            dnnl::memory::desc({K, N}, dt::s8, tag::ba),
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            attr);
        auto result = cache.emplace(key, dnnl::matmul(pd));
        it = result.first;
    }
    return it->second;
}

inline dnnl::matmul& q8_matmul_nn_primitive(GpuEngine& ctx, int M, int K, int N) {
    static std::unordered_map<Q8MatmulKey, dnnl::matmul, Q8MatmulKeyHash> cache;
    Q8MatmulKey key{ctx.index, M, K, N};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {1, 32}, dt::f32);
        attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, /*apply_to_int=*/true);
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            dnnl::memory::desc({K, N}, dt::s8, tag::ab),
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            attr);
        auto result = cache.emplace(key, dnnl::matmul(pd));
        it = result.first;
    }
    return it->second;
}

inline void matmul_q8_0(
    const bf16* A,
    int M,
    int K,
    const Q8Linear& W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.in_features != K)
        throw std::runtime_error("matmul_q8_0: K does not match weight shape");
    if (K % W.group_size != 0 || W.group_size != 32)
        throw std::runtime_error("matmul_q8_0: K must be divisible by 32");

    int N = W.out_features;
    int G = K / 32;
    const int8_t* weights = W.weight_qs.data();
    const float* scales = W.weight_scale.data();

    if (q8_backend() == Q8Backend::Custom) {
        matmul_q8_0_custom_kernel(ctx.queue, A, M, K, weights, scales, N, C);
        return;
    }

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto src_md = dnnl::memory::desc({M, K}, dt::bf16, tag::ab);
    auto weights_md = dnnl::memory::desc({K, N}, dt::s8, tag::ba);
    auto scales_md = dnnl::memory::desc({G, N}, dt::f32, tag::ab);
    auto dst_md = dnnl::memory::desc({M, N}, dt::bf16, tag::ab);

    dnnl::matmul& prim = q8_matmul_primitive(ctx, M, K, N);
    prim.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<int8_t*>(weights))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<float*>(scales))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

inline void matmul_q8_0_raw(
    const bf16* A,
    int M,
    int K,
    const int8_t* weights,
    const float* scales,
    int N,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (K % 32 != 0)
        throw std::runtime_error("matmul_q8_0_raw: K must be divisible by 32");

    int G = K / 32;

    if (q8_backend() == Q8Backend::Custom) {
        matmul_q8_0_custom_kernel(ctx.queue, A, M, K, weights, scales, N, C);
        return;
    }

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto src_md = dnnl::memory::desc({M, K}, dt::bf16, tag::ab);
    auto weights_md = dnnl::memory::desc({K, N}, dt::s8, tag::ba);
    auto scales_md = dnnl::memory::desc({G, N}, dt::f32, tag::ab);
    auto dst_md = dnnl::memory::desc({M, N}, dt::bf16, tag::ab);

    dnnl::matmul& prim = q8_matmul_primitive(ctx, M, K, N);
    prim.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<int8_t*>(weights))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<float*>(scales))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

inline void matmul_q8_0_expert(
    const bf16* A,
    int M,
    int K,
    const Q8BatchedLinear& W,
    int expert,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.in_features != K || expert < 0 || expert >= W.batch)
        throw std::runtime_error("matmul_q8_0_expert: shape or expert mismatch");
    int N = W.out_features;
    int G = K / 32;
    const int8_t* weights = W.weight_qs.data() + (size_t)expert * N * K;
    const float* scales = W.weight_scale.data() + (size_t)expert * G * N;
    matmul_q8_0_raw(A, M, K, weights, scales, N, C, ctx);
}

inline void matmul_q8_0_grouped_expert(
    const bf16* A,
    int K,
    const Q8BatchedLinear& W,
    const int32_t* block_slot,
    const int32_t* block_expert,
    int blocks,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.in_features != K)
        throw std::runtime_error("matmul_q8_0_grouped_expert: input shape mismatch");
    if (K % 32 != 0 || W.out_features % 16 != 0)
        throw std::runtime_error("matmul_q8_0_grouped_expert: unsupported shape");
    if (blocks == 0) return;

    int N = W.out_features;
    int G = K / 32;
    const int8_t* weights = W.weight_qs.data();
    const float* scales = W.weight_scale.data();
    ctx.queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)blocks, (size_t)N),
                                         sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int b = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                int slot0 = block_slot[b];
                int expert = block_expert[b];
                const int8_t* Wbase = weights + (size_t)expert * N * K;
                const float* Sbase = scales + (size_t)expert * G * N;
                q8_v8f c = {0,0,0,0,0,0,0,0};
                for (int k0 = 0; k0 < K; k0 += 16) {
                    float scale = Sbase[(size_t)(k0 / 32) * N + n];
                    const int8_t* wrow = Wbase + (size_t)n * K + k0;
                    q8_v8i bv;
                    for (int j = 0; j < 8; ++j) {
                        uint16_t lo = float_to_bf16((float)wrow[2 * j] * scale);
                        uint16_t hi = float_to_bf16((float)wrow[2 * j + 1] * scale);
                        bv[j] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
                    }
                    q8_v8s av;
                    int kk = k0 + lane;
                    for (int m = 0; m < 8; ++m)
                        av[m] = (short)A[(size_t)(slot0 + m) * K + kk];
                    c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, av, bv, c, kQ8DpasBF16);
                }
                for (int m = 0; m < 8; ++m)
                    C[(size_t)(slot0 + m) * N + n] = float_to_bf16(c[m]);
            });
    });
}

inline void matmul_q8_0_grouped_expert_gateup_geglu(
    const bf16* A,
    int K,
    const Q8BatchedLinear& W,
    const int32_t* block_slot,
    const int32_t* block_expert,
    int blocks,
    int inter,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.in_features != K || W.out_features != 2 * inter)
        throw std::runtime_error("matmul_q8_0_grouped_expert_gateup_geglu: shape mismatch");
    if (K % 32 != 0 || inter % 16 != 0)
        throw std::runtime_error("matmul_q8_0_grouped_expert_gateup_geglu: unsupported shape");
    if (blocks == 0) return;

    int N = W.out_features;
    int G = K / 32;
    const int8_t* weights = W.weight_qs.data();
    const float* scales = W.weight_scale.data();
    ctx.queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)blocks, (size_t)inter),
                                         sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int b = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                int slot0 = block_slot[b];
                int expert = block_expert[b];
                const int8_t* Wbase = weights + (size_t)expert * N * K;
                const float* Sbase = scales + (size_t)expert * G * N;
                q8_v8f cg = {0,0,0,0,0,0,0,0};
                q8_v8f cu = {0,0,0,0,0,0,0,0};
                for (int k0 = 0; k0 < K; k0 += 16) {
                    int sg = k0 / 32;
                    const int8_t* wg = Wbase + (size_t)n * K + k0;
                    const int8_t* wu = Wbase + (size_t)(inter + n) * K + k0;
                    float scale_g = Sbase[(size_t)sg * N + n];
                    float scale_u = Sbase[(size_t)sg * N + inter + n];
                    q8_v8i bg, bu;
                    for (int j = 0; j < 8; ++j) {
                        uint16_t glo = float_to_bf16((float)wg[2 * j] * scale_g);
                        uint16_t ghi = float_to_bf16((float)wg[2 * j + 1] * scale_g);
                        uint16_t ulo = float_to_bf16((float)wu[2 * j] * scale_u);
                        uint16_t uhi = float_to_bf16((float)wu[2 * j + 1] * scale_u);
                        bg[j] = (int)((uint32_t)glo | ((uint32_t)ghi << 16));
                        bu[j] = (int)((uint32_t)ulo | ((uint32_t)uhi << 16));
                    }
                    q8_v8s av;
                    int kk = k0 + lane;
                    for (int m = 0; m < 8; ++m)
                        av[m] = (short)A[(size_t)(slot0 + m) * K + kk];
                    cg = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, av, bg, cg, kQ8DpasBF16);
                    cu = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, av, bu, cu, kQ8DpasBF16);
                }
                constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
                constexpr float COEF = 0.044715f;
                for (int m = 0; m < 8; ++m) {
                    float g = cg[m];
                    float u = cu[m];
                    float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
                    C[(size_t)(slot0 + m) * inter + n] =
                        float_to_bf16(0.5f * g * (1.0f + sycl::tanh(inner)) * u);
                }
            });
    });
}

inline void matmul_q8_0_expert_gateup_geglu(
    const bf16* A,
    int M,
    int K,
    const Q8BatchedLinear& W,
    int expert,
    int inter,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.in_features != K || W.out_features != 2 * inter ||
        expert < 0 || expert >= W.batch)
        throw std::runtime_error("matmul_q8_0_expert_gateup_geglu: shape or expert mismatch");
    if ((M % 8) != 0 || (inter % 16) != 0) {
        GpuBuffer<bf16> gate_up((size_t)M * 2 * inter, ctx.queue);
        matmul_q8_0_expert(A, M, K, W, expert, gate_up.data(), ctx);
        const bf16* gate_up_ptr = gate_up.data();
        ctx.queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>((size_t)M, (size_t)inter), [=](sycl::id<2> id) {
                int r = (int)id[0];
                int d = (int)id[1];
                const bf16* row = gate_up_ptr + (size_t)r * 2 * inter;
                float g = bf16_to_float(row[d]);
                float u = bf16_to_float(row[inter + d]);
                constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
                constexpr float COEF = 0.044715f;
                float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
                C[(size_t)r * inter + d] =
                    float_to_bf16(0.5f * g * (1.0f + sycl::tanh(inner)) * u);
            });
        });
        ctx.queue.wait();
        return;
    }

    int G = K / 32;
    int Mt = M / 8;
    int out_features = W.out_features;
    const int8_t* Wbase = W.weight_qs.data() + (size_t)expert * out_features * K;
    const float* Sbase = W.weight_scale.data() + (size_t)expert * G * out_features;
    ctx.queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)Mt, (size_t)inter),
                                         sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int m0 = (int)it.get_group(0) * 8;
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                q8_v8f cg = {0,0,0,0,0,0,0,0};
                q8_v8f cu = {0,0,0,0,0,0,0,0};
                for (int k0 = 0; k0 < K; k0 += 16) {
                    int sg = k0 / 32;
                    const int8_t* wg = Wbase + (size_t)n * K + k0;
                    const int8_t* wu = Wbase + (size_t)(inter + n) * K + k0;
                    float scale_g = Sbase[(size_t)sg * out_features + n];
                    float scale_u = Sbase[(size_t)sg * out_features + inter + n];
                    q8_v8i bg, bu;
                    for (int j = 0; j < 8; ++j) {
                        uint16_t glo = float_to_bf16((float)wg[2 * j] * scale_g);
                        uint16_t ghi = float_to_bf16((float)wg[2 * j + 1] * scale_g);
                        uint16_t ulo = float_to_bf16((float)wu[2 * j] * scale_u);
                        uint16_t uhi = float_to_bf16((float)wu[2 * j + 1] * scale_u);
                        bg[j] = (int)((uint32_t)glo | ((uint32_t)ghi << 16));
                        bu[j] = (int)((uint32_t)ulo | ((uint32_t)uhi << 16));
                    }
                    q8_v8s av;
                    int kk = k0 + lane;
                    for (int m = 0; m < 8; ++m)
                        av[m] = (short)A[(size_t)(m0 + m) * K + kk];
                    cg = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, av, bg, cg, kQ8DpasBF16);
                    cu = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, av, bu, cu, kQ8DpasBF16);
                }
                constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
                constexpr float COEF = 0.044715f;
                for (int m = 0; m < 8; ++m) {
                    float g = cg[m];
                    float u = cu[m];
                    float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
                    C[(size_t)(m0 + m) * inter + n] =
                        float_to_bf16(0.5f * g * (1.0f + sycl::tanh(inner)) * u);
                }
            });
    });
}

inline dnnl::matmul& q8_matmul_batched_primitive(
    GpuEngine& ctx, int B, int M, int K, int N)
{
    static std::unordered_map<Q8BatchedMatmulKey, dnnl::matmul,
                              Q8BatchedMatmulKeyHash> cache;
    Q8BatchedMatmulKey key{ctx.index, B, M, K, N};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt = dnnl::memory::data_type;
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS,
                        (1 << 0) | (1 << 1) | (1 << 2),
                        {1, 32, 1}, dt::f32);
        attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, /*apply_to_int=*/true);
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({B, M, K}, dt::bf16,
                dnnl::memory::dims{(dnnl_dim_t)M * K, (dnnl_dim_t)K, 1}),
            dnnl::memory::desc({B, K, N}, dt::s8,
                dnnl::memory::dims{(dnnl_dim_t)N * K, 1, (dnnl_dim_t)K}),
            dnnl::memory::desc({B, M, N}, dt::bf16,
                dnnl::memory::dims{(dnnl_dim_t)M * N, (dnnl_dim_t)N, 1}),
            attr);
        auto result = cache.emplace(key, dnnl::matmul(pd));
        it = result.first;
    }
    return it->second;
}

inline void matmul_q8_0_batched(
    const bf16* A,
    int B,
    int M,
    int K,
    const Q8BatchedLinear& W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.batch != B || W.in_features != K)
        throw std::runtime_error("matmul_q8_0_batched: shape mismatch");
    if (K % W.group_size != 0 || W.group_size != 32)
        throw std::runtime_error("matmul_q8_0_batched: K must be divisible by 32");

    int N = W.out_features;
    int G = K / 32;

    if (q8_backend() == Q8Backend::Custom) {
        matmul_q8_0_batched_custom_kernel(
            ctx.queue, A, B, M, K, W.weight_qs.data(), W.weight_scale.data(), N, C);
        return;
    }

    using dt = dnnl::memory::data_type;
    auto src_md = dnnl::memory::desc({B, M, K}, dt::bf16,
        dnnl::memory::dims{(dnnl_dim_t)M * K, (dnnl_dim_t)K, 1});
    auto weights_md = dnnl::memory::desc({B, K, N}, dt::s8,
        dnnl::memory::dims{(dnnl_dim_t)N * K, 1, (dnnl_dim_t)K});
    auto scales_md = dnnl::memory::desc({B, G, N}, dt::f32,
        dnnl::memory::dims{(dnnl_dim_t)G * N, (dnnl_dim_t)N, 1});
    auto dst_md = dnnl::memory::desc({B, M, N}, dt::bf16,
        dnnl::memory::dims{(dnnl_dim_t)M * N, (dnnl_dim_t)N, 1});

    dnnl::matmul& prim = q8_matmul_batched_primitive(ctx, B, M, K, N);
    prim.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<int8_t*>(W.weight_qs.data()))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<float*>(W.weight_scale.data()))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

inline void matmul_q8_0_nn(
    const bf16* A,
    int M,
    int K,
    const Q8Linear& W,
    int N,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (W.out_features != K || W.in_features != N)
        throw std::runtime_error("matmul_q8_0_nn: weight shape mismatch");
    if (N % 32 != 0 || W.weight_scale_rows.empty())
        throw std::runtime_error("matmul_q8_0_nn: missing row-major Q8_0 scales");

    // The NN (no-transpose) path contracts over the vocab dimension (K = V).
    // oneDNN's s8 matmul is pathological for that layout (~20x slower than the
    // custom DPAS kernel), so soft_next stays on the custom kernel regardless of
    // the global q8 backend; only an explicit DIFF_Q8_SOFT_NEXT_BACKEND opts in.
    if (!q8_soft_next_use_onednn()) {
        matmul_q8_0_nn_custom_kernel(
            ctx.queue, A, M, K, W.weight_qs.data(), W.weight_scale_rows.data(), N, C);
        return;
    }

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto src_md = dnnl::memory::desc({M, K}, dt::bf16, tag::ab);
    auto weights_md = dnnl::memory::desc({K, N}, dt::s8, tag::ab);
    auto scales_md = dnnl::memory::desc({K, N / 32}, dt::f32, tag::ab);
    auto dst_md = dnnl::memory::desc({M, N}, dt::bf16, tag::ab);

    dnnl::matmul& prim = q8_matmul_nn_primitive(ctx, M, K, N);
    prim.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<int8_t*>(W.weight_qs.data()))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            scales_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<float*>(W.weight_scale_rows.data()))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

inline void embedding_lookup_q8_0(
    sycl::queue& q,
    const Q8Linear& table,
    const int32_t* ids,
    bf16* out,
    int seq_len, int H,
    float scale)
{
    if (table.in_features != H || table.weight_scale_rows.empty())
        throw std::runtime_error("embedding_lookup_q8_0: invalid Q8_0 embedding table");
    const int8_t* qs = table.weight_qs.data();
    const float* sc = table.weight_scale_rows.data();
    int groups = H / 32;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(seq_len, H), [=](sycl::id<2> id) {
            int tok = id[0];
            int d = id[1];
            int token_id = ids[tok];
            float v = (float)qs[(size_t)token_id * H + d] *
                      sc[(size_t)token_id * groups + d / 32] * scale;
            out[(size_t)tok * H + d] = float_to_bf16(v);
        });
    });
}

// out[t, :] = scale * sum_{s<k} w[t,s] * dequant(table[idx[t,s], :])   (Q8_0 table)
// Top-k weighted embedding gather; counterpart to weighted_embed_gather for BF16.
inline void weighted_embed_gather_q8_0(
    sycl::queue& q,
    const Q8Linear& table,
    const int32_t* idx,    // (seq, k)
    const float* w,        // (seq, k)
    bf16* out,             // (seq, H)
    int seq, int k, int H,
    float scale)
{
    if (table.in_features != H || table.weight_scale_rows.empty())
        throw std::runtime_error("weighted_embed_gather_q8_0: invalid Q8_0 embedding table");
    const int8_t* qs = table.weight_qs.data();
    const float* sc = table.weight_scale_rows.data();
    int groups = H / 32;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(seq, H), [=](sycl::id<2> id) {
            int t = (int)id[0], d = (int)id[1];
            float acc = 0.0f;
            for (int s = 0; s < k; ++s) {
                int tok = idx[(size_t)t * k + s];
                float val = (float)qs[(size_t)tok * H + d] *
                            sc[(size_t)tok * groups + d / 32];
                acc += w[(size_t)t * k + s] * val;
            }
            out[(size_t)t * H + d] = float_to_bf16(acc * scale);
        });
    });
}

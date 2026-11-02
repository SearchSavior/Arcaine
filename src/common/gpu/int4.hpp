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
#include <cstring>
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
    // Optional per-(K-group, N) signed zero points, logical shape
    // (in_features / group_size, out_features) as s8. Present for asymmetric
    // compressed-tensors int4 checkpoints; empty (has_zero_point=false) for the
    // symmetric zero-point-8 case (diffusion_gemma).
    GpuBuffer<int8_t> weight_zero_point;
    bool has_zero_point = false;

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
    bool has_zp;
    bool operator==(const Int4MatmulKey& o) const {
        return gpu == o.gpu && M == o.M && K == o.K && N == o.N && group == o.group &&
               has_zp == o.has_zp;
    }
};

struct Int4MatmulKeyHash {
    size_t operator()(const Int4MatmulKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.group) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}((int)k.has_zp) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Int4MatmulEntry {
    dnnl::matmul primitive;
    dnnl::memory::desc weights_md;
};

inline Int4MatmulEntry& int4_matmul_entry(GpuEngine& ctx, int M, int K, int N, int group,
                                          bool has_zp = false) {
    static std::unordered_map<Int4MatmulKey, Int4MatmulEntry, Int4MatmulKeyHash> cache;
    Int4MatmulKey key{ctx.index, M, K, N, group, has_zp};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::primitive_attr attr;
        // Per-group weight scales along K (group along dim 0, per-channel on N).
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {group, 1}, dt::bf16);
        // Per-(K-group, N) weights zero points for asymmetric int4. s8 keeps the
        // fast jit decompression kernel (s32 falls back to ocl:ref).
        if (has_zp)
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {group, 1}, dt::s8);
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
    return W.weight_any.data();
}

// ---------------------------------------------------------------------------
// W4A8 prefill path (DIFF_INT4_W4A8_PREFILL).
//
// The default INT4 path keeps BF16 activations (W4A16) and lets oneDNN
// decompress the s4 weights into a BF16 GEMM, which exercises the BF16 XMX
// pipe.  At large M (prefill) BMG can do better: quantize the activations to
// s8 per-token and run oneDNN's native mixed s8xs4 DPAS.  Pre-Xe3p hardware
// (BMG) supports that instruction directly as long as the s4 matrix carries no
// zero points -- our loader folds the compressed-tensors zero-point-8 into the
// signed nibbles, so no zero-point argument is present.  The weights stay s4,
// the weight dequant happens in the GEMM epilogue, and the raw tag::ba buffer
// is consumed as-is (no reorder).
//
// Gate: DIFF_INT4_W4A8_PREFILL=1, applied only when M >= DIFF_INT4_W4A8_MIN_M
// (default 128) so the tiny per-expert decode buckets stay on W4A16.
// ---------------------------------------------------------------------------

inline bool diff_int4_w4a8_prefill_enabled() {
    static bool enabled = [] {
        const char* e = std::getenv("DIFF_INT4_W4A8_PREFILL");
        return e && std::strcmp(e, "0") && std::strcmp(e, "off") &&
               std::strcmp(e, "false") && std::strcmp(e, "no");
    }();
    return enabled;
}

inline int diff_int4_w4a8_min_m() {
    static int min_m = [] {
        const char* e = std::getenv("DIFF_INT4_W4A8_MIN_M");
        if (!e) return 128;
        int v = std::atoi(e);
        return v > 0 ? v : 128;
    }();
    return min_m;
}

struct Int4W4A8Key {
    int gpu, M, K, N, group;
    bool operator==(const Int4W4A8Key& o) const {
        return gpu == o.gpu && M == o.M && K == o.K && N == o.N && group == o.group;
    }
};

struct Int4W4A8KeyHash {
    size_t operator()(const Int4W4A8Key& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.group) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Int4W4A8Entry {
    dnnl::matmul primitive;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc src_scale_md;
};

inline Int4W4A8Entry& int4_w4a8_matmul_entry(GpuEngine& ctx, int M, int K, int N,
                                             int group) {
    static std::unordered_map<Int4W4A8Key, Int4W4A8Entry, Int4W4A8KeyHash> cache;
    Int4W4A8Key key{ctx.index, M, K, N, group};
    auto it = cache.find(key);
    if (it == cache.end()) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::primitive_attr attr;
        // Symmetric per-token activation scale: one f32 per row (group covers K).
        attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1), {1, K}, dt::f32);
        // Per-group weight scales along K, as the W4A16 path.
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {group, 1}, dt::bf16);
        // No fpmath_mode and no zero points: both would push oneDNN off the
        // native mixed s8xs4 DPAS onto an s4->s8 upconversion (pd.cpp
        // has_s8s4_dpas requires the s4 matrix to be zero-point free).
        auto weights_md = (int4_weight_layout() == Int4WeightLayout::Any)
            ? dnnl::memory::desc({K, N}, dt::s4, tag::any)
            : dnnl::memory::desc({K, N}, dt::s4, tag::ba);
        dnnl::matmul::primitive_desc pd(ctx.engine,
            dnnl::memory::desc({M, K}, dt::s8, tag::ab),
            weights_md,
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            attr);
        auto result = cache.emplace(key,
            Int4W4A8Entry{dnnl::matmul(pd), pd.weights_desc(),
                          dnnl::memory::desc({M, 1}, dt::f32, tag::ab)});
        it = result.first;
    }
    return it->second;
}

// Growable per-engine scratch for the quantized activations + per-token scales.
// The queue is in-order, so reuse is safe: the next quantization is enqueued
// after the previous GEMM that read the buffer.
struct Int4W4A8Scratch {
    GpuBuffer<int8_t> aq;
    GpuBuffer<float> scales;
    size_t aq_cap = 0;
    size_t scale_cap = 0;
};

inline Int4W4A8Scratch& int4_w4a8_scratch(GpuEngine& ctx, size_t aq_needed,
                                          size_t scale_needed) {
    static std::unordered_map<int, Int4W4A8Scratch> cache;
    Int4W4A8Scratch& s = cache[ctx.index];
    if (s.aq_cap < aq_needed) {
        s.aq = GpuBuffer<int8_t>(aq_needed, ctx.queue);
        s.aq_cap = aq_needed;
    }
    if (s.scale_cap < scale_needed) {
        s.scales = GpuBuffer<float>(scale_needed, ctx.queue);
        s.scale_cap = scale_needed;
    }
    return s;
}

// Per-token symmetric bf16 -> s8: q = round(x / (amax/127)), clamp [-127,127].
// A zero row gets scale 1 and q 0.  Vectorized (8 bf16 / 16 B loads) so the
// pass is issue-light even at K=8192.
inline void quantize_bf16_to_s8_per_token(
    sycl::queue& q, const bf16* A, int M, int K, int8_t* Aq, float* scales)
{
    if (M <= 0 || K <= 0) return;
    size_t local = 256;
    constexpr int V = 8;
    int Kvec = (K / V) * V;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmax(local, h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)M * local, local),
            [=](sycl::nd_item<1> it) {
                int row = (int)it.get_group(0);
                int lid = (int)it.get_local_id(0);
                int lsz = (int)it.get_local_range(0);
                const bf16* arow = A + (size_t)row * K;
                float mx = 0.0f;
                for (int k = lid * V; k < Kvec; k += lsz * V) {
                    const bf16* p = arow + k;
#pragma unroll
                    for (int j = 0; j < V; ++j)
                        mx = sycl::fmax(mx, sycl::fabs(bf16_to_float(p[j])));
                }
                for (int k = Kvec + lid; k < K; k += lsz)
                    mx = sycl::fmax(mx, sycl::fabs(bf16_to_float(arow[k])));
                lmax[lid] = mx;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = lsz >> 1; s > 0; s >>= 1) {
                    if (lid < s)
                        lmax[lid] = sycl::fmax(lmax[lid], lmax[lid + s]);
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float scale = lmax[0] > 0.0f ? lmax[0] / 127.0f : 1.0f;
                if (lid == 0) scales[row] = scale;
                float inv = 1.0f / scale;
                int8_t* qrow = Aq + (size_t)row * K;
                for (int k = lid * V; k < Kvec; k += lsz * V) {
                    const bf16* p = arow + k;
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        int qi = (int)sycl::round(bf16_to_float(p[j]) * inv);
                        qrow[k + j] = (int8_t)sycl::min(127, sycl::max(-127, qi));
                    }
                }
                for (int k = Kvec + lid; k < K; k += lsz) {
                    int qi = (int)sycl::round(bf16_to_float(arow[k]) * inv);
                    qrow[k] = (int8_t)sycl::min(127, sycl::max(-127, qi));
                }
            });
    });
}

// W4A8: C = quant_s8(A) @ dequant_s4(W)^T with per-token s8 activation scales.
// Same weight buffer + weight-scale layout as matmul_int4; only the activation
// side changes (BF16 -> s8 + scales).
inline void matmul_int4_w4a8(
    const bf16* A, int M, int K, const Int4Linear& W, bf16* C, GpuEngine& ctx)
{
    if (W.in_features != K)
        throw std::runtime_error("matmul_int4_w4a8: K does not match weight shape");
    if (K % W.group_size != 0)
        throw std::runtime_error("matmul_int4_w4a8: K must be divisible by group_size");

    int N = W.out_features;
    int G = K / W.group_size;

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    auto wscale_md = dnnl::memory::desc({G, N}, dt::bf16, tag::ab);

    auto& scratch = int4_w4a8_scratch(ctx, (size_t)M * K, (size_t)M);
    quantize_bf16_to_s8_per_token(ctx.queue, A, M, K, scratch.aq.data(),
                                  scratch.scales.data());

    auto& entry = int4_w4a8_matmul_entry(ctx, M, K, N, W.group_size);
    const uint8_t* weight_data = int4_weight_data(W, entry.weights_md, K, N, ctx);

    entry.primitive.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({M, K}, dt::s8, tag::ab),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            scratch.aq.data())},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            entry.weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(weight_data))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            entry.src_scale_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            scratch.scales.data())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            wscale_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(W.weight_scale.data()))},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
}

// W4A16 baseline: C (M,N) = A (M,K) @ dequant(W)^T, where W is logical (N,K)
// s4 with per-group BF16 scales.  A and C are BF16.  Async on ctx's stream.
inline void matmul_int4_w4a16(
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

    auto& entry = int4_matmul_entry(ctx, M, K, N, W.group_size, W.has_zero_point);
    const uint8_t* weight_data = int4_weight_data(W, entry.weights_md, K, N, ctx);

    std::unordered_map<int, dnnl::memory> args;
    args.insert({DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
        src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
        const_cast<bf16*>(A))});
    args.insert({DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
        entry.weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
        const_cast<uint8_t*>(weight_data))});
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
        wscale_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
        const_cast<bf16*>(W.weight_scale.data()))});
    if (W.has_zero_point) {
        auto wzp_md = dnnl::memory::desc({G, N}, dt::s8, tag::ab);
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
            dnnl::sycl_interop::make_memory(
                wzp_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
                const_cast<int8_t*>(W.weight_zero_point.data()))});
    }
    args.insert({DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
        dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)});

    entry.primitive.execute(ctx.stream, args);
}

// Public entry point: picks W4A8 for large-M (prefill) when enabled, otherwise
// the W4A16 baseline.  Three guards, all measured on BMG, keep it from
// regressing:
//   - M >= DIFF_INT4_W4A8_MIN_M: the small per-expert decode buckets stay on
//     W4A16, where the s8xs4 DPAS tile is not profitable;
//   - K >= 1024: very small-K GEMMs (the 704-wide expert down projection) are
//     launch/quant-bound and lose 0.6-0.9x;
//   - N >= K: oneDNN's mixed s8xs4 kernel beats BF16-decompression only on
//     wide-N GEMMs (q/k/v, fused qkv/qk).  The narrow output projections
//     (o_proj has N < K) are 0.5-0.9x and stay W4A16.
inline void matmul_int4(
    const bf16* A,
    int M,
    int K,
    const Int4Linear& W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0))
{
    if (!W.has_zero_point && diff_int4_w4a8_prefill_enabled() &&
        M >= diff_int4_w4a8_min_m() && K >= 1024 && W.out_features >= K) {
        matmul_int4_w4a8(A, M, K, W, C, ctx);
        return;
    }
    matmul_int4_w4a16(A, M, K, W, C, ctx);
}

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
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/engine.hpp"
#include "runtime/gpu/ops.hpp"
#include "runtime/profiling/launch_prof.hpp"

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

    // Asymmetric zero-point correction, logical shape
    // (in_features / group_size, out_features) BF16:
    //   zp_offset[g,n] = scale[g,n] * (zp_u[g,n] - 8)
    // where zp_u is the checkpoint's unsigned stored zero-point nibble
    // (compressed-tensors pack-quantized packs zp along the output dim N).
    // The packed weights are still rebased q' = q_u - 8 (XOR 0x88) so oneDNN's
    // s4 path runs unchanged; matmul_int4 subtracts the residual
    // rowsum(A)[m,g] * zp_offset[g,n] afterwards. Empty when the checkpoint is
    // symmetric (zp_u == 8 everywhere) — the correction is then a no-op.
    GpuBuffer<bf16> zp_offset;

    bool has_zp() const { return !zp_offset.empty(); }

    // oneDNN impl-preferred weight layout, materialized once on first use under
    // DIFF_INT4_WEIGHT_LAYOUT=any. The raw tag::ba layout forces oneDNN's
    // generic decompress kernel; the reordered blocked layout unlocks the fast
    // one. Cached per-GPU so a sharded weight reorders on its owning device.
    mutable GpuBuffer<uint8_t> weight_any;
    mutable size_t weight_any_bytes = 0;
    mutable int weight_any_gpu = -1;

    // Dequantized BF16 copy w[n,k] = scale[g,n]*q_s4[n,k] - zp_offset[g,n],
    // materialized lazily for the prefill fast path (ARCAINE_QWEN35_DEQUANT_BF16).
    // Folding the asymmetric-zp correction into the weight removes the rowsum +
    // corr GEMM + subtract launches and lets oneDNN run the GEMM at its native
    // BF16 rate (~155 vs ~100 TFLOPS on BMG). N*K BF16 = 4x the packed bytes.
    // Cached per-GPU like weight_any; the dense path is single-threaded per
    // engine (matches the existing weight_any lazy pattern).
    mutable GpuBuffer<bf16> weight_bf16;
    mutable int weight_bf16_gpu = -1;

    bool empty() const { return weight_packed.empty(); }

    // Dequantize on ctx's queue (async; the next GEMM on the same queue is
    // ordered after it). Returns a (N,K) BF16 row-major buffer consumable by
    // matmul_bf16's tag::ba logical (K,N). First use per GPU pays the one-time
    // dequant; subsequent calls return the cached buffer.
    const bf16* ensure_dequantized_bf16(GpuEngine& ctx) const {
        if (weight_bf16.empty() || weight_bf16_gpu != ctx.index) {
            weight_bf16 = GpuBuffer<bf16>((size_t)out_features * in_features,
                                          ctx.queue);
            weight_bf16_gpu = ctx.index;
            int N = out_features, K = in_features, gs = group_size;
            uint8_t* packed = weight_packed.data();
            bf16* scale = weight_scale.data();
            const bf16* zp = has_zp() ? zp_offset.data() : nullptr;
            sycl::queue& q = ctx.queue;
            auto* wout = weight_bf16.data();
            auto _ev_deq = q.submit([=](sycl::handler& h) {
                h.parallel_for(sycl::range<2>((size_t)N, (size_t)K),
                               [=](sycl::id<2> id) {
                    int n = (int)id[0], k = (int)id[1];
                    int g = k / gs;
                    size_t byte = (size_t)n * (K / 2) + (k / 2);
                    int nib = (packed[byte] >> (4 * (k & 1))) & 0xF;
                    if (nib >= 8) nib -= 16;
                    float v = bf16_to_float(scale[(size_t)g * N + n]) * (float)nib;
                    if (zp) v -= bf16_to_float(zp[(size_t)g * N + n]);
                    wout[(size_t)n * K + k] = float_to_bf16(v);
                });
            });
            launchprof::record("int4_dequant_bf16", _ev_deq);
        }
        return weight_bf16.data();
    }
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

// ---------------------------------------------------------------------------
// Fused-corr path (ARCAINE_QWEN35_INT4_CORR_FUSED): split matmul_int4 so the
// rowsum*zp_offset correction can be folded into the consumer kernel instead of
// a dedicated subtract launch. matmul_int4_gemm runs only the s4 GEMM;
// matmul_int4_zp_corr additionally computes corr = rowsum(A) @ zp_offset into a
// caller-owned (M, N) buffer. Bit-identical to matmul_int4's correction when
// the consumer applies corr the same way (bf16-round the difference once).
// ---------------------------------------------------------------------------
inline void matmul_int4_gemm(const bf16* A, int M, int K, const Int4Linear& W,
                             bf16* C, GpuEngine& ctx = GpuEngine::get(0)) {
    if (W.in_features != K)
        throw std::runtime_error("matmul_int4_gemm: K does not match weight shape");
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    int N = W.out_features;
    auto src_md = dnnl::memory::desc({M, K}, dt::bf16, tag::ab);
    auto dst_md = dnnl::memory::desc({M, N}, dt::bf16, tag::ab);
    auto wscale_md = dnnl::memory::desc({K / W.group_size, N}, dt::bf16, tag::ab);

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
    launchprof::record("int4_gemm");
}

// s4 GEMM + rowsum + corr GEMM into corr_out (caller-owned M*N). The caller
// consumes (C, corr) with a fused kernel; nothing is subtracted here.
// Defined after int4_rowsum (see below).
inline void matmul_int4_zp_corr(const bf16* A, int M, int K, const Int4Linear& W,
                                bf16* C, GpuEngine& ctx, bf16* rowsum_scratch,
                                bf16* corr_out);

// ---------------------------------------------------------------------------
// W4A8 accuracy probe (ARCAINE_INT4_ACT_QUANT_S8=off|stats|apply).
//
// Simulates the activation side of the planned ngen s8x4 DPAS kernel: A is
// quantized per-token (per-row) symmetric s8 (scale = absmax/127, RTN,
// clamp [-127,127]) and dequantized back to bf16 before the GEMM, so the
// existing oneDNN path executes with the same rounding a real s8x4 kernel
// would see. Modes:
//   off   - default, untouched baseline path.
//   apply - run the GEMM on the s8-round-tripped activations (end-to-end
//           simulation of W4A8).
//   stats - keep the baseline output in C, additionally run the quantized
//           GEMM into scratch and print per-call error metrics (call order
//           follows layer order). Costs 2x GEMM.
enum class Int4ActQuantMode { Off = 0, Stats = 1, Apply = 2 };

inline Int4ActQuantMode& int4_act_quant_mode_ref() {
    static Int4ActQuantMode mode = [] {
        const char* env = std::getenv("ARCAINE_INT4_ACT_QUANT_S8");
        if (env) {
            std::string v(env);
            if (v == "stats") return Int4ActQuantMode::Stats;
            if (v == "apply" || v == "1" || v == "on") return Int4ActQuantMode::Apply;
        }
        return Int4ActQuantMode::Off;
    }();
    return mode;
}

inline Int4ActQuantMode int4_act_quant_mode() { return int4_act_quant_mode_ref(); }
inline void int4_act_quant_set(Int4ActQuantMode m) { int4_act_quant_mode_ref() = m; }

// Outlier clip factor for the s8 scale: scale = clip * absmax / 127. clip < 1
// clips the row absmax (values beyond clip*absmax saturate at +-127), trading
// outlier saturation for finer resolution on the bulk. Settable for sweeps.
inline float& int4_act_quant_clip_ref() {
    static float clip = [] {
        if (const char* env = std::getenv("ARCAINE_INT4_ACT_QUANT_S8_CLIP"))
            return std::max(0.05f, std::min(1.0f, (float)std::atof(env)));
        return 1.0f;
    }();
    return clip;
}
inline float int4_act_quant_clip() { return int4_act_quant_clip_ref(); }
inline void int4_act_quant_set_clip(float c) { int4_act_quant_clip_ref() = c; }

// Quantization granularity knob: ARCAINE_INT4_ACT_QUANT_S8_GRAN=g128 uses
// per-(row, K-group) s8 scales with the given group size (default: per-row /
// per-token). Per-group activation scales are kernel-feasible via per-K-slab
// fp rescaling of the s32 accumulator, at some DPAS utilization cost.
inline int int4_act_quant_gran() {
    static int gran = [] {
        const char* env = std::getenv("ARCAINE_INT4_ACT_QUANT_S8_GRAN");
        if (env && env[0] == 'g') {
            int g = std::atoi(env + 1);
            if (g >= 32) return g;
        }
        return 0;  // 0 = per-token
    }();
    return gran;
}

// Ablation knob: ARCAINE_INT4_ACT_QUANT_S8_SKIP="17408x5120,6144x5120" keeps
// the listed (K,N) GEMM signatures on the baseline bf16-activation path even
// in apply/stats mode, to isolate which projection drives end-to-end error.
inline bool int4_act_quant_skipped(int K, int N) {
    static std::string spec = [] {
        const char* env = std::getenv("ARCAINE_INT4_ACT_QUANT_S8_SKIP");
        return env ? std::string(env) : std::string();
    }();
    if (spec.empty()) return false;
    char pat[32];
    std::snprintf(pat, sizeof(pat), "%dx%d", K, N);
    return spec.find(pat) != std::string::npos;
}

// Q[m,:] = bf16(round_s8(A[m,:]) * scale_m) with scale_m = absmax(A[m,:])/127.
inline void act_quant_s8_roundtrip(const bf16* A, bf16* Q, int M, int K,
                                   sycl::queue& queue) {
    float clip = int4_act_quant_clip();
    int gran = int4_act_quant_gran();
    if (gran > 0 && K % gran == 0) {
        int GA = K / gran;
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>((size_t)M, (size_t)GA),
                           [=](sycl::id<2> id) {
                const bf16* row = A + id[0] * (size_t)K + id[1] * (size_t)gran;
                float absmax = 0.0f;
                for (int k = 0; k < gran; ++k)
                    absmax = sycl::fmax(absmax, sycl::fabs(bf16_to_float(row[k])));
                float scale = absmax > 0.0f ? (clip * absmax) / 127.0f : 1.0f;
                bf16* out = Q + id[0] * (size_t)K + id[1] * (size_t)gran;
                for (int k = 0; k < gran; ++k) {
                    float v = sycl::round(bf16_to_float(row[k]) / scale);
                    v = sycl::fmin(sycl::fmax(v, -127.0f), 127.0f);
                    out[k] = float_to_bf16(v * scale);
                }
            });
        });
        return;
    }
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)M), [=](sycl::id<1> id) {
            const bf16* row = A + id[0] * (size_t)K;
            float absmax = 0.0f;
            for (int k = 0; k < K; ++k)
                absmax = sycl::fmax(absmax, sycl::fabs(bf16_to_float(row[k])));
            float scale = absmax > 0.0f ? (clip * absmax) / 127.0f : 1.0f;
            bf16* out = Q + id[0] * (size_t)K;
            for (int k = 0; k < K; ++k) {
                float v = sycl::round(bf16_to_float(row[k]) / scale);
                v = sycl::fmin(sycl::fmax(v, -127.0f), 127.0f);
                out[k] = float_to_bf16(v * scale);
            }
        });
    });
}

// Compare ref vs test (both M*N bf16) on device and print one stats line.
inline void act_quant_s8_report(const bf16* ref, const bf16* test, size_t n,
                                int call, int M, int K, int N,
                                sycl::queue& queue) {
    GpuBuffer<float> stats(3, queue);  // max_abs | sum_sq_diff | sum_sq_ref
    queue.memset(stats.data(), 0, 3 * sizeof(float));
    float* s_ptr = stats.data();
    unsigned* s_max = reinterpret_cast<unsigned*>(s_ptr);  // nonneg floats: bit-max == max
    const size_t threads = 65536;
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(threads), [=](sycl::id<1> id) {
            float lmax = 0.0f, lsd = 0.0f, lsr = 0.0f;
            for (size_t i = id[0]; i < n; i += threads) {
                float r = bf16_to_float(ref[i]);
                float d = sycl::fabs(r - bf16_to_float(test[i]));
                lmax = sycl::fmax(lmax, d);
                lsd += d * d;
                lsr += r * r;
            }
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                a_max(*s_max);
            a_max.fetch_max(__builtin_bit_cast(unsigned, lmax));
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                a_sd(s_ptr[1]), a_sr(s_ptr[2]);
            a_sd.fetch_add(lsd);
            a_sr.fetch_add(lsr);
        });
    });
    float host[3];
    queue.memcpy(host, s_ptr, sizeof(host)).wait();
    double rms_diff = std::sqrt((double)host[1] / (double)n);
    double rms_ref = std::sqrt((double)host[2] / (double)n);
    std::printf("[actq-s8] call=%d M=%d K=%d N=%d max_abs=%.6g rms_abs=%.6g "
                "rms_rel=%.6g\n",
                call, M, K, N, (double)host[0], rms_diff,
                rms_diff / (rms_ref + 1e-12));
}

// ---------------------------------------------------------------------------
// Asymmetric zp correction scratch (ARCAINE_QWEN35_INT4_ZP_SCRATCH, default on).
//
// The dense qwen3_5 path calls matmul_int4 ~4x per layer; every call with
// asymmetric zp previously heap-allocated rowsum + corr GpuBuffers, violating
// the workspace contract ("no layer-forward path allocates device memory").
// This switch uses caller-provided workspace buffers instead. Measured neutral
// (warm) vs the heap path on BMG; strictly fewer device allocations. Set to 0
// to restore per-call allocation.
// ---------------------------------------------------------------------------
inline bool& int4_zp_scratch_ref() {
    static bool scratch = [] {
        const char* env = std::getenv("ARCAINE_QWEN35_INT4_ZP_SCRATCH");
        if (env && (std::string(env) == "0" || std::string(env) == "off"))
            return false;
        return true;
    }();
    return scratch;
}
inline bool int4_zp_scratch() { return int4_zp_scratch_ref(); }
inline void int4_zp_scratch_set(bool v) { int4_zp_scratch_ref() = v; }

// rowsum[m,g] = bf16(sum_{k in group g} A[m,k]), group size gs. Shared by every
// INT4 projection consuming A (hoisted once per activation). Must be called on
// ctx's queue before the GEMM it feeds.
inline void int4_rowsum(const bf16* A, int M, int K, int gs, bf16* out,
                        sycl::queue& queue) {
    int G = K / gs;
    auto _ev = queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)M, (size_t)G),
                       [=](sycl::id<2> id) {
            size_t m = id[0], g = id[1];
            float acc = 0.0f;
            const bf16* row = A + m * (size_t)K + g * (size_t)gs;
            for (int k = 0; k < gs; ++k) acc += bf16_to_float(row[k]);
            out[m * (size_t)G + g] = float_to_bf16(acc);
        });
    });
    launchprof::record("int4_rowsum", _ev);
}

// ---------------------------------------------------------------------------
// W4A8 path (ARCAINE_QWEN35_W4A8): per-token s8 activations x s4 weights via the
// oneDNN s8x s4 fast path (~2x the bf16-activation s4 GEMM). oneDNN 3.13 rejects
// per-token src scales (only per-tensor is supported), so the GEMM runs with a
// per-tensor src scale of 1.0 and each output row is multiplied by its act scale
// afterwards. The zp-correction path is unchanged (rowsum over the original bf16
// A), so C - corr still equals A @ dequant(W)^T up to per-token act rounding.
// ---------------------------------------------------------------------------

inline bool int4_w4a8_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("ARCAINE_QWEN35_W4A8");
        if (!env) return false;
        std::string v(env);
        return v != "0" && v != "off" && v != "false" && v != "no";
    }();
    return enabled;
}

// A (bf16, M x K) -> A_s8 (s8, per-token symmetric), scale[m] = absmax(A[m])/127.
// Two-pass and fully parallel (chunked per-row absmax reduction + M*K quantize)
// so it does not collapse to a single thread at M=1 (decode).
inline void act_quant_s8_s8(const bf16* A, int8_t* Q, float* scale, int M, int K,
                            sycl::queue& queue) {
    const int CHUNK = 128;
    int nchunks = (K + CHUNK - 1) / CHUNK;

    // Pass 1: per-row absmax (scale[] zeroed, then atomic max over chunks).
    queue.memset(scale, 0, (size_t)M * sizeof(float));
    {
        unsigned* scale_u = reinterpret_cast<unsigned*>(scale);
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>((size_t)M, (size_t)nchunks),
                           [=](sycl::id<2> id) {
                int m = (int)id[0], c = (int)id[1];
                int k0 = c * CHUNK;
                int k1 = sycl::min(k0 + CHUNK, K);
                float mx = 0.0f;
                const bf16* row = A + (size_t)m * K;
                for (int k = k0; k < k1; ++k)
                    mx = sycl::fmax(mx, sycl::fabs(bf16_to_float(row[k])));
                sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    a(scale_u[m]);
                a.fetch_max(__builtin_bit_cast(unsigned, mx));
            });
        });
    }

    // Pass 2: quantize (fully parallel over M*K).
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)M * K), [=](sycl::id<1> id) {
            size_t i = id[0];
            int m = (int)(i / (size_t)K);
            float s = scale[m] > 0.0f ? scale[m] / 127.0f : 1.0f;
            float v = sycl::round(bf16_to_float(A[i]) / s);
            v = sycl::fmin(sycl::fmax(v, -127.0f), 127.0f);
            Q[i] = (int8_t)v;
        });
    });
    launchprof::record("w4a8_act_quant");
}

// C[m,n] *= scale[m] (per-token post-scale of the GEMM output).
inline void scale_rows(bf16* C, const float* scale, int M, int N,
                       sycl::queue& queue) {
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)M, (size_t)N), [=](sycl::id<2> id) {
            int m = (int)id[0], n = (int)id[1];
            C[(size_t)m * N + n] =
                float_to_bf16(bf16_to_float(C[(size_t)m * N + n]) * scale[m]);
        });
    });
    launchprof::record("w4a8_post_scale");
}

// oneDNN s8 x s4 -> bf16 GEMM, per-tensor src scale (1.0) + per-group weight
// scales. C = (A_s8 * 1.0) @ (W_s4 * scale). The per-token act scale is applied
// by the caller via scale_rows (not here).
inline void matmul_int4_s8x4_gemm(const int8_t* A, int M, int K,
                                  const Int4Linear& W, bf16* C,
                                  GpuEngine& ctx = GpuEngine::get(0)) {
    if (W.in_features != K)
        throw std::runtime_error("matmul_int4_s8x4_gemm: K does not match weight shape");
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    int N = W.out_features;
    auto src_md = dnnl::memory::desc({M, K}, dt::s8, tag::ab);
    auto dst_md = dnnl::memory::desc({M, N}, dt::bf16, tag::ab);
    auto wscale_md = dnnl::memory::desc({K / W.group_size, N}, dt::bf16, tag::ab);

    static std::unordered_map<Int4MatmulKey, Int4MatmulEntry, Int4MatmulKeyHash> cache_w4a8;
    Int4MatmulKey key{ctx.index, M, K, N, W.group_size};
    auto it = cache_w4a8.find(key);
    if (it == cache_w4a8.end()) {
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_SRC, 0, {}, dt::f32);   // per-tensor (1 scalar)
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {W.group_size, 1}, dt::bf16);
        auto weights_md = (int4_weight_layout() == Int4WeightLayout::Any)
            ? dnnl::memory::desc({K, N}, dt::s4, tag::any)
            : dnnl::memory::desc({K, N}, dt::s4, tag::ba);
        dnnl::matmul::primitive_desc pd(ctx.engine, src_md, weights_md, dst_md, attr);
        it = cache_w4a8.emplace(key,
            Int4MatmulEntry{dnnl::matmul(pd), pd.weights_desc()}).first;
    }
    auto& entry = it->second;
    const uint8_t* weight_data = int4_weight_data(W, entry.weights_md, K, N, ctx);

    static GpuBuffer<float> one_buf;
    if (one_buf.empty()) {
        one_buf = GpuBuffer<float>(1, ctx.queue);
        float one = 1.0f;
        one_buf.upload(&one, 1);
    }

    entry.primitive.execute(ctx.stream, {
        {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            src_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<int8_t*>(A))},
        {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            entry.weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<uint8_t*>(weight_data))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
            wscale_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
            const_cast<bf16*>(W.weight_scale.data()))},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({1}, dt::f32, tag::a), ctx.engine,
            dnnl::sycl_interop::memory_kind::usm, one_buf.data())},
        {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
            dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C)}
    });
    launchprof::record("w4a8_s8x4_gemm");
}

// Full W4A8 GEMM: quantize A per-token -> s8, run s8 x s4, post-scale rows.
// Drop-in replacement for matmul_int4_gemm (same (A, W, C) contract).
inline void matmul_int4_w4a8(const bf16* A, int M, int K, const Int4Linear& W,
                             bf16* C, GpuEngine& ctx = GpuEngine::get(0)) {
    static GpuBuffer<int8_t> a_s8;
    static GpuBuffer<float> a_scale;
    if (a_s8.count() < (size_t)M * K) a_s8 = GpuBuffer<int8_t>((size_t)M * K, ctx.queue);
    if (a_scale.count() < (size_t)M) a_scale = GpuBuffer<float>((size_t)M, ctx.queue);
    act_quant_s8_s8(A, a_s8.data(), a_scale.data(), M, K, ctx.queue);
    matmul_int4_s8x4_gemm(a_s8.data(), M, K, W, C, ctx);
    scale_rows(C, a_scale.data(), M, W.out_features, ctx.queue);
}

// s4 GEMM + rowsum + corr GEMM into corr_out (caller-owned M*N). The caller
// consumes (C, corr) with a fused kernel; nothing is subtracted here.
inline void matmul_int4_zp_corr(const bf16* A, int M, int K, const Int4Linear& W,
                                bf16* C, GpuEngine& ctx, bf16* rowsum_scratch,
                                bf16* corr_out) {
    if (W.in_features != K)
        throw std::runtime_error("matmul_int4_zp_corr: K does not match weight shape");
    matmul_int4_gemm(A, M, K, W, C, ctx);
    if (W.has_zp()) {
        int G = K / W.group_size;
        int4_rowsum(A, M, K, W.group_size, rowsum_scratch, ctx.queue);
        matmul_bf16_nn(rowsum_scratch, M, G, W.zp_offset.data(),
                       W.out_features, corr_out, ctx);
    }
}

// C (M,N) = A (M,K) @ dequant(W)^T, where W is logical (N,K) s4 with per-group
// BF16 scales.  A and C are BF16.  Async on ctx's stream.
//
// Optional trailing args (qwen3_5 dense uses these; other callers keep default):
//   rowsum_scratch     - caller-owned M*ceil(K/group) bf16 buffer. Avoids the
//                        per-call heap allocation for the zp rowsum. May be
//                        nullptr (heap fallback).
//   corr_scratch       - caller-owned M*N bf16 buffer for the zp correction
//                        GEMM output. Avoids the per-call heap allocation.
//                        May be nullptr (heap fallback).
//   precomputed_rowsum - rowsum(A) already computed (e.g. shared by two
//                        projections on the same activation). nullptr = compute.
inline void matmul_int4(
    const bf16* A,
    int M,
    int K,
    const Int4Linear& W,
    bf16* C,
    GpuEngine& ctx = GpuEngine::get(0),
    bf16* rowsum_scratch = nullptr,
    bf16* corr_scratch = nullptr,
    const bf16* precomputed_rowsum = nullptr)
{
    if (W.in_features != K)
        throw std::runtime_error("matmul_int4: K does not match weight shape");
    if (K % W.group_size != 0)
        throw std::runtime_error("matmul_int4: K must be divisible by group_size");

    int N = W.out_features;
    int G = K / W.group_size;

    auto run = [&](const bf16* A_in, bf16* C_out) {
        if (int4_w4a8_enabled()) {
            matmul_int4_w4a8(A_in, M, K, W, C_out, ctx);
        } else {
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
                    const_cast<bf16*>(A_in))},
                {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
                    entry.weights_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
                    const_cast<uint8_t*>(weight_data))},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
                    wscale_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
                    const_cast<bf16*>(W.weight_scale.data()))},
                {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
                    dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, C_out)}
            });
            launchprof::record("int4_gemm");
        }

        if (W.has_zp()) {
            // True dequant is w = scale * (q_u - zp_u); the s4 GEMM above computed
            // scale * (q_u - 8), so subtract scale * (zp_u - 8) summed over each
            // K-group: C[m,n] -= rowsum(A)[m,g] * zp_offset[g,n].
            const bf16* rs = precomputed_rowsum;
            GpuBuffer<bf16> rowsum_heap, corr_heap;
            bf16* rowsum_ptr = int4_zp_scratch() ? rowsum_scratch : nullptr;
            if (!rs) {
                if (!rowsum_ptr) {
                    rowsum_heap = GpuBuffer<bf16>((size_t)M * G, ctx.queue);
                    rowsum_ptr = rowsum_heap.data();
                }
                int4_rowsum(A_in, M, K, W.group_size, rowsum_ptr, ctx.queue);
                rs = rowsum_ptr;
            }
            bf16* corr_ptr = int4_zp_scratch() ? corr_scratch : nullptr;
            if (!corr_ptr) {
                corr_heap = GpuBuffer<bf16>((size_t)M * N, ctx.queue);
                corr_ptr = corr_heap.data();
            }
            // The corr GEMM launch itself is recorded as bf16_gemm_nn inside
            // matmul_bf16_nn (runtime/gpu/ops.hpp) — the authoritative census.
            matmul_bf16_nn(rs, M, G, W.zp_offset.data(), N, corr_ptr, ctx);

            bf16* C_ptr = C_out;
            const bf16* corr_data = corr_ptr;
            auto corr_ev = ctx.queue.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>((size_t)M * N), [=](sycl::id<1> id) {
                    C_ptr[id[0]] = float_to_bf16(
                        bf16_to_float(C_ptr[id[0]]) - bf16_to_float(corr_data[id[0]]));
                });
            });
            launchprof::record("int4_corr_sub", corr_ev);
        }
    };

    Int4ActQuantMode mode = int4_act_quant_mode();
    if (mode != Int4ActQuantMode::Off && int4_act_quant_skipped(K, N))
        mode = Int4ActQuantMode::Off;
    if (mode == Int4ActQuantMode::Off) {
        run(A, C);
        return;
    }

    GpuBuffer<bf16> aq((size_t)M * K, ctx.queue);
    act_quant_s8_roundtrip(A, aq.data(), M, K, ctx.queue);

    if (mode == Int4ActQuantMode::Apply) {
        run(aq.data(), C);
        return;
    }

    // Stats: baseline stays in C; quantized GEMM goes to scratch for comparison.
    run(A, C);
    GpuBuffer<bf16> cq((size_t)M * N, ctx.queue);
    run(aq.data(), cq.data());
    static int call = 0;
    act_quant_s8_report(C, cq.data(), (size_t)M * N, call++, M, K, N, ctx.queue);
}

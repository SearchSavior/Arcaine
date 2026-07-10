#pragma once
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph/command_graph.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <mutex>
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
// __spirv_SubgroupMatrixMultiplyAccumulateINTEL has no host definition (it's a
// SPIRV intrinsic materialized only under __SYCL_DEVICE_ONLY__). The #else
// stub exists so host-side translation units that pull in this header compile,
// never run. Suppress the bogus -Wundefined-inline on the host stub.
#if defined(__clang__) || defined(__INTEL_CLANG_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
#endif
SYCL_EXTERNAL inline diff_dpas_v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, diff_dpas_v8s A, diff_dpas_v8i B, diff_dpas_v8f C, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return diff_dpas_v8f{}; }
#endif
#if defined(__clang__) || defined(__INTEL_CLANG_COMPILER)
#pragma clang diagnostic pop
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
inline bool nvfp4_sycl_graph_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_NVFP4_SYCL_GRAPH");
        return env && std::strcmp(env, "0") && std::strcmp(env, "off") &&
               std::strcmp(env, "false") && std::strcmp(env, "no");
    }();
    return enabled;
}
inline size_t nvfp4_sycl_graph_cache_limit() {
    static size_t limit = [] {
        const char* env = std::getenv("DIFF_NVFP4_SYCL_GRAPH_CACHE_LIMIT");
        if (!env) return size_t{512};
        char* end = nullptr;
        unsigned long parsed = std::strtoul(env, &end, 10);
        return end != env && parsed > 0 ? static_cast<size_t>(parsed) : size_t{512};
    }();
    return limit;
}
struct Nvfp4SyclGraphKey {
    const sycl::queue* queue = nullptr;
    int kind = 0;
    std::vector<uintptr_t> args;
    bool operator==(const Nvfp4SyclGraphKey& other) const {
        return queue == other.queue && kind == other.kind && args == other.args;
    }
};
struct Nvfp4SyclGraphKeyHash {
    size_t operator()(const Nvfp4SyclGraphKey& key) const {
        size_t hash = std::hash<const sycl::queue*>{}(key.queue);
        auto mix = [&](uintptr_t value) {
            hash ^= std::hash<uintptr_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        };
        mix(static_cast<uintptr_t>(key.kind));
        for (uintptr_t value : key.args) mix(value);
        return hash;
    }
};
struct Nvfp4SyclGraphEntry {
    using Modifiable = sycl::ext::oneapi::experimental::command_graph<>;
    using Executable = sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>;
    std::unique_ptr<Modifiable> graph;
    std::unique_ptr<Executable> executable;
    bool unavailable = false;
};
struct Nvfp4SyclGraphCache {
    std::mutex mutex;
    std::unordered_map<Nvfp4SyclGraphKey, Nvfp4SyclGraphEntry,
                       Nvfp4SyclGraphKeyHash> entries;
    size_t captures = 0;
    size_t replays = 0;
    size_t fallbacks = 0;
    size_t capacity_bypasses = 0;
};
inline Nvfp4SyclGraphCache& nvfp4_sycl_graph_cache() {
    static Nvfp4SyclGraphCache cache;
    return cache;
}
inline uintptr_t nvfp4_sycl_graph_float_key(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// ---------------------------------------------------------------------------
// Nvfp4GraphSession
//
// The per-call micro-graph cache below (nvfp4_sycl_graph_submit) captures and
// replays ONE kernel at a time, keyed by that kernel's own buffer pointers.
// That's fine for amortizing a single pack/matmul launch, but a denoising
// step calls many of these in sequence (attention -> router -> gather/pack ->
// grouped GEMM -> combine), and today each still gets its own
// begin_recording/end_recording/finalize and its own q.ext_oneapi_graph()
// dispatch. A session instead lets a caller open ONE graph, have every kernel
// in the step record into it (including attention kernels defined outside
// this file), and replay the whole step with a single graph launch -- which
// is the piece that actually removes host round-trips between kernels within
// a step, not just between repeated steps.
//
// Usage:
//   Nvfp4GraphSession session;
//   if (session.begin(q, step_key)) {
//       attn_forward(..., &session);
//       moe_forward(..., &session);
//       session.end_and_replay();
//   } else {
//       // cache hit: session.begin() already replayed the cached graph.
//   }
//
// Layer types are static per DiffusionGemma config (layer_types is fixed at
// model-build time), so the sequence of kernels captured for a given
// (layer_index, is_sliding) step_key never changes shape across replays --
// this is what makes whole-step capture safe to cache and replay verbatim,
// the same way the per-kernel cache below already relies on stable shapes.
// Queue-keyed registry of *actively recording* sessions. nvfp4_sycl_graph_submit()
// consults this as a backstop: when a session is recording on a queue, ANY
// per-kernel micro-capture attempt on that queue would call begin_recording on
// an already-recording queue, which throws (intel-llvm graph_impl.cpp:2238) and
// would poison the per-kernel cache (entry marked `unavailable` forever).
// The registry lets such call sites stand down to a bare submit() -- which the
// active session's recording captures as a node -- even when the caller didn't
// receive an explicit session pointer (e.g. the dense-MLP pack_bf16_to_nvfp4
// and the MoE Xe-pack paths, which thread through matmul_nvfp4 / expert_parallel
// without an explicit session argument). Explicit session threading (Phases 1-2)
// remains the preferred fast path; this registry is the correctness backstop.
struct Nvfp4GraphSession;
inline std::mutex& nvfp4_session_registry_mutex() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<const sycl::queue*, Nvfp4GraphSession*>&
nvfp4_session_registry() {
    static std::unordered_map<const sycl::queue*, Nvfp4GraphSession*> r;
    return r;
}
// Returns the session currently recording on `q`, or nullptr if none. Called
// on the hot path of nvfp4_sycl_graph_submit; the map is normally empty when no
// session is active, so the lookup is a fast miss.
inline Nvfp4GraphSession* nvfp4_active_session(const sycl::queue& q) {
    std::lock_guard<std::mutex> lock(nvfp4_session_registry_mutex());
    auto it = nvfp4_session_registry().find(&q);
    return it == nvfp4_session_registry().end() ? nullptr : it->second;
}
// True iff a Nvfp4GraphSession is currently recording on `q`. Forward-declarable
// (no Nvfp4GraphSession definition needed) so lightweight headers such as
// utils/profile.hpp can call it without including nvfp4.hpp's DPAS intrinsics.
// Used to skip queue waits during recording (q.wait()/stream.wait() throw on a
// recording queue: "wait cannot be called for a queue which is recording to a
// command graph").
//
// NOTE: declared here, defined out-of-line in common/gpu/expert_parallel.cpp.
// It must NOT be `inline`: a TU that only forward-declares it (e.g. profile.cpp)
// emits an external reference; an `inline` definition is only emitted as
// linkonce_odr by TUs that include this header and use it, and if all such uses
// are inlined away there is no standalone symbol to satisfy that reference.
bool nvfp4_session_recording(const sycl::queue& q);
struct Nvfp4GraphSession {
    using Modifiable = sycl::ext::oneapi::experimental::command_graph<>;
    using Executable = sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>;

    sycl::queue* queue_ = nullptr;
    std::unique_ptr<Modifiable> graph_;
    bool recording_ = false;
    // True once begin() has opened a *new* recording that the caller must
    // populate and close with end_and_replay(). False if begin() found a
    // cached executable and already replayed it (nothing left to record).
    bool needs_recording_ = false;

    // True while this session is actively recording -- kernels submitted via
    // nvfp4_sycl_graph_submit() check this and skip their own micro-graph
    // capture, just enqueuing directly so they land inside the session graph.
    bool active() const { return recording_; }

    // Returns true if the caller must record the step's kernels (cache miss);
    // false if a cached graph was found and already replayed (cache hit).
    bool begin(sycl::queue& q, const Nvfp4SyclGraphKey& step_key) {
        queue_ = &q;
        if (!nvfp4_sycl_graph_enabled()) {
            needs_recording_ = false;
            recording_ = false;
            return true;  // graphs disabled: caller runs kernels eagerly every time
        }
        auto& cache = nvfp4_sycl_graph_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto found = cache.entries.find(step_key);
        if (found != cache.entries.end() && found->second.executable) {
            ++cache.replays;
            q.ext_oneapi_graph(*found->second.executable);
            needs_recording_ = false;
            recording_ = false;
            return false;
        }
        // Either not present, or present-but-unavailable (previous capture
        // failed) -- try recording fresh. Reserve/replace the slot now so a
        // racing session doesn't also try to record the same key.
        if (found == cache.entries.end() &&
            cache.entries.size() >= nvfp4_sycl_graph_cache_limit()) {
            ++cache.capacity_bypasses;
            needs_recording_ = false;
            recording_ = false;
            return true;  // caller runs kernels eagerly, no capture this time
        }
        key_ = step_key;
        // assume_buffer_outlives_graph: oneDNN's SYCL interop may internally use
        // sycl::buffer for some primitives (e.g. non-batched matmul_bf16). SYCL
        // throws "Cannot use buffers in a graph without ... assume_buffer_outlives_graph"
        // if a buffer-accessor is recorded into a graph lacking this property. All
        // persistent data (weights, KV cache, arena workspaces) is USM and outlives
        // the graph; the property is the documented oneDNN-interop capture enabler.
        graph_ = std::make_unique<Modifiable>(q,
            sycl::property_list{
                sycl::ext::oneapi::experimental::property::graph::assume_buffer_outlives_graph{}});
        try {
            graph_->begin_recording(q);
        } catch (const std::exception& error) {
            graph_.reset();
            needs_recording_ = false;
            recording_ = false;
            if (nvfp4_verbose())
                std::fprintf(stderr, "[nvfp4-graph] session capture disabled: %s\n",
                             error.what());
            return true;  // caller runs kernels eagerly this time
        }
        recording_ = true;
        needs_recording_ = true;
        {
            std::lock_guard<std::mutex> lock(nvfp4_session_registry_mutex());
            nvfp4_session_registry()[&q] = this;
        }
        return true;
    }

    // Closes recording, finalizes, caches, and replays the freshly-captured
    // graph. Call exactly once, after the caller has submitted every kernel
    // for the step, only when begin() returned true AND active() is true
    // (i.e. begin() did not already replay a cached hit).
    void end_and_replay() {
        if (!recording_) return;  // begin() already replayed a cached graph
        auto& cache = nvfp4_sycl_graph_cache();
        // Unregister from the active-session registry now that recording has
        // ended, so nvfp4_sycl_graph_submit stops standing down for this queue.
        {
            std::lock_guard<std::mutex> lock(nvfp4_session_registry_mutex());
            nvfp4_session_registry().erase(queue_);
        }
        try {
            graph_->end_recording(*queue_);
            recording_ = false;
            auto executable = std::make_unique<Executable>(graph_->finalize());
            std::lock_guard<std::mutex> lock(cache.mutex);
            Nvfp4SyclGraphEntry entry;
            entry.graph = std::move(graph_);
            entry.executable = std::move(executable);
            ++cache.captures;
            queue_->ext_oneapi_graph(*entry.executable);
            cache.entries[key_] = std::move(entry);
        } catch (const std::exception& error) {
            if (recording_) {
                try { graph_->end_recording(*queue_); } catch (...) {}
                recording_ = false;
            }
            std::lock_guard<std::mutex> lock(cache.mutex);
            ++cache.fallbacks;
            Nvfp4SyclGraphEntry entry;
            entry.unavailable = true;
            cache.entries[key_] = std::move(entry);
            if (nvfp4_verbose())
                std::fprintf(stderr, "[nvfp4-graph] session finalize failed: %s\n",
                             error.what());
            // Kernels already ran eagerly on queue_ during recording attempts
            // in SYCL graph semantics recording still enqueues work, so no
            // separate re-run is needed here; the step's results are correct,
            // only the *caching* of it for next time failed.
        }
    }

    Nvfp4SyclGraphKey key_;
};

// Builds a step-level cache key from a layer index and its (static) attention
// kind, so sliding-window layers and global layers -- which submit different
// kernel sequences -- are never conflated into the same cached graph even
// though both are "one decoder layer step" at the call site.
inline Nvfp4SyclGraphKey nvfp4_step_key(const sycl::queue& q, int layer_index,
                                       bool is_sliding, int denoise_step) {
    // denoise_step is intentionally NOT part of the key for the *replay*
    // path -- the kernel sequence for "decoder layer L, sliding" is identical
    // on every denoising step, so one capture at step 0 should serve all
    // subsequent steps. It's accepted here only so callers that want
    // per-step keys (e.g. while validating that shapes truly don't change
    // across steps) can opt into it via kind, without changing this
    // function's signature.
    (void)denoise_step;
    Nvfp4SyclGraphKey key;
    key.queue = &q;
    key.kind = 1000 + layer_index * 2 + (is_sliding ? 0 : 1);
    return key;
}

template <class Submit>
inline void nvfp4_sycl_graph_submit(sycl::queue& q, int kind,
                                    std::initializer_list<uintptr_t> args,
                                    Submit&& submit,
                                    Nvfp4GraphSession* session = nullptr) {
    if (session && session->active()) {
        // An outer Nvfp4GraphSession is already recording this step's graph;
        // just enqueue -- it gets captured as part of that larger graph
        // rather than owning its own begin/end/finalize here.
        submit();
        return;
    }
    if (Nvfp4GraphSession* active = nvfp4_active_session(q)) {
        // Backstop: a session is recording on this queue but the caller didn't
        // carry an explicit session pointer (e.g. dense-MLP pack_bf16_to_nvfp4
        // via matmul_nvfp4, or MoE pack paths via expert_parallel). Stand down
        // the same way -- a bare submit() is captured as a node of the active
        // session's graph. Without this, the begin_recording() below would
        // throw on the already-recording queue and poison this per-kernel
        // cache entry (unavailable forever).
        (void)active;
        submit();
        return;
    }
    if (!nvfp4_sycl_graph_enabled()) {
        submit();
        return;
    }
    Nvfp4SyclGraphKey key{&q, kind, args};
    auto& cache = nvfp4_sycl_graph_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto found = cache.entries.find(key);
    if (found != cache.entries.end()) {
        if (found->second.executable) {
            ++cache.replays;
            q.ext_oneapi_graph(*found->second.executable);
            return;
        }
        ++cache.fallbacks;
        submit();
        return;
    }
    if (cache.entries.size() >= nvfp4_sycl_graph_cache_limit()) {
        ++cache.capacity_bypasses;
        submit();
        return;
    }
    auto [it, inserted] = cache.entries.emplace(std::move(key), Nvfp4SyclGraphEntry{});
    auto& entry = it->second;
    bool recording = false;
    try {
        entry.graph = std::make_unique<Nvfp4SyclGraphEntry::Modifiable>(q);
        entry.graph->begin_recording(q);
        recording = true;
        submit();
        entry.graph->end_recording(q);
        recording = false;
        entry.executable = std::make_unique<Nvfp4SyclGraphEntry::Executable>(entry.graph->finalize());
        ++cache.captures;
        q.ext_oneapi_graph(*entry.executable);
    } catch (const std::exception& error) {
        if (recording) {
            try {
                entry.graph->end_recording(q);
            } catch (...) {
            }
        }
        entry.graph.reset();
        entry.executable.reset();
        entry.unavailable = true;
        ++cache.fallbacks;
        if (nvfp4_verbose())
            std::fprintf(stderr, "[nvfp4-graph] capture disabled: %s\n", error.what());
        submit();
    }
}
inline void nvfp4_sycl_graph_report() {
    if (!nvfp4_sycl_graph_enabled()) return;
    auto& cache = nvfp4_sycl_graph_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    std::fprintf(stderr,
                 "[nvfp4-graph] captures=%zu replays=%zu fallbacks=%zu cache-bypasses=%zu entries=%zu\n",
                 cache.captures, cache.replays, cache.fallbacks,
                 cache.capacity_bypasses, cache.entries.size());
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
    float input_global_scale,
    Nvfp4GraphSession* session = nullptr)
{
    if (K % 16 != 0) throw std::runtime_error("NVFP4 activation K must be divisible by 16");
    int G = K / 16;
    nvfp4_sycl_graph_submit(q, 1, {
        reinterpret_cast<uintptr_t>(src), reinterpret_cast<uintptr_t>(packed),
        reinterpret_cast<uintptr_t>(scales), static_cast<uintptr_t>(M),
        static_cast<uintptr_t>(K), nvfp4_sycl_graph_float_key(input_global_scale)
    }, [&] {
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
    }, session);
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
//     shared [G,N] layout is REJECTED by oneDNN at execute -- do not use it)
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
    GpuEngine& ctx = GpuEngine::get(0),
    uint8_t* A_packed_buf = nullptr,
    uint8_t* A_scale_buf = nullptr)
{
    if (W.in_features != K)
        throw std::runtime_error("matmul_nvfp4: K does not match weight shape");
    if (K % 16 != 0)
        throw std::runtime_error("matmul_nvfp4: K must be divisible by 16");
    int G = K / 16;
    auto& q = ctx.queue;
    // When caller-provided (stable, arena-backed) workspaces are passed, use
    // them directly: no per-call sycl::malloc_device (which would be truly
    // freed at scope exit and so dangle at graph-replay time) and no
    // ctx.stream.wait() (the in-order queue serializes the pack before the
    // matmul, and the workspace is owned by the caller so there is no
    // use-after-free on free). This makes matmul_nvfp4 safe to capture inside a
    // Nvfp4GraphSession. Without caller workspaces, fall back to the original
    // transient-GpuBuffer + synchronous-wait behavior (unchanged for existing
    // callers that don't opt in).
    if (A_packed_buf && A_scale_buf) {
        pack_bf16_to_nvfp4(q, A, A_packed_buf, A_scale_buf, M, K,
                           W.input_global_scale);
        matmul_nvfp4_packed(A_packed_buf, A_scale_buf, M, K, W, C, ctx);
        return;
    }
    GpuBuffer<uint8_t> A_packed((size_t)M * K / 2, q);
    GpuBuffer<uint8_t> A_scale((size_t)M * G, q);
    pack_bf16_to_nvfp4(q, A, A_packed.data(), A_scale.data(), M, K,
                       W.input_global_scale);
    matmul_nvfp4_packed(A_packed.data(), A_scale.data(), M, K, W, C, ctx);
    // A_packed/A_scale are temporary workspaces owned by this call. Keep the
    // execution synchronous until callers pass reusable workspaces explicitly.
    if (!nvfp4_session_recording(q)) ctx.stream.wait();
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
// Vectorized (ESIMD simd) arithmetic e2m1 dequant -- no LUT, no scalar loop.
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
// Vectorized (ESIMD simd) arithmetic e4m3 dequant -- bit-exact vs nvfp4_e4m3_fast.
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
    uint8_t* scales,
    Nvfp4GraphSession* session = nullptr)
{
    if (K % 16 != 0) throw std::runtime_error("grouped NVFP4 activation K must be divisible by 16");
    int G = K / 16;
    nvfp4_sycl_graph_submit(q, 2, {
        reinterpret_cast<uintptr_t>(src), reinterpret_cast<uintptr_t>(row_slot),
        reinterpret_cast<uintptr_t>(row_expert), reinterpret_cast<uintptr_t>(input_global_scale),
        reinterpret_cast<uintptr_t>(packed), reinterpret_cast<uintptr_t>(scales),
        static_cast<uintptr_t>(K), static_cast<uintptr_t>(rows)
    }, [&] {
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
    }, session);
}
// Fuse GeGLU (gate_up -> act) with the down-projection input pack (act -> nvfp4).
// Eliminates the bf16 `act` intermediate and one kernel launch. Bit-exact with
// the unfused sequence (geglu_strided_grouped + pack_bf16_to_nvfp4_grouped):
// the geglu float result is rounded through bf16 (write->read round-trip)
// before the identical per-16-group quant math.
inline void geglu_pack_nvfp4_grouped(
    sycl::queue& q,
    const bf16* gate_up,          // [total_rows, 2*inter]  (gate | up), indexed by row_slot
    int inter,
    const int32_t* row_slot,      // [rows] -> slot offset in [0, total_rows)
    const int32_t* row_expert,    // [rows] -> expert id (for input_global_scale)
    int rows,
    const float* input_global_scale,  // [num_experts] (down proj's per-expert input scale)
    uint8_t* packed,              // [total_rows, inter/2]
    uint8_t* scales,              // [total_rows, inter/16]
    Nvfp4GraphSession* session = nullptr)
{
    if (inter % 16 != 0)
        throw std::runtime_error("geglu_pack_nvfp4_grouped: inter must be divisible by 16");
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEF = 0.044715f;
    int G = inter / 16;
    nvfp4_sycl_graph_submit(q, 3, {
        reinterpret_cast<uintptr_t>(gate_up), reinterpret_cast<uintptr_t>(row_slot),
        reinterpret_cast<uintptr_t>(row_expert), reinterpret_cast<uintptr_t>(input_global_scale),
        reinterpret_cast<uintptr_t>(packed), reinterpret_cast<uintptr_t>(scales),
        static_cast<uintptr_t>(inter), static_cast<uintptr_t>(rows)
    }, [&] {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)rows, (size_t)G), [=](sycl::id<2> id) {
            int r = (int)id[0];
            int g = (int)id[1];
            int slot = row_slot[r];
            int expert = row_expert[r];
            int k0 = g * 16;
            const bf16* row = gate_up + (size_t)slot * 2 * inter;
            uint8_t* packed_row = packed + (size_t)slot * (inter / 2);
            uint8_t* scale_row = scales + (size_t)slot * G;
            float input_scale = input_global_scale[expert];
            float vals[16];
            float max_abs = 0.0f;
            for (int i = 0; i < 16; ++i) {
                float gg = bf16_to_float(row[k0 + i]);
                float uu = bf16_to_float(row[inter + k0 + i]);
                float inner = SQRT_2_OVER_PI * (gg + COEF * gg * gg * gg);
                // Round-trip through bf16 matches the unfused write->read exactly.
                float v = bf16_to_float(float_to_bf16(0.5f * gg * (1.0f + sycl::tanh(inner)) * uu));
                vals[i] = v;
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
                    float v0 = vals[i] * input_scale / scale;
                    float v1 = vals[i + 1] * input_scale / scale;
                    lo = nvfp4_encode_e2m1(v0);
                    hi = nvfp4_encode_e2m1(v1);
                }
                packed_row[(k0 + i) / 2] = lo | (hi << 4);
            }
        });
    });
    }, session);
}
// Fuse the hidden-state scatter with the gate/up-projection input pack.
// Replaces: scatter hidden->Xe (bf16) + pack_bf16_to_nvfp4_grouped(Xe,...).
// Iterates source assignments a in [0, A_all); for each valid a (slot >= 0),
// gathers 16 bf16 hidden values for token (a / top_k) and packs them straight
// to nvfp4 at slot[a]. Padding bucket rows (rounded rows beyond count[e],
// which no valid a maps to) are NOT written here -- caller must memset
// `packed`/`scales` to 0 first, which matches the unfused behavior (Xe is
// memset to 0, so pack produces zeros for those rows).
//
// Device-routes mode only: relies on `slot` being the atomic-computed dest
// slot per source assignment (0xFFFFFFFF/-1 for invalid assignments).
inline void scatter_pack_nvfp4_grouped(
    sycl::queue& q,
    const bf16* hidden,          // [seq, H]
    int H,
    const int32_t* slot,         // [A_all] dest slot per source assignment
    const int32_t* idx_dev,      // [A_all] expert id per source assignment
    int first_expert,
    const float* input_global_scale,  // [num_experts] (gate/up proj's per-expert input scale)
    int A_all,
    int top_k,
    uint8_t* packed,             // [total_rows, H/2]
    uint8_t* scales,             // [total_rows, H/16]
    Nvfp4GraphSession* session = nullptr)
{
    if (H % 16 != 0)
        throw std::runtime_error("scatter_pack_nvfp4_grouped: H must be divisible by 16");
    int G = H / 16;
    nvfp4_sycl_graph_submit(q, 4, {
        reinterpret_cast<uintptr_t>(hidden), reinterpret_cast<uintptr_t>(slot),
        reinterpret_cast<uintptr_t>(idx_dev), reinterpret_cast<uintptr_t>(input_global_scale),
        reinterpret_cast<uintptr_t>(packed), reinterpret_cast<uintptr_t>(scales),
        static_cast<uintptr_t>(H), static_cast<uintptr_t>(first_expert),
        static_cast<uintptr_t>(A_all), static_cast<uintptr_t>(top_k)
    }, [&] {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)A_all, (size_t)G), [=](sycl::id<2> id) {
            int a = (int)id[0];
            int g = (int)id[1];
            int s = slot[a];
            if (s < 0) return;                 // invalid assignment (slot init 0xFF)
            int expert = idx_dev[a] - first_expert;
            if (expert < 0) return;            // safety (slot<0 check above covers this)
            int token = a / top_k;
            int k0 = g * 16;
            const bf16* src_row = hidden + (size_t)token * H;
            uint8_t* packed_row = packed + (size_t)s * (H / 2);
            uint8_t* scale_row = scales + (size_t)s * G;
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
    }, session);
}
// ---------------------------------------------------------------------------
// Grouped-GEMM entry points (custom scalar / DPAS gateup-geglu-pack / Xe2).
// These submit raw sycl kernels via q.submit (no oneDNN primitive, no per-kernel
// nvfp4_sycl_graph_submit wrapper). Session-awareness (Phase 2): each accepts a
// trailing `session` pointer for API consistency with the pack functions and so
// the orchestrator (Phase 3) can pass an active Nvfp4GraphSession through the
// MoE call chain. They are NOT individually wrapped in nvfp4_sycl_graph_submit
// (task Phase-2 option a) because:
//   * Under an active session recording, bare q.submit calls are captured
//     automatically as graph nodes (the canonical SYCL graph capture path;
//     oneDNN primitives were separately confirmed capture-safe in Phase 0).
//   * Wrapping them (option b) for per-kernel micro-caching on the no-session
//     path is deferred: it requires (1) per-kernel capture+replay validation of
//     these DPAS/custom ESIMD kernels (Phase 0 tested oneDNN only) and (2) the
//     Phase-4 confirmation that the arena yields stable USM offsets across
//     denoising steps (the existing pack micro-cache's captures=512/replays=4651
//     ratio is empirical evidence this holds for the fixed-shape path). The
//     `session` argument is therefore accepted but not yet consumed here; the
//     registry backstop in nvfp4_sycl_graph_submit already protects any nested
//     per-kernel capture from conflicting with an active session.
// ---------------------------------------------------------------------------
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
    bf16* C,
    Nvfp4GraphSession* /*session*/ = nullptr)
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
    uint8_t* scales,
    Nvfp4GraphSession* session = nullptr)
{
    if (K % 16 != 0) throw std::runtime_error("counted grouped NVFP4 activation K must be divisible by 16");
    int G = K / 16;
    nvfp4_sycl_graph_submit(q, 5, {
        reinterpret_cast<uintptr_t>(src), reinterpret_cast<uintptr_t>(row_expert),
        reinterpret_cast<uintptr_t>(input_global_scale), reinterpret_cast<uintptr_t>(packed),
        reinterpret_cast<uintptr_t>(scales), static_cast<uintptr_t>(K),
        static_cast<uintptr_t>(max_rows)
    }, [&] {
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
    }, session);
}
// ============================================================================
// Grouped DPAS gate/up + GeGLU + down-pack fused kernel.
//
// One custom grouped-DPAS kernel that replaces, in the hybrid path:
//   (1) the per-expert oneDNN gate/up GEMM loop  -> `gu` [rows, 2*inter] bf16
//   (2) geglu_strided_grouped                     -> `act` [rows, inter] bf16
//   (3) pack_bf16_to_nvfp4_grouped               -> `act_packed`/`act_scale`
// The `gu` and `act` bf16 intermediates are eliminated entirely; the GeGLU
// result is packed straight to nvfp4 in the epilogue.
//
// Layout (matches the hybrid branch's compute_slot/compute_expert arrays,
// built per-expert in 8-row-aligned runs so every 8-row m-tile shares one
// expert):
//   A_packed/A_scale : xe_packed/xe_scale at slot = row_slot[r]  (the scattered
//                      hidden state, already nvfp4-packed by scatter+pack).
//   W_coal_by_expert : nvfp4_coalesced_weight(gate_up_proj_fp4[e], H, 2*inter)
//                      -- coalesced [2*inter, K] per expert (cached on the linear).
//   W_scale_by_expert: gate_up_proj_fp4[e].weight_scale  ([G=K/16, 2*inter])
//   dst_scale        : gate_up_proj_fp4[e].dst_scale ([1] f32)
//   down_input_global_scale : down_proj_fp4[e].input_global_scale (for the pack)
//
// Grid: nd_range<2>({mtiles*KS, inter}, {KS, 16})  -- one WG per (8-row m-tile,
// gate group g in [0, inter/16)). KS threads split K; 16 lanes own the 16
// columns of the gate/up tile. Gate (cols ng..ng+15) and up (cols inter+ng..)
// share the SAME a_tile (one input load, two DPAS -- llm-scaler gate+up fusion).
// After SLM K-reduce, (s==0, lane==0) does GeGLU+pack for the 8 rows of group g.
//
// Selection: DIFF_NVFP4_EXPERT_KERNEL=grouped-dpas (new codepath; the hybrid
// oneDNN loop remains the default and is untouched).
// ============================================================================
inline void matmul_nvfp4_grouped_dpas_gateup_geglu_pack(
    GpuEngine& ctx,
    const uint8_t* A_packed,          // xe_packed [total_rows, H/2]
    const uint8_t* A_scale,          // xe_scale  [total_rows, H/16]
    int H,                            // K (in_features of gate/up)
    const int32_t* row_slot,         // compute_slot [rows] -> slot offset
    const int32_t* row_expert,       // compute_expert [rows] -> expert id
    int rows,
    const uint8_t* const* W_coal_by_expert,    // [localE] coalesced [2*inter, K]
    const uint8_t* const* W_scale_by_expert,   // [localE] [G=K/16, 2*inter]
    const float* const* dst_scale_by_expert,    // [localE] [1]
    int inter,
    const float* down_input_global_scale,      // [localE] down proj input scale
    uint8_t* act_packed,              // [total_rows, inter/2]
    uint8_t* act_scale,              // [total_rows, inter/16]
    Nvfp4GraphSession* /*session*/ = nullptr)
{
    if (H % 16 != 0 || inter % 16 != 0)
        throw std::runtime_error("grouped-dpas gateup: H and inter must be divisible by 16");
    if (rows <= 0) return;
    auto& q = ctx.queue;
    int halfK = H / 2;
    int ktiles = H / 16;
    int mtiles = (rows + 7) / 8;
    int G_inter = inter / 16;
    int Ngu = 2 * inter;                 // gate|up out_features
    int KS = nvfp4_dpas_ksplit_factor(mtiles, ktiles, G_inter / 16);
    const uint16_t* lut = nvfp4_dequant_lut(ctx);
    q.submit([&](sycl::handler& h) {
        // SLM holds two accumulators (gate, up): KS threads * 8 rows * 16 lanes.
        sycl::local_accessor<float, 1> slm((size_t)2 * KS * 8 * 16, h);
        float* slm_g = slm.get_pointer();
        float* slm_u = slm_g + (size_t)KS * 8 * 16;
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)mtiles * KS, (size_t)inter),
                              sycl::range<2>((size_t)KS, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int m0   = (int)it.get_group(0) * 8;
                int s    = (int)it.get_local_id(0);
                int lane = (int)it.get_local_id(1);
                int g    = (int)it.get_group(1);          // gate group index
                int ng   = g * 16;
                int gate_n = ng;                          // gate cols [ng, ng+16)
                int up_n   = inter + ng;                  // up   cols [inter+ng, ...)
                int r0    = m0;
                int expert = (r0 < rows) ? row_expert[r0] : -1;
                diff_dpas_v8f cg = {0,0,0,0,0,0,0,0};
                diff_dpas_v8f cu = {0,0,0,0,0,0,0,0};
                if (expert >= 0) {
                    const uint8_t* wcoal = W_coal_by_expert[expert];
                    const uint8_t* wscale = W_scale_by_expert[expert];
                    for (int kt = s; kt < ktiles; kt += KS) {
                        // Gate and up B-tiles (two coalesced dequants, same K-tile).
                        diff_dpas_v8i bg = nvfp4_dequant_b_coal(
                            wcoal, lut, wscale, gate_n, lane, kt, H, Ngu);
                        diff_dpas_v8i bu = nvfp4_dequant_b_coal(
                            wcoal, lut, wscale, up_n, lane, kt, H, Ngu);
                        // A-tile: computed ONCE, shared by gate and up DPAS.
                        int k0 = kt * 16;
                        int kk = k0 + lane;
                        diff_dpas_v8s a;
                        for (int m = 0; m < 8; ++m) {
                            int r = r0 + m;
                            uint16_t av = 0;
                            if (r < rows && row_expert[r] == expert) {
                                uint8_t byte = A_packed[(size_t)row_slot[r] * halfK + kk / 2];
                                uint8_t nib = (kk & 1) ? (byte >> 4) : (byte & 0x0f);
                                av = float_to_bf16(
                                    nvfp4_e2m1_fast(nib) *
                                    nvfp4_e4m3_fast(A_scale[(size_t)row_slot[r] * ktiles + kt]));
                            }
                            a[m] = (short)av;
                        }
                        cg = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, a, bg, cg, kNvfp4DpasBF16);
                        cu = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                            16, a, bu, cu, kNvfp4DpasBF16);
                    }
                }
                // Stash partials (gate + up) for the K-split reduction.
                for (int m = 0; m < 8; ++m) {
                    slm_g[(size_t)(s * 8 + m) * 16 + lane] = cg[m];
                    slm_u[(size_t)(s * 8 + m) * 16 + lane] = cu[m];
                }
                it.barrier(sycl::access::fence_space::local_space);
                // Epilogue: one thread (s==0, lane==0) reduces K, GeGLUs, and
                // packs group g for the 8 rows. Pack needs all 16 GeGLU values
                // per row (max_abs over the group), so a single thread does it
                // rather than the 16-lane parallel write used by the plain GEMM.
                if (s == 0 && lane == 0 && expert >= 0) {
                    float inv_dst = 1.0f / dst_scale_by_expert[expert][0];
                    float down_in_scale = down_input_global_scale[expert];
                    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
                    constexpr float COEF = 0.044715f;
                    for (int m = 0; m < 8; ++m) {
                        int r = r0 + m;
                        if (r >= rows || row_expert[r] != expert) continue;
                        int slot = row_slot[r];
                        float vals[16];
                        float max_abs = 0.0f;
                        for (int i = 0; i < 16; ++i) {
                            float gg = 0.0f, uu = 0.0f;
                            for (int ss = 0; ss < KS; ++ss) {
                                gg += slm_g[(size_t)(ss * 8 + m) * 16 + i];
                                uu += slm_u[(size_t)(ss * 8 + m) * 16 + i];
                            }
                            gg *= inv_dst;
                            uu *= inv_dst;
                            float inner = SQRT_2_OVER_PI * (gg + COEF * gg * gg * gg);
                            // Round-trip through bf16 matches the unfused path
                            // (gu bf16 -> geglu reads bf16).
                            float v = bf16_to_float(float_to_bf16(
                                0.5f * gg * (1.0f + sycl::tanh(inner)) * uu));
                            vals[i] = v;
                            max_abs = sycl::fmax(max_abs, sycl::fabs(v));
                        }
                        uint8_t scale_bits = 0;
                        float scale = 0.0f;
                        if (max_abs > 0.0f) {
                            float raw_scale = max_abs * down_in_scale / 6.0f;
                            scale_bits = nvfp4_encode_e4m3_positive(raw_scale);
                            scale = nvfp4_e4m3_to_float(scale_bits);
                        }
                        act_scale[(size_t)slot * G_inter + g] = scale_bits;
                        uint8_t* packed_row = act_packed + (size_t)slot * (inter / 2);
                        int k0 = ng;
                        for (int i = 0; i < 16; i += 2) {
                            uint8_t lo = 0, hi = 0;
                            if (scale > 0.0f) {
                                float v0 = vals[i] * down_in_scale / scale;
                                float v1 = vals[i + 1] * down_in_scale / scale;
                                lo = nvfp4_encode_e2m1(v0);
                                hi = nvfp4_encode_e2m1(v1);
                            }
                            packed_row[(k0 + i) / 2] = lo | (hi << 4);
                        }
                    }
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
    bf16* C,
    Nvfp4GraphSession* /*session*/ = nullptr)
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
    bf16* C,
    Nvfp4GraphSession* /*session*/ = nullptr)
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
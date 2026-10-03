#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph/command_graph.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Quant-neutral SYCL command-graph session machinery.
//
// Shared by every capture path in the tree (the NVFP4 diffusion layer sessions
// and per-kernel micro-captures, and the INT4 diffusion attention-sub-layer
// captures). Historically this lived in nvfp4.hpp under Nvfp4* names; it is
// independent of any quantization format, so it moved to its own lightweight
// header that does NOT drag in nvfp4.hpp's DPAS/SPIRV intrinsics -- that is
// what lets utils/profile.cpp forward-declare the recording check.
//
// Capture policy pieces shared by all users:
//   * A DiffGraphSession opens ONE graph over a queue: every kernel submitted
//     to that queue between begin() and end_and_replay() becomes a node of
//     the graph (raw q.submit, and oneDNN executes via sycl_interop over the
//     same queue). Finalized graphs are cached under DiffGraphKey and replayed
//     with a single queue.ext_oneapi_graph() dispatch.
//   * Replays re-execute the exact recorded kernel sequence with the exact
//     recorded pointer arguments, so callers must only capture step sequences
//     whose kernel sequence and buffer addresses are step-static (see the
//     diffusion_gemma INT4 attention capture for the arena-coalescing argument
//     and the anchor check that verifies it).
//   * Failure of any capture/finalize marks the cache entry `unavailable` and
//     the caller falls back to eager submission forever (that step's results
//     are still correct: recording enqueues work on the queue as it goes).
// ---------------------------------------------------------------------------

// Scope tag: graph keys from different quant codepaths never collide.
enum class DiffGraphScope : int {
    Nvfp4 = 1,
    Int4 = 2,
};

inline bool diff_graph_verbose() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_SYCL_GRAPH_VERBOSE");
        if (env) return env[0] && std::strcmp(env, "0") != 0;
        // Legacy knob: the nvfp4 path printed under its own env.
        return std::getenv("DIFF_NVFP4_VERBOSE") != nullptr;
    }();
    return enabled;
}

// Cache capacity shared by every capture path. Honors the neutral
// DIFF_SYCL_GRAPH_CACHE_LIMIT first and the historical
// DIFF_NVFP4_SYCL_GRAPH_CACHE_LIMIT second.
inline size_t diff_graph_cache_limit() {
    static size_t limit = [] {
        const char* env = std::getenv("DIFF_SYCL_GRAPH_CACHE_LIMIT");
        if (!env) env = std::getenv("DIFF_NVFP4_SYCL_GRAPH_CACHE_LIMIT");
        if (!env) return size_t{512};
        char* end = nullptr;
        unsigned long parsed = std::strtoul(env, &end, 10);
        return end != env && parsed > 0 ? static_cast<size_t>(parsed) : size_t{512};
    }();
    return limit;
}

// Scope-disable depth: while nonzero, every capture attempt runs eager.
// Structured decision reads use this to bypass capture entirely.
inline int& diff_graph_capture_disable_depth() {
    static thread_local int depth = 0;
    return depth;
}
inline bool diff_graph_capture_disabled() {
    return diff_graph_capture_disable_depth() > 0;
}
struct DiffGraphEagerScope {
    DiffGraphEagerScope() { ++diff_graph_capture_disable_depth(); }
    ~DiffGraphEagerScope() { --diff_graph_capture_disable_depth(); }
    DiffGraphEagerScope(const DiffGraphEagerScope&) = delete;
    DiffGraphEagerScope& operator=(const DiffGraphEagerScope&) = delete;
};

struct DiffGraphKey {
    const sycl::queue* queue = nullptr;
    int scope = 0;              // DiffGraphScope
    int kind = 0;               // scope-local kind space
    std::vector<uintptr_t> args;
    bool operator==(const DiffGraphKey& other) const {
        return queue == other.queue && scope == other.scope &&
               kind == other.kind && args == other.args;
    }
};
struct DiffGraphKeyHash {
    size_t operator()(const DiffGraphKey& key) const {
        size_t hash = std::hash<const sycl::queue*>{}(key.queue);
        auto mix = [&](uintptr_t value) {
            hash ^= std::hash<uintptr_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        };
        mix(static_cast<uintptr_t>(key.scope));
        mix(static_cast<uintptr_t>(key.kind));
        for (uintptr_t value : key.args) mix(value);
        return hash;
    }
};

struct DiffGraphEntry {
    using Modifiable = sycl::ext::oneapi::experimental::command_graph<>;
    using Executable = sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>;
    std::unique_ptr<Modifiable> graph;
    std::unique_ptr<Executable> executable;
    bool unavailable = false;
    // Warm policy: entry created by an eager "warm" invocation; the next
    // invocation with the same key records. See DiffGraphSession::begin.
    bool warmed = false;
    // Optional capture-time pointer recorded by the call site (e.g. the
    // pipeline buffer feeding the captured segment). Replays that pass a
    // different pointer are a stale-address replay bug -> detected, thrown.
    uintptr_t anchor = 0;
};

struct DiffGraphCache {
    std::mutex mutex;
    std::unordered_map<DiffGraphKey, DiffGraphEntry, DiffGraphKeyHash> entries;
    size_t captures = 0;
    size_t replays = 0;
    size_t fallbacks = 0;
    size_t capacity_bypasses = 0;
};
inline DiffGraphCache& diff_graph_cache() {
    static DiffGraphCache cache;
    return cache;
}

// Bit-cast a float into a key argument so kernels parameterized by an fp
// scalar (temperature, scales, ...) are keyed exactly.
inline uintptr_t diff_graph_float_key(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct DiffGraphSession;

// ---------------------------------------------------------------------------
// DiffGraphSession
//
// One command_graph capture scoped to a caller-determined region (for the
// diffusion INT4 path: one decoder layer's attention sub-layer). Repeated
// identical regions are captured once and replayed; a region is identified by
// the caller-provided DiffGraphKey (queue + scope + kind + args), so callers
// must fold every shape/position input that changes the region's kernel
// sequence or pointer arguments into the key.
//
// Usage:
//   DiffGraphSession session;
//   if (session.begin(q, key, graphs_enabled_gate)) {
//       ...submit the region's kernels...      // may instead run warm/eager
//       if (session.active()) session.end_and_replay();
//   } else {
//       // cache hit: begin() already replayed the graph; skip submission.
//   }
//
// Queue-keyed registry of *actively recording* sessions. Per-kernel
// micro-capture helpers (nvfp4_sycl_graph_submit) consult this as a
// backstop: when a session is recording on a queue, ANY nested per-kernel
// begin_recording on that queue would throw and poison the micro-cache, so
// they stand down to a bare submit() -- which the active session's recording
// captures as a node.
inline std::mutex& diff_graph_session_registry_mutex() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<const sycl::queue*, DiffGraphSession*>&
diff_graph_session_registry() {
    static std::unordered_map<const sycl::queue*, DiffGraphSession*> r;
    return r;
}
// Returns the session currently recording on `q`, or nullptr if none. Called
// on hot paths; the map is empty when no session is active, so this is a
// fast miss.
inline DiffGraphSession* diff_graph_active_session(const sycl::queue& q) {
    std::lock_guard<std::mutex> lock(diff_graph_session_registry_mutex());
    auto it = diff_graph_session_registry().find(&q);
    return it == diff_graph_session_registry().end() ? nullptr : it->second;
}
// True iff a DiffGraphSession is currently recording on `q`. Forward-declarable
// (no DiffGraphSession definition needed) so lightweight headers such as
// utils/profile.hpp can call it without including SYCL graph internals.
// Used to skip queue waits during recording (q.wait()/stream.wait() throw on
// a recording queue: "wait cannot be called for a queue which is recording to
// a command graph").
//
// NOTE: declared here, defined out-of-line in common/gpu/sycl_graph_session.cpp.
// It must NOT be `inline`: a TU that only forward-declares it (e.g.
// profile.cpp) emits an external reference; an `inline` definition is only
// emitted as linkonce_odr by TUs that include this header and use it, and if
// all such uses are inlined away there is no standalone symbol to satisfy
// that reference.
bool diff_graph_recording(const sycl::queue& q);

struct DiffGraphSession {
    using Modifiable = DiffGraphEntry::Modifiable;
    using Executable = DiffGraphEntry::Executable;

    sycl::queue* queue_ = nullptr;
    std::unique_ptr<Modifiable> graph_;
    bool recording_ = false;
    // True once begin() has opened a *new* recording that the caller must
    // populate and close with end_and_replay(). False if begin() found a
    // cached executable and already replayed it (nothing left to record).
    bool needs_recording_ = false;
    uintptr_t anchor_ = 0;

    // True while this session is actively recording -- kernels submitted via
    // per-kernel capture helpers check this and simply enqueue directly so
    // they land inside this session's graph rather than starting their own.
    bool active() const { return recording_; }

    // Returns true if the caller must run the region's kernels (eager run,
    // warm run, or a fresh recording); false if a cached graph was found and
    // already replayed (cache hit).
    //
    // graphs_enabled: the caller's env gate (quant-scoped). Checked here so
    //   that gating semantics travel with every call site.
    // warm_steps: number of initial invocations per key that must run eager
    //   before the first capture. First invocations are dirty with lazy
    //   artifacts that must never be captured or waited on inside a
    //   recording (lazy oneDNN layout reorders + stream.wait, primitive/pd
    //   creation, scratch growth, arena chunk growth); the warm pass runs
    //   them eagerly, and the graph records from a steady state. 0 = previous
    //   behavior (capture on first invocation).
    // anchor / check_anchor: optional call-site pointer (e.g. the pipeline
    //   buffer feeding the captured region) recorded with the entry. A later
    //   cache-hit invocation carrying a different pointer throws: replaying
    //   a captured graph with different buffer addresses is silent data
    //   corruption, and this turns it into a loud failure.
    bool begin(sycl::queue& q, const DiffGraphKey& step_key, bool graphs_enabled,
              int warm_steps = 0, uintptr_t anchor = 0, bool check_anchor = false) {
        queue_ = &q;
        anchor_ = anchor;
        if (diff_graph_capture_disabled()) {
            needs_recording_ = false;
            recording_ = false;
            return true;  // per-call eager policy: caller runs kernels eagerly
        }
        if (!graphs_enabled) {
            needs_recording_ = false;
            recording_ = false;
            return true;  // graphs disabled: caller runs kernels eagerly every time
        }
        auto& cache = diff_graph_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto found = cache.entries.find(step_key);
        if (found != cache.entries.end() && found->second.executable) {
            if (check_anchor && found->second.anchor != 0 && anchor != 0 &&
                found->second.anchor != anchor) {
                throw std::runtime_error(
                    "diff-graph replay address drift: captured key replayed "
                    "with a different pipeline buffer (arena handed a "
                    "different address this step) -- graph replay would "
                    "silently corrupt; refusing");
            }
            ++cache.replays;
            q.ext_oneapi_graph(*found->second.executable);
            needs_recording_ = false;
            recording_ = false;
            return false;
        }
        if (found == cache.entries.end() &&
            cache.entries.size() >= diff_graph_cache_limit()) {
            ++cache.capacity_bypasses;
            needs_recording_ = false;
            recording_ = false;
            return true;  // caller runs kernels eagerly, no capture this time
        }
        if (found == cache.entries.end() && warm_steps > 0) {
            // First invocation for this key runs warm (eager): everything
            // lazily initialized on this call -- oneDNN reorders/waits,
            // scratch growth, arena chunk growth -- stays outside the graph.
            // Mark the slot so the next invocation records.
            DiffGraphEntry warm;
            warm.warmed = true;
            cache.entries.emplace(step_key, std::move(warm));
            needs_recording_ = false;
            recording_ = false;
            return true;
        }
        if (found != cache.entries.end() && warm_steps > 0 &&
            !found->second.warmed) {
            found->second.warmed = true;
            needs_recording_ = false;
            recording_ = false;
            return true;
        }
        if (found != cache.entries.end() && found->second.unavailable &&
            warm_steps > 0) {
            // Warm policy + previously failed capture: stay eager for this
            // key instead of retrying a capture that already failed (each
            // failed retry costs a recorded-then-dropped region plus the
            // forced eager re-run). Old (warm_steps==0) semantics retry.
            needs_recording_ = false;
            recording_ = false;
            return true;
        }
        // No executable under this key: either a warm marker (warm policy),
        // or a previously failed capture with old (warm_steps == 0) semantics
        // retrying. Recording starts below.
        key_ = step_key;
        // assume_buffer_outlives_graph: oneDNN's SYCL interop may internally
        // use sycl::buffer for some primitives. SYCL throws "Cannot use
        // buffers in a graph without ... assume_buffer_outlives_graph" if a
        // buffer-accessor is recorded into a graph lacking this property. All
        // persistent data (weights, KV cache, arena workspaces) is USM and
        // outlives the graph; the property is the documented
        // oneDNN-interop capture enabler.
        graph_ = std::make_unique<Modifiable>(q,
            sycl::property_list{
                sycl::ext::oneapi::experimental::property::graph::assume_buffer_outlives_graph{}});
        try {
            graph_->begin_recording(q);
        } catch (const std::exception& error) {
            graph_.reset();
            needs_recording_ = false;
            recording_ = false;
            if (diff_graph_verbose())
                std::fprintf(stderr, "[diff-graph] session capture disabled: %s\n",
                             error.what());
            return true;  // caller runs kernels eagerly this time
        }
        recording_ = true;
        needs_recording_ = true;
        {
            std::lock_guard<std::mutex> lock(diff_graph_session_registry_mutex());
            diff_graph_session_registry()[&q] = this;
        }
        return true;
    }

    // Closes recording, finalizes, caches, and replays the freshly-captured
    // graph. Call exactly once, after the caller has submitted every kernel
    // for the region, only when begin() returned true AND active() is true
    // (i.e. begin() did not already replay a cached hit).
    //
    // DPC++ record-replay semantics (queue_impl.cpp submit_command_to_graph):
    // kernels submitted while recording are stored as graph nodes and are
    // NOT executed; the queue_->ext_oneapi_graph() dispatch below is the
    // region's FIRST and ONLY execution of this step. Legacy form (no
    // callback): if finalize/dispatch throws, the recorded region is dropped
    // and the step silently proceeds WITHOUT this region's work.
    void end_and_replay() {
        if (!recording_) return;  // begin() already replayed a cached graph
        try {
            finalize_and_dispatch();
        } catch (const std::exception& error) {
            finalize_failed(error);
        }
    }

    // Failsafe form: `rerun()` re-submits the region's kernels eagerly if the
    // finalize/dispatch failed, so a dropped recording can never skip the
    // region's work. Returns true if the graph was finalized and dispatched
    // (or recording was already closed), false if the region was re-run by
    // the callback instead.
    template <class Rerun>
    bool end_and_replay(Rerun&& rerun) {
        if (!recording_) return true;
        try {
            finalize_and_dispatch();
            return true;
        } catch (const std::exception& error) {
            finalize_failed(error);
            // Recording has been closed; re-running the caller's region now
            // submits eagerly.
            rerun();
            return false;
        }
    }

private:
    void finalize_and_dispatch() {
        auto& cache = diff_graph_cache();
        if (recording_) {
            // Unregister from the active-session registry now that recording
            // has ended, so per-kernel capture helpers stop standing down.
            graph_->end_recording(*queue_);
            recording_ = false;
            {
                std::lock_guard<std::mutex> lock(diff_graph_session_registry_mutex());
                diff_graph_session_registry().erase(queue_);
            }
        }
        auto executable = std::make_unique<Executable>(graph_->finalize());
        std::lock_guard<std::mutex> lock(cache.mutex);
        DiffGraphEntry entry;
        entry.graph = std::move(graph_);
        entry.executable = std::move(executable);
        entry.warmed = true;
        entry.anchor = anchor_;
        ++cache.captures;
        queue_->ext_oneapi_graph(*entry.executable);
        cache.entries[key_] = std::move(entry);
    }

    void finalize_failed(const std::exception& error) {
        if (recording_) {
            try { graph_->end_recording(*queue_); } catch (...) {}
            recording_ = false;
            std::lock_guard<std::mutex> lock(diff_graph_session_registry_mutex());
            diff_graph_session_registry().erase(queue_);
        }
        auto& cache = diff_graph_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        ++cache.fallbacks;
        DiffGraphEntry entry;
        entry.unavailable = true;
        entry.warmed = true;
        cache.entries[key_] = std::move(entry);
        if (diff_graph_verbose())
            std::fprintf(stderr, "[diff-graph] session finalize failed: %s\n",
                         error.what());
    }

public:

    DiffGraphKey key_;
};

// End-of-run summary of the shared cache. Prints only if anything happened so
// eager baselines stay noise-free.
inline void diff_graph_report() {
    auto& cache = diff_graph_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.captures == 0 && cache.replays == 0 && cache.fallbacks == 0 &&
        cache.capacity_bypasses == 0 && cache.entries.empty())
        return;
    std::fprintf(stderr,
                 "[diff-graph] captures=%zu replays=%zu fallbacks=%zu "
                 "cache-bypasses=%zu entries=%zu\n",
                 cache.captures, cache.replays, cache.fallbacks,
                 cache.capacity_bypasses, cache.entries.size());
}

#pragma once
// Kernel-launch profiler: launch census + event-based per-launch GPU time.
//
// Gated by ARCAINE_LAUNCH_PROFILE (1/on). When off, every record() is a single
// enabled() bool check — no locks, no event queries, no queue changes.
//
// When on:
//   - GpuEngine constructs the SYCL queue with enable_profiling (see
//     runtime/gpu/engine.hpp), so every captured sycl::event carries
//     command_start/command_end timestamps.
//   - Instrumented call sites capture the submit event and call
//     launchprof::record(name, ev). Per-launch GPU time is read from the event
//     WITHOUT waiting on the queue — no pipeline serialization (unlike the
//     wait-based diffprof timers).
//   - oneDNN primitive.execute() returns void, so those sites call
//     record(name) for the census only; their internal timing is visible via
//     DNNL_VERBOSE=2.
//
// report() prints a per-name table sorted by total GPU time. State is
// header-local (inline function statics) — all binaries that link GpuEngine
// share one instance.
#ifndef ARCAINE_LAUNCH_PROF_HPP
#define ARCAINE_LAUNCH_PROF_HPP

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace launchprof {

struct Entry {
    long count = 0;
    long long total_ns = 0;
    long long min_ns = 0;
    long long max_ns = 0;
};

inline bool& enabled_ref() {
    static bool e = [] {
        const char* v = std::getenv("ARCAINE_LAUNCH_PROFILE");
        if (!v) return false;
        return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 &&
               std::strcmp(v, "false") != 0 && std::strcmp(v, "no") != 0;
    }();
    return e;
}
inline bool enabled() { return enabled_ref(); }

// True when GpuEngine built the queue with enable_profiling (only when the env
// is on). record(name, ev) uses this to decide whether the event carries
// timestamps; record(name) call sites never touch events.
inline bool& queue_profiling_ref() {
    static bool q = false;
    return q;
}
inline bool queue_profiling() { return queue_profiling_ref(); }
inline void mark_queue_profiling(bool v) { queue_profiling_ref() = v; }

inline std::map<std::string, Entry>& state() {
    static std::map<std::string, Entry> m;
    return m;
}
inline std::mutex& state_mutex() {
    static std::mutex m;
    return m;
}

inline void record(const char* name) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock(state_mutex());
    auto& e = state()[name];
    e.count += 1;
}

inline void record(const char* name, const sycl::event& ev) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock(state_mutex());
    auto& e = state()[name];
    e.count += 1;
    if (!queue_profiling()) return;
    try {
        long long start = (long long)ev.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        long long end = (long long)ev.get_profiling_info<
            sycl::info::event_profiling::command_end>();
        if (end >= start) {
            long long ns = end - start;
            e.total_ns += ns;
            if (e.min_ns == 0 || ns < e.min_ns) e.min_ns = ns;
            if (ns > e.max_ns) e.max_ns = ns;
        }
    } catch (...) {
        // Non-profiling event (e.g. some backends return no timestamps for
        // D2D memcpy): keep the census count, skip the timing.
    }
}

inline void reset() {
    std::lock_guard<std::mutex> lock(state_mutex());
    state().clear();
}

inline long total_launches() {
    std::lock_guard<std::mutex> lock(state_mutex());
    long n = 0;
    for (auto& kv : state()) n += kv.second.count;
    return n;
}

inline void report() {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock(state_mutex());
    auto& m = state();
    if (m.empty()) {
        std::fprintf(stderr, "[launchprof] no launches recorded\n");
        return;
    }
    long long total_ns = 0;
    long total = 0;
    for (auto& kv : m) {
        total_ns += kv.second.total_ns;
        total += kv.second.count;
    }
    std::vector<std::pair<std::string, Entry>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_ns > b.second.total_ns;
              });
    std::fprintf(stderr,
                 "\n[launchprof] ARCAINE_LAUNCH_PROFILE report: %ld launches, "
                 "%.3f ms GPU time (profiled events only)\n",
                 total, (double)total_ns * 1e-6);
    std::fprintf(stderr, "%-42s %8s %10s %9s %7s %7s\n", "kernel", "count",
                 "total_ms", "avg_us", "%time", "%cnt");
    for (auto& kv : v) {
        const Entry& e = kv.second;
        double ms = (double)e.total_ns * 1e-6;
        double avg = e.count ? (double)e.total_ns * 1e-3 / e.count : 0.0;
        double pt = total_ns ? 100.0 * e.total_ns / total_ns : 0.0;
        double pc = total ? 100.0 * e.count / total : 0.0;
        std::fprintf(stderr, "%-42s %8ld %10.3f %9.2f %6.1f%% %6.1f%%\n",
                     kv.first.c_str(), e.count, ms, avg, pt, pc);
    }
    std::fflush(stderr);
}

}  // namespace launchprof

#endif  // ARCAINE_LAUNCH_PROF_HPP

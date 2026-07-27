#pragma once
#include <sycl/sycl.hpp>
#include <chrono>

// Lightweight forward-pass phase profiler, gated by DIFF_PROFILE=1. A scoped
// timer waits the queue at entry and exit (so it measures actual GPU work for
// the enqueued ops) and accumulates wall time per named phase across the whole
// generation. Because each scope waits, this serializes the pipeline and
// removes cross-GPU/async overlap — totals are inflated vs a real run, but the
// per-phase *breakdown* (where the time goes) is the signal. report() prints a
// sorted table; call reset() before a run and report() after.
namespace diffprof {
bool enabled();
void add(const char* name, double seconds);
void reset();
void report();

// Manual span timing for sections that can't be wrapped in a scope (e.g. that
// create buffers used later). tic() waits the queue and returns the start;
// toc() waits and accumulates. No-ops (no waits) when DIFF_PROFILE is off.
std::chrono::steady_clock::time_point tic(sycl::queue& q);
void toc(sycl::queue& q, const char* name, std::chrono::steady_clock::time_point t0);

struct ScopedGpu {
    sycl::queue* q;
    const char* name;
    std::chrono::steady_clock::time_point t0;
    bool on;
    ScopedGpu(sycl::queue& queue, const char* nm);
    ~ScopedGpu();
};
}  // namespace diffprof

#define DIFFPROF_CONCAT2(a, b) a##b
#define DIFFPROF_CONCAT(a, b) DIFFPROF_CONCAT2(a, b)
// Time the enclosing scope under DIFF_PROFILE; no-op cost otherwise (just the
// enabled() bool check, no waits).
#define DIFF_PROF(q, name) diffprof::ScopedGpu DIFFPROF_CONCAT(_dprof_, __LINE__)(q, name)

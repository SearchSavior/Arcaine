#pragma once
#include <cstddef>

// Lightweight accessor for the shared SYCL graph cache counters. Declared here
// (without any SYCL graph internals) so HTTP/diagnostic code can observe
// capture/replay activity -- in particular to verify that structured decision
// reads bypass graph capture even when a capture gate is enabled.
// Implemented out-of-line in common/gpu/sycl_graph_session.cpp.
struct DiffGraphCounts {
    size_t captures = 0;          // graphs finalized and cached
    size_t replays = 0;           // cached-graph dispatches
    size_t fallbacks = 0;         // capture failed / entry unavailable
    size_t capacity_bypasses = 0; // cache full -> eager submit
};

DiffGraphCounts diff_graph_capture_counts();

#pragma once
#include <cstddef>

// Lightweight accessor for the NVFP4 SYCL graph cache counters.  Declared here
// (without nvfp4.hpp's DPAS intrinsics) so HTTP/diagnostic code can observe
// capture/replay activity -- in particular to verify that structured decision
// reads bypass graph capture even when DIFF_NVFP4_SYCL_GRAPH is enabled.
// Implemented out-of-line in common/gpu/nvfp4_session.cpp.
struct Nvfp4GraphCounts {
    size_t captures = 0;          // graphs finalized and cached
    size_t replays = 0;           // cached-graph dispatches
    size_t fallbacks = 0;         // capture failed / entry unavailable
    size_t capacity_bypasses = 0; // cache full -> eager submit
};

Nvfp4GraphCounts nvfp4_graph_capture_counts();

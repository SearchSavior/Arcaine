// Out-of-line definitions for the shared SYCL-graph session machinery
// (declared in common/gpu/sycl_graph_session.hpp).
//
// diff_graph_recording is deliberately NON-inline and lives in its own TU so
// that every binary that may call it links exactly one external definition:
//   - utils/profile.cpp forward-declares it (avoids pulling the SYCL graph
//     internals into the lightweight profile header) and references it from
//     tic/toc/ScopedGpu to skip q.wait() while a command_graph is recording;
//   - src/common/gpu/expert_parallel.cpp references it to make run_shard's
//     trailing q.wait() / owner_q.wait() session-conditional;
//   - matmul_nvfp4 (qwen3_5_moe/model.cpp, shared with the diffusion dense-MLP)
//     references it to make its ctx.stream.wait() session-conditional.
// An `inline` definition would only be emitted as linkonce_odr by TUs that
// include sycl_graph_session.hpp and USE the function; if all such uses are
// inlined away (as happens for the LLM targets, which never record a graph),
// no standalone symbol exists to satisfy a forward-decl-only TU's external
// reference.
//
// This TU includes only the session header + engine helpers: no DPAS/SPIRV
// intrinsics, so it needs no -Xspirv-translator spirv-ext option of its own;
// the device link of any target it joins already requires that option for
// other TUs.
#include "sycl_graph_session.hpp"
#include "diff_graph_stats.hpp"

bool diff_graph_recording(const sycl::queue& q) {
    return diff_graph_active_session(q) != nullptr;
}

DiffGraphCounts diff_graph_capture_counts() {
    auto& cache = diff_graph_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    DiffGraphCounts out;
    out.captures = cache.captures;
    out.replays = cache.replays;
    out.fallbacks = cache.fallbacks;
    out.capacity_bypasses = cache.capacity_bypasses;
    return out;
}

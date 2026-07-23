// Out-of-line definition of nvfp4_session_recording (declared in nvfp4.hpp).
//
// This function is deliberately NON-inline and lives in its own TU so that
// every binary that may call it links exactly one external definition:
//   - utils/profile.cpp forward-declares it (avoids pulling nvfp4.hpp's DPAS
//     intrinsics into the lightweight profile header) and references it from
//     tic/toc/ScopedGpu to skip q.wait() while a command_graph is recording;
//   - src/common/gpu/expert_parallel.cpp references it to make run_shard's
//     trailing q.wait() / owner_q.wait() session-conditional;
//   - matmul_nvfp4 (qwen3_5_moe/model.cpp, shared with the diffusion dense-MLP)
//     references it to make its ctx.stream.wait() session-conditional.
// An `inline` definition would only be emitted as linkonce_odr by TUs that
// include nvfp4.hpp and USE the function; if all such uses are inlined away
// (as happens for the LLM targets, which never record a graph), no standalone
// symbol exists to satisfy a forward-decl-only TU's external reference.
//
// This TU includes nvfp4.hpp, so it sees the inline nvfp4_active_session and
// the nvfp4_session_registry helpers. It touches no DPAS intrinsics itself,
// so it needs no -Xspirv-translator spirv-ext option of its own; the device
// link of any target it joins already requires that option for other TUs.
#include "nvfp4.hpp"

bool nvfp4_session_recording(const sycl::queue& q) {
    return nvfp4_active_session(q) != nullptr;
}

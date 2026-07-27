#pragma once

#include <cstdint>
#include <vector>

namespace arcaine::qwen3_5_moe {

// Private, model-specific resolved request. Never leaves this module.
// Qwen3.5-MoE is text-only: no image/audio fields.
struct QwenMoeInvocation {
    std::vector<int> input_ids;

    float temperature = 1.0f;
    int   top_k        = 64;
    float top_p        = 0.95f;

    int           max_output_tokens = 0;
    std::uint64_t seed = 0;

    bool has_tools     = false;
    bool stream        = false;
    bool include_usage = false;
};

}  // namespace arcaine::qwen3_5_moe

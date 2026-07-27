#pragma once

#include <cstdint>
#include <vector>

#include "inference/contracts/media_input.hpp"

namespace arcaine::gemma4_unified {

// Private, model-specific resolved request. Never leaves this module.
struct Gemma4Invocation {
    std::vector<int>         input_ids;
    std::vector<inference::ImageInput> images;
    std::vector<inference::AudioInput> audio;
    std::vector<int32_t>    mm_token_type_ids;

    // Resolved sampler options (defaults already applied from config).
    float                   temperature = 1.0f;
    int                     top_k       = 64;
    float                   top_p       = 0.95f;

    int                     max_output_tokens = 0;
    std::uint64_t           seed = 0;

    bool                    has_tools = false;
    bool                    stream    = false;
    bool                    include_usage = false;
};

}  // namespace arcaine::gemma4_unified

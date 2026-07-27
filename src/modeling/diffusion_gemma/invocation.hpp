#pragma once

#include <cstdint>
#include <vector>

namespace arcaine::diffusion_gemma {

// Private, model-specific resolved request. Never leaves this module.
// DiffusionGemma is text-only (block-diffusion): no image/audio fields.
struct DiffusionGemmaInvocation {
    std::vector<int> encoder_ids;

    int  output_length     = 0;   // max new tokens
    int  denoising_steps   = -1;  // -1 = use model config default
    std::uint64_t seed     = 0;
    bool stream_drafts     = false;  // emit per-step draft events

    bool has_tools     = false;
    bool stream        = false;
    bool include_usage = false;
};

}  // namespace arcaine::diffusion_gemma

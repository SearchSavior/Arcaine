#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "inference/contracts/capabilities.hpp"

namespace arcaine::inference {

// Describes a registered model implementation. The registry matches a config
// `model_type` string against `model_types`; applications use the capability
// masks only for validation/listing, never for dispatching generation.
struct ModelDescriptor {
    // config.json `model_type` strings this implementation answers to.
    // A module may live in `qwen3_5_moe/` but register the `qwen3_5_moe_text`
    // model_type, so model_types is a vector.
    std::vector<std::string> model_types;

    // Stable, unique id of this implementation (e.g. "qwen3_5_moe").
    std::string implementation_id;

    InferenceContract inference_contract = InferenceContract::CausalGenerationV1;

    std::uint32_t input_capabilities  = 0;
    std::uint32_t output_capabilities = 0;

    bool supports_streaming = false;
};

}  // namespace arcaine::inference

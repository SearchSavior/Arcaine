#pragma once

#include <cstdint>

namespace arcaine::inference {

// Public inference contract a model implementation commits to. The server and
// registry use this only to *describe* a model (e.g. for capability checks and
// listings); they never switch on it to choose a generation algorithm.
enum class InferenceContract : std::uint8_t {
    // Causal (autoregressive) token generation: prefill -> decode loop -> EOS.
    CausalGenerationV1,
    // Block-diffusion generation: encoder + denoising loop + draft/commit.
    BlockDiffusionGenerationV1,
};

// Bitmask of input forms a model accepts. Models may set several.
enum class InputCapability : std::uint32_t {
    None       = 0,
    PromptText = 1u << 0,
    Messages   = 1u << 1,
    TokenIds   = 1u << 2,
    Images     = 1u << 3,
    Audio      = 1u << 4,
    Tools      = 1u << 5,
};

inline InputCapability operator|(InputCapability a, InputCapability b) {
    return static_cast<InputCapability>(
        static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

inline std::uint32_t operator|(std::uint32_t a, InputCapability b) {
    return a | static_cast<std::uint32_t>(b);
}

inline bool has_input_capability(std::uint32_t mask, InputCapability cap) {
    return (mask & static_cast<std::uint32_t>(cap)) != 0;
}

// Bitmask of output forms a model can produce. Models may set several.
enum class OutputCapability : std::uint32_t {
    None        = 0,
    TokenIds    = 1u << 0,
    Text        = 1u << 1,
    TextDeltas  = 1u << 2,
    ToolCalls   = 1u << 3,
    DraftTokens = 1u << 4,
};

inline OutputCapability operator|(OutputCapability a, OutputCapability b) {
    return static_cast<OutputCapability>(
        static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

inline std::uint32_t operator|(std::uint32_t a, OutputCapability b) {
    return a | static_cast<std::uint32_t>(b);
}

inline bool has_output_capability(std::uint32_t mask, OutputCapability cap) {
    return (mask & static_cast<std::uint32_t>(cap)) != 0;
}

}  // namespace arcaine::inference

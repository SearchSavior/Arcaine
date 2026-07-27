#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "inference/contracts/generation_metrics.hpp"

namespace arcaine::inference {

// Typed generation event emitted by a model session through GenerationSink.
// The transport layer (SSE encoder) pattern-matches on the variant; it never
// inspects model-internal state.

struct StartedEvent {};

struct TokenEvent {
    int token_id = 0;
};

struct TextDeltaEvent {
    std::string delta;
};

struct ToolCallDeltaEvent {
    int         index           = 0;   // position within the tool_calls array
    std::string id;
    std::string name;
    std::string arguments_delta;
};

// Diffusion draft / denoising-step event. Carries current DiffusionGemma draft
// information: block, step, temperature, mean entropy, and either draft token
// ids or draft text.
struct DraftEvent {
    int                  block          = 0;
    int                  step           = 0;
    float                temperature    = 0.0f;
    float                mean_entropy   = 0.0f;
    std::vector<int>     draft_token_ids;
    std::string          draft_text;
};

struct MetricsEvent {
    GenerationMetrics metrics;
};

struct CompletedEvent {
    std::string finish_reason;
    // Snapshot of usage for the final transport chunk; the authoritative
    // GenerationResult is the return value of InferenceSession::generate().
    int prompt_tokens   = 0;
    int completion_tokens = 0;
};

using GenerationEvent = std::variant<
    StartedEvent,
    TokenEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    DraftEvent,
    MetricsEvent,
    CompletedEvent
>;

}  // namespace arcaine::inference

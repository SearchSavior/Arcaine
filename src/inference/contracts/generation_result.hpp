#pragma once

#include <string>
#include <vector>

#include "inference/contracts/generation_metrics.hpp"

namespace arcaine::inference {

// A parsed model-native tool call. The model's output_parser produces these
// from the model's native tool-call syntax; the transport layer only encodes
// them.
struct ParsedToolCall {
    std::string id;         // OpenAI tool_call id (generated if absent)
    std::string name;
    std::string arguments;  // JSON string
};

struct Usage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
};

// Final result of one generate() call.
struct GenerationResult {
    std::vector<int>             output_token_ids;
    std::string                  text;           // public, channel-stripped text
    std::vector<ParsedToolCall>  tool_calls;
    std::string                  finish_reason;  // "stop" | "length" | "tool_calls"
    Usage                        usage;
    GenerationMetrics            metrics;
};

}  // namespace arcaine::inference

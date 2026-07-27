#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace arcaine::inference {

// Normalized OpenAI-style message. The request decoder produces a list of
// these from the raw HTTP `messages` array; a model's request mapper renders
// the model chat template from them. Tool-related fields are carried so the
// chat template can render tool calls / tool responses.
struct MessageInput {
    std::string role;                 // "system" | "user" | "assistant" | "tool"
    std::string content;              // text content (may be empty for tool calls)
    // assistant tool_calls (when role == "assistant")
    std::string tool_calls_json;      // serialized JSON array, if any
    // tool response (when role == "tool")
    std::string tool_call_id;
    std::string name;                 // optional tool name
};

}  // namespace arcaine::inference

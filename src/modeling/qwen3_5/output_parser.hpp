#pragma once

#include <string>
#include <vector>

#include "inference/contracts/generation_result.hpp"

namespace arcaine::qwen3_5 {

struct Qwen35AssistantOutput {
    std::string                                      content;
    std::vector<arcaine::inference::ParsedToolCall>  tool_calls;
};

// Qwen3.5 model-native output parsing. The Gemma-family tool-call syntax and
// boundary accounting do not apply to Qwen3.5 (different tokenizer + chat
// template), so this parser is intentionally distinct from the Gemma
// output_parser. The full Qwen3.5-native tool-call parsing is deferred; for
// now this is a pass-through that returns the decoded text as public content
// with no parsed tool calls, matching the prior transport behavior for this
// model (no structured tool extraction).
Qwen35AssistantOutput parse_assistant_output(const std::string& raw_text);

}  // namespace arcaine::qwen3_5

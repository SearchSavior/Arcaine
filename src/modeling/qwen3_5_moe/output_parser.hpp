#pragma once

#include <string>
#include <vector>

#include "inference/contracts/generation_result.hpp"

namespace arcaine::qwen3_5_moe {

struct QwenMoeAssistantOutput {
    std::string                                      content;
    std::vector<arcaine::inference::ParsedToolCall>  tool_calls;
};

// Qwen3.5-MoE model-native output parsing. Distinct from the Gemma-family
// parser (different tokenizer + chat template). The full MoE-native tool-call
// parsing is deferred; for now this is a pass-through that returns the decoded
// text as public content with no parsed tool calls, matching the prior
// transport behavior for this model.
QwenMoeAssistantOutput parse_assistant_output(const std::string& raw_text);

}  // namespace arcaine::qwen3_5_moe

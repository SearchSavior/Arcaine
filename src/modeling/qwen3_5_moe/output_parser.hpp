#pragma once

#include <string>
#include <variant>
#include <vector>

#include "inference/contracts/generation_result.hpp"

namespace arcaine::qwen3_5_moe {

struct QwenMoeAssistantOutput {
    std::string                                      content;
    std::vector<arcaine::inference::ParsedToolCall>  tool_calls;
};

// Qwen3.5-MoE model-native output parsing. The chat template renders tool
// calls as XML-ish blocks (see chat_template.jinja):
//
//   <tool_call>
//   <function=NAME>
//   <parameter=KEY>
//   VALUE
//   </parameter>
//   ...
//   </function>
//   </tool_call>
//
// Multiple blocks are allowed; natural-language content may precede or follow
// them. Parameter values that are not plain strings are rendered as raw JSON
// by the template, so each value is json::parse'd with a string fallback.
QwenMoeAssistantOutput parse_assistant_output(const std::string& raw_text);

// Incremental stream parser for the same format. Feed per-token decoded text;
// content is emitted as it becomes safe (a potential partial "<tool_call>"
// prefix at the tail is held back), and each completed tool-call block yields
// exactly one tool-call output carrying the full arguments JSON.
class QwenMoeStreamParser {
public:
    struct TextDelta { std::string text; };
    struct ToolCall {
        int         index = 0;
        std::string id;
        std::string name;
        std::string arguments;  // JSON object string
    };
    using Output = std::variant<TextDelta, ToolCall>;

    std::vector<Output> feed(const std::string& text);
    // Emit anything still held back (truncated tail). Call once at end of
    // generation.
    std::vector<Output> flush();

private:
    bool        in_tool_call_ = false;
    std::string pending_;      // held-back content suffix / content remainder
    std::string tool_buf_;     // accumulates one tool-call block's body
    int         tool_index_ = 0;
};

}  // namespace arcaine::qwen3_5_moe

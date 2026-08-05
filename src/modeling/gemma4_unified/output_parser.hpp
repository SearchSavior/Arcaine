#pragma once

#include <string>
#include <variant>
#include <vector>

#include "inference/contracts/generation_result.hpp"

namespace arcaine::gemma4_unified {

// Gemma4 token-boundary accounting (thought vs response channels).
struct Gemma4BoundaryCounts {
    int reasoning_tokens = 0;
    int response_tokens  = 0;
};

// Parsed Gemma4 assistant output: public, channel-stripped text + tool calls
// (Gemma-native <|tool_call>{...}<tool_call|> syntax).
struct Gemma4AssistantOutput {
    std::string                                  content;
    std::vector<inference::ParsedToolCall>       tool_calls;
};

Gemma4AssistantOutput parse_assistant_output(const std::string& raw_text);

// Incremental stream parser for the Gemma-native format. Feed per-token
// decoded text; content is emitted as it becomes safe (a potential partial
// "<|tool_call>" marker at the tail is held back), and each completed
// <|tool_call>...<tool_call|> block yields one tool-call output carrying the
// full arguments JSON. Content deltas use the same channel-stripped
// prefix-diff semantics as the final parse.
class Gemma4StreamParser {
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
    std::string pending_;      // held-back content suffix (partial marker)
    std::string tool_buf_;     // accumulates one tool-call block's body
    int         tool_index_ = 0;
    std::string content_raw_;  // all content-region text seen so far
    std::string emitted_;      // stripped content already emitted
};

class Gemma4BoundaryParser {
public:
    explicit Gemma4BoundaryParser(const std::string& model_dir);
    Gemma4BoundaryCounts count(const std::vector<int>& token_ids) const;

private:
    std::vector<int>                 thought_start_ids_;
    std::vector<std::vector<int>>    response_channel_start_ids_;
    std::vector<int>                 ignored_response_ids_;
    int                              channel_start_id_ = -1;
    int                              thought_end_id_   = -1;

    bool has_thought_markers() const;
    bool matches_at(const std::vector<int>& token_ids, size_t pos,
                    const std::vector<int>& pattern) const;
    const std::vector<int>* matching_response_channel_at(
        const std::vector<int>& token_ids, size_t pos) const;
    bool is_ignored_response_id(int token_id) const;
};

}  // namespace arcaine::gemma4_unified

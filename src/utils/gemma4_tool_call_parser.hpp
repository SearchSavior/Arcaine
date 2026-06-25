#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

struct ParsedAssistantOutput {
    std::string content;
    std::vector<ParsedToolCall> tool_calls;
};

ParsedAssistantOutput parse_assistant_output(const std::string& raw_text);

struct Gemma4TokenBoundaryCounts {
    int reasoning_tokens = 0;
    int response_tokens = 0;
};

class Gemma4TokenBoundaryParser {
public:
    explicit Gemma4TokenBoundaryParser(const std::string& model_dir);

    Gemma4TokenBoundaryCounts count(const std::vector<int>& token_ids) const;

private:
    std::vector<int> thought_start_ids_;
    std::vector<std::vector<int>> response_channel_start_ids_;
    std::vector<int> ignored_response_ids_;
    int channel_start_id_ = -1;
    int thought_end_id_ = -1;

    bool has_thought_markers() const;
    bool matches_at(const std::vector<int>& token_ids, size_t pos,
                    const std::vector<int>& pattern) const;
    const std::vector<int>* matching_response_channel_at(
        const std::vector<int>& token_ids, size_t pos) const;
    bool is_ignored_response_id(int token_id) const;
};

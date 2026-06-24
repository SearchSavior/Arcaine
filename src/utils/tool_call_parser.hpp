#pragma once

#include <string>
#include <vector>

struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

struct ParsedAssistantOutput {
    std::string content;
    std::string reasoning_content;
    std::vector<ParsedToolCall> tool_calls;
};

ParsedAssistantOutput parse_assistant_output(const std::string& raw_text);

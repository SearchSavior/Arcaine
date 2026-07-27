#include "modeling/qwen3_5_moe/output_parser.hpp"

#include <algorithm>
#include <cctype>

namespace arcaine::qwen3_5_moe {
namespace {
std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) { return std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [&](unsigned char c) { return !is_ws(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [&](unsigned char c) { return !is_ws(c); }).base(), s.end());
    return s;
}
}  // namespace

// Pass-through: public content is the decoded text (trimmed), no tool calls.
// The Qwen3.5-MoE-native tool-call parser is deferred (see output_parser.hpp).
QwenMoeAssistantOutput parse_assistant_output(const std::string& raw_text) {
    QwenMoeAssistantOutput out;
    out.content = trim_copy(raw_text);
    return out;
}

}  // namespace arcaine::qwen3_5_moe

#include "modeling/qwen3_5/output_parser.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

namespace arcaine::qwen3_5 {
namespace {

using json = nlohmann::ordered_json;

const std::string kToolCallOpen  = "<tool_call>";
const std::string kToolCallClose = "</tool_call>";

std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) { return std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [&](unsigned char c) { return !is_ws(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [&](unsigned char c) { return !is_ws(c); }).base(), s.end());
    return s;
}

json parameter_value(const std::string& raw) {
    try {
        return json::parse(raw);
    } catch (const std::exception&) {
        return json(raw);
    }
}

// Parse one tool-call block body (text between <tool_call> and </tool_call>):
//   <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>
// Best-effort: a block without a function name yields no call; missing
// terminators (truncated generation) are tolerated.
std::optional<arcaine::inference::ParsedToolCall>
parse_tool_call_block(const std::string& body, int index) {
    const std::string fn_open = "<function=";
    size_t fn = body.find(fn_open);
    if (fn == std::string::npos) return std::nullopt;
    size_t name_begin = fn + fn_open.size();
    size_t name_end = body.find('>', name_begin);
    if (name_end == std::string::npos) return std::nullopt;
    std::string name = trim_copy(body.substr(name_begin, name_end - name_begin));
    if (name.empty()) return std::nullopt;

    json arguments = json::object();
    const std::string p_open  = "<parameter=";
    const std::string p_close = "</parameter>";
    size_t pos = name_end;
    while (true) {
        size_t p = body.find(p_open, pos);
        if (p == std::string::npos) break;
        size_t key_begin = p + p_open.size();
        size_t key_end = body.find('>', key_begin);
        if (key_end == std::string::npos) break;
        std::string key = trim_copy(body.substr(key_begin, key_end - key_begin));
        size_t value_begin = key_end + 1;
        if (value_begin < body.size() && body[value_begin] == '\n') ++value_begin;
        size_t value_end = body.find(p_close, value_begin);
        bool truncated = (value_end == std::string::npos);
        if (truncated) value_end = body.size();
        std::string value = body.substr(value_begin, value_end - value_begin);
        if (!value.empty() && value.back() == '\n') value.pop_back();
        if (!key.empty()) arguments[key] = parameter_value(value);
        if (truncated) break;
        pos = value_end + p_close.size();
    }

    arcaine::inference::ParsedToolCall call;
    call.id        = "call_" + std::to_string(index);
    call.name      = std::move(name);
    call.arguments = arguments.dump();
    return call;
}

// Longest suffix of `s` that is a proper prefix of `marker` (possibly a
// partial marker at the tail of the stream).
size_t marker_prefix_suffix_len(const std::string& s, const std::string& marker) {
    size_t max = std::min(s.size(), marker.size() - 1);
    for (size_t len = max; len > 0; --len)
        if (s.compare(s.size() - len, len, marker, 0, len) == 0) return len;
    return 0;
}

}  // namespace

Qwen35AssistantOutput parse_assistant_output(const std::string& raw_text) {
    Qwen35AssistantOutput out;
    std::string content;
    size_t pos = 0;
    int index = 0;
    while (true) {
        size_t open = raw_text.find(kToolCallOpen, pos);
        if (open == std::string::npos) {
            content += raw_text.substr(pos);
            break;
        }
        content += raw_text.substr(pos, open - pos);
        size_t body_begin = open + kToolCallOpen.size();
        size_t close = raw_text.find(kToolCallClose, body_begin);
        size_t body_end = (close == std::string::npos) ? raw_text.size() : close;
        if (auto call = parse_tool_call_block(
                raw_text.substr(body_begin, body_end - body_begin), index)) {
            out.tool_calls.push_back(std::move(*call));
            ++index;
        } else {
            // Not a usable tool call: keep the raw text as content.
            content += raw_text.substr(open, body_end - open);
        }
        if (close == std::string::npos) break;
        pos = close + kToolCallClose.size();
    }
    out.content = trim_copy(std::move(content));
    return out;
}

std::vector<Qwen35StreamParser::Output>
Qwen35StreamParser::feed(const std::string& text) {
    std::vector<Output> out;
    std::string rest = text;

    // Loop: one feed may cross several content/tool-call boundaries.
    while (true) {
        if (!in_tool_call_) {
            // Content mode: emit everything that cannot be part of a
            // "<tool_call>" marker; hold back a trailing partial-marker suffix.
            pending_ += rest;
            rest.clear();
            size_t open = pending_.find(kToolCallOpen);
            if (open == std::string::npos) {
                size_t hold = marker_prefix_suffix_len(pending_, kToolCallOpen);
                size_t safe = pending_.size() - hold;
                if (safe > 0) out.push_back(TextDelta{pending_.substr(0, safe)});
                pending_.erase(0, safe);
                break;
            }
            if (open > 0) out.push_back(TextDelta{pending_.substr(0, open)});
            tool_buf_ += pending_.substr(open + kToolCallOpen.size());
            pending_.clear();
            in_tool_call_ = true;
        }

        // Tool-call mode: accumulate until the block closes, emit one
        // tool-call output, then return to content mode with the remainder.
        tool_buf_ += rest;
        rest.clear();
        size_t close = tool_buf_.find(kToolCallClose);
        if (close == std::string::npos) break;
        if (auto call = parse_tool_call_block(tool_buf_.substr(0, close),
                                              tool_index_)) {
            out.push_back(ToolCall{tool_index_, std::move(call->id),
                                   std::move(call->name),
                                   std::move(call->arguments)});
            ++tool_index_;
        } else {
            out.push_back(TextDelta{kToolCallOpen + tool_buf_.substr(0, close) +
                                    kToolCallClose});
        }
        rest = tool_buf_.substr(close + kToolCallClose.size());
        tool_buf_.clear();
        in_tool_call_ = false;
        if (rest.empty()) break;
    }
    return out;
}

std::vector<Qwen35StreamParser::Output> Qwen35StreamParser::flush() {
    std::vector<Output> out;
    if (in_tool_call_) {
        // Truncated tool call (hit max_tokens mid-block): best-effort parse.
        if (auto call = parse_tool_call_block(tool_buf_, tool_index_)) {
            out.push_back(ToolCall{tool_index_, std::move(call->id),
                                   std::move(call->name),
                                   std::move(call->arguments)});
        } else if (!tool_buf_.empty()) {
            out.push_back(TextDelta{kToolCallOpen + tool_buf_});
        }
        tool_buf_.clear();
        in_tool_call_ = false;
    }
    if (!pending_.empty()) {
        out.push_back(TextDelta{std::move(pending_)});
        pending_.clear();
    }
    return out;
}

}  // namespace arcaine::qwen3_5

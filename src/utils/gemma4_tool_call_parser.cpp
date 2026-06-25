#include "gemma4_tool_call_parser.hpp"

#include "../common/preprocess/tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

using json = nlohmann::ordered_json;

namespace {

struct Gemma4BoundaryTokenStrings {
    std::string bos = "<bos>";
    std::string eos = "<eos>";
    std::string eot = "<turn|>";
    std::string soc = "<|channel>";
    std::string eoc = "<channel|>";
    std::string tool_response_start = "<|tool_response>";
    std::string tool_response_end = "<tool_response|>";
};

std::string string_value(const json& j, const char* key, const std::string& fallback) {
    return j.contains(key) && j.at(key).is_string()
        ? j.at(key).get<std::string>()
        : fallback;
}

Gemma4BoundaryTokenStrings load_boundary_token_strings(const std::string& model_dir) {
    Gemma4BoundaryTokenStrings out;
    std::ifstream f(model_dir + "/tokenizer_config.json");
    if (!f) return out;

    json config = json::parse(f);
    out.bos = string_value(config, "bos_token", out.bos);
    out.eos = string_value(config, "eos_token", out.eos);
    out.eot = string_value(config, "eot_token", out.eot);
    out.soc = string_value(config, "soc_token", out.soc);
    out.eoc = string_value(config, "eoc_token", out.eoc);

    if (config.contains("model_specific_special_tokens") &&
        config.at("model_specific_special_tokens").is_object()) {
        const auto& special = config.at("model_specific_special_tokens");
        out.eot = string_value(special, "eot_token", out.eot);
        out.soc = string_value(special, "soc_token", out.soc);
        out.eoc = string_value(special, "eoc_token", out.eoc);
        out.tool_response_start = string_value(special, "str_token", out.tool_response_start);
        out.tool_response_end = string_value(special, "etr_token", out.tool_response_end);
    }

    out.tool_response_start = string_value(config, "str_token", out.tool_response_start);
    return out;
}

void add_if_present(const Tokenizer& tokenizer, const std::string& token,
                    std::vector<int>& token_ids) {
    if (tokenizer.has_token(token))
        token_ids.push_back(tokenizer.token_id(token));
}

void add_channel_pattern(const Tokenizer& tokenizer, const std::string& channel_token,
                         const std::string& label,
                         std::vector<std::vector<int>>& patterns) {
    std::vector<int> ids = tokenizer.encode(channel_token + label + "\n",
                                            /*add_bos=*/false);
    if (!ids.empty())
        patterns.push_back(std::move(ids));
}

std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c) { return std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](unsigned char c) { return !is_ws(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](unsigned char c) { return !is_ws(c); }).base(), s.end());
    return s;
}

class GemmaToolArgsParser {
public:
    explicit GemmaToolArgsParser(std::string s) : s_(std::move(s)) {}

    json parse() {
        skip_ws();
        json value = parse_value();
        skip_ws();
        if (pos_ != s_.size()) throw std::runtime_error("trailing characters in tool arguments");
        return value;
    }

private:
    void skip_ws() {
        while (pos_ < s_.size() && std::isspace((unsigned char)s_[pos_])) ++pos_;
    }

    bool consume(char c) {
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) throw std::runtime_error(std::string("expected '") + c + "'");
    }

    std::string parse_key() {
        skip_ws();
        if (starts_with("<|\"|>")) return parse_string();
        size_t start = pos_;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.') ++pos_;
            else break;
        }
        if (start == pos_) throw std::runtime_error("expected object key");
        return s_.substr(start, pos_ - start);
    }

    std::string parse_string() {
        const std::string delim = "<|\"|>";
        if (!starts_with(delim)) throw std::runtime_error("expected tokenizer string delimiter");
        pos_ += delim.size();
        size_t end = s_.find(delim, pos_);
        if (end == std::string::npos) throw std::runtime_error("unterminated tokenizer string");
        std::string out = s_.substr(pos_, end - pos_);
        pos_ = end + delim.size();
        return out;
    }

    json parse_object() {
        expect('{');
        json out = json::object();
        skip_ws();
        if (consume('}')) return out;
        while (true) {
            std::string key = parse_key();
            expect(':');
            out[key] = parse_value();
            if (consume('}')) break;
            expect(',');
        }
        return out;
    }

    json parse_array() {
        expect('[');
        json out = json::array();
        skip_ws();
        if (consume(']')) return out;
        while (true) {
            out.push_back(parse_value());
            if (consume(']')) break;
            expect(',');
        }
        return out;
    }

    json parse_atom() {
        size_t start = pos_;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ',' || c == '}' || c == ']' || std::isspace((unsigned char)c)) break;
            ++pos_;
        }
        if (start == pos_) throw std::runtime_error("expected value");
        std::string atom = s_.substr(start, pos_ - start);
        if (atom == "true") return true;
        if (atom == "false") return false;
        if (atom == "null") return nullptr;
        char* end = nullptr;
        long long i = std::strtoll(atom.c_str(), &end, 10);
        if (end && *end == '\0') return i;
        double d = std::strtod(atom.c_str(), &end);
        if (end && *end == '\0') return d;
        return atom;
    }

    json parse_value() {
        skip_ws();
        if (pos_ >= s_.size()) throw std::runtime_error("expected value");
        if (starts_with("<|\"|>")) return parse_string();
        if (s_[pos_] == '{') return parse_object();
        if (s_[pos_] == '[') return parse_array();
        return parse_atom();
    }

    bool starts_with(const std::string& prefix) const {
        return s_.compare(pos_, prefix.size(), prefix) == 0;
    }

    std::string s_;
    size_t pos_ = 0;
};

std::string balanced_object_prefix(const std::string& text, size_t start) {
    if (start >= text.size() || text[start] != '{')
        throw std::runtime_error("tool arguments must start with object");
    const std::string string_delim = "<|\"|>";
    bool in_string = false;
    int depth = 0;
    for (size_t i = start; i < text.size();) {
        if (text.compare(i, string_delim.size(), string_delim) == 0) {
            in_string = !in_string;
            i += string_delim.size();
            continue;
        }
        if (!in_string) {
            if (text[i] == '{') {
                ++depth;
            } else if (text[i] == '}') {
                --depth;
                if (depth == 0)
                    return text.substr(start, i - start + 1);
            }
        }
        ++i;
    }
    return text.substr(start);
}

std::string collapse_duplicate_braces_outside_strings(const std::string& text) {
    const std::string string_delim = "<|\"|>";
    std::string out;
    out.reserve(text.size());
    bool in_string = false;
    for (size_t i = 0; i < text.size();) {
        if (text.compare(i, string_delim.size(), string_delim) == 0) {
            in_string = !in_string;
            out += string_delim;
            i += string_delim.size();
            continue;
        }
        if (!in_string && (text[i] == '{' || text[i] == '}')) {
            char brace = text[i];
            out.push_back(brace);
            while (i < text.size() && text[i] == brace)
                ++i;
            continue;
        }
        out.push_back(text[i++]);
    }
    return out;
}

json parse_tool_args(std::string args_src) {
    try {
        return GemmaToolArgsParser(args_src).parse();
    } catch (const std::exception&) {
        return GemmaToolArgsParser(
            collapse_duplicate_braces_outside_strings(args_src)).parse();
    }
}

std::optional<ParsedToolCall> parse_one_tool_call(const std::string& body, size_t index) {
    const std::string prefix = "call:";
    size_t p = body.find(prefix);
    if (p == std::string::npos) return std::nullopt;
    p += prefix.size();
    size_t brace = body.find('{', p);
    if (brace == std::string::npos) return std::nullopt;
    std::string name = trim_copy(body.substr(p, brace - p));
    std::string args_src = balanced_object_prefix(body, brace);
    json args = parse_tool_args(std::move(args_src));
    if (!args.is_object()) args = json{{"value", args}};
    return ParsedToolCall{
        "call_" + std::to_string(index),
        name,
        args.dump(),
    };
}

void erase_all(std::string& text, const std::string& needle) {
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos)
        text.erase(pos, needle.size());
}

void strip_gemma_channels(std::string& content) {
    const std::string channel_start = "<|channel>";
    const std::string thought_start = "<|channel>thought\n";
    const std::string channel_end = "<channel|>";

    while (true) {
        size_t s = content.find(thought_start);
        if (s == std::string::npos) break;
        size_t body = s + thought_start.size();
        size_t e = content.find(channel_end, body);
        size_t next_channel = content.find(channel_start, body);
        if (e == std::string::npos ||
            (next_channel != std::string::npos && next_channel < e)) {
            if (next_channel == std::string::npos) {
                content.erase(s);
                break;
            }
            content.erase(s, next_channel - s);
        } else {
            content.erase(s, e + channel_end.size() - s);
        }
    }

    size_t search = 0;
    while (true) {
        size_t s = content.find(channel_start, search);
        if (s == std::string::npos) break;
        size_t label_end = content.find('\n', s + channel_start.size());
        if (label_end == std::string::npos) {
            content.erase(s, channel_start.size());
            search = s;
        } else {
            content.erase(s, label_end + 1 - s);
            search = s;
        }
    }
    erase_all(content, channel_end);
}

} // namespace

ParsedAssistantOutput parse_assistant_output(const std::string& raw_text) {
    ParsedAssistantOutput out;
    std::string content = raw_text;
    const std::string start = "<|tool_call>";
    const std::string end = "<tool_call|>";
    size_t search = 0;
    size_t idx = 0;
    while (true) {
        size_t s = content.find(start, search);
        if (s == std::string::npos) break;
        size_t body_start = s + start.size();
        size_t e = content.find(end, body_start);
        if (e == std::string::npos) break;
        std::string body = content.substr(body_start, e - body_start);
        try {
            if (auto call = parse_one_tool_call(body, idx++))
                out.tool_calls.push_back(std::move(*call));
        } catch (const std::exception&) {
            // Keep malformed calls as text. The model may still have produced a normal answer.
        }
        content.erase(s, e + end.size() - s);
        search = s;
    }

    strip_gemma_channels(content);

    erase_all(content, "<turn|>");
    erase_all(content, "<|tool_response>");
    erase_all(content, "<tool_response|>");
    erase_all(content, "<eos>");
    erase_all(content, "<bos>");
    out.content = trim_copy(content);
    return out;
}

Gemma4TokenBoundaryParser::Gemma4TokenBoundaryParser(const std::string& model_dir) {
    Tokenizer tokenizer = Tokenizer::from_json(model_dir + "/tokenizer.json");
    Gemma4BoundaryTokenStrings tokens = load_boundary_token_strings(model_dir);

    if (tokenizer.has_token(tokens.soc) && tokenizer.has_token(tokens.eoc)) {
        channel_start_id_ = tokenizer.token_id(tokens.soc);
        thought_start_ids_ = tokenizer.encode(tokens.soc + "thought\n", /*add_bos=*/false);
        thought_end_id_ = tokenizer.token_id(tokens.eoc);
        add_channel_pattern(tokenizer, tokens.soc, "final", response_channel_start_ids_);
        add_channel_pattern(tokenizer, tokens.soc, "answer", response_channel_start_ids_);
    }

    add_if_present(tokenizer, tokens.bos, ignored_response_ids_);
    add_if_present(tokenizer, tokens.eos, ignored_response_ids_);
    add_if_present(tokenizer, tokens.eot, ignored_response_ids_);
    add_if_present(tokenizer, tokens.soc, ignored_response_ids_);
    add_if_present(tokenizer, tokens.eoc, ignored_response_ids_);
    add_if_present(tokenizer, tokens.tool_response_start, ignored_response_ids_);
    add_if_present(tokenizer, tokens.tool_response_end, ignored_response_ids_);
    std::sort(ignored_response_ids_.begin(), ignored_response_ids_.end());
    ignored_response_ids_.erase(
        std::unique(ignored_response_ids_.begin(), ignored_response_ids_.end()),
        ignored_response_ids_.end());
}

Gemma4TokenBoundaryCounts
Gemma4TokenBoundaryParser::count(const std::vector<int>& token_ids) const {
    Gemma4TokenBoundaryCounts counts;
    size_t i = 0;
    while (i < token_ids.size()) {
        if (has_thought_markers() && matches_at(token_ids, i, thought_start_ids_)) {
            i += thought_start_ids_.size();
            while (i < token_ids.size() && token_ids[i] != thought_end_id_ &&
                   token_ids[i] != channel_start_id_) {
                ++counts.reasoning_tokens;
                ++i;
            }
            if (i < token_ids.size() && token_ids[i] == thought_end_id_)
                ++i;
            continue;
        }

        if (const std::vector<int>* channel = matching_response_channel_at(token_ids, i)) {
            i += channel->size();
            continue;
        }

        if (!is_ignored_response_id(token_ids[i]))
            ++counts.response_tokens;
        ++i;
    }
    return counts;
}

bool Gemma4TokenBoundaryParser::has_thought_markers() const {
    return !thought_start_ids_.empty() && thought_end_id_ >= 0;
}

bool Gemma4TokenBoundaryParser::matches_at(const std::vector<int>& token_ids, size_t pos,
                                           const std::vector<int>& pattern) const {
    if (pattern.empty() || pos + pattern.size() > token_ids.size())
        return false;
    for (size_t i = 0; i < pattern.size(); ++i)
        if (token_ids[pos + i] != pattern[i])
            return false;
    return true;
}

const std::vector<int>* Gemma4TokenBoundaryParser::matching_response_channel_at(
    const std::vector<int>& token_ids, size_t pos) const {
    for (const auto& pattern : response_channel_start_ids_)
        if (matches_at(token_ids, pos, pattern))
            return &pattern;
    return nullptr;
}

bool Gemma4TokenBoundaryParser::is_ignored_response_id(int token_id) const {
    return std::binary_search(ignored_response_ids_.begin(),
                              ignored_response_ids_.end(),
                              token_id);
}

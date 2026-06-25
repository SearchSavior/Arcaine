#include "tool_call_parser.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

using json = nlohmann::ordered_json;

namespace {

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

std::optional<ParsedToolCall> parse_one_tool_call(const std::string& body, size_t index) {
    const std::string prefix = "call:";
    size_t p = body.find(prefix);
    if (p == std::string::npos) return std::nullopt;
    p += prefix.size();
    size_t brace = body.find('{', p);
    if (brace == std::string::npos) return std::nullopt;
    std::string name = trim_copy(body.substr(p, brace - p));
    std::string args_src = body.substr(brace);
    json args = GemmaToolArgsParser(args_src).parse();
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

    // Strip the model's thought channel from public content. An unterminated
    // block means we are mid-stream and still inside the reasoning, so drop
    // the remainder for now.
    const std::string thought_start = "<|channel>thought\n";
    const std::string thought_end = "<channel|>";
    while (true) {
        size_t s = content.find(thought_start);
        if (s == std::string::npos) break;
        size_t body = s + thought_start.size();
        size_t e = content.find(thought_end, body);
        if (e == std::string::npos) {
            content.erase(s);
            break;
        }
        content.erase(s, e + thought_end.size() - s);
    }

    erase_all(content, "<turn|>");
    erase_all(content, "<|tool_response>");
    erase_all(content, "<tool_response|>");
    erase_all(content, "<eos>");
    erase_all(content, "<bos>");
    out.content = trim_copy(content);
    return out;
}

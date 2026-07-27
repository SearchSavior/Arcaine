#pragma once

#include <nlohmann/json.hpp>

#include "inference/contracts/generation_metrics.hpp"
#include "inference/contracts/generation_result.hpp"

namespace arcaine::openai {

using json = nlohmann::ordered_json;

// Build the OpenAI error envelope.
inline json error_body(const std::string& message, const std::string& type,
                       const std::string& code) {
    return {
        {"error", {
            {"message", message},
            {"type", type},
            {"param", nullptr},
            {"code", code},
        }}
    };
}

inline json usage_json(int prompt_tokens, int completion_tokens) {
    return {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + completion_tokens},
    };
}

inline json metrics_json(const arcaine::inference::GenerationMetrics& m) {
    return {
        {"input_token", m.input_token},
        {"new_token", m.new_token},
        {"ttft", m.ttft},
        {"tpot", m.tpot},
        {"prefill_throughput", m.prefill_throughput},
        {"decode_throuput", m.decode_throuput},
        {"duration", m.duration},
    };
}

inline json tool_calls_json(const std::vector<arcaine::inference::ParsedToolCall>& calls) {
    json out = json::array();
    for (const auto& call : calls) {
        out.push_back({
            {"id", call.id},
            {"type", "function"},
            {"function", {
                {"name", call.name},
                {"arguments", call.arguments},
            }},
        });
    }
    return out;
}

}  // namespace arcaine::openai

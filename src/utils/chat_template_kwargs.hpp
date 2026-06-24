#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

// Parse the --chat-template-kwargs CLI value: a JSON object whose keys are
// injected into the chat template's minja extra_context (mirrors vLLM's
// --chat-template-kwargs). Throws std::runtime_error on malformed JSON or any
// non-object value.
inline nlohmann::ordered_json parse_chat_template_kwargs(const std::string& value) {
    nlohmann::ordered_json parsed;
    try {
        parsed = nlohmann::ordered_json::parse(value);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(
            "--chat-template-kwargs must be valid JSON: " + std::string(e.what()));
    }
    if (!parsed.is_object())
        throw std::runtime_error("--chat-template-kwargs must be a JSON object");
    return parsed;
}

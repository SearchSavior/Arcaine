#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Architecture-independent chat templating. Minja renders the model's
// chat_template.jinja and the native tokenizer.json loader maps text to ids,
// so callers only need the model directory, not per-architecture config.

struct PromptBuildResult {
    std::vector<int> tokens;
    std::vector<int32_t> mm_token_type_ids;  // 0=text, 1=image, 2=video, 3=audio
};

struct ChatTemplateMessage {
    std::string role;
    std::string content;
};

PromptBuildResult build_chat_prompt(
    const std::string& model_dir,
    const std::string& user_prompt,
    const std::vector<int>& image_token_counts,
    const std::vector<int>& audio_token_counts,
    bool add_generation_prompt = true,
    bool enable_thinking = false
);

PromptBuildResult build_chat_prompt(
    const std::string& model_dir,
    const std::vector<ChatTemplateMessage>& messages,
    bool add_generation_prompt = true,
    bool enable_thinking = false
);

PromptBuildResult build_chat_prompt_json(
    const std::string& model_dir,
    nlohmann::ordered_json messages,
    nlohmann::ordered_json tools = nlohmann::ordered_json::array(),
    bool add_generation_prompt = true,
    bool enable_thinking = false
);

std::string decode_tokens(
    const std::string& model_dir,
    const std::vector<int>& token_ids
);

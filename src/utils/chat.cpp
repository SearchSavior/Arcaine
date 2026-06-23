#include "chat.hpp"
#include "../common/preprocess/chat_template.hpp"

TokenizerBridge::TokenizerBridge(const std::string& model_dir)
    : model_dir_(model_dir), tokenizer_(Tokenizer::from_json(model_dir + "/tokenizer.json")) {}

std::vector<int> TokenizerBridge::build_prompt(const std::string& user_prompt) {
    auto built = build_chat_prompt(model_dir_, user_prompt, {}, {},
        /*add_generation_prompt=*/true, /*enable_thinking=*/false);
    return std::move(built.tokens);
}

std::vector<int> TokenizerBridge::build_prompt(const std::vector<ChatTemplateMessage>& messages) {
    auto built = build_chat_prompt(model_dir_, messages,
        /*add_generation_prompt=*/true, /*enable_thinking=*/false);
    return std::move(built.tokens);
}

std::string TokenizerBridge::decode(const std::vector<int>& token_ids) {
    return tokenizer_.decode(token_ids, /*skip_special=*/true);
}

std::vector<std::string> TokenizerBridge::pieces(const std::vector<int>& token_ids) {
    std::vector<std::string> out;
    out.reserve(token_ids.size());
    for (int id : token_ids)
        out.push_back(tokenizer_.decode({id}, /*skip_special=*/false,
                                        /*strip_leading_space=*/false));
    return out;
}

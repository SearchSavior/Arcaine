#include "chat.hpp"
#include "../common/preprocess/chat_template.hpp"
#include "../common/io/gguf.hpp"
#include <cstdlib>
#include <stdexcept>

TokenizerBridge::TokenizerBridge(const std::string& model_dir)
    : model_dir_(model_dir) {
    const char* gguf = std::getenv("DIFF_GGUF_Q8_WEIGHTS");
    if (gguf && gguf[0]) {
        gguf_path_ = gguf;
        GgufFile gg(gguf);
        std::string tmpl;
        if (!gg.get_str("tokenizer.chat_template", tmpl))
            throw std::runtime_error("GGUF: missing tokenizer.chat_template");
        template_source_ = std::move(tmpl);
        uint32_t bos_id, eos_id;
        if (!gg.get_u32("tokenizer.ggml.bos_token_id", bos_id))
            throw std::runtime_error("GGUF: missing tokenizer.ggml.bos_token_id");
        if (!gg.get_u32("tokenizer.ggml.eos_token_id", eos_id))
            throw std::runtime_error("GGUF: missing tokenizer.ggml.eos_token_id");
        std::vector<std::string> tokens;
        if (!gg.get_str_array("tokenizer.ggml.tokens", tokens))
            throw std::runtime_error("GGUF: missing tokenizer.ggml.tokens");
        bos_token_ = tokens.size() > bos_id ? tokens[bos_id] : "";
        eos_token_ = tokens.size() > eos_id ? tokens[eos_id] : "";
        tokenizer_ = Tokenizer::from_gguf(gguf);
    } else {
        tokenizer_ = Tokenizer::from_json(model_dir + "/tokenizer.json");
    }
}

std::vector<int> TokenizerBridge::build_prompt(const std::string& user_prompt) {
    if (!gguf_path_.empty()) {
        std::vector<ChatTemplateMessage> messages = {{"user", user_prompt}};
        auto built = build_chat_prompt_from_gguf(gguf_path_, tokenizer_,
            template_source_, bos_token_, eos_token_, messages,
            /*add_generation_prompt=*/true, /*enable_thinking=*/false);
        return std::move(built.tokens);
    }
    auto built = build_chat_prompt(model_dir_, user_prompt, {}, {},
        /*add_generation_prompt=*/true, /*enable_thinking=*/false);
    return std::move(built.tokens);
}

std::vector<int> TokenizerBridge::build_prompt(const std::vector<ChatTemplateMessage>& messages) {
    if (!gguf_path_.empty()) {
        auto built = build_chat_prompt_from_gguf(gguf_path_, tokenizer_,
            template_source_, bos_token_, eos_token_, messages,
            /*add_generation_prompt=*/true, /*enable_thinking=*/false);
        return std::move(built.tokens);
    }
    auto built = build_chat_prompt(model_dir_, messages,
        /*add_generation_prompt=*/true, /*enable_thinking=*/false);
    return std::move(built.tokens);
}

std::vector<int> TokenizerBridge::build_prompt_json(
        const nlohmann::ordered_json& messages,
        const nlohmann::ordered_json& tools,
        const nlohmann::ordered_json& chat_template_kwargs) {
    if (!gguf_path_.empty()) {
        auto built = build_chat_prompt_from_gguf_json(gguf_path_, tokenizer_,
            template_source_, bos_token_, eos_token_, messages, tools,
            /*add_generation_prompt=*/true, /*enable_thinking=*/false,
            chat_template_kwargs);
        return std::move(built.tokens);
    }
    auto built = build_chat_prompt_json(model_dir_, messages, tools,
        /*add_generation_prompt=*/true, /*enable_thinking=*/false,
        chat_template_kwargs);
    return std::move(built.tokens);
}

std::string TokenizerBridge::decode(const std::vector<int>& token_ids) {
    return tokenizer_.decode(token_ids, /*skip_special=*/true);
}

std::string TokenizerBridge::decode_raw(const std::vector<int>& token_ids) {
    return tokenizer_.decode(token_ids, /*skip_special=*/false,
                             /*strip_leading_space=*/false);
}

std::vector<std::string> TokenizerBridge::pieces(const std::vector<int>& token_ids) {
    std::vector<std::string> out;
    out.reserve(token_ids.size());
    for (int id : token_ids)
        out.push_back(tokenizer_.decode({id}, /*skip_special=*/false,
                                        /*strip_leading_space=*/false));
    return out;
}

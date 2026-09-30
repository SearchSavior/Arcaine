#pragma once
#include "../common/preprocess/tokenizer.hpp"
#include "../common/preprocess/chat_template.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Native chat templating/tokenization bridge.  Minja renders the model's
// chat_template.jinja and the local tokenizer.json implementation maps text to ids.
class TokenizerBridge {
public:
    explicit TokenizerBridge(const std::string& model_dir);

    TokenizerBridge(const TokenizerBridge&) = delete;
    TokenizerBridge& operator=(const TokenizerBridge&) = delete;

    std::vector<int> build_prompt(const std::string& user_prompt);
    std::vector<int> build_prompt(const std::vector<ChatTemplateMessage>& messages);
    // The same messages rendered with add_generation_prompt=false. The trailing
    // tokens that differ from build_prompt() are the assistant generation
    // suffix, isolated from the state and instructions content.
    std::vector<int> build_prompt_no_generation(const std::vector<ChatTemplateMessage>& messages);
    // Render only (no tokenizing). Use this with encode_raw to avoid tokenizing
    // a long shared prefix twice.
    std::string build_prompt_text(const std::vector<ChatTemplateMessage>& messages,
                                  bool add_generation_prompt);
    std::vector<int> build_prompt_json(
        const nlohmann::ordered_json& messages,
        const nlohmann::ordered_json& tools,
        const nlohmann::ordered_json& chat_template_kwargs =
            nlohmann::ordered_json::object());
    std::string decode(const std::vector<int>& token_ids);
    std::string decode_raw(const std::vector<int>& token_ids);
    std::vector<std::string> pieces(const std::vector<int>& token_ids);

    // Raw tokenizer access for structured-read template compilation:
    // BOS-free encoding and direct token lookup (special tokens included).
    // `add_prefix_space=false` encodes a continuation segment (appended after
    // an existing special token) without a SentencePiece text-start sentinel.
    std::vector<int> encode_raw(const std::string& text, bool add_bos = false,
                                bool add_prefix_space = true);
    int  token_id(const std::string& token) const;
    bool has_token(const std::string& token) const;
    int  vocab_size() const;
    bool is_special_token(int id) const;

private:
    std::string model_dir_;
    Tokenizer tokenizer_;
};

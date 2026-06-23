#pragma once
#include "../common/preprocess/tokenizer.hpp"
#include "../common/preprocess/chat_template.hpp"
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
    std::string decode(const std::vector<int>& token_ids);
    std::vector<std::string> pieces(const std::vector<int>& token_ids);

private:
    std::string model_dir_;
    Tokenizer tokenizer_;
};

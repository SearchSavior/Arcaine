#include "chat_template.hpp"
#include "tokenizer.hpp"

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using json = nlohmann::ordered_json;

struct TokenizerMetadata {
    std::string bos_token;
    std::string eos_token;
    std::string boi_token;
    std::string eoi_token;
    std::string image_token;
    std::string boa_token;
    std::string eoa_token;
    std::string audio_token;
    std::string video_token;
};

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

TokenizerMetadata load_tokenizer_metadata(const std::string& model_dir) {
    std::ifstream f(model_dir + "/tokenizer_config.json");
    if (!f) throw std::runtime_error("Cannot open " + model_dir + "/tokenizer_config.json");
    auto t = json::parse(f);

    TokenizerMetadata meta;
    meta.bos_token   = t.at("bos_token").get<std::string>();
    meta.eos_token   = t.at("eos_token").get<std::string>();
    meta.boi_token   = t.value("boi_token", std::string());
    meta.eoi_token   = t.value("eoi_token", std::string());
    meta.image_token = t.value("image_token", std::string());
    meta.boa_token   = t.value("boa_token", std::string());
    meta.eoa_token   = t.value("eoa_token", std::string());
    meta.audio_token = t.value("audio_token", std::string());
    if (t.contains("video_token")) {
        meta.video_token = t.at("video_token").get<std::string>();
    } else if (t.contains("extra_special_tokens") && !t.at("extra_special_tokens").empty()) {
        meta.video_token = t.at("extra_special_tokens").at(0).get<std::string>();
    }
    return meta;
}

void replace_one(std::string& text, const std::string& needle, const std::string& replacement) {
    size_t pos = text.find(needle);
    if (pos == std::string::npos)
        throw std::runtime_error("chat template did not emit placeholder " + needle);
    text.replace(pos, needle.size(), replacement);
}

std::string repeat_token(const std::string& token, int count) {
    std::string out;
    out.reserve(token.size() * (size_t)count);
    for (int i = 0; i < count; ++i) out += token;
    return out;
}

PromptBuildResult render_chat_prompt_core(
    const TokenizerMetadata& meta,
    const std::string& template_source,
    const Tokenizer& tok,
    json messages,
    json tools,
    const std::vector<int>& image_token_counts,
    const std::vector<int>& audio_token_counts,
    bool add_generation_prompt,
    bool enable_thinking,
    json chat_template_kwargs
) {
    minja::chat_template tmpl(template_source, meta.bos_token, meta.eos_token);
    minja::chat_template_inputs inputs;
    inputs.messages = std::move(messages);
    inputs.tools = std::move(tools);
    inputs.add_generation_prompt = add_generation_prompt;
    // Start from the dedicated enable_thinking flag, then let explicit
    // --chat-template-kwargs override it (and add any other template vars).
    json extra_context = {{"enable_thinking", enable_thinking}};
    if (chat_template_kwargs.is_object())
        for (auto& [key, value] : chat_template_kwargs.items())
            extra_context[key] = value;
    inputs.extra_context = std::move(extra_context);

    std::string rendered = tmpl.apply(inputs);

    if (!meta.image_token.empty()) {
        for (int count : image_token_counts) {
            replace_one(rendered, meta.image_token,
                meta.boi_token + repeat_token(meta.image_token, count) + meta.eoi_token);
        }
    }
    if (!meta.audio_token.empty()) {
        for (int count : audio_token_counts) {
            replace_one(rendered, meta.audio_token,
                meta.boa_token + repeat_token(meta.audio_token, count) + meta.eoa_token);
        }
    }

    PromptBuildResult out;
    out.tokens = tok.encode(rendered, /*add_bos=*/false);
    out.mm_token_type_ids.reserve(out.tokens.size());

    const int image_id = meta.image_token.empty() ? -1 : tok.token_id(meta.image_token);
    const int audio_id = meta.audio_token.empty() ? -1 : tok.token_id(meta.audio_token);
    const int video_id = meta.video_token.empty() || !tok.has_token(meta.video_token) ? -1 : tok.token_id(meta.video_token);
    for (int token_id : out.tokens) {
        if (token_id == image_id) {
            out.mm_token_type_ids.push_back(1);
        } else if (token_id == video_id) {
            out.mm_token_type_ids.push_back(2);
        } else if (token_id == audio_id) {
            out.mm_token_type_ids.push_back(3);
        } else {
            out.mm_token_type_ids.push_back(0);
        }
    }
    return out;
}

PromptBuildResult render_chat_prompt(
    const std::string& model_dir,
    json messages,
    json tools,
    const std::vector<int>& image_token_counts,
    const std::vector<int>& audio_token_counts,
    bool add_generation_prompt,
    bool enable_thinking,
    json chat_template_kwargs = json::object()
) {
    const TokenizerMetadata meta = load_tokenizer_metadata(model_dir);
    const std::string source = read_file(model_dir + "/chat_template.jinja");
    Tokenizer tok = Tokenizer::from_json(model_dir + "/tokenizer.json");
    return render_chat_prompt_core(meta, source, tok, std::move(messages), std::move(tools),
        image_token_counts, audio_token_counts, add_generation_prompt, enable_thinking,
        std::move(chat_template_kwargs));
}

} // namespace

PromptBuildResult build_chat_prompt(
    const std::string& model_dir,
    const std::string& user_prompt,
    const std::vector<int>& image_token_counts,
    const std::vector<int>& audio_token_counts,
    bool add_generation_prompt,
    bool enable_thinking
) {
    json content = json::array();
    for (size_t i = 0; i < image_token_counts.size(); ++i)
        content.push_back({{"type", "image"}});
    for (size_t i = 0; i < audio_token_counts.size(); ++i)
        content.push_back({{"type", "audio"}});
    content.push_back({{"type", "text"}, {"text", user_prompt}});

    json messages = json::array({{{"role", "user"}, {"content", content}}});
    return render_chat_prompt(model_dir, std::move(messages), json::array(),
                              image_token_counts, audio_token_counts,
                              add_generation_prompt, enable_thinking);
}


PromptBuildResult build_chat_prompt(
    const std::string& model_dir,
    const std::vector<ChatTemplateMessage>& messages,
    bool add_generation_prompt,
    bool enable_thinking
) {
    if (messages.empty())
        throw std::runtime_error("chat prompt needs at least one message");

    json rendered_messages = json::array();
    for (const ChatTemplateMessage& message : messages) {
        if (message.role.empty())
            throw std::runtime_error("chat message role cannot be empty");
        json content = json::array({{{"type", "text"}, {"text", message.content}}});
        rendered_messages.push_back({{"role", message.role}, {"content", content}});
    }

    return render_chat_prompt(model_dir, std::move(rendered_messages), json::array(),
                              {}, {}, add_generation_prompt, enable_thinking);
}

PromptBuildResult build_chat_prompt_json(
    const std::string& model_dir,
    json messages,
    json tools,
    bool add_generation_prompt,
    bool enable_thinking,
    json chat_template_kwargs
) {
    if (!messages.is_array() || messages.empty())
        throw std::runtime_error("chat prompt needs at least one message");
    if (!tools.is_array())
        throw std::runtime_error("chat prompt tools must be an array");
    return render_chat_prompt(model_dir, std::move(messages), std::move(tools),
                              {}, {}, add_generation_prompt, enable_thinking,
                              std::move(chat_template_kwargs));
}

PromptBuildResult build_chat_prompt_from_gguf(
    const std::string& gguf_path,
    const Tokenizer& tok,
    const std::string& template_source,
    const std::string& bos_token,
    const std::string& eos_token,
    const std::vector<ChatTemplateMessage>& messages,
    bool add_generation_prompt,
    bool enable_thinking
) {
    (void)gguf_path;
    if (messages.empty())
        throw std::runtime_error("chat prompt needs at least one message");

    TokenizerMetadata meta;
    meta.bos_token = bos_token;
    meta.eos_token = eos_token;

    json rendered_messages = json::array();
    for (const ChatTemplateMessage& message : messages) {
        if (message.role.empty())
            throw std::runtime_error("chat message role cannot be empty");
        json content = json::array({{{"type", "text"}, {"text", message.content}}});
        rendered_messages.push_back({{"role", message.role}, {"content", content}});
    }

    return render_chat_prompt_core(meta, template_source, tok,
        std::move(rendered_messages), json::array(),
        {}, {}, add_generation_prompt, enable_thinking, json::object());
}

PromptBuildResult build_chat_prompt_from_gguf_json(
    const std::string& gguf_path,
    const Tokenizer& tok,
    const std::string& template_source,
    const std::string& bos_token,
    const std::string& eos_token,
    json messages,
    json tools,
    bool add_generation_prompt,
    bool enable_thinking,
    json chat_template_kwargs
) {
    (void)gguf_path;
    if (!messages.is_array() || messages.empty())
        throw std::runtime_error("chat prompt needs at least one message");
    if (!tools.is_array())
        throw std::runtime_error("chat prompt tools must be an array");

    TokenizerMetadata meta;
    meta.bos_token = bos_token;
    meta.eos_token = eos_token;

    return render_chat_prompt_core(meta, template_source, tok,
        std::move(messages), std::move(tools),
        {}, {}, add_generation_prompt, enable_thinking,
        std::move(chat_template_kwargs));
}

std::string decode_tokens(
    const std::string& model_dir,
    const std::vector<int>& token_ids
) {
    Tokenizer tok = Tokenizer::from_json(model_dir + "/tokenizer.json");
    return tok.decode(token_ids, /*skip_special=*/true);
}

#include "modeling/qwen3_5/request_mapper.hpp"

#include "modeling/qwen3_5/model.hpp"
#include "utils/chat.hpp"

#include <stdexcept>
#include <utility>

namespace arcaine::qwen3_5 {

Qwen35Invocation map_request(const arcaine::inference::GenerationRequest& req,
                             Qwen35Model& model,
                             TokenizerBridge& tokenizer,
                             const ModelInfo& info) {
    Qwen35Invocation inv;
    inv.max_output_tokens = req.max_output_tokens > 0 ? req.max_output_tokens : 256;
    inv.seed              = req.seed;
    inv.has_tools         = !req.tools.empty();
    inv.stream            = req.stream.enabled;
    inv.include_usage     = req.stream.include_usage;

    inv.temperature = req.sampling.overrides_temperature() ? req.sampling.temperature
                                                            : info.temperature;
    inv.top_k        = req.sampling.overrides_top_k()       ? req.sampling.top_k
                                                            : info.top_k;
    inv.top_p        = req.sampling.overrides_top_p()       ? req.sampling.top_p
                                                            : info.top_p;

    if (!req.messages.empty()) {
        // Chat path: render the model chat template from the normalized OpenAI
        // messages + tools. AR chat is text-only.
        inv.input_ids = tokenizer.build_prompt_json(req.messages, req.tools,
                                                    req.chat_template.kwargs);
    } else if (!req.prompt_text.empty()) {
        // CLI path: the model applies its chat template + preprocesses media.
        auto prepared = model.prepare_input(req.prompt_text, req.image_paths,
                                             /*audio_paths=*/{}, req.vad_model_path);
        inv.input_ids         = std::move(prepared.tokens);
        inv.images            = std::move(prepared.images);
        inv.mm_token_type_ids = std::move(prepared.mm_token_type_ids);
    } else if (!req.input_token_ids.empty()) {
        inv.input_ids = req.input_token_ids;
    } else {
        throw std::runtime_error(
            "qwen3_5: request has no input (messages/prompt_text/input_token_ids)");
    }
    return inv;
}

}  // namespace arcaine::qwen3_5

#include "modeling/gemma4_unified/request_mapper.hpp"

#include "modeling/gemma4_unified/model.hpp"
#include "utils/chat.hpp"

#include <stdexcept>
#include <utility>

namespace arcaine::gemma4_unified {

Gemma4Invocation map_request(const inference::GenerationRequest& req,
                             Gemma4Model& model,
                             TokenizerBridge& tokenizer,
                             const ModelInfo& info) {
    Gemma4Invocation inv;
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
        // messages + tools. AR chat is text-only (matches the prior transport).
        inv.input_ids = tokenizer.build_prompt_json(req.messages, req.tools,
                                                    req.chat_template.kwargs);
    } else if (!req.prompt_text.empty()) {
        // CLI path: the model applies its chat template + preprocesses media.
        auto prepared = model.prepare_input(req.prompt_text, req.image_paths,
                                             req.audio_paths, req.vad_model_path);
        inv.input_ids         = std::move(prepared.tokens);
        inv.images            = std::move(prepared.images);
        inv.audio             = std::move(prepared.audio);
        inv.mm_token_type_ids = std::move(prepared.mm_token_type_ids);
    } else if (!req.input_token_ids.empty()) {
        inv.input_ids = req.input_token_ids;
    } else {
        throw std::runtime_error(
            "gemma4_unified: request has no input (messages/prompt_text/input_token_ids)");
    }
    return inv;
}

}  // namespace arcaine::gemma4_unified

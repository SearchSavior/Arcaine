#include "modeling/diffusion_gemma/request_mapper.hpp"

#include "utils/chat.hpp"

#include <stdexcept>
#include <utility>

namespace arcaine::diffusion_gemma {

DiffusionGemmaInvocation map_request(
    const arcaine::inference::GenerationRequest& req,
    TokenizerBridge& tokenizer) {
    DiffusionGemmaInvocation inv;
    inv.output_length    = req.max_output_tokens > 0 ? req.max_output_tokens : 256;
    inv.denoising_steps  = req.denoising_steps;  // -1 = model default (resolved at generate)
    inv.seed             = req.seed;
    inv.stream_drafts    = req.stream.stream_drafts;
    inv.has_tools        = !req.tools.empty();
    inv.stream           = req.stream.enabled;
    inv.include_usage    = req.stream.include_usage;

    if (!req.messages.empty()) {
        // Chat path: render the model chat template from the normalized OpenAI
        // messages + tools. DiffusionGemma is Gemma-family (Gemma chat template
        // + Gemma tool-call syntax).
        inv.encoder_ids = tokenizer.build_prompt_json(req.messages, req.tools,
                                                       req.chat_template.kwargs);
    } else if (!req.prompt_text.empty()) {
        // CLI path: apply the chat template to a single user prompt.
        inv.encoder_ids = tokenizer.build_prompt(req.prompt_text);
    } else if (!req.input_token_ids.empty()) {
        inv.encoder_ids = req.input_token_ids;
    } else {
        throw std::runtime_error(
            "diffusion_gemma: request has no input (messages/prompt_text/input_token_ids)");
    }
    return inv;
}

}  // namespace arcaine::diffusion_gemma

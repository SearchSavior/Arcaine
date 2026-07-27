#pragma once

#include "modeling/diffusion_gemma/invocation.hpp"
#include "inference/contracts/generation_request.hpp"

class DiffusionGemmaModel;  // global (modeling/diffusion_gemma/model.hpp)
class TokenizerBridge;      // global (utils/chat.hpp)

namespace arcaine::diffusion_gemma {

// Resolve a public GenerationRequest into a private DiffusionGemmaInvocation:
// applies the chat template (or CLI prompt path), tokenizes, and resolves
// denoising-step / output-length / draft-streaming options. The transport
// never tokenizes or applies a model chat template.
DiffusionGemmaInvocation map_request(
    const arcaine::inference::GenerationRequest& request,
    TokenizerBridge& tokenizer);

}  // namespace arcaine::diffusion_gemma

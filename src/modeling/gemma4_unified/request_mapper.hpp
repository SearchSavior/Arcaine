#pragma once

#include "invocation.hpp"
#include "inference/contracts/generation_request.hpp"

class Gemma4Model;      // global (modeling/gemma4_unified/model.hpp)
class TokenizerBridge;  // global (utils/chat.hpp)
struct ModelInfo;       // global (common/model_interface.hpp)

namespace arcaine::gemma4_unified {

// Resolve a public GenerationRequest into a private Gemma4Invocation: applies
// the chat template (or CLI prompt path), tokenizes, preprocesses media, and
// resolves sampler defaults from the model config. The transport never
// tokenizes or applies a model chat template.
Gemma4Invocation map_request(const inference::GenerationRequest& request,
                             Gemma4Model& model,
                             TokenizerBridge& tokenizer,
                             const ModelInfo& info);

}  // namespace arcaine::gemma4_unified

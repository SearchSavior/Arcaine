#pragma once

#include "modeling/qwen3_5/invocation.hpp"
#include "inference/contracts/generation_request.hpp"

class Qwen35Model;      // global (modeling/qwen3_5/model.hpp)
class TokenizerBridge;  // global (utils/chat.hpp)
struct ModelInfo;       // global (common/model_interface.hpp)

namespace arcaine::qwen3_5 {

// Resolve a public GenerationRequest into a private Qwen35Invocation: applies
// the chat template (or CLI prompt path), tokenizes, preprocesses media, and
// resolves sampler defaults from the model config. The transport never
// tokenizes or applies a model chat template.
Qwen35Invocation map_request(const arcaine::inference::GenerationRequest& request,
                             Qwen35Model& model,
                             TokenizerBridge& tokenizer,
                             const ModelInfo& info);

}  // namespace arcaine::qwen3_5

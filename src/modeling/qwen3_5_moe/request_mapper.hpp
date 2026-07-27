#pragma once

#include "modeling/qwen3_5_moe/invocation.hpp"
#include "inference/contracts/generation_request.hpp"

class QwenModel;      // global (modeling/qwen3_5_moe/model.hpp)
class TokenizerBridge;  // global (utils/chat.hpp)
struct ModelInfo;       // global (common/model_interface.hpp)

namespace arcaine::qwen3_5_moe {

// Resolve a public GenerationRequest into a private QwenMoeInvocation: applies
// the chat template (or CLI prompt path), tokenizes, and resolves sampler
// defaults from the model config. The transport never tokenizes or applies a
// model chat template.
QwenMoeInvocation map_request(const arcaine::inference::GenerationRequest& request,
                              QwenModel& model,
                              TokenizerBridge& tokenizer,
                              const ModelInfo& info);

}  // namespace arcaine::qwen3_5_moe

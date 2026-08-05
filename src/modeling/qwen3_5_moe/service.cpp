#include "modeling/qwen3_5_moe/service.hpp"

#include "modeling/qwen3_5_moe/session.hpp"
#include "modeling/qwen3_5_moe/model.hpp"
#include "utils/chat.hpp"
#include "inference/contracts/capabilities.hpp"

#include <utility>

namespace arcaine::qwen3_5_moe {

arcaine::inference::ModelDescriptor qwen_moe_descriptor() {
    arcaine::inference::ModelDescriptor d;
    // "qwen3_5_moe" is the wrapped VLM config (text-only loading; see
    // QwenConfig::from_dir).
    d.model_types        = {"qwen3_5_moe_text", "qwen3_5_moe"};
    d.implementation_id  = "qwen3_5_moe";
    d.inference_contract = arcaine::inference::InferenceContract::CausalGenerationV1;
    d.input_capabilities = static_cast<std::uint32_t>(
        arcaine::inference::InputCapability::Messages |
        arcaine::inference::InputCapability::PromptText |
        arcaine::inference::InputCapability::TokenIds |
        arcaine::inference::InputCapability::Tools);
    d.output_capabilities = static_cast<std::uint32_t>(
        arcaine::inference::OutputCapability::Text |
        arcaine::inference::OutputCapability::TextDeltas |
        arcaine::inference::OutputCapability::ToolCalls);
    d.supports_streaming = true;
    return d;
}

QwenMoeService::QwenMoeService(const arcaine::inference::ModelLoadRequest& req) {
    model_      = std::make_unique<QwenModel>(req.model_dir, req.max_seq_len);
    tokenizer_  = std::make_unique<TokenizerBridge>(req.model_dir);
    descriptor_ = qwen_moe_descriptor();
}

QwenMoeService::~QwenMoeService() = default;

std::unique_ptr<arcaine::inference::InferenceSession>
QwenMoeService::create_session(const arcaine::inference::SessionCreateRequest& req) {
    return std::make_unique<QwenMoeSession>(*this, req);
}

}  // namespace arcaine::qwen3_5_moe

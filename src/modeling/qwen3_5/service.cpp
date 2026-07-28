#include "modeling/qwen3_5/service.hpp"

#include "modeling/qwen3_5/session.hpp"
#include "modeling/qwen3_5/model.hpp"
#include "utils/chat.hpp"
#include "inference/contracts/capabilities.hpp"

#include <utility>

namespace arcaine::qwen3_5 {

arcaine::inference::ModelDescriptor qwen35_descriptor() {
    arcaine::inference::ModelDescriptor d;
    d.model_types        = {"qwen3_5"};
    d.implementation_id  = "qwen3_5";
    d.inference_contract = arcaine::inference::InferenceContract::CausalGenerationV1;
    d.input_capabilities = static_cast<std::uint32_t>(
        arcaine::inference::InputCapability::Messages |
        arcaine::inference::InputCapability::PromptText |
        arcaine::inference::InputCapability::TokenIds |
        arcaine::inference::InputCapability::Images |
        arcaine::inference::InputCapability::Tools);
    d.output_capabilities = static_cast<std::uint32_t>(
        arcaine::inference::OutputCapability::Text |
        arcaine::inference::OutputCapability::TextDeltas);
    // NOTE: ToolCalls intentionally NOT declared. The qwen3_5 output_parser is a
    // pass-through stub (native qwen tool-call syntax not yet implemented); the
    // model cannot produce structured tool_calls. Add ToolCalls back when a
    // native qwen tool-call parser lands (D2).
    d.supports_streaming = true;
    return d;
}

Qwen35Service::Qwen35Service(const arcaine::inference::ModelLoadRequest& req) {
    model_      = std::make_unique<Qwen35Model>(req.model_dir, req.max_seq_len);
    tokenizer_  = std::make_unique<TokenizerBridge>(req.model_dir);
    descriptor_ = qwen35_descriptor();
}

Qwen35Service::~Qwen35Service() = default;

std::unique_ptr<arcaine::inference::InferenceSession>
Qwen35Service::create_session(const arcaine::inference::SessionCreateRequest& req) {
    return std::make_unique<Qwen35Session>(*this, req);
}

}  // namespace arcaine::qwen3_5

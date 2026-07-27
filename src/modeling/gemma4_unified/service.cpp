#include "modeling/gemma4_unified/service.hpp"

#include "modeling/gemma4_unified/session.hpp"
#include "modeling/gemma4_unified/model.hpp"
#include "modeling/gemma4_unified/output_parser.hpp"
#include "utils/chat.hpp"
#include "inference/contracts/capabilities.hpp"

#include <utility>

namespace arcaine::gemma4_unified {

arcaine::inference::ModelDescriptor gemma4_descriptor() {
    arcaine::inference::ModelDescriptor d;
    d.model_types        = {"gemma4_unified"};
    d.implementation_id  = "gemma4_unified";
    d.inference_contract = arcaine::inference::InferenceContract::CausalGenerationV1;
    d.input_capabilities = static_cast<std::uint32_t>(
        arcaine::inference::InputCapability::Messages |
        arcaine::inference::InputCapability::PromptText |
        arcaine::inference::InputCapability::TokenIds |
        arcaine::inference::InputCapability::Images |
        arcaine::inference::InputCapability::Audio |
        arcaine::inference::InputCapability::Tools);
    d.output_capabilities = static_cast<std::uint32_t>(
        arcaine::inference::OutputCapability::Text |
        arcaine::inference::OutputCapability::TextDeltas |
        arcaine::inference::OutputCapability::ToolCalls);
    d.supports_streaming = true;
    return d;
}

Gemma4Service::Gemma4Service(const arcaine::inference::ModelLoadRequest& req) {
    model_           = std::make_unique<Gemma4Model>(req.model_dir, req.max_seq_len);
    tokenizer_       = std::make_unique<TokenizerBridge>(req.model_dir);
    boundary_parser_ = std::make_unique<Gemma4BoundaryParser>(req.model_dir);
    descriptor_      = gemma4_descriptor();
}

Gemma4Service::~Gemma4Service() = default;

std::unique_ptr<arcaine::inference::InferenceSession>
Gemma4Service::create_session(const arcaine::inference::SessionCreateRequest& req) {
    return std::make_unique<Gemma4Session>(*this, req);
}

Gemma4Model&          Gemma4Service::model() noexcept           { return *model_; }
TokenizerBridge&      Gemma4Service::tokenizer() noexcept       { return *tokenizer_; }
Gemma4BoundaryParser& Gemma4Service::boundary_parser() noexcept { return *boundary_parser_; }

}  // namespace arcaine::gemma4_unified

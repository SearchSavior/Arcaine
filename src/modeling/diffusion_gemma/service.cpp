#include "modeling/diffusion_gemma/service.hpp"

#include "modeling/diffusion_gemma/session.hpp"
#include "modeling/diffusion_gemma/model.hpp"
#include "modeling/diffusion_gemma/output_parser.hpp"
#include "utils/chat.hpp"
#include "inference/contracts/capabilities.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace arcaine::diffusion_gemma {

arcaine::inference::ModelDescriptor diffusion_gemma_descriptor() {
    arcaine::inference::ModelDescriptor d;
    d.model_types        = {"diffusion_gemma"};
    d.implementation_id  = "diffusion_gemma";
    d.inference_contract = arcaine::inference::InferenceContract::BlockDiffusionGenerationV1;
    d.input_capabilities = static_cast<std::uint32_t>(
        arcaine::inference::InputCapability::Messages |
        arcaine::inference::InputCapability::PromptText |
        arcaine::inference::InputCapability::TokenIds |
        arcaine::inference::InputCapability::Tools);
    d.output_capabilities = static_cast<std::uint32_t>(
        arcaine::inference::OutputCapability::Text |
        arcaine::inference::OutputCapability::TextDeltas |
        arcaine::inference::OutputCapability::ToolCalls |
        arcaine::inference::OutputCapability::DraftTokens);
    d.supports_streaming = true;
    return d;
}

namespace {
void apply_layers_spec(const std::string& v, DiffPlacementOptions& p) {
    if (v == "auto") { p.layer_mode = DiffLayerPlacementMode::Auto; p.layer_split = -1; }
    else if (v == "single") { p.layer_mode = DiffLayerPlacementMode::Single; p.layer_split = -1; }
    else if (v.rfind("split:", 0) == 0) {
        p.layer_mode = DiffLayerPlacementMode::Split;
        p.layer_split = std::stoi(v.substr(6));
    } else {
        throw std::runtime_error("Unsupported --layers value: " + v);
    }
}
void apply_experts_spec(const std::string& v, DiffPlacementOptions& p) {
    if (v == "auto")        p.expert_mode = DiffExpertPlacementMode::Auto;
    else if (v == "layer-owner") p.expert_mode = DiffExpertPlacementMode::LayerOwner;
    else if (v == "shard")       p.expert_mode = DiffExpertPlacementMode::Shard;
    else throw std::runtime_error("Unsupported --experts value: " + v);
}
}  // namespace

DiffPlacementOptions resolve_placement_overrides(
    const arcaine::inference::ModelLoadRequest& req) {
    DiffPlacementOptions p;
    for (const auto& kv : req.placement_overrides) {
        if (kv.first == "layers")  apply_layers_spec(kv.second, p);
        else if (kv.first == "experts") apply_experts_spec(kv.second, p);
        else if (kv.first == "gpus") { /* only "all" supported; ignore */ }
        // Unknown keys ignored (neutral contract).
    }
    return p;
}

DiffusionGemmaService::DiffusionGemmaService(
    const arcaine::inference::ModelLoadRequest& req) {
    DiffPlacementOptions placement = resolve_placement_overrides(req);
    model_           = std::make_unique<DiffusionGemmaModel>(
        req.model_dir, req.max_seq_len, placement, req.print_placement);
    tokenizer_       = std::make_unique<TokenizerBridge>(req.model_dir);
    boundary_parser_ = std::make_unique<DiffusionGemmaBoundaryParser>(req.model_dir);
    descriptor_      = diffusion_gemma_descriptor();
}

DiffusionGemmaService::~DiffusionGemmaService() = default;

std::unique_ptr<arcaine::inference::InferenceSession>
DiffusionGemmaService::create_session(const arcaine::inference::SessionCreateRequest& req) {
    return std::make_unique<DiffusionGemmaSession>(*this, req);
}

DiffusionGemmaModel& DiffusionGemmaService::model() noexcept { return *model_; }
TokenizerBridge& DiffusionGemmaService::tokenizer() noexcept { return *tokenizer_; }
DiffusionGemmaBoundaryParser& DiffusionGemmaService::boundary_parser() noexcept {
    return *boundary_parser_;
}

}  // namespace arcaine::diffusion_gemma

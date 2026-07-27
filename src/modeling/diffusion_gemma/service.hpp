#pragma once

#include <memory>

#include "inference/contracts/model_descriptor.hpp"
#include "inference/contracts/model_load_request.hpp"
#include "inference/service/model_service.hpp"
#include "inference/placement.hpp"  // DiffPlacementOptions

class DiffusionGemmaModel;   // global (modeling/diffusion_gemma/model.hpp)
class TokenizerBridge;       // global (utils/chat.hpp)

namespace arcaine::diffusion_gemma {

class DiffusionGemmaBoundaryParser;

// The public descriptor for the diffusion_gemma implementation.
arcaine::inference::ModelDescriptor diffusion_gemma_descriptor();

// Resolve the neutral ModelLoadRequest::placement_overrides into the
// diffusion-specific DiffPlacementOptions. Recognized keys: "layers"
// (auto|single|split:N), "experts" (auto|layer-owner|shard), "gpus" (all).
DiffPlacementOptions resolve_placement_overrides(
    const arcaine::inference::ModelLoadRequest& req);

// One loaded diffusion_gemma implementation. Owns the architecture engine
// (DiffusionGemmaModel, including its denoising loop, canvas, encoder,
// expert routing + execution, entropy/acceptance, and placement), the
// tokenizer + chat-template bridge, and the boundary parser; creates
// per-request sessions.
class DiffusionGemmaService : public arcaine::inference::ModelService {
public:
    explicit DiffusionGemmaService(const arcaine::inference::ModelLoadRequest& request);
    ~DiffusionGemmaService() override;

    const arcaine::inference::ModelDescriptor& descriptor() const noexcept override {
        return descriptor_;
    }
    std::unique_ptr<arcaine::inference::InferenceSession> create_session(
        const arcaine::inference::SessionCreateRequest& req) override;

    DiffusionGemmaModel&          model() noexcept;
    TokenizerBridge&             tokenizer() noexcept;
    DiffusionGemmaBoundaryParser& boundary_parser() noexcept;

private:
    std::unique_ptr<DiffusionGemmaModel>          model_;
    std::unique_ptr<TokenizerBridge>              tokenizer_;
    std::unique_ptr<DiffusionGemmaBoundaryParser> boundary_parser_;
    arcaine::inference::ModelDescriptor           descriptor_;
};

}  // namespace arcaine::diffusion_gemma

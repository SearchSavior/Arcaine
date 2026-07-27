#pragma once

#include <memory>

#include "inference/service/inference_session.hpp"
#include "inference/service/session_create.hpp"

namespace arcaine::diffusion_gemma {

class DiffusionGemmaService;

class DiffusionGemmaSession : public arcaine::inference::InferenceSession {
public:
    DiffusionGemmaSession(DiffusionGemmaService& service,
                          const arcaine::inference::SessionCreateRequest& req);
    ~DiffusionGemmaSession() override = default;

    arcaine::inference::GenerationResult generate(
        const arcaine::inference::GenerationRequest& request,
        arcaine::inference::GenerationSink& sink) override;

    void cancel() override;

private:
    DiffusionGemmaService&               service_;
    arcaine::inference::CancellationToken* cancel_ = nullptr;
};

}  // namespace arcaine::diffusion_gemma

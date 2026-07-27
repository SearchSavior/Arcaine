#pragma once

#include <memory>

#include "inference/service/inference_session.hpp"
#include "inference/service/session_create.hpp"

namespace arcaine::qwen3_5_moe {

class QwenMoeService;

class QwenMoeSession : public arcaine::inference::InferenceSession {
public:
    QwenMoeSession(QwenMoeService& service,
                   const arcaine::inference::SessionCreateRequest& req);
    ~QwenMoeSession() override = default;

    arcaine::inference::GenerationResult generate(
        const arcaine::inference::GenerationRequest& request,
        arcaine::inference::GenerationSink& sink) override;

    void cancel() override;

private:
    QwenMoeService&                      service_;
    arcaine::inference::CancellationToken* cancel_ = nullptr;
};

}  // namespace arcaine::qwen3_5_moe

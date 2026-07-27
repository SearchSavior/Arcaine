#pragma once

#include <memory>

#include "inference/service/inference_session.hpp"
#include "inference/service/session_create.hpp"

namespace arcaine::qwen3_5 {

class Qwen35Service;

class Qwen35Session : public arcaine::inference::InferenceSession {
public:
    Qwen35Session(Qwen35Service& service, const arcaine::inference::SessionCreateRequest& req);
    ~Qwen35Session() override = default;

    arcaine::inference::GenerationResult generate(
        const arcaine::inference::GenerationRequest& request,
        arcaine::inference::GenerationSink& sink) override;

    void cancel() override;

private:
    Qwen35Service&                service_;
    arcaine::inference::CancellationToken* cancel_ = nullptr;
};

}  // namespace arcaine::qwen3_5

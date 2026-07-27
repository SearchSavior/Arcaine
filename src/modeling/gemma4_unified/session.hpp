#pragma once

#include <memory>

#include "inference/service/inference_session.hpp"
#include "inference/service/session_create.hpp"

namespace arcaine::gemma4_unified {

class Gemma4Service;

class Gemma4Session : public inference::InferenceSession {
public:
    Gemma4Session(Gemma4Service& service, const inference::SessionCreateRequest& req);
    ~Gemma4Session() override = default;

    inference::GenerationResult generate(
        const inference::GenerationRequest& request,
        inference::GenerationSink& sink) override;

    void cancel() override;

private:
    Gemma4Service&                 service_;
    inference::CancellationToken*  cancel_;
};

}  // namespace arcaine::gemma4_unified

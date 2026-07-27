#pragma once

#include <memory>

#include "inference/contracts/model_descriptor.hpp"
#include "inference/service/inference_session.hpp"
#include "inference/service/session_create.hpp"

namespace arcaine::inference {

// One loaded model implementation. Owns persistent resources: loaded weights,
// model configuration, tokenizer + chat-template resources, device
// assignment, persistent GPU allocations, kernel implementation choices, and
// model metadata.
class ModelService {
public:
    virtual ~ModelService() = default;

    virtual const ModelDescriptor& descriptor() const noexcept = 0;

    virtual std::unique_ptr<InferenceSession> create_session(
        const SessionCreateRequest& request) = 0;
};

}  // namespace arcaine::inference

#pragma once

#include <string>
#include <vector>

#include "inference/registry/model_registration.hpp"
#include "inference/service/loaded_model.hpp"

namespace arcaine::inference {

// Resolves config.json `model_type` strings to model implementations and
// constructs ModelService instances.
//
// The registry only matches strings and calls factories. It does not perform
// generation, own sampling, own a decode loop, inspect diffusion state,
// inspect KV cache state, or branch on InferenceContract.
class ModelRegistry {
public:
    // Register one model implementation. Duplicate model_type strings across
    // registrations are rejected by validate(), not here.
    void register_model(ModelRegistration registration);

    // Reject duplicate model_type strings. Throws std::runtime_error listing
    // the colliding types. Called once after all built-ins are registered.
    void validate();

    // Reads <request.model_dir>/config.json, extracts "model_type", and
    // dispatches to the matching factory. Throws if the type is unknown,
    // listing the registered model types.
    LoadedModel create(const ModelLoadRequest& request);

    std::vector<ModelDescriptor> registered_descriptors() const;
    std::vector<std::string>      registered_model_types() const;

private:
    std::vector<ModelRegistration> registrations_;
};

}  // namespace arcaine::inference

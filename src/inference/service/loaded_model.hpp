#pragma once

#include <memory>

#include "inference/contracts/model_descriptor.hpp"
#include "inference/contracts/model_load_request.hpp"
#include "inference/service/model_service.hpp"

namespace arcaine::inference {

// A model resolved + constructed by the registry, bundled with the descriptor
// and the load request that produced it. Applications hold one of these.
struct LoadedModel {
    std::unique_ptr<ModelService> service;
    ModelDescriptor               descriptor;
    ModelLoadRequest              load_request;
};

}  // namespace arcaine::inference

#pragma once

#include <functional>
#include <memory>

#include "inference/contracts/model_descriptor.hpp"
#include "inference/contracts/model_load_request.hpp"
#include "inference/service/model_service.hpp"

namespace arcaine::inference {

// Factory that constructs a ModelService from a ModelLoadRequest. Provided by
// each model module's registration.cpp.
using ModelServiceFactory =
    std::function<std::unique_ptr<ModelService>(const ModelLoadRequest&)>;

// One registered model implementation.
struct ModelRegistration {
    ModelDescriptor      descriptor;
    ModelServiceFactory  factory;
};

}  // namespace arcaine::inference

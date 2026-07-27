#include "modeling/gemma4_unified/registration.hpp"

#include "modeling/gemma4_unified/service.hpp"
#include "inference/registry/model_registry.hpp"
#include "inference/registry/model_registration.hpp"

#include <utility>

namespace arcaine::gemma4_unified {

void register_model(arcaine::inference::ModelRegistry& registry) {
    arcaine::inference::ModelRegistration reg;
    reg.descriptor = gemma4_descriptor();
    reg.factory = [](const arcaine::inference::ModelLoadRequest& req)
        -> std::unique_ptr<arcaine::inference::ModelService> {
        return std::make_unique<Gemma4Service>(req);
    };
    registry.register_model(std::move(reg));
}

}  // namespace arcaine::gemma4_unified

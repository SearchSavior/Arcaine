#include "modeling/diffusion_gemma/registration.hpp"

#include "modeling/diffusion_gemma/service.hpp"
#include "inference/registry/model_registry.hpp"
#include "inference/registry/model_registration.hpp"

#include <utility>

namespace arcaine::diffusion_gemma {

void register_model(arcaine::inference::ModelRegistry& registry) {
    arcaine::inference::ModelRegistration reg;
    reg.descriptor = diffusion_gemma_descriptor();
    reg.factory = [](const arcaine::inference::ModelLoadRequest& req)
        -> std::unique_ptr<arcaine::inference::ModelService> {
        return std::make_unique<DiffusionGemmaService>(req);
    };
    registry.register_model(std::move(reg));
}

}  // namespace arcaine::diffusion_gemma

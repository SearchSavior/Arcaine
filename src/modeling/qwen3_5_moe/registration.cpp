#include "modeling/qwen3_5_moe/registration.hpp"

#include "modeling/qwen3_5_moe/service.hpp"
#include "inference/registry/model_registry.hpp"
#include "inference/registry/model_registration.hpp"

#include <utility>

namespace arcaine::qwen3_5_moe {

void register_model(arcaine::inference::ModelRegistry& registry) {
    arcaine::inference::ModelRegistration reg;
    reg.descriptor = qwen_moe_descriptor();
    reg.factory = [](const arcaine::inference::ModelLoadRequest& req)
        -> std::unique_ptr<arcaine::inference::ModelService> {
        return std::make_unique<QwenMoeService>(req);
    };
    registry.register_model(std::move(reg));
}

}  // namespace arcaine::qwen3_5_moe

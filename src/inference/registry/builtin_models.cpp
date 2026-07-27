#include "inference/registry/builtin_models.hpp"

#include "inference/registry/model_registry.hpp"

#include "modeling/gemma4_unified/registration.hpp"
#include "modeling/qwen3_5/registration.hpp"
#include "modeling/qwen3_5_moe/registration.hpp"
#include "modeling/diffusion_gemma/registration.hpp"

namespace arcaine::inference {

void register_builtin_models(ModelRegistry& registry) {
#ifdef ARCAINE_MODEL_GEMMA4_UNIFIED
    arcaine::gemma4_unified::register_model(registry);
#endif

#ifdef ARCAINE_MODEL_QWEN3_5
    arcaine::qwen3_5::register_model(registry);
#endif

#ifdef ARCAINE_MODEL_QWEN3_5_MOE
    arcaine::qwen3_5_moe::register_model(registry);
#endif

#ifdef ARCAINE_MODEL_DIFFUSION_GEMMA
    arcaine::diffusion_gemma::register_model(registry);
#endif

    registry.validate();
}

}  // namespace arcaine::inference

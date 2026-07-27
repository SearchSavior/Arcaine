#pragma once

namespace arcaine::inference {
class ModelRegistry;
}

namespace arcaine::diffusion_gemma {

// Registers the diffusion_gemma implementation (model_type "diffusion_gemma")
// with the registry.
void register_model(arcaine::inference::ModelRegistry& registry);

}  // namespace arcaine::diffusion_gemma

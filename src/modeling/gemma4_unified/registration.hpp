#pragma once

namespace arcaine::inference {
class ModelRegistry;
}

namespace arcaine::gemma4_unified {

// Registers the gemma4_unified implementation (model_type "gemma4_unified")
// with the registry.
void register_model(arcaine::inference::ModelRegistry& registry);

}  // namespace arcaine::gemma4_unified

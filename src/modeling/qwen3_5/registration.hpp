#pragma once

namespace arcaine::inference {
class ModelRegistry;
}

namespace arcaine::qwen3_5 {

// Registers the qwen3_5 implementation (model_type "qwen3_5") with the registry.
void register_model(arcaine::inference::ModelRegistry& registry);

}  // namespace arcaine::qwen3_5

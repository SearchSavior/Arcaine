#pragma once

namespace arcaine::inference {
class ModelRegistry;
}

namespace arcaine::qwen3_5_moe {

// Registers the qwen3_5_moe implementation (model_type "qwen3_5_moe_text")
// with the registry.
void register_model(arcaine::inference::ModelRegistry& registry);

}  // namespace arcaine::qwen3_5_moe

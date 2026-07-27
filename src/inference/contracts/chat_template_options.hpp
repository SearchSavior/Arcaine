#pragma once

#include <nlohmann/json.hpp>

namespace arcaine::inference {

// Per-request chat-template overrides forwarded to the model's chat template
// renderer (minja). Empty object = use the template's defaults.
struct ChatTemplateOptions {
    nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
};

}  // namespace arcaine::inference

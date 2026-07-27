#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace arcaine::inference {

// Load-time request handed to a ModelServiceFactory. Placement is carried as
// a neutral string-keyed map so the contract does not depend on any model's
// placement option types; each owning model interprets the keys it recognizes.
struct ModelLoadRequest {
    std::string model_dir;
    int         max_seq_len = 2048;

    // Level Zero device selection spec (e.g. "0" or "" for all).
    std::string device_selection;

    // Neutral placement overrides, e.g. {"layers","split:16"}, {"experts","shard"}.
    // Each model interprets only the keys it owns; unknown keys are ignored.
    std::vector<std::pair<std::string, std::string>> placement_overrides;

    // Chat-template kwargs applied at template render time (load-time default;
    // per-request overrides win).
    nlohmann::ordered_json chat_template_kwargs = nlohmann::ordered_json::object();

    bool print_placement = true;
    bool verbose         = false;
};

}  // namespace arcaine::inference

#pragma once

#include <ctime>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "inference/contracts/model_load_request.hpp"
#include "inference/registry/model_registry.hpp"
#include "inference/service/loaded_model.hpp"
#include "utils/chat_template_kwargs.hpp"

namespace arcaine::server {

struct ServerOptions {
    std::string model_dir;
    std::string served_model_name;
    std::string host = "127.0.0.1";
    int port = 8000;
    int max_seq = 2048;
    int default_max_tokens = 256;
    int steps = -1;            // default denoising steps (-1 = model default)
    unsigned seed = 42;
    bool print_placement = true;
    bool debug = false;
    std::string debug_log_path = "arcaine_debug.log";
    // Neutral placement overrides (e.g. {"layers","split:16"}), interpreted by
    // the loaded model.
    std::vector<std::pair<std::string, std::string>> placement_overrides;
    nlohmann::ordered_json chat_template_kwargs = nlohmann::ordered_json::object();
};

struct AppState {
    ServerOptions                          opts;
    std::time_t                            created = std::time_t(std::time(nullptr));
    std::string                            api_key;
    arcaine::inference::ModelRegistry      registry;
    arcaine::inference::LoadedModel        model;
    std::mutex                             generate_mu;

    explicit AppState(ServerOptions o);
};

}  // namespace arcaine::server

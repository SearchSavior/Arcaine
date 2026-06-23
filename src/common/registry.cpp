#include "registry.hpp"
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry r;
    return r;
}

void ModelRegistry::register_arch(std::string model_type, Factory f) {
    factories_[std::move(model_type)] = std::move(f);
}

std::unique_ptr<Model> ModelRegistry::create(const std::string& model_dir,
                                             int max_seq_len) {
    std::ifstream f(model_dir + "/config.json");
    if (!f) throw std::runtime_error("Cannot open " + model_dir + "/config.json");
    auto j = nlohmann::json::parse(f);
    if (!j.contains("model_type"))
        throw std::runtime_error("config.json is missing \"model_type\"");

    std::string mt = j.at("model_type").get<std::string>();
    auto it = factories_.find(mt);
    if (it == factories_.end()) {
        std::string known;
        for (auto& kv : factories_) known += " " + kv.first;
        throw std::runtime_error(
            "No model architecture registered for model_type='" + mt +
            "'. Registered:" + (known.empty() ? " (none)" : known));
    }
    return it->second(model_dir, max_seq_len);
}

std::vector<std::string> ModelRegistry::registered() const {
    std::vector<std::string> out;
    out.reserve(factories_.size());
    for (auto& kv : factories_) out.push_back(kv.first);
    return out;
}

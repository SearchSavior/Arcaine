#include "inference/registry/model_registry.hpp"

#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace arcaine::inference {

void ModelRegistry::register_model(ModelRegistration registration) {
    registrations_.push_back(std::move(registration));
}

void ModelRegistry::validate() {
    std::set<std::string> seen;
    std::string duplicate;
    for (const auto& reg : registrations_) {
        for (const auto& mt : reg.descriptor.model_types) {
            if (!seen.insert(mt).second) {
                duplicate = mt;
                break;
            }
        }
        if (!duplicate.empty()) break;
    }
    if (!duplicate.empty()) {
        throw std::runtime_error(
            "Duplicate model_type '" + duplicate +
            "' registered by more than one model implementation.");
    }
}

LoadedModel ModelRegistry::create(const ModelLoadRequest& request) {
    const std::string cfg_path = request.model_dir + "/config.json";
    std::ifstream f(cfg_path);
    if (!f) {
        throw std::runtime_error("Cannot open " + cfg_path);
    }
    auto j = nlohmann::json::parse(f);
    if (!j.contains("model_type")) {
        throw std::runtime_error("config.json is missing \"model_type\"");
    }
    const std::string mt = j.at("model_type").get<std::string>();

    for (const auto& reg : registrations_) {
        for (const auto& candidate : reg.descriptor.model_types) {
            if (candidate == mt) {
                auto service = reg.factory(request);
                LoadedModel loaded;
                loaded.descriptor  = reg.descriptor;
                loaded.load_request = request;
                loaded.service      = std::move(service);
                return loaded;
            }
        }
    }

    std::string known;
    for (const auto& mt : registered_model_types()) {
        if (!known.empty()) known += ", ";
        known += mt;
    }
    throw std::runtime_error(
        "Model type \"" + mt +
        "\" is not compiled into this Arcaine build. Registered model types: " +
        (known.empty() ? "(none)" : known));
}

std::vector<ModelDescriptor> ModelRegistry::registered_descriptors() const {
    std::vector<ModelDescriptor> out;
    out.reserve(registrations_.size());
    for (const auto& reg : registrations_) out.push_back(reg.descriptor);
    return out;
}

std::vector<std::string> ModelRegistry::registered_model_types() const {
    std::vector<std::string> out;
    for (const auto& reg : registrations_) {
        for (const auto& mt : reg.descriptor.model_types) out.push_back(mt);
    }
    return out;
}

}  // namespace arcaine::inference

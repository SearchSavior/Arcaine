#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "model_interface.hpp"

// ---------------------------------------------------------------------------
// Maps a config.json `model_type` string to a factory that builds the matching
// Model. Architectures register themselves through register_arch(); the set of
// built-ins is wired up by register_builtin_architectures() (see
// src/register_builtins.cpp), which a frontend calls once at startup.
// ---------------------------------------------------------------------------
class ModelRegistry {
public:
    using Factory = std::function<std::unique_ptr<Model>(
        const std::string& model_dir, int max_seq_len)>;

    static ModelRegistry& instance();

    void register_arch(std::string model_type, Factory f);

    // Reads <model_dir>/config.json, extracts "model_type", and dispatches to
    // the registered factory. Throws if the type is unknown.
    std::unique_ptr<Model> create(const std::string& model_dir, int max_seq_len);

    std::vector<std::string> registered() const;

private:
    std::unordered_map<std::string, Factory> factories_;
};

// Registers every architecture compiled into this binary. Defined in
// src/register_builtins.cpp (outside common/, so common/ stays arch-agnostic).
void register_builtin_architectures();

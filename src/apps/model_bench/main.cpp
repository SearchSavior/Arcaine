// src/apps/model_bench/main.cpp
//
// Central model-benchmark dispatcher. Resolves config.json::model_type from the
// -m/--model dir and forwards argv to the model-owned benchmark registered for
// that type (modeling/<m>/benchmarks/model_bench.cpp via REGISTER_MODEL_BENCH).
// The dispatcher owns no model-specific benchmark logic — each model's bench
// drives its own concrete engine (forward/reset_cache/generate) and prints its
// own usage (see the llama-bench-style help text in benchmarks/model_bench_util.hpp).
//
//   ./build/arcaine_mbench -m <dir> [model-owned opts passed through]
//
// A model type with no registered bench means its model_bench.cpp was not
// compiled into this build (the model was disabled via ARCAINE_MODELS).

#include "benchmarks/model_bench_registry.hpp"
#include "benchmarks/model_bench_util.hpp"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

static std::string resolve_model_type(const std::string& model_dir) {
    const std::string cfg_path = model_dir + "/config.json";
    std::ifstream f(cfg_path);
    if (!f) throw std::runtime_error("Cannot open " + cfg_path);
    auto j = nlohmann::json::parse(f);
    if (!j.contains("model_type"))
        throw std::runtime_error("config.json is missing \"model_type\"");
    return j.at("model_type").get<std::string>();
}

static std::string find_model_arg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" || a == "--model") {
            if (i + 1 < argc) return argv[i + 1];
            return std::string();  // missing value; reported by the bench
        }
    }
    return std::string();
}

int main(int argc, char** argv) {
    const std::string model_dir = find_model_arg(argc, argv);
    if (model_dir.empty()) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-h" || a == "--help") {
                arcaine::bench::print_mbench_usage(argc > 0 ? argv[0] : "arcaine_mbench", nullptr);
                return 0;
            }
        }
        std::fprintf(stderr,
            "Usage: arcaine_mbench -m, --model <dir> [options]\n"
            "The model benchmark owns the full flag set; run\n"
            "  arcaine_mbench -m <dir> --help\n"
            "to see the llama-bench-style options for that model.\n");
        return 2;
    }

    std::string model_type;
    try {
        model_type = resolve_model_type(model_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const auto* bench = arcaine::bench::ModelBenchRegistry::get().find(model_type);
    if (!bench) {
        std::string known;
        for (const auto& b : arcaine::bench::ModelBenchRegistry::get().all()) {
            if (!known.empty()) known += ", ";
            known += b.model_type;
        }
        std::fprintf(stderr,
            "No model benchmark registered for model_type \"%s\". It is not\n"
            "compiled into this Arcaine build. Registered model benchmarks: %s\n",
            model_type.c_str(),
            known.empty() ? "(none)" : known.c_str());
        return 2;
    }
    // Forward the FULL argv: the model-owned bench re-parses -m/--model and its
    // own flags.
    return bench->run(argc, argv);
}

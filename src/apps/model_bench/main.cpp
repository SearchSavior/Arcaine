// src/apps/model_bench/main.cpp
//
// Central model-benchmark dispatcher. Resolves config.json::model_type from the
// model dir and forwards argv to the model-owned benchmark registered for that
// type (modeling/<m>/benchmarks/model_bench.cpp via REGISTER_MODEL_BENCH). The
// dispatcher owns no model-specific benchmark logic and no cache semantics —
// each model's bench drives its own concrete engine (forward/reset_cache/
// generate) and defines its own flags.
//
//   ./build/arcaine_mbench --list
//   ./build/arcaine_mbench --model <dir> [model-owned opts passed through]
//
// A model type with no registered bench means its model_bench.cpp was not
// compiled into this build (the model was disabled via ARCAINE_MODELS).

#include "benchmarks/model_bench_registry.hpp"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

static int print_list() {
    const auto& regs = arcaine::bench::ModelBenchRegistry::get().all();
    if (regs.empty()) {
        std::printf("(no model benchmarks registered — no models enabled in this build)\n");
        return 0;
    }
    std::printf("%-22s %s\n", "model_type", "description");
    std::printf("%s\n", std::string(70, '-').c_str());
    for (const auto& b : regs)
        std::printf("%-22s %s\n", b.model_type, b.description);
    return 0;
}

static std::string resolve_model_type(const std::string& model_dir) {
    const std::string cfg_path = model_dir + "/config.json";
    std::ifstream f(cfg_path);
    if (!f) throw std::runtime_error("Cannot open " + cfg_path);
    auto j = nlohmann::json::parse(f);
    if (!j.contains("model_type"))
        throw std::runtime_error("config.json is missing \"model_type\"");
    return j.at("model_type").get<std::string>();
}

int main(int argc, char** argv) {
    std::string model_dir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--list") return print_list();
        if (a == "-h" || a == "--help") {
            std::printf(
                "Usage: arcaine_mbench --model <dir> [model-owned options]\n"
                "       arcaine_mbench --list\n\n"
                "Resolves config.json::model_type from <dir> and forwards argv to\n"
                "the model-owned benchmark (see arcaine_mbench --list). Each model\n"
                "defines its own flags; run 'arcaine_mbench --model <dir> --help' for\n"
                "the per-model usage.\n");
            return 0;
        }
        if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
    }
    if (model_dir.empty()) {
        std::fprintf(stderr,
            "Usage: arcaine_mbench --model <dir> [opts]\n"
            "Run 'arcaine_mbench --list' to see registered model benchmarks.\n");
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
    // Forward the FULL argv: the model-owned bench re-parses --model and its
    // own flags.
    return bench->run(argc, argv);
}

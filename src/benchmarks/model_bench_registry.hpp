#pragma once
//
// Model-benchmark registry. Each model module registers a model-owned
// benchmark entry (modeling/<m>/benchmarks/model_bench.cpp) that drives the
// module's private engine/cache APIs directly. The central arcaine_mbench
// binary resolves config.json::model_type to one of these entries and forwards
// argv; it owns no model-specific benchmark logic.
//
//   REGISTER_MODEL_BENCH("<model_type>", "<description>", run)   // at file scope
//   ModelBenchRegistry::get().find("<model_type>")               // -> const ModelBench*
//   ModelBenchRegistry::get().all()                              // registered entries
//

#include <cstring>
#include <string>
#include <vector>

namespace arcaine::bench {

struct ModelBench {
    const char* model_type;     // matches config.json::model_type
    const char* description;
    int (*run)(int argc, char** argv);
};

class ModelBenchRegistry {
public:
    static ModelBenchRegistry& get() {
        static ModelBenchRegistry r;
        return r;
    }
    void add(ModelBench b) { benches_.push_back(b); }
    const std::vector<ModelBench>& all() const { return benches_; }

    // Returns the entry whose model_type matches, or nullptr if this model is
    // not compiled into the build (its model_bench.cpp was not linked).
    const ModelBench* find(const std::string& model_type) const {
        for (const auto& b : benches_)
            if (b.model_type == model_type) return &b;
        return nullptr;
    }

private:
    std::vector<ModelBench> benches_;
};

struct ModelBenchRegistrar {
    ModelBenchRegistrar(const char* model_type, const char* desc, int (*fn)(int, char**)) {
        ModelBenchRegistry::get().add({model_type, desc, fn});
    }
};

}  // namespace arcaine::bench

#define REGISTER_MODEL_BENCH(model_type, desc, fn) \
    static ::arcaine::bench::ModelBenchRegistrar \
        arcaine_model_bench_registrar_##__COUNTER__(model_type, desc, fn);

// Verifies register_builtin_models wires every enabled model implementation
// through the same ModelRegistry. GPU-free (registers only; no generation).
// Built only when all four model targets exist (the all-models build).
#include "check.hpp"

#include "inference/registry/builtin_models.hpp"
#include "inference/registry/model_registry.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace arcaine::inference;

int main() {
    ModelRegistry registry;
    register_builtin_models(registry);

    auto descs = registry.registered_descriptors();
    std::vector<std::string> types;
    for (const auto& d : descs)
        for (const auto& t : d.model_types) types.push_back(t);

    // All four built-in model types must be registered through the same path.
    auto has = [&](const std::string& t) {
        return std::find(types.begin(), types.end(), t) != types.end();
    };
    CHECK(has("gemma4_unified"));
    CHECK(has("qwen3_5"));
    CHECK(has("qwen3_5_moe_text"));
    CHECK(has("diffusion_gemma"));
    CHECK_EQ(types.size(), 4u);

    // The diffusion descriptor carries the block-diffusion contract + draft
    // output capability; causal models carry the causal contract.
    for (const auto& d : descs) {
        if (d.model_types[0] == "diffusion_gemma") {
            CHECK(d.inference_contract == InferenceContract::BlockDiffusionGenerationV1);
            CHECK(has_output_capability(d.output_capabilities, OutputCapability::DraftTokens));
        } else {
            CHECK(d.inference_contract == InferenceContract::CausalGenerationV1);
        }
        CHECK(d.supports_streaming);
    }

    // Duplicate registration must be rejected (validate() already ran inside
    // register_builtin_models; re-registering a causal model and validating
    // must throw).
    ModelRegistry dup;
    register_builtin_models(dup);  // ok (fresh)
    bool threw = false;
    try {
        // Manually re-add one and validate to confirm duplicate detection.
        ModelRegistration r;
        r.descriptor.model_types = {"gemma4_unified"};
        r.descriptor.implementation_id = "gemma4_dup";
        r.factory = [](const ModelLoadRequest&) { return nullptr; };
        dup.register_model(std::move(r));
        dup.validate();
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);

    RETURN_TESTS();
}

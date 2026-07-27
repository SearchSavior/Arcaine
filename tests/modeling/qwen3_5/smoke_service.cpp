// GPU smoke test for the qwen3_5 ModelService. Loads through the new
// ModelRegistry and runs greedy + tool-call generations through ModelService
// + InferenceSession. Not a ctest target (needs a GPU + model dir); run:
//   build/tests/arcaine_smoke_qwen35 /workspace/models/unsloth_Qwen3.6-27B-NVFP4
#include "collecting_sink.hpp"

#include "modeling/qwen3_5/registration.hpp"
#include "inference/contracts/generation_request.hpp"
#include "inference/registry/model_registry.hpp"
#include "inference/service/session_create.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model_dir>\n", argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];

    arcaine::inference::ModelRegistry registry;
    arcaine::qwen3_5::register_model(registry);
    registry.validate();

    arcaine::inference::ModelLoadRequest load;
    load.model_dir   = model_dir;
    load.max_seq_len = 4096;
    auto loaded = registry.create(load);
    auto session = loaded.service->create_session({});

    arcaine::inference::GenerationRequest req;
    req.served_model_name = "qwen3.5";
    req.messages = nlohmann::ordered_json::array();
    req.messages.push_back({{"role", "user"},
                            {"content", "What is the capital of France? Answer with one word."}});
    req.max_output_tokens    = 32;
    req.seed                 = 42;
    req.sampling.temperature = 0.0f;

    arcaine::test::CollectingGenerationSink sink;
    arcaine::inference::GenerationResult result = session->generate(req, sink);
    std::printf("finish_reason=%s\n", result.finish_reason.c_str());
    std::printf("text=%s\n", result.text.c_str());
    std::printf("usage: prompt=%d completion=%d\n",
                result.usage.prompt_tokens, result.usage.completion_tokens);
    const bool ok = (result.text.find("Paris") != std::string::npos) &&
                    result.finish_reason == "stop" &&
                    result.usage.completion_tokens > 0;
    std::printf("%s\n", ok ? "SMOKE_OK" : "SMOKE_FAIL");
    return ok ? 0 : 1;
}

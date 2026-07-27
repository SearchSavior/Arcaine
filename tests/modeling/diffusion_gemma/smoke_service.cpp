// GPU smoke test for the diffusion_gemma ModelService. Loads through the new
// ModelRegistry (same path as all other models) and runs one block-diffusion
// generation through ModelService + InferenceSession. Not a ctest target (needs
// a GPU + model dir); run:
//   build/tests/arcaine_smoke_diffusion /workspace/models/diffusiongemma-26B-A4B-it
#include "collecting_sink.hpp"

#include "modeling/diffusion_gemma/registration.hpp"
#include "inference/contracts/generation_event.hpp"
#include "inference/contracts/generation_request.hpp"
#include "inference/registry/model_registry.hpp"
#include "inference/service/session_create.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model_dir>\n", argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];

    arcaine::inference::ModelRegistry registry;
    arcaine::diffusion_gemma::register_model(registry);
    registry.validate();

    arcaine::inference::ModelLoadRequest load;
    load.model_dir   = model_dir;
    load.max_seq_len = 4096;
    auto loaded = registry.create(load);
    auto session = loaded.service->create_session({});

    arcaine::inference::GenerationRequest req;
    req.served_model_name = "diffusion_gemma";
    req.prompt_text = "The capital of France is";
    req.max_output_tokens = 64;
    req.seed              = 42;
    req.denoising_steps   = -1;  // model default (48)

    arcaine::test::CollectingGenerationSink sink;
    arcaine::inference::GenerationResult result = session->generate(req, sink);
    std::printf("finish_reason=%s\n", result.finish_reason.c_str());
    std::printf("text=%s\n", result.text.c_str());
    std::printf("usage: prompt=%d completion=%d\n",
                result.usage.prompt_tokens, result.usage.completion_tokens);
    std::printf("metrics: input_token=%d new_token=%d ttft=%.4f decode_tps=%.2f duration=%.4f\n",
                result.metrics.input_token, result.metrics.new_token,
                result.metrics.ttft, result.metrics.decode_throuput, result.metrics.duration);
    const bool ok = (result.text.find("Paris") != std::string::npos) &&
                    result.usage.completion_tokens > 0;
    std::printf("%s\n", ok ? "SMOKE_OK" : "SMOKE_FAIL");
    return ok ? 0 : 1;
}

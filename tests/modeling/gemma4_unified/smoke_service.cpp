// GPU smoke test for the gemma4_unified ModelService. Loads the model through
// the new ModelRegistry, runs one greedy generation through ModelService +
// InferenceSession, and checks the output. Not a ctest target (needs a GPU +
// model dir); run manually:
//   ./build/arcaine_smoke_gemma4 /workspace/models/models/gemma-4-12B-it
#include "collecting_sink.hpp"

#include "modeling/gemma4_unified/registration.hpp"
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
    arcaine::gemma4_unified::register_model(registry);
    registry.validate();

    arcaine::inference::ModelLoadRequest load;
    load.model_dir   = model_dir;
    load.max_seq_len = 2048;
    auto loaded = registry.create(load);

    auto session = loaded.service->create_session({});

    arcaine::inference::GenerationRequest req;
    req.served_model_name = "gemma4";
    req.messages = nlohmann::ordered_json::array();
    req.messages.push_back({{"role", "user"},
                            {"content", "What is the capital of France? Answer with one word."}});
    req.max_output_tokens   = 32;
    req.seed                = 42;
    req.sampling.temperature = 0.0f;  // greedy

    arcaine::test::CollectingGenerationSink sink;
    arcaine::inference::GenerationResult result = session->generate(req, sink);

    std::printf("finish_reason=%s\n", result.finish_reason.c_str());
    std::printf("text=%s\n", result.text.c_str());
    std::printf("usage: prompt=%d completion=%d\n",
                result.usage.prompt_tokens, result.usage.completion_tokens);
    std::printf("metrics: input_token=%d new_token=%d ttft=%.4f tpot=%.4f "
                "prefill_tps=%.2f decode_tps=%.2f duration=%.4f\n",
                result.metrics.input_token, result.metrics.new_token,
                result.metrics.ttft, result.metrics.tpot,
                result.metrics.prefill_throughput, result.metrics.decode_throuput,
                result.metrics.duration);

    const bool ok = (result.text.find("Paris") != std::string::npos) &&
                    result.finish_reason == "stop" &&
                    result.usage.completion_tokens > 0;
    std::printf("%s\n", ok ? "SMOKE_OK" : "SMOKE_FAIL");

    // ---- Tool-call request ----
    arcaine::inference::GenerationRequest treq;
    treq.served_model_name = "gemma4";
    treq.messages = nlohmann::ordered_json::array();
    treq.messages.push_back({{"role", "user"},
                            {"content", "Use get_weather for Paris. Do not answer directly."}});
    treq.tools = nlohmann::ordered_json::array();
    treq.tools.push_back({
        {"type", "function"},
        {"function", {
            {"name", "get_weather"},
            {"description", "Get the current weather for a city."},
            {"parameters", {
                {"type", "object"},
                {"properties", {{"city", {{"type", "string"}}}}},
                {"required", nlohmann::ordered_json::array({"city"})},
            }},
        }},
    });
    treq.max_output_tokens   = 128;
    treq.seed                = 42;
    treq.sampling.temperature = 0.0f;

    arcaine::test::CollectingGenerationSink tsink;
    arcaine::inference::GenerationResult tresult = session->generate(treq, tsink);
    std::printf("tool finish_reason=%s tool_calls=%zu content=%s\n",
                tresult.finish_reason.c_str(), tresult.tool_calls.size(),
                tresult.text.c_str());
    for (const auto& tc : tresult.tool_calls)
        std::printf("  tool_call: name=%s args=%s\n", tc.name.c_str(), tc.arguments.c_str());

    const bool tok = !tresult.tool_calls.empty() &&
                     tresult.finish_reason == "tool_calls" &&
                     tresult.text.find("<|tool_call>") == std::string::npos;
    std::printf("%s\n", tok ? "TOOL_OK" : "TOOL_FAIL");
    return (ok && tok) ? 0 : 1;
}

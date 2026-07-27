#include "check.hpp"

#include "inference/contracts/model_descriptor.hpp"
#include "inference/contracts/generation_event.hpp"
#include "inference/contracts/generation_request.hpp"
#include "inference/contracts/generation_result.hpp"

#include <variant>

using namespace arcaine::inference;

int main() {
    ModelDescriptor d;
    d.model_types           = {"qwen3_5_moe_text"};
    d.implementation_id     = "qwen3_5_moe";
    d.inference_contract     = InferenceContract::BlockDiffusionGenerationV1;
    d.input_capabilities    = static_cast<std::uint32_t>(
        InputCapability::Messages | InputCapability::Images);
    d.output_capabilities   = static_cast<std::uint32_t>(
        OutputCapability::Text | OutputCapability::ToolCalls);
    d.supports_streaming    = true;

    CHECK_EQ(d.model_types.size(), 1u);
    CHECK_EQ(d.model_types[0], "qwen3_5_moe_text");
    CHECK(d.inference_contract == InferenceContract::BlockDiffusionGenerationV1);
    CHECK(d.supports_streaming);
    CHECK(has_input_capability(d.input_capabilities, InputCapability::Images));
    CHECK(has_output_capability(d.output_capabilities, OutputCapability::ToolCalls));

    // Event variant behavior.
    GenerationEvent ev = StartedEvent{};
    CHECK(std::holds_alternative<StartedEvent>(ev));

    ev = TokenEvent{42};
    CHECK_EQ(std::get<TokenEvent>(ev).token_id, 42);

    ev = TextDeltaEvent{"hello"};
    CHECK_EQ(std::get<TextDeltaEvent>(ev).delta, "hello");

    ev = ToolCallDeltaEvent{0, "call_1", "get_weather", "{\"city\":"};
    auto& tc = std::get<ToolCallDeltaEvent>(ev);
    CHECK_EQ(tc.index, 0);
    CHECK_EQ(tc.name, "get_weather");

    ev = DraftEvent{2, 5, 0.7f, 0.1f, {1, 2, 3}, "draft"};
    auto& de = std::get<DraftEvent>(ev);
    CHECK_EQ(de.block, 2);
    CHECK_EQ(de.step, 5);
    CHECK_EQ(de.draft_token_ids.size(), 3u);
    CHECK_EQ(de.draft_text, "draft");

    ev = CompletedEvent{"tool_calls", 10, 3};
    CHECK_EQ(std::get<CompletedEvent>(ev).finish_reason, "tool_calls");

    // Request + result construction roundtrip.
    GenerationRequest req;
    req.request_id       = "abc";
    req.served_model_name = "qwen";
    req.max_output_tokens = 128;
    req.seed              = 7;
    req.denoising_steps   = -1;
    CHECK_EQ(req.max_output_tokens, 128);
    CHECK(req.sampling.overrides_temperature() == false);

    GenerationResult res;
    res.output_token_ids = {1, 2, 3};
    res.text             = "hi";
    res.finish_reason    = "stop";
    res.usage.prompt_tokens = 10;
    res.metrics.input_token  = 10;
    CHECK_EQ(res.output_token_ids.size(), 3u);
    CHECK_EQ(res.finish_reason, "stop");
    CHECK_EQ(res.metrics.decode_throuput, 0.0);  // spelling preserved

    RETURN_TESTS();
}

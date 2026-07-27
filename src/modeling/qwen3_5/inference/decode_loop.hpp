#pragma once

#include <vector>

#include "modeling/qwen3_5/invocation.hpp"
#include "modeling/qwen3_5/output_parser.hpp"
#include "inference/contracts/generation_metrics.hpp"
#include "inference/service/cancellation_token.hpp"
#include "inference/service/generation_sink.hpp"

class Qwen35Model;      // global (modeling/qwen3_5/model.hpp)
class TokenizerBridge;  // global (utils/chat.hpp)

namespace arcaine::qwen3_5 {

struct Qwen35DecodeResult {
    std::vector<int>      generated_ids;
    Qwen35AssistantOutput parsed;
    double                ttft_s     = 0.0;
    double                duration_s = 0.0;
    double                prefill_s  = 0.0;
    double                decode_s   = 0.0;
};

// The model-owned autoregressive decode loop: prefill, per-step sample, EOS
// handling, streaming TextDelta emission, cache reset, and tool parsing.
// Independent of any other model's decode loop.
Qwen35DecodeResult run_decode_loop(Qwen35Model& model,
                                   const Qwen35Invocation& inv,
                                   TokenizerBridge& tokenizer,
                                   arcaine::inference::GenerationSink& sink,
                                   const arcaine::inference::CancellationToken* cancel);

}  // namespace arcaine::qwen3_5

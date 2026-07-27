#pragma once

#include <vector>

#include "modeling/gemma4_unified/invocation.hpp"
#include "modeling/gemma4_unified/output_parser.hpp"
#include "inference/contracts/generation_metrics.hpp"
#include "inference/service/cancellation_token.hpp"
#include "inference/service/generation_sink.hpp"

class Gemma4Model;      // global (modeling/gemma4_unified/model.hpp)
class TokenizerBridge;  // global (utils/chat.hpp)

namespace arcaine::gemma4_unified {

struct Gemma4DecodeResult {
    std::vector<int>         generated_ids;
    Gemma4AssistantOutput    parsed;
    Gemma4BoundaryCounts     boundary_counts;
    double                   ttft_s     = 0.0;
    double                   duration_s = 0.0;
    double                   prefill_s  = 0.0;
    double                   decode_s   = 0.0;
};

// The model-owned autoregressive decode loop: prefill, per-step sample, EOS
// handling, streaming TextDelta emission, KV reset, and tool/boundary parsing.
// Returns the generated ids, parsed output, boundary counts, and timings.
Gemma4DecodeResult run_decode_loop(Gemma4Model& model,
                                   const Gemma4Invocation& inv,
                                   TokenizerBridge& tokenizer,
                                   const Gemma4BoundaryParser& boundaries,
                                   inference::GenerationSink& sink,
                                   const inference::CancellationToken* cancel);

}  // namespace arcaine::gemma4_unified

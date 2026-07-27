#pragma once

#include <vector>

#include "modeling/qwen3_5_moe/invocation.hpp"
#include "modeling/qwen3_5_moe/output_parser.hpp"
#include "inference/contracts/generation_metrics.hpp"
#include "inference/service/cancellation_token.hpp"
#include "inference/service/generation_sink.hpp"

class QwenModel;      // global (modeling/qwen3_5_moe/model.hpp)
class TokenizerBridge;  // global (utils/chat.hpp)

namespace arcaine::qwen3_5_moe {

struct QwenMoeDecodeResult {
    std::vector<int>        generated_ids;
    QwenMoeAssistantOutput  parsed;
    double                  ttft_s     = 0.0;
    double                  duration_s = 0.0;
    double                  prefill_s  = 0.0;
    double                  decode_s   = 0.0;
};

// The model-owned autoregressive decode loop: prefill, per-step sample, EOS
// handling, streaming TextDelta emission, cache reset, and tool parsing.
// Independent of any other model's decode loop; owns its MoE routing and
// expert execution via the architecture engine.
QwenMoeDecodeResult run_decode_loop(QwenModel& model,
                                    const QwenMoeInvocation& inv,
                                    TokenizerBridge& tokenizer,
                                    arcaine::inference::GenerationSink& sink,
                                    const arcaine::inference::CancellationToken* cancel);

}  // namespace arcaine::qwen3_5_moe

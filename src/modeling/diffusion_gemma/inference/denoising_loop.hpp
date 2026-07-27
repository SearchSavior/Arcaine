#pragma once

#include <vector>

#include "modeling/diffusion_gemma/invocation.hpp"
#include "modeling/diffusion_gemma/output_parser.hpp"
#include "inference/contracts/generation_metrics.hpp"
#include "inference/service/cancellation_token.hpp"
#include "inference/service/generation_sink.hpp"

class DiffusionGemmaModel;  // global (modeling/diffusion_gemma/model.hpp)
class TokenizerBridge;      // global (utils/chat.hpp)

namespace arcaine::diffusion_gemma {

struct DiffusionGemmaDecodeResult {
    std::vector<int>               generated_ids;
    DiffusionGemmaAssistantOutput  parsed;
    DiffusionGemmaBoundaryCounts   boundary_counts;
    double                         ttft_s     = 0.0;
    double                         duration_s = 0.0;
    double                         prefill_s  = 0.0;
    double                         decode_s   = 0.0;
    int                            prefill_tokens = 0;
};

// The model-owned block-diffusion denoising loop: encoder pass, per-step
// denoising, draft/commit event emission through the sink, EOS-bound
// committing, sampling, entropy/acceptance, and metrics. The architecture
// engine (DiffusionGemmaModel) owns the denoiser forward + expert execution;
// this wrapper owns the generation orchestration + event mapping.
DiffusionGemmaDecodeResult run_denoising_loop(
    DiffusionGemmaModel& model,
    const DiffusionGemmaInvocation& inv,
    TokenizerBridge& tokenizer,
    const DiffusionGemmaBoundaryParser& boundaries,
    arcaine::inference::GenerationSink& sink,
    const arcaine::inference::CancellationToken* cancel);

}  // namespace arcaine::diffusion_gemma

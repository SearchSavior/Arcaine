#include "modeling/gemma4_unified/session.hpp"

#include "modeling/gemma4_unified/service.hpp"
#include "modeling/gemma4_unified/model.hpp"
#include "modeling/gemma4_unified/output_parser.hpp"
#include "modeling/gemma4_unified/request_mapper.hpp"
#include "modeling/gemma4_unified/inference/decode_loop.hpp"
#include "modeling/gemma4_unified/inference/cache.hpp"
#include "utils/chat.hpp"
#include "inference/contracts/generation_event.hpp"
#include "inference/contracts/generation_result.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

namespace arcaine::gemma4_unified {

namespace {
double round4(double v) {
    if (!std::isfinite(v)) return 0.0;
    return std::round(v * 10000.0) / 10000.0;
}
std::string finish_reason_for(size_t generated, int max_tokens) {
    return generated >= static_cast<size_t>(max_tokens) ? "length" : "stop";
}
}  // namespace

Gemma4Session::Gemma4Session(Gemma4Service& service,
                             const inference::SessionCreateRequest& req)
    : service_(service), cancel_(req.cancellation) {}

inference::GenerationResult Gemma4Session::generate(
    const inference::GenerationRequest& request, inference::GenerationSink& sink) {
    if (request.cancellation) cancel_ = request.cancellation;

    Gemma4Invocation inv = map_request(request, service_.model(),
                                      service_.tokenizer(), service_.model().info());

    if (static_cast<int>(inv.input_ids.size()) + inv.max_output_tokens >
        service_.model().info().max_seq_len) {
        throw std::runtime_error("prompt_tokens + max_tokens exceeds model max_seq");
    }

    sink.emit(inference::StartedEvent{});

    Gemma4Cache cache(service_.model());
    cache.reset();

    Gemma4DecodeResult r = run_decode_loop(service_.model(), inv,
                                           service_.tokenizer(),
                                           service_.boundary_parser(),
                                            sink, cancel_);

    inference::GenerationResult out;
    out.output_token_ids = std::move(r.generated_ids);
    out.text             = std::move(r.parsed.content);
    out.tool_calls       = std::move(r.parsed.tool_calls);
    out.finish_reason    = !out.tool_calls.empty()
                              ? "tool_calls"
                              : finish_reason_for(out.output_token_ids.size(),
                                                  inv.max_output_tokens);
    out.usage.prompt_tokens     = static_cast<int>(inv.input_ids.size());
    out.usage.completion_tokens = static_cast<int>(out.output_token_ids.size());

    const int in_t  = static_cast<int>(inv.input_ids.size());
    const int new_t = static_cast<int>(out.output_token_ids.size());
    out.metrics.input_token        = in_t;
    out.metrics.new_token          = new_t;
    out.metrics.ttft               = round4(new_t > 0 ? r.ttft_s : 0.0);
    out.metrics.tpot               = round4(new_t > 1
        ? ((r.duration_s - r.ttft_s) * 1000.0) / (new_t - 1) : 0.0);
    out.metrics.prefill_throughput = round4(r.prefill_s > 0 ? in_t / r.prefill_s : 0.0);
    out.metrics.decode_throuput    = round4(r.decode_s > 0 ? new_t / r.decode_s : 0.0);
    out.metrics.duration           = round4(r.duration_s);

    sink.emit(inference::MetricsEvent{out.metrics});
    sink.emit(inference::CompletedEvent{out.finish_reason,
                                        out.usage.prompt_tokens,
                                        out.usage.completion_tokens});
    return out;
}

void Gemma4Session::cancel() {
    if (cancel_) cancel_->set();
}

}  // namespace arcaine::gemma4_unified

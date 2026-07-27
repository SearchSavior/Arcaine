#include "modeling/qwen3_5/session.hpp"

#include "modeling/qwen3_5/service.hpp"
#include "modeling/qwen3_5/model.hpp"
#include "modeling/qwen3_5/output_parser.hpp"
#include "modeling/qwen3_5/request_mapper.hpp"
#include "modeling/qwen3_5/inference/decode_loop.hpp"
#include "modeling/qwen3_5/inference/cache.hpp"
#include "utils/chat.hpp"
#include "inference/contracts/generation_event.hpp"
#include "inference/contracts/generation_result.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

namespace arcaine::qwen3_5 {

namespace {
double round4(double v) {
    if (!std::isfinite(v)) return 0.0;
    return std::round(v * 10000.0) / 10000.0;
}
std::string finish_reason_for(std::size_t generated, int max_tokens) {
    return generated >= static_cast<std::size_t>(max_tokens) ? "length" : "stop";
}
}  // namespace

Qwen35Session::Qwen35Session(Qwen35Service& service,
                             const arcaine::inference::SessionCreateRequest& req)
    : service_(service), cancel_(req.cancellation) {}

arcaine::inference::GenerationResult Qwen35Session::generate(
    const arcaine::inference::GenerationRequest& request,
    arcaine::inference::GenerationSink& sink) {
    if (request.cancellation) cancel_ = request.cancellation;

    Qwen35Invocation inv = map_request(request, service_.model(),
                                       service_.tokenizer(), service_.model().info());

    if (static_cast<int>(inv.input_ids.size()) + inv.max_output_tokens >
        service_.model().info().max_seq_len) {
        throw std::runtime_error("prompt_tokens + max_tokens exceeds model max_seq");
    }

    sink.emit(arcaine::inference::StartedEvent{});

    Qwen35Cache cache(service_.model());
    cache.reset();

    Qwen35DecodeResult r = run_decode_loop(service_.model(), inv,
                                           service_.tokenizer(),
                                           sink, cancel_);

    // Streaming with tools: emit the buffered output as a final delta.
    bool ok = true;
    if (inv.stream && inv.has_tools) {
        if (!r.parsed.tool_calls.empty()) {
            for (size_t i = 0; i < r.parsed.tool_calls.size() && ok; ++i) {
                arcaine::inference::ToolCallDeltaEvent tc;
                tc.index           = static_cast<int>(i);
                tc.id              = r.parsed.tool_calls[i].id;
                tc.name            = r.parsed.tool_calls[i].name;
                tc.arguments_delta = r.parsed.tool_calls[i].arguments;
                if (!sink.emit(std::move(tc))) ok = false;
            }
        } else if (!r.parsed.content.empty()) {
            if (!sink.emit(arcaine::inference::TextDeltaEvent{r.parsed.content})) ok = false;
        }
    }

    arcaine::inference::GenerationResult out;
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

    sink.emit(arcaine::inference::MetricsEvent{out.metrics});
    sink.emit(arcaine::inference::CompletedEvent{out.finish_reason,
                                                 out.usage.prompt_tokens,
                                                 out.usage.completion_tokens});
    return out;
}

void Qwen35Session::cancel() {
    if (cancel_) cancel_->set();
}

}  // namespace arcaine::qwen3_5

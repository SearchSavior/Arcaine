#include "modeling/qwen3_5/inference/decode_loop.hpp"

#include "modeling/qwen3_5/model.hpp"
#include "modeling/qwen3_5/output_parser.hpp"
#include "modeling/qwen3_5/inference/sampling.hpp"
#include "utils/chat.hpp"
#include "inference/contracts/generation_event.hpp"

#include <chrono>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace arcaine::qwen3_5 {

namespace {
bool emit_stream_outputs(
    arcaine::inference::GenerationSink& sink,
    std::vector<Qwen35StreamParser::Output>&& outputs) {
    for (auto& o : outputs) {
        if (std::holds_alternative<Qwen35StreamParser::TextDelta>(o)) {
            auto& d = std::get<Qwen35StreamParser::TextDelta>(o);
            if (d.text.empty()) continue;
            if (!sink.emit(arcaine::inference::TextDeltaEvent{std::move(d.text)}))
                return false;
        } else {
            auto& tc = std::get<Qwen35StreamParser::ToolCall>(o);
            arcaine::inference::ToolCallDeltaEvent ev;
            ev.index           = tc.index;
            ev.id              = std::move(tc.id);
            ev.name            = std::move(tc.name);
            ev.arguments_delta = std::move(tc.arguments);
            if (!sink.emit(std::move(ev))) return false;
        }
    }
    return true;
}
}  // namespace

Qwen35DecodeResult run_decode_loop(Qwen35Model& model,
                                   const Qwen35Invocation& inv,
                                   TokenizerBridge& tokenizer,
                                   arcaine::inference::GenerationSink& sink,
                                   const arcaine::inference::CancellationToken* cancel) {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    Qwen35DecodeResult r;

    const ModelInfo& info = model.info();

    // --- Prefill ---
    auto t0 = Clock::now();
    std::vector<float> logits = model.forward(ForwardInput{
        std::cref(inv.input_ids), /*past_len=*/0,
        inv.images.empty()           ? nullptr : &inv.images,
        /*audio=*/nullptr,
        inv.mm_token_type_ids.empty() ? nullptr : &inv.mm_token_type_ids});
    double prefill_s = std::chrono::duration<double>(Clock::now() - t0).count();
    int past = (int)inv.input_ids.size();
    auto tgen0 = Clock::now();

    std::mt19937 rng(static_cast<unsigned>(inv.seed));
    Qwen35StreamParser stream_parser;
    bool saw_first = false;
    bool ok = true;

    for (int step = 0; step < inv.max_output_tokens && ok; ++step) {
        if (cancel && cancel->is_set()) break;
        for (int id : info.suppress_tokens)
            if (id >= 0 && id < (int)logits.size())
                logits[id] = -std::numeric_limits<float>::infinity();
        int next = sample_token(logits.data(), (int)logits.size(),
                                inv.temperature, inv.top_k, inv.top_p, rng);
        if (info.is_eos(next)) break;
        if (!saw_first) {
            r.ttft_s = std::chrono::duration<double>(Clock::now() - start).count();
            saw_first = true;
        }
        r.generated_ids.push_back(next);

        // Stream content/tool-call deltas per token; the stream parser holds
        // back only a potential partial "<tool_call>" marker suffix.
        if (inv.stream) {
            std::vector<int> one{next};
            if (!emit_stream_outputs(sink,
                    stream_parser.feed(tokenizer.decode_raw(one)))) {
                ok = false;
                break;
            }
        }

        std::vector<int> step_tok{next};
        logits = model.forward(ForwardInput{std::cref(step_tok), past});
        ++past;
    }

    if (inv.stream && ok)
        ok = emit_stream_outputs(sink, stream_parser.flush());

    r.duration_s = std::chrono::duration<double>(Clock::now() - start).count();
    r.decode_s   = std::chrono::duration<double>(Clock::now() - tgen0).count();
    if (!saw_first && !r.generated_ids.empty()) r.ttft_s = r.duration_s;

    r.parsed = parse_assistant_output(tokenizer.decode_raw(r.generated_ids));
    r.prefill_s = prefill_s;
    return r;
}

}  // namespace arcaine::qwen3_5

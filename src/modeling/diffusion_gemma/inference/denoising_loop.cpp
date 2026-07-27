#include "modeling/diffusion_gemma/inference/denoising_loop.hpp"

#include "modeling/diffusion_gemma/model.hpp"
#include "modeling/diffusion_gemma/output_parser.hpp"
#include "utils/chat.hpp"
#include "inference/contracts/generation_event.hpp"
#include "inference/service/cancellation_token.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace arcaine::diffusion_gemma {

namespace {
std::string channel_delta(const std::string& full, std::string& emitted) {
    std::string delta;
    if (full.size() >= emitted.size() &&
        full.compare(0, emitted.size(), emitted) == 0) {
        delta = full.substr(emitted.size());
    } else {
        delta = full;
    }
    emitted = full;
    return delta;
}
}  // namespace

DiffusionGemmaDecodeResult run_denoising_loop(
    DiffusionGemmaModel& model,
    const DiffusionGemmaInvocation& inv,
    TokenizerBridge& tokenizer,
    const DiffusionGemmaBoundaryParser& boundaries,
    arcaine::inference::GenerationSink& sink,
    const arcaine::inference::CancellationToken* cancel) {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    DiffusionGemmaDecodeResult r;

    const int n_steps = (inv.denoising_steps > 0)
                            ? inv.denoising_steps
                            : model.config().gen.max_denoising_steps;

    // State captured by the per-step callback.
    bool saw_first_commit = false;
    double ttft_s = 0.0;
    std::vector<int> emitted_ids;
    std::string emitted_content;
    bool ok = true;

    DiffStreamCallback on_step = [&](const DiffStepEvent& ev) {
        if (!ok) return;
        if (cancel && cancel->is_set()) return;
        if (ev.committed && !saw_first_commit) {
            ttft_s = std::chrono::duration<double>(Clock::now() - start).count();
            saw_first_commit = true;
        }
        // Draft (per-step) events, only when streaming + draft streaming enabled.
        if (inv.stream && inv.stream_drafts && !ev.committed && ev.canvas) {
            arcaine::inference::DraftEvent draft;
            draft.block        = ev.block;
            draft.step         = ev.cur_step;
            draft.temperature  = ev.temperature;
            draft.mean_entropy = ev.mean_entropy;
            draft.draft_token_ids = *ev.canvas;
            draft.draft_text    = tokenizer.decode(*ev.canvas);
            if (!sink.emit(std::move(draft))) { ok = false; return; }
        }
        if (inv.has_tools || !ev.committed || !ev.canvas) return;
        if (emitted_ids.size() >= static_cast<size_t>(inv.output_length)) return;

        // Commit: accumulate committed tokens, decode + parse, prefix-diff
        // the public content, emit a TextDelta. (Mirrors the prior transport's
        // per-commit prefix-diff against the freshly parsed full buffer.)
        size_t remaining = static_cast<size_t>(inv.output_length) - emitted_ids.size();
        size_t take = std::min(remaining, ev.canvas->size());
        emitted_ids.insert(emitted_ids.end(), ev.canvas->begin(),
                           ev.canvas->begin() +
                               static_cast<std::vector<int>::difference_type>(take));

        DiffusionGemmaAssistantOutput parsed =
            parse_assistant_output(tokenizer.decode_raw(emitted_ids));
        std::string delta = channel_delta(parsed.content, emitted_content);
        if (!delta.empty()) {
            if (!sink.emit(arcaine::inference::TextDeltaEvent{std::move(delta)})) {
                ok = false;
                return;
            }
        }
    };

    std::vector<int> generated = model.generate(
        inv.encoder_ids, inv.output_length, n_steps,
        static_cast<unsigned>(inv.seed), /*verbose=*/false, on_step,
        /*ignore_eos=*/false, cancel ? &cancel->flag() : nullptr);

    double duration_s = std::chrono::duration<double>(Clock::now() - start).count();
    if (generated.size() > static_cast<size_t>(inv.output_length))
        generated.resize(static_cast<size_t>(inv.output_length));
    if (!saw_first_commit && !generated.empty()) ttft_s = duration_s;

    const DiffPerfStats& st = model.stats();
    r.generated_ids  = std::move(generated);
    r.parsed         = parse_assistant_output(tokenizer.decode_raw(r.generated_ids));
    r.boundary_counts = boundaries.count(r.generated_ids);
    r.ttft_s         = ttft_s;
    r.duration_s     = duration_s;
    r.prefill_s      = st.prefill_s;
    r.decode_s       = st.decode_s;
    r.prefill_tokens = st.prefill_tokens;
    return r;
}

}  // namespace arcaine::diffusion_gemma

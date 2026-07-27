#include "apps/server/openai/sse_event_sink.hpp"

#include "apps/server/openai/schemas.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <string>
#include <variant>

namespace arcaine::server {
namespace {
using json = nlohmann::ordered_json;

json chunk(const std::string& id, std::time_t created, const std::string& model,
           json delta, json finish_reason = nullptr, json usage = nullptr,
           json metrics = nullptr) {
    return {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array({{
            {"index", 0},
            {"delta", std::move(delta)},
            {"finish_reason", std::move(finish_reason)},
        }})},
        {"usage", std::move(usage)},
        {"metrics", std::move(metrics)},
    };
}

json usage_chunk(const std::string& id, std::time_t created, const std::string& model,
                 int prompt_tokens, int completion_tokens,
                 const arcaine::inference::GenerationMetrics& m) {
    return {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array()},
        {"usage", {{"prompt_tokens", prompt_tokens},
                   {"completion_tokens", completion_tokens},
                   {"total_tokens", prompt_tokens + completion_tokens}}},
        {"metrics", {{"input_token", m.input_token}, {"new_token", m.new_token},
                     {"ttft", m.ttft}, {"tpot", m.tpot},
                     {"prefill_throughput", m.prefill_throughput},
                     {"decode_throuput", m.decode_throuput}, {"duration", m.duration}}},
    };
}
}  // namespace

SseEventSink::SseEventSink(httplib::DataSink& sink, std::string id, std::time_t created,
                           std::string model, bool include_usage,
                           arcaine::inference::CancellationToken* cancel)
    : sink_(sink), id_(std::move(id)), created_(created), model_(std::move(model)),
      include_usage_(include_usage), cancel_(cancel) {}

bool SseEventSink::write_raw(const char* data, size_t len) {
    if (!sink_.write(data, len)) {
        if (cancel_) cancel_->set();
        return false;
    }
    return true;
}

bool SseEventSink::write_sse(const json& data, const char* event) {
    std::string payload;
    if (event) { payload += "event: "; payload += event; payload += "\n"; }
    payload += "data: ";
    payload += data.dump();
    payload += "\n\n";
    return write_raw(payload.data(), payload.size());
}

bool SseEventSink::emit(const arcaine::inference::GenerationEvent& ev) {
    using namespace arcaine::inference;
    if (std::holds_alternative<StartedEvent>(ev)) {
        if (!saw_role_) {
            saw_role_ = true;
            return write_sse(chunk(id_, created_, model_, {{"role", "assistant"}}));
        }
        return true;
    }
    if (std::holds_alternative<TextDeltaEvent>(ev)) {
        return write_sse(chunk(id_, created_, model_,
                               {{"content", std::get<TextDeltaEvent>(ev).delta}}));
    }
    if (std::holds_alternative<ToolCallDeltaEvent>(ev)) {
        const auto& tc = std::get<ToolCallDeltaEvent>(ev);
        json call = {{"index", tc.index}, {"id", tc.id}, {"type", "function"},
                     {"function", {{"name", tc.name}, {"arguments", tc.arguments_delta}}}};
        return write_sse(chunk(id_, created_, model_,
                               {{"tool_calls", json::array({std::move(call)})}}));
    }
    if (std::holds_alternative<DraftEvent>(ev)) {
        const auto& d = std::get<DraftEvent>(ev);
        json draft = {{"block", d.block}, {"step", d.step},
                      {"temperature", d.temperature}, {"mean_entropy", d.mean_entropy},
                      {"text", d.draft_text}};
        return write_sse(draft, "arcaine.diffusion_step");
    }
    if (std::holds_alternative<MetricsEvent>(ev)) {
        metrics_ = std::get<MetricsEvent>(ev).metrics;
        have_metrics_ = true;
        return true;
    }
    if (std::holds_alternative<CompletedEvent>(ev)) {
        const auto& c = std::get<CompletedEvent>(ev);
        json metrics_json = have_metrics_
            ? json({{"input_token", metrics_.input_token}, {"new_token", metrics_.new_token},
                    {"ttft", metrics_.ttft}, {"tpot", metrics_.tpot},
                    {"prefill_throughput", metrics_.prefill_throughput},
                    {"decode_throuput", metrics_.decode_throuput},
                    {"duration", metrics_.duration}})
            : json(nullptr);
        if (!write_sse(chunk(id_, created_, model_, json::object(),
                             c.finish_reason, nullptr, metrics_json)))
            return false;
        if (include_usage_ && !write_sse(usage_chunk(id_, created_, model_,
                                                     c.prompt_tokens,
                                                     c.completion_tokens, metrics_)))
            return false;
        return write_done();
    }
    return true;  // TokenEvent: not streamed as content (deltas carry text).
}

bool SseEventSink::write_done() {
    static const char done[] = "data: [DONE]\n\n";
    return write_raw(done, sizeof(done) - 1);
}

}  // namespace arcaine::server

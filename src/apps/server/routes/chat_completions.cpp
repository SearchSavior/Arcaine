#include "apps/server/routes/chat_completions.hpp"

#include "apps/server/app_state.hpp"
#include "apps/server/openai/error_encoder.hpp"
#include "apps/server/openai/request_decoder.hpp"
#include "apps/server/openai/response_encoder.hpp"
#include "apps/server/openai/schemas.hpp"
#include "apps/server/openai/sse_event_sink.hpp"
#include "inference/contracts/generation_event.hpp"
#include "inference/service/cancellation_token.hpp"
#include "inference/service/generation_sink.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <ctime>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace arcaine::server {
namespace {
using json = nlohmann::ordered_json;

std::string completion_id() {
    static std::atomic<unsigned long long> seq{0};
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "chatcmpl-arcaine-" + std::to_string((unsigned long long)now) +
           "-" + std::to_string(seq.fetch_add(1));
}

// Collecting sink for non-streaming generation.
class CollectingGenerationSink : public arcaine::inference::GenerationSink {
public:
    bool emit(const arcaine::inference::GenerationEvent& event) override { return true; }
};
}  // namespace

void handle_chat_completions(const httplib::Request& req, httplib::Response& res,
                             AppState& app) {
    using namespace arcaine::openai;
    try {
        arcaine::inference::GenerationRequest gen =
            decode_chat_completion_request(req, app);

        std::lock_guard<std::mutex> lock(app.generate_mu);
        auto cancel = std::make_shared<arcaine::inference::CancellationToken>();
        gen.request_id = completion_id();
        auto session = std::shared_ptr<arcaine::inference::InferenceSession>(
            app.model.service->create_session({gen.request_id, cancel.get()}));

        const std::time_t created = std::time(nullptr);

        if (gen.stream.enabled) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider(
                "text/event-stream",
                [gen, created, session, cancel, &mu = app.generate_mu]
                (size_t, httplib::DataSink& sink) -> bool {
                    SseEventSink sse(sink, gen.request_id, created,
                                     gen.served_model_name, gen.stream.include_usage,
                                     cancel.get());
                    // Hold the single-generation mutex for the whole generate()
                    // call. The handler's lock_guard released when this provider
                    // was *registered* (set_chunked_content_provider returns
                    // immediately); the actual generation runs here, in the
                    // provider, so the mutex must be taken here to serialize
                    // concurrent streaming generations.
                    std::lock_guard<std::mutex> gen_lock(mu);
                    try {
                        session->generate(gen, sse);
                    } catch (const std::exception& e) {
                        json err = error_body(e.what(), "server_error", "internal_error");
                        sse.write_sse(err, "error");
                        sse.write_done();
                        sink.done();   // terminate the httplib chunked-provider loop
                        return false;
                    }
                    bool ok = sse.write_done();
                    sink.done();       // signal completion; otherwise httplib re-invokes the provider
                    return ok;
                });
        } else {
            CollectingGenerationSink sink;
            arcaine::inference::GenerationResult result = session->generate(gen, sink);
            encode_chat_completion_response(
                res, gen.request_id, created, gen, result);
        }
    } catch (const OpenAiError& e) {
        res.status = e.status;
        res.set_content(error_body(e.what(), e.type, e.code).dump(),
                        "application/json");
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(error_body(e.what(), "server_error",
                                   "internal_error").dump(),
                        "application/json");
    }
}

}  // namespace arcaine::server

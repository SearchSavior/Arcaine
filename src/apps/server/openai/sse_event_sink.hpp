#pragma once

#include <ctime>
#include <string>

#include "inference/contracts/generation_event.hpp"
#include "inference/contracts/generation_request.hpp"
#include "inference/service/cancellation_token.hpp"
#include "inference/service/generation_sink.hpp"
#include "apps/server/openai/schemas.hpp"

namespace httplib { class DataSink; }

namespace arcaine::server {

// A GenerationSink that translates GenerationEvent objects into an OpenAI SSE
// stream on an httplib chunked content provider. Does NOT drive generation;
// the model session owns the generation loop and pushes events here.
class SseEventSink : public arcaine::inference::GenerationSink {
public:
    SseEventSink(httplib::DataSink& sink, std::string id, std::time_t created,
                 std::string model, bool include_usage,
                 arcaine::inference::CancellationToken* cancel);

    bool emit(const arcaine::inference::GenerationEvent& event) override;

    // Emit the trailing [DONE] sentinel. Called by the route after generate().
    bool write_done();

    // Emit a raw SSE event (used for the error path). Returns false on write
    // failure (sink disconnected).
    bool write_sse(const arcaine::openai::json& data, const char* event = nullptr);

private:
    bool write_raw(const char* data, size_t len);

    httplib::DataSink&                              sink_;
    std::string                                     id_;
    std::time_t                                     created_;
    std::string                                     model_;
    bool                                            include_usage_;
    arcaine::inference::CancellationToken*          cancel_;
    arcaine::inference::GenerationMetrics          metrics_;
    bool                                            have_metrics_ = false;
    bool                                            saw_role_     = false;
};

}  // namespace arcaine::server

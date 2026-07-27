#include "apps/server/openai/response_encoder.hpp"

#include "apps/server/openai/schemas.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <ctime>
#include <string>

namespace arcaine::server {
namespace {
using json = nlohmann::ordered_json;

json chat_response(const std::string& id, std::time_t created, const std::string& model,
                   const arcaine::inference::GenerationResult& result) {
    using arcaine::openai::json;
    json message = {
        {"role", "assistant"},
        {"content", result.text.empty() && !result.tool_calls.empty()
                        ? json(nullptr)
                        : json(result.text)},
    };
    if (!result.tool_calls.empty())
        message["tool_calls"] = arcaine::openai::tool_calls_json(result.tool_calls);

    return {
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model},
        {"choices", json::array({{
            {"index", 0},
            {"message", std::move(message)},
            {"logprobs", nullptr},
            {"finish_reason", result.finish_reason},
        }})},
        {"usage", arcaine::openai::usage_json(result.usage.prompt_tokens,
                                              result.usage.completion_tokens)},
        {"metrics", arcaine::openai::metrics_json(result.metrics)},
    };
}
}  // namespace

void encode_chat_completion_response(
    httplib::Response& res, const std::string& id, std::time_t created,
    const arcaine::inference::GenerationRequest& gen,
    const arcaine::inference::GenerationResult& result) {
    json body = chat_response(id, created, gen.served_model_name, result);
    res.status = 200;
    res.set_content(body.dump(), "application/json");
}

}  // namespace arcaine::server

#pragma once

#include "inference/contracts/generation_request.hpp"
#include "inference/contracts/generation_result.hpp"

namespace httplib { struct Response; }

namespace arcaine::server {

// Translate a GenerationResult into the non-streaming OpenAI chat-completion
// response. Does not parse raw model output — uses the result's already-parsed
// text + tool calls.
void encode_chat_completion_response(
    httplib::Response& res, const std::string& id, std::time_t created,
    const arcaine::inference::GenerationRequest& gen,
    const arcaine::inference::GenerationResult& result);

}  // namespace arcaine::server

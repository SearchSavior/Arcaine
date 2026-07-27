#pragma once

#include "inference/contracts/generation_request.hpp"

namespace httplib { struct Request; }

namespace arcaine::server {

struct AppState;

// Translate an OpenAI chat-completions HTTP request into a GenerationRequest.
// Validates the OpenAI schema, normalizes messages/tools, parses stream /
// sampler / chat-template overrides and Arcaine extensions, and constructs
// media inputs. Does NOT run a tokenizer, apply a model chat template, sample
// tokens, or inspect model config classes.
arcaine::inference::GenerationRequest decode_chat_completion_request(
    const httplib::Request& req, const AppState& app);

}  // namespace arcaine::server

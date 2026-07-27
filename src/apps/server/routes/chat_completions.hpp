#pragma once

#include "inference/contracts/generation_result.hpp"

namespace httplib { struct Request; struct Response; }

namespace arcaine::server {

struct AppState;

void handle_chat_completions(const httplib::Request& req, httplib::Response& res,
                              AppState& app);

}  // namespace arcaine::server

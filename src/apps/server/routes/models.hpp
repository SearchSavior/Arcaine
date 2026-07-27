#pragma once

namespace httplib { struct Request; struct Response; }

namespace arcaine::server {

struct AppState;

void handle_models(const httplib::Request& req, httplib::Response& res,
                   const AppState& app);

}  // namespace arcaine::server

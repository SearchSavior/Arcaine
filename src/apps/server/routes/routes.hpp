#pragma once

namespace httplib { class Server; struct Request; struct Response; }

namespace arcaine::server {

struct AppState;

void register_routes(httplib::Server& server, AppState& app);

}  // namespace arcaine::server

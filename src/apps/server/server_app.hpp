#pragma once

namespace arcaine::server {

struct AppState;
struct ServerOptions;

void init_debug_log(const ServerOptions& opts);
int run_server(AppState& app);

}  // namespace arcaine::server

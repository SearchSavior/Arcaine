#include "apps/server/server_app.hpp"

#include "apps/server/app_state.hpp"
#include "apps/server/openai/error_encoder.hpp"
#include "apps/server/openai/schemas.hpp"
#include "apps/server/routes/chat_completions.hpp"
#include "apps/server/routes/models.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>

namespace arcaine::server {
namespace {
using json = nlohmann::ordered_json;

std::string now_string() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}
void log_line(const char* level, const std::string& message) {
    std::fprintf(stderr, "[api][%s][%s] %s\n", now_string().c_str(), level, message.c_str());
    std::fflush(stderr);
}
std::string request_label(const httplib::Request& req) {
    return req.method + " " + req.path + " from " + req.remote_addr;
}
bool authorized(const httplib::Request& req, const AppState& app) {
    if (app.api_key.empty()) return true;
    return req.get_header_value("Authorization") == "Bearer " + app.api_key;
}
}  // namespace

void init_debug_log(const ServerOptions& opts) {
    if (!opts.debug) return;
    std::ofstream out(opts.debug_log_path, std::ios::trunc);
    if (out) out << "[api][" << now_string() << "][debug] debug log started\n";
}

int run_server(AppState& app) {
    httplib::Server server;

    server.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        log_line("info", request_label(req) + " received");
        if (!authorized(req, app)) {
            log_line("error", request_label(req) + " -> HTTP 401 invalid_api_key");
            res.status = 401;
            res.set_content(arcaine::openai::error_body("invalid or missing bearer token",
                                                       "authentication_error",
                                                       "invalid_api_key").dump(),
                            "application/json");
            return;
        }
        handle_models(req, res, app);
        log_line("info", request_label(req) + " -> HTTP 200");
    });

    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        log_line("info", request_label(req) + " received");
        if (!authorized(req, app)) {
            log_line("error", request_label(req) + " -> HTTP 401 invalid_api_key");
            res.status = 401;
            res.set_content(arcaine::openai::error_body("invalid or missing bearer token",
                                                       "authentication_error",
                                                       "invalid_api_key").dump(),
                            "application/json");
            return;
        }
        handle_chat_completions(req, res, app);
    });

    server.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        log_line("error", request_label(req) + " -> HTTP " + std::to_string(res.status) + " not_found");
        res.set_content(arcaine::openai::error_body("not found", "invalid_request_error",
                                                    "not_found").dump(),
                        "application/json");
    });
    server.set_exception_handler([](const httplib::Request& req, httplib::Response& res,
                                    std::exception_ptr ep) {
        std::string message = "unknown exception";
        if (ep) {
            try { std::rethrow_exception(ep); }
            catch (const std::exception& e) { message = e.what(); }
            catch (...) { message = "non-standard exception"; }
        }
        log_line("error", request_label(req) + " -> HTTP 500 unhandled exception: " + message);
        res.status = 500;
        res.set_content(arcaine::openai::error_body(message, "server_error",
                                                    "internal_error").dump(),
                        "application/json");
    });

    log_line("info", "listening on http://" + app.opts.host + ":" + std::to_string(app.opts.port));
    if (!server.listen(app.opts.host, app.opts.port)) {
        log_line("error", "failed to listen on " + app.opts.host + ":" + std::to_string(app.opts.port));
        return 1;
    }
    return 0;
}

}  // namespace arcaine::server

#include "apps/server/routes/models.hpp"

#include "apps/server/app_state.hpp"
#include "apps/server/openai/schemas.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

namespace arcaine::server {
namespace {
using json = nlohmann::ordered_json;
}  // namespace

void handle_models(const httplib::Request& /*req*/, httplib::Response& res,
                   const AppState& app) {
    json body = {
        {"object", "list"},
        {"data", json::array({{
            {"id", app.opts.served_model_name},
            {"object", "model"},
            {"created", app.created},
            {"owned_by", "arcaine"},
        }})},
    };
    res.status = 200;
    res.set_content(body.dump(), "application/json");
}

}  // namespace arcaine::server

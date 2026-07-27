#include "apps/server/app_state.hpp"

#include "inference/registry/builtin_models.hpp"

#include <cstdlib>
#include <utility>

namespace arcaine::server {

AppState::AppState(ServerOptions o) : opts(std::move(o)) {
    api_key = std::getenv("ARCAINE_API_KEY") ? std::getenv("ARCAINE_API_KEY") : "";

    arcaine::inference::register_builtin_models(registry);

    arcaine::inference::ModelLoadRequest load;
    load.model_dir          = opts.model_dir;
    load.max_seq_len       = opts.max_seq;
    load.placement_overrides = opts.placement_overrides;
    load.chat_template_kwargs = opts.chat_template_kwargs;
    load.print_placement   = opts.print_placement;
    load.verbose           = opts.debug;
    model = registry.create(load);
}

}  // namespace arcaine::server

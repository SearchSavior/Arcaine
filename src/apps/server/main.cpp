// Arcaine OpenAI-compatible HTTP API server.
//
// One model, one execution path. The server loads a single model through
// ModelRegistry (which dispatches by config.json::model_type), then serves
// /v1/models and /v1/chat/completions. Every request runs through
// ModelService::create_session + InferenceSession::generate; the server does
// not tokenize, apply a chat template, sample, or parse model-native output,
// and it does not branch on whether the model is autoregressive or
// block-diffusion.
#include "apps/server/app_state.hpp"
#include "apps/server/server_app.hpp"

#include "utils/chat_template_kwargs.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace arcaine::server {
namespace {

std::string basename_of(std::string path) {
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s --model <dir> [options]\n"
        "  --model <dir>              model directory containing config/tokenizer/weights\n"
        "  --served-model-name <id>   API model id (default: basename of --model)\n"
        "  --host <addr>              listen address (default: 127.0.0.1)\n"
        "  --port <N>                 listen port (default: 8000)\n"
        "  --max-seq <N>              KV cache capacity (default: 2048)\n"
        "  --max-tokens <N>           default max completion tokens (default: 256)\n"
        "  --steps <N>                default denoising steps (default: model config)\n"
        "  --seed <S>                 default RNG seed (default: 42)\n"
        "  --layers <spec>            layer placement: auto, single, split:N\n"
        "  --experts <spec>           expert placement: auto, layer-owner, shard\n"
        "  --gpus <spec>              GPU selection: all (default: all)\n"
        "  --print-placement          print resolved placement during model load (default)\n"
        "  --no-print-placement       suppress placement report during model load\n"
        "  --debug                    write full debug responses to ./arcaine_debug.log\n"
        "  --chat-template-kwargs <json>  JSON object merged into chat template vars\n",
        p);
}

void apply_layers_spec(const std::string& v, ServerOptions& opts) {
    if (v == "auto" || v == "single" || v.rfind("split:", 0) == 0) {
        opts.placement_overrides.push_back({"layers", v});
    } else {
        throw std::runtime_error("--layers must be one of: auto, single, split:N");
    }
}
void apply_experts_spec(const std::string& v, ServerOptions& opts) {
    if (v == "auto" || v == "layer-owner" || v == "shard") {
        opts.placement_overrides.push_back({"experts", v});
    } else {
        throw std::runtime_error("--experts must be one of: auto, layer-owner, shard");
    }
}

ServerOptions parse_args(int argc, char** argv) {
    ServerOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if      (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else if (a == "--model")              opts.model_dir = next();
        else if (a == "--served-model-name")  opts.served_model_name = next();
        else if (a == "--host")               opts.host = next();
        else if (a == "--port")               opts.port = std::stoi(next());
        else if (a == "--max-seq")            opts.max_seq = std::stoi(next());
        else if (a == "--max-tokens")         opts.default_max_tokens = std::stoi(next());
        else if (a == "--steps")              opts.steps = std::stoi(next());
        else if (a == "--seed")               opts.seed = (unsigned)std::stoul(next());
        else if (a == "--layers")             apply_layers_spec(next(), opts);
        else if (a == "--experts")            apply_experts_spec(next(), opts);
        else if (a == "--gpus") {
            const std::string v = next();
            if (v != "all") throw std::runtime_error("--gpus only supports 'all'");
            opts.placement_overrides.push_back({"gpus", v});
        }
        else if (a == "--print-placement")    opts.print_placement = true;
        else if (a == "--no-print-placement")  opts.print_placement = false;
        else if (a == "--debug")               opts.debug = true;
        else if (a == "--chat-template-kwargs") opts.chat_template_kwargs = parse_chat_template_kwargs(next());
        else throw std::runtime_error("unknown arg: " + a);
    }
    if (opts.model_dir.empty()) throw std::runtime_error("--model is required");
    if (opts.served_model_name.empty()) opts.served_model_name = basename_of(opts.model_dir);
    if (opts.port <= 0 || opts.port > 65535) throw std::runtime_error("--port must be 1..65535");
    if (opts.max_seq <= 0) throw std::runtime_error("--max-seq must be greater than 0");
    if (opts.default_max_tokens <= 0) throw std::runtime_error("--max-tokens must be greater than 0");
    return opts;
}

}  // namespace
}  // namespace arcaine::server

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        arcaine::server::ServerOptions opts = arcaine::server::parse_args(argc, argv);
        arcaine::server::init_debug_log(opts);
        std::printf("[api] loading model from %s ...\n", opts.model_dir.c_str());
        arcaine::server::AppState app(std::move(opts));
        std::printf("[api] serving model id '%s'\n", app.opts.served_model_name.c_str());
        std::printf("[api] auth: %s\n", app.api_key.empty() ? "disabled" : "ARCAINE_API_KEY");
        return arcaine::server::run_server(app);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[api][error] %s\n", e.what());
        arcaine::server::usage(argv[0]);
        return 1;
    }
}

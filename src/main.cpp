#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <cstdio>
#include <random>
#include <limits>
#include <memory>

#include "common/model.hpp"
#include "common/registry.hpp"
#include "common/sampler.hpp"
#include "common/preprocess/chat_template.hpp"
#include "common/preprocess/image_proc.hpp"
#include "common/preprocess/audio_proc.hpp"

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --model <dir> --prompt <text> [options]\n"
        "Options:\n"
        "  --image <path>        Image file (PNG/JPEG)\n"
        "  --audio <path>        Audio file (WAV or raw float32 PCM at configured sample rate)\n"
        "  --vad <path>          Silero VAD model (silero_vad.onnx) to strip silence\n"
        "  --max-tokens <N>      Max new tokens to generate (default: 200)\n"
        "  --temp <T>            Temperature (default: generation_config.json)\n"
        "  --top-k <K>           Top-K (default: generation_config.json)\n"
        "  --top-p <P>           Top-P (default: generation_config.json)\n"
        "  --seed <S>            RNG seed (default: 42)\n"
        "  --max-seq <N>         KV cache capacity (default: 2048)\n",
        prog);
}

int main(int argc, char** argv) {
    std::string model_dir;
    std::string prompt;
    std::string image_path;
    std::string audio_path;
    std::string vad_model;
    int   max_new_tokens = 200;
    float temp           = 0.0f;
    int   top_k          = 0;
    float top_p          = 0.0f;
    bool  temp_set       = false;
    bool  top_k_set      = false;
    bool  top_p_set      = false;
    int   seed           = 42;
    int   max_seq        = 2048;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "Missing value for %s\n", argv[i]); std::exit(1); }
            return argv[++i];
        };
        if      (arg == "--model")      model_dir = next();
        else if (arg == "--prompt")     prompt    = next();
        else if (arg == "--image")      image_path = next();
        else if (arg == "--audio")      audio_path = next();
        else if (arg == "--vad")        vad_model  = next();
        else if (arg == "--max-tokens") max_new_tokens = std::stoi(next());
        else if (arg == "--temp")       { temp = std::stof(next()); temp_set = true; }
        else if (arg == "--top-k")      { top_k = std::stoi(next()); top_k_set = true; }
        else if (arg == "--top-p")      { top_p = std::stof(next()); top_p_set = true; }
        else if (arg == "--seed")       seed = std::stoi(next());
        else if (arg == "--max-seq")    max_seq = std::stoi(next());
        else { std::fprintf(stderr, "Unknown argument: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }

    if (model_dir.empty() || prompt.empty()) { print_usage(argv[0]); return 1; }

    register_builtin_architectures();

    std::printf("[main] Loading model from %s\n", model_dir.c_str());
    std::unique_ptr<Model> model = ModelRegistry::instance().create(model_dir, max_seq);
    const ModelInfo& info = model->info();
    if (!temp_set)  temp  = info.temperature;
    if (!top_k_set) top_k = info.top_k;
    if (!top_p_set) top_p = info.top_p;
    std::printf("[main] Model and tokenizer ready.\n");

    // Prepare inputs: chat template + any multimodal preprocessing (handled by the arch).
    std::vector<std::string> image_paths;
    std::vector<std::string> audio_paths;
    if (!image_path.empty()) image_paths.push_back(image_path);
    if (!audio_path.empty()) audio_paths.push_back(audio_path);

    auto prepared = model->prepare_input(prompt, image_paths, audio_paths, vad_model);
    const std::vector<int>& tokens = prepared.tokens;

    std::printf("[main] Prompt token count: %d\n", (int)tokens.size());

    // Prefill
    auto t0 = std::chrono::steady_clock::now();
    auto logits = model->forward(ForwardInput{
        tokens, 0,
        prepared.images.empty() ? nullptr : &prepared.images,
        prepared.audio.empty()  ? nullptr : &prepared.audio,
        &prepared.mm_token_type_ids});

    auto t1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double prefill_tps = tokens.size() / (prefill_ms / 1000.0);
    std::printf("[main] Prefill: %.1f ms (%d tokens, %.1f tok/s)\n",
                prefill_ms, (int)tokens.size(), prefill_tps);

    // Generation loop
    std::mt19937 rng(seed);
    std::vector<int> generated;
    int past = (int)tokens.size();

    auto tgen0 = std::chrono::steady_clock::now();
    for (int step = 0; step < max_new_tokens; ++step) {
        for (int id : info.suppress_tokens) {
            if (id >= 0 && id < (int)logits.size())
                logits[id] = -std::numeric_limits<float>::infinity();
        }
        int next = sample_token(logits.data(), (int)logits.size(),
                                temp, top_k, top_p, rng);

        if (info.is_eos(next)) break;
        generated.push_back(next);

        std::vector<int> step_tok{next};
        logits = model->forward(ForwardInput{step_tok, past});
        past++;
    }
    auto tgen1 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(tgen1 - tgen0).count();
    if (!generated.empty()) {
        double tps = generated.size() / (gen_ms / 1000.0);
        std::printf("[main] Generated %d tokens in %.1f ms (%.1f tok/s)\n",
                    (int)generated.size(), gen_ms, tps);
    }

    std::cout << decode_tokens(info.model_dir, generated) << "\n";
    return 0;
}

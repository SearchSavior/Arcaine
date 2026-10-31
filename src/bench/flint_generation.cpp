#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "modeling/flint/model.hpp"
#include "modeling/flint/sampling.hpp"
#include "common/preprocess/chat_template.hpp"

int run_flint_generation(int argc, char** argv) {
    std::string model_dir = "/workspace/models/flint-int4";
    std::string prompt = "The capital of France is";
    int max_new = 16;
    int ctx = 4096;
    int runs = 1;
    bool sample_ab = false;
    bool greedy = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--prompt") prompt = next();
        else if (a == "--max-new") max_new = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--greedy") greedy = true;
        else if (a == "--runs") runs = std::stoi(next());
        else if (a == "--sample-ab") sample_ab = true;
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: arcaine_mbench flint-generate [model_dir] [--prompt TEXT] [--max-new N] [--ctx N] [--greedy] [--runs N] [--sample-ab]\n");
            return 0;
        }
        else if (a.rfind("--", 0) != 0) model_dir = a;
        else { std::fprintf(stderr, "Unknown option: %s\n", a.c_str()); return 1; }
    }

    FlintModel model(model_dir, ctx);
    auto in = model.prepare_input(prompt, {}, {}, "");
    std::printf("[flint] prompt tokens: %zu\n", in.tokens.size());
    std::printf("[flint] prompt: %s\n", prompt.c_str());

    std::printf("[flint] decoded prompt tokens: %s\n",
                decode_tokens(model_dir, in.tokens).c_str());
    if(runs<=0 || (sample_ab && (!model.device_generation_enabled() || greedy || max_new<=0))) {
        std::fprintf(stderr,"--runs must be positive; --sample-ab requires stochastic GPU generation\n");
        return 1;
    }
    std::vector<int> sample_reference;
    uint64_t sample_reference_rng=0;
    bool sample_reference_ready=false;
    for(int phase=0;phase<(sample_ab?2:1);++phase) {
        if(sample_ab) ::setenv("ARCAINE_FLINT_DECODE_PARALLEL_SAMPLE",phase?"1":"0",1);
        for(int run=0;run<runs;++run) {
            if(runs>1 || sample_ab)
                std::printf("[flint] generation run %d/%d parallel_sample=%d\n",run+1,runs,
                            sample_ab?phase:(std::getenv("ARCAINE_FLINT_DECODE_PARALLEL_SAMPLE")!=nullptr));
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;
    if (model.device_generation_enabled()) model.forward_device({in.tokens, 0});
    else logits = model.forward({in.tokens, 0});
    auto t1 = std::chrono::steady_clock::now();
    std::printf("[flint] prefill: %.1f ms (%zu tokens)\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                in.tokens.size());

    uint64_t rng = 0x9e3779b97f4a7c15ull;
    std::vector<int> generated;
    int past = (int)in.tokens.size();
    auto t2 = std::chrono::steady_clock::now();
    if (model.device_generation_enabled())
        generated = model.generate_device(max_new, greedy ? 0.f : 1.f, 20, 0.95f, rng);
    else for (int i = 0; i < max_new; ++i) {
        const int tok = flint_sample(logits, greedy ? 0.f : 1.0f, 20, 0.95f, rng);
        if (tok < 0) break;
        generated.push_back(tok);
        if (model.info().is_eos(tok)) break;
        std::vector<int> one{tok};
        logits = model.forward({one, past});
        ++past;
    }
    auto t3 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    if (const char* trace = std::getenv("ARCAINE_FLINT_TRACE")) model.dump_trace(trace);
    std::printf("[flint] decode: %d tokens in %.1f ms (%.2f ms/token)\n",
                (int)generated.size(), ms, generated.empty() ? 0.0 : ms / generated.size());

    std::printf("[flint] tokens:");
    for (int t : generated) std::printf(" %d", t);
    std::printf("\n");
    try {
        std::printf("[flint] output: %s\n", decode_tokens(model_dir, generated).c_str());
    } catch (const std::exception& e) {
        std::printf("[flint] detokenize failed: %s\n", e.what());
    }
    if(sample_ab) {
        if(!sample_reference_ready) {
            sample_reference=generated; sample_reference_rng=rng; sample_reference_ready=true;
        } else if(generated!=sample_reference || rng!=sample_reference_rng) {
            std::fprintf(stderr,"[flint] GPU sampler A/B token or RNG mismatch\n");
            return 1;
        }
    }
        }
    }
    if(sample_ab) std::printf("[flint] GPU sampler A/B token and RNG parity PASS\n");
    return 0;
}

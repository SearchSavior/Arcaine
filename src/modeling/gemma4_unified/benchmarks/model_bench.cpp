// gemma4_unified — model-owned inference benchmark (llama-bench approach).
// Drives the module's concrete Gemma4Model engine directly (forward/reset_cache)
// through the shared AR driver in benchmarks/model_bench_util.hpp: synthetic
// BOS+random payloads, per-cell warmup at test size, -pchunk batching, depth
// prefill, and a markdown result table. Registered with the central
// arcaine_mbench dispatcher as "gemma4_unified" (matches config.json::model_type).
//
//   ./build/arcaine_mbench -m <dir> [options]
//   (see arcaine_mbench -m <dir> --help for the flag list)

#include <cstdio>
#include <string>

#include "modeling/gemma4_unified/model.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "benchmarks/model_bench_registry.hpp"
#include "benchmarks/model_bench_util.hpp"

static int run(int argc, char* argv[]) {
    using namespace arcaine::bench;

    BenchArgs args;
    int r = parse_bench_args(argc, argv, nullptr, 0, nullptr, args);
    if (r == 1) return 0;  // help shown
    if (r != 0) return 1;  // parse error

    if (args.device_set) {
        try { gpu_device_control::apply_device_index(args.device); }
        catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    }

    const int max_seq = compute_max_seq(args);
    std::printf("loading model from %s ...\n", args.model.c_str());
    double t0 = now_ms();
    Gemma4Model model(args.model, max_seq);
    double load_s = (now_ms() - t0) * 0.001;
    const ModelInfo& info = model.info();

    std::string backend_line = "SYCL + oneDNN | GPUs: " + std::to_string(GpuEngine::count()) +
                              " | max_seq: " + std::to_string(max_seq);
    if (const char* gpus = gpu_device_control::active_gpus_spec())
        backend_line += std::string(" | ZE_AFFINITY_MASK=") + gpus;

    ArEngine eng;
    eng.bos_id   = info.bos_token_id;
    eng.n_vocab  = info.vocab_size;
    eng.reset_cache = [&] { model.reset_cache(); };
    eng.forward = [&](const std::vector<int>& ids, int past_len) {
        model.forward(ForwardInput{ids, past_len});
    };

    return run_ar_bench(args, eng, info.description, backend_line.c_str(), load_s);
}

REGISTER_MODEL_BENCH("gemma4_unified", "Gemma4 Unified AR (llama-bench pp/tg KV-depth throughput)", run)

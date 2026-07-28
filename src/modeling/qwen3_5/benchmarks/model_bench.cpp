// qwen3_5 — model-owned inference benchmark (llama-bench style).
// Drives the module's concrete Qwen35Model engine directly (forward/reset_cache)
// to measure real GPU prefill (PP) and decode (TG) throughput at various KV-cache
// depths. Registered with the central arcaine_mbench dispatcher as "qwen3_5"
// (matches config.json::model_type).
//
//   ./build/arcaine_mbench --model <dir> [options]
//     -p, --p P,...   prefill prompt sizes (default: 512,1024,2048,4096)
//     -n, --n N,...   new-token counts      (default: 128)
//     -d D,...        starting KV depths     (default: 0,512,1024,2048)
//     -r, --r R       timed repetitions      (default: 3)
//     -w, --w W       warmup runs            (default: 1)
//     --max-seq N     KV cache capacity      (default: auto)
//     --device N      restrict to one Level Zero GPU

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "modeling/qwen3_5/model.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "benchmarks/model_bench_registry.hpp"
#include "benchmarks/model_bench_util.hpp"

using Clk = std::chrono::high_resolution_clock;
using Ms  = std::chrono::duration<double, std::milli>;
static double now_ms() { return Ms(Clk::now().time_since_epoch()).count(); }

static const char* USAGE =
    "Usage: arcaine_mbench --model <dir> [options]   (qwen3_5)\n"
    "  -h, --help    show this help text\n"
    "  -p, --p P,... prefill sizes          (default: 512,1024,2048,4096)\n"
    "  -n, --n N,... new-token counts       (default: 128)\n"
    "  -d D,...      KV-cache depths        (default: 0,512,1024,2048)\n"
    "  -r, --r R     timed repetitions      (default: 3)\n"
    "  -w, --w W     warmup runs            (default: 1)\n"
    "  --max-seq N   KvCache capacity       (default: auto)\n"
    "  --device N    run with one visible Level Zero GPU\n";

static int run(int argc, char* argv[]) {
    std::string model_dir;
    std::vector<int> pp_list = {512, 1024, 2048, 4096};
    std::vector<int> tg_list = {128};
    std::vector<int> depths  = {0, 512, 1024, 2048};
    std::string device_index;
    int reps = 3, warmup = 1, max_seq = -1;
    bool device_index_set = false;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model") && i+1<argc) model_dir = argv[++i];
        else if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "--p")) && i+1<argc)
            pp_list = arcaine::bench::parse_int_csv(argv[++i]);
        else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n")) && i+1<argc)
            tg_list = arcaine::bench::parse_int_csv(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i+1<argc) depths = arcaine::bench::parse_int_csv(argv[++i]);
        else if ((!strcmp(argv[i], "-r") || !strcmp(argv[i], "--r") || !strcmp(argv[i], "--reps")) && i+1<argc) reps = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-w") || !strcmp(argv[i], "--w") || !strcmp(argv[i], "--warmup")) && i+1<argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq") && i+1<argc) max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i+1<argc) { device_index = argv[++i]; device_index_set = true; }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { std::fputs(USAGE, stderr); return 0; }
        else if (argv[i][0] != '-' && model_dir.empty()) model_dir = argv[i];
        else { std::fprintf(stderr, "Unknown argument: %s\n", argv[i]); std::fputs(USAGE, stderr); return 1; }
    }

    if (model_dir.empty() || pp_list.empty() || tg_list.empty() || depths.empty() || reps <= 0 || warmup < 0) {
        std::fputs(USAGE, stderr); return 1;
    }
    for (int v : pp_list) if (v <= 0) { std::fputs("--p values must be positive\n", stderr); return 1; }
    for (int v : tg_list) if (v <= 0) { std::fputs("--n values must be positive\n", stderr); return 1; }

    try { if (device_index_set) gpu_device_control::apply_device_index(device_index); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }

    if (max_seq < 0) {
        int max_d = *std::max_element(depths.begin(), depths.end());
        int max_p = *std::max_element(pp_list.begin(), pp_list.end());
        int max_n = *std::max_element(tg_list.begin(), tg_list.end());
        max_seq = std::max(max_d + max_n, max_p);
    }

    std::printf("loading model from %s ...\n", model_dir.c_str());
    double t0 = now_ms();
    Qwen35Model model(model_dir, max_seq);
    double load_s = (now_ms() - t0) * 0.001;
    const ModelInfo& info = model.info();
    std::printf("model   : %s\n", info.description.c_str());
    std::printf("backend : SYCL + oneDNN | GPUs: %d | max_seq: %d",
                GpuEngine::count(), max_seq);
    if (const char* active_gpus = gpu_device_control::active_gpus_spec())
        std::printf(" | ZE_AFFINITY_MASK=%s", active_gpus);
    std::printf("\nload    : %.1f s\n", load_s);

    const int bos_id = info.bos_token_id;
    const std::vector<int> single(1, bos_id);
    arcaine::bench::print_pp_tg_header();

    for (int pp : pp_list) {
        if (pp > max_seq) {
            char name[32]; std::snprintf(name, sizeof name, "pp %d", pp);
            std::printf(" %-18s %9s   [skip: pp=%d > max_seq=%d]\n", name, "—", pp, max_seq);
            continue;
        }
        const std::vector<int> prompt(pp, bos_id);
        std::vector<double> times;
        for (int r = 0; r < warmup + reps; ++r) {
            model.reset_cache();
            double t = now_ms();
            model.forward(ForwardInput{prompt, 0});
            double dt = now_ms() - t;
            if (r >= warmup) times.push_back(dt);
        }
        char name[32]; std::snprintf(name, sizeof name, "pp %d", pp);
        arcaine::bench::print_pp_tg_row(name, "—", arcaine::bench::compute_pp_tg_stats(times, pp));
    }

    for (int tg : tg_list) {
        for (int depth : depths) {
            char name[32]; std::snprintf(name, sizeof name, "tg %d", tg);
            char dstr[32]; std::snprintf(dstr, sizeof dstr, "%d", depth);
            if (depth + tg > max_seq) {
                arcaine::bench::print_pp_tg_row(name, dstr, {}, /*skipped=*/true);
                continue;
            }
            std::vector<double> times;
            for (int r = 0; r < warmup + reps; ++r) {
                model.reset_cache();
                if (depth > 0) {
                    constexpr int CHUNK = 512;
                    for (int pos = 0; pos < depth; pos += CHUNK) {
                        int sz = std::min(CHUNK, depth - pos);
                        std::vector<int> chunk(sz, bos_id);
                        model.forward(ForwardInput{chunk, pos});
                    }
                }
                double t = now_ms();
                for (int step = 0; step < tg; ++step)
                    model.forward(ForwardInput{single, depth + step});
                double dt = now_ms() - t;
                if (r >= warmup) times.push_back(dt);
            }
            arcaine::bench::print_pp_tg_row(name, dstr, arcaine::bench::compute_pp_tg_stats(times, tg));
        }
    }
    std::printf("\n");
    return 0;
}

REGISTER_MODEL_BENCH("qwen3_5", "Qwen3.5 AR DeltaNet (PP/TG KV-depth throughput)", run)

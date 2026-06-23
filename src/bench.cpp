// bench.cpp — Gemma4 inference benchmark, styled after llama-bench.
//
// Measures real GPU throughput for prefill (PP) and decode (TG) at
// various KV-cache depths.  Each row times N actual forward passes and
// reports mean t/s ± σ so you can see how decode slows as the cache fills.
//
// Usage: ./build/bench <model_dir> [options]
//   -p P,...      prefill prompt sizes to test   (default: 128,512)
//   -n N          max new tokens (decode steps)  (default: 128)
//   -d D,...      starting KV-cache depth(s)     (default: 0,512,1024,2048)
//   --reps  R     timed repetitions              (default: 3)
//   --warmup W    discarded warmup runs          (default: 1)
//   --max-seq N   KvCache allocation size        (default: auto)
//   --device N    restrict visible GPUs to one Level Zero device

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include <memory>

#include "common/model.hpp"
#include "common/registry.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"

// ---------------------------------------------------------------------------
using Clk = std::chrono::high_resolution_clock;
using Ms  = std::chrono::duration<double, std::milli>;

static double now_ms() { return Ms(Clk::now().time_since_epoch()).count(); }

// Per-benchmark statistics computed from a set of wall-clock timings.
struct Stats {
    double mean_ms, sd_ms;     // raw timing
    double mean_tps, sd_tps;   // derived tokens/sec (each rep computed independently)
    double ms_per_tok;         // mean_ms / n_toks
    int    n_toks;
};

static Stats compute_stats(const std::vector<double>& ms, int n_toks) {
    int n = (int)ms.size();

    // ms stats
    double s = 0, s2 = 0;
    for (double t : ms) { s += t; s2 += t * t; }
    double mean_ms = s / n;
    double sd_ms   = std::sqrt(std::max(0.0, s2 / n - mean_ms * mean_ms));

    // t/s stats (computed per-rep to get σ in t/s space)
    double ts = 0, ts2 = 0;
    for (double t : ms) {
        double v = n_toks / (t * 0.001);
        ts += v; ts2 += v * v;
    }
    double mean_tps = ts / n;
    double sd_tps   = std::sqrt(std::max(0.0, ts2 / n - mean_tps * mean_tps));

    return {mean_ms, sd_ms, mean_tps, sd_tps, mean_ms / n_toks, n_toks};
}

// Parse "a,b,c" → {a,b,c}.
static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    char buf[512]; strncpy(buf, s, 511); buf[511] = 0;
    for (char* tok = strtok(buf, ","); tok; tok = strtok(nullptr, ","))
        v.push_back(atoi(tok));
    return v;
}

// ---------------------------------------------------------------------------
static void print_header() {
    printf("\n %-18s %9s   %10s   %7s   %9s   %8s\n",
           "test", "kv-depth", "t/s", "± sd", "ms/tok", "time(s)");
    printf(" %-18s %9s   %10s   %7s   %9s   %8s\n",
           "──────────────────", "─────────",
           "──────────", "───────", "─────────", "────────");
}

static void print_row(const char* test, const char* depth,
                      const Stats& s, bool skipped = false) {
    if (skipped) {
        printf(" %-18s %9s   [skipped: depth+tg > max_seq]\n", test, depth);
        return;
    }
    printf(" %-18s %9s   %10.2f   %7.2f   %9.3f   %8.3f\n",
           test, depth, s.mean_tps, s.sd_tps, s.ms_per_tok, s.mean_ms * 0.001);
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    static const char* USAGE =
        "Usage: bench <model_dir> [options]\n"
        "  -h, --help    show this help text\n"
        "  -p P,...      prefill sizes          (default: 128,512)\n"
        "  -n N          max new tokens         (default: 128)\n"
        "  -d D,...      KV-cache depths        (default: 0,512,1024,2048)\n"
        "  --reps R      timed repetitions      (default: 3)\n"
        "  --warmup W    warmup runs            (default: 1)\n"
        "  --max-seq N   KvCache capacity       (default: auto)\n"
        "  --device N    run with one visible Level Zero GPU\n";

    if (argc < 2) { fputs(USAGE, stderr); return 1; }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        fputs(USAGE, stderr);
        return 0;
    }

    std::string model_dir = argv[1];
    std::vector<int> pp_list  = {128, 512};
    std::vector<int> depths   = {0, 512, 1024, 2048};
    std::string device_index;
    int tg      = 128;
    int reps    = 3;
    int warmup  = 1;
    int max_seq = -1;  // computed after arg parsing
    bool device_index_set = false;

    for (int i = 2; i < argc; ++i) {
        if      (!strcmp(argv[i], "-p")         && i+1<argc) pp_list = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "-n")         && i+1<argc) tg      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d")         && i+1<argc) depths  = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--reps")    && i+1<argc) reps    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")  && i+1<argc) warmup  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq") && i+1<argc) max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device")  && i+1<argc) { device_index = argv[++i]; device_index_set = true; }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { fputs(USAGE, stderr); return 0; }
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); fputs(USAGE, stderr); return 1; }
    }

    try {
        if (device_index_set) gpu_device_control::apply_device_index(device_index);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    // Derive max_seq from the largest depth + n, plus the largest pp size,
    // if the user did not override it with --max-seq.
    if (max_seq < 0) {
        int max_d = depths.empty() ? 0 : *std::max_element(depths.begin(), depths.end());
        int max_p = pp_list.empty() ? 0 : *std::max_element(pp_list.begin(), pp_list.end());
        max_seq = std::max(max_d + tg, max_p);
    }

    // ── Load ────────────────────────────────────────────────────────────────
    register_builtin_architectures();

    printf("loading model from %s ...\n", model_dir.c_str());
    double t0 = now_ms();
    std::unique_ptr<Model> model = ModelRegistry::instance().create(model_dir, max_seq);
    double load_s = (now_ms() - t0) * 0.001;

    const ModelInfo& info = model->info();
    printf("model   : %s\n", info.description.c_str());
    printf("backend : SYCL + oneDNN | GPUs: %d | max_seq: %d",
           GpuEngine::count(), max_seq);
    if (const char* active_gpus = gpu_device_control::active_gpus_spec())
        printf(" | ZE_AFFINITY_MASK=%s", active_gpus);
    printf("\n");
    printf("load    : %.1f s\n", load_s);

    // Placeholder token: BOS from the loaded model config.
    const int bos_id = info.bos_token_id;
    const std::vector<int> single(1, bos_id);

    print_header();

    // ── Prefill (PP) ────────────────────────────────────────────────────────
    for (int pp : pp_list) {
        if (pp > max_seq) {
            char name[32]; snprintf(name, sizeof name, "pp %d", pp);
            printf(" %-18s %9s   [skip: pp=%d > max_seq=%d]\n", name, "—", pp, max_seq);
            continue;
        }
        const std::vector<int> prompt(pp, bos_id);
        std::vector<double> times;

        for (int r = 0; r < warmup + reps; ++r) {
            model->reset_cache();
            double t = now_ms();
            model->forward(ForwardInput{prompt, 0});
            double dt = now_ms() - t;
            if (r >= warmup) times.push_back(dt);
        }

        char name[32]; snprintf(name, sizeof name, "pp %d", pp);
        print_row(name, "—", compute_stats(times, pp));
    }

    // ── Decode (TG) at various KV-cache depths ──────────────────────────────
    for (int depth : depths) {
        char name[32];  snprintf(name, sizeof name, "tg %d", tg);
        char dstr[32];  snprintf(dstr, sizeof dstr, "%d", depth);

        if (depth + tg > max_seq) {
            print_row(name, dstr, {}, /*skipped=*/true);
            continue;
        }

        std::vector<double> times;

        for (int r = 0; r < warmup + reps; ++r) {
            model->reset_cache();

            // Pre-fill KV cache to `depth` tokens in chunks (untimed).
            // A single forward of `depth` tokens allocates scores of shape
            // (nq, depth, depth) — O(depth²) and gigantic at depth=16k+.
            // Chunking caps peak activation memory at O(CHUNK × kv_len).
            if (depth > 0) {
                constexpr int CHUNK = 512;
                for (int pos = 0; pos < depth; pos += CHUNK) {
                    int sz = std::min(CHUNK, depth - pos);
                    std::vector<int> chunk(sz, bos_id);
                    model->forward(ForwardInput{chunk, pos});
                }
            }

            // Time `tg` single-token decode steps.
            // The cache grows from depth to depth+tg during this window,
            // which is realistic for mid-generation latency measurement.
            double t = now_ms();
            for (int step = 0; step < tg; ++step)
                model->forward(ForwardInput{single, depth + step});
            double dt = now_ms() - t;

            if (r >= warmup) times.push_back(dt);
        }

        print_row(name, dstr, compute_stats(times, tg));
    }

    printf("\n");
    return 0;
}

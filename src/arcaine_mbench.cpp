// bench.cpp — Gemma4 inference benchmark, styled after llama-bench.
//
// Measures real GPU throughput for prefill (PP) and decode (TG) at
// various KV-cache depths.  Each row times N actual forward passes and
// reports mean t/s ± σ so you can see how decode slows as the cache fills.
//
// Usage: ./build/bench --model <model_dir> [options]
//   -p, --p P,... prefill prompt sizes to test   (default: 128,512)
//   -n, --n N,... new-token counts to test       (default: 128)
//   -d D,...      starting KV-cache depth(s)     (default: 0,512,1024,2048)
//   -r, --r R     timed repetitions              (default: 3)
//   -w, --w W     discarded warmup runs          (default: 1)
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

#include "common/model_interface.hpp"
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

// --- llama-bench-style output ----------------------------------------------
// A single `test` column encodes pp/tg and folds the KV-cache depth in as a
// `@ d N` suffix (matching llama-bench's `tg128 @ d512`); the `t/s` column is
// `mean ± sd`.  Output format is selectable with -o (md default, also csv/json).
enum class OutputFormat { Markdown, Csv, Json };

static OutputFormat parse_output_format(const char* s) {
    std::string f = s;
    for (auto& c : f) c = (char)std::tolower((unsigned char)c);
    if (f == "md" || f == "markdown") return OutputFormat::Markdown;
    if (f == "csv")                    return OutputFormat::Csv;
    if (f == "json")                   return OutputFormat::Json;
    std::fprintf(stderr, "unknown output format: %s (use md|csv|json)\n", s);
    std::exit(1);
}

static std::string model_basename(const std::string& dir) {
    size_t pos = dir.find_last_of('/');
    return dir.substr(pos == std::string::npos ? 0 : pos + 1);
}

// "pp 512", "tg 128", "tg 128 @ d 512" — depth folded into the test name.
static std::string test_name(const char* kind, int n, int depth) {
    char buf[64];
    if (depth > 0) std::snprintf(buf, sizeof buf, "%s %d @ d %d", kind, n, depth);
    else           std::snprintf(buf, sizeof buf, "%s %d", kind, n);
    return buf;
}

struct BenchRow {
    std::string test;
    double mean_tps = 0, sd_tps = 0;
    bool skipped = false;
};

static void print_results(OutputFormat fmt, const std::string& model, int gpus,
                          const std::vector<BenchRow>& rows) {
    if (rows.empty()) return;
    if (fmt == OutputFormat::Markdown) {
        std::printf("\n| model | gpus | test | t/s |\n");
        std::printf("|---|---:|---|---:|\n");
        for (const auto& r : rows) {
            if (r.skipped)
                std::printf("| %s | %d | %s | N/A |\n", model.c_str(), gpus, r.test.c_str());
            else
                std::printf("| %s | %d | %s | %.2f ± %.2f |\n",
                            model.c_str(), gpus, r.test.c_str(), r.mean_tps, r.sd_tps);
        }
    } else if (fmt == OutputFormat::Csv) {
        std::printf("model,gpus,test,t/s,sd\n");
        for (const auto& r : rows) {
            if (r.skipped)
                std::printf("%s,%d,%s,,\n", model.c_str(), gpus, r.test.c_str());
            else
                std::printf("%s,%d,%s,%.4f,%.4f\n",
                            model.c_str(), gpus, r.test.c_str(), r.mean_tps, r.sd_tps);
        }
    } else { // Json
        std::printf("[\n");
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            std::printf("  {\"model\":\"%s\",\"gpus\":%d,\"test\":\"%s\",",
                        model.c_str(), gpus, r.test.c_str());
            if (r.skipped) std::printf("\"t/s\":null,\"sd\":null");
            else           std::printf("\"t/s\":%.4f,\"sd\":%.4f", r.mean_tps, r.sd_tps);
            std::printf("}%s\n", (i + 1 < rows.size()) ? "," : "");
        }
        std::printf("]\n");
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    static const char* USAGE =
        "Usage: bench --model <model_dir> [options]\n"
        "  -h, --help    show this help text\n"
        "  -p, --p P,... prefill sizes          (default: 128,512)\n"
        "  -n, --n N,... new-token counts       (default: 128)\n"
        "  -d D,...      KV-cache depths (applies to pp and tg) (default: 0,512,1024,2048)\n"
        "  -r, --r R     timed repetitions      (default: 3)\n"
        "  -w, --w W     warmup runs            (default: 1)\n"
        "  --max-seq N   KvCache capacity       (default: auto)\n"
        "  --device N    run with one visible Level Zero GPU\n"
        "  -o, --output <md|csv|json>  output format (default: md)\n";

    std::string model_dir;
    std::vector<int> pp_list  = {128, 512};
    std::vector<int> tg_list  = {128};
    std::vector<int> depths   = {0, 512, 1024, 2048};
    std::string device_index;
    int reps    = 3;
    int warmup  = 1;
    int max_seq = -1;  // computed after arg parsing
    bool device_index_set = false;
    OutputFormat out_fmt = OutputFormat::Markdown;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model")    && i+1<argc) model_dir = argv[++i];
        else if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "--p")) && i+1<argc)
            pp_list = parse_list(argv[++i]);
        else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n")) && i+1<argc)
            tg_list = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "-d")         && i+1<argc) depths  = parse_list(argv[++i]);
        else if ((!strcmp(argv[i], "-r") || !strcmp(argv[i], "--r") ||
                  !strcmp(argv[i], "--reps")) && i+1<argc) reps = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-w") || !strcmp(argv[i], "--w") ||
                  !strcmp(argv[i], "--warmup")) && i+1<argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq") && i+1<argc) max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device")  && i+1<argc) { device_index = argv[++i]; device_index_set = true; }
        else if ((!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output")) && i+1<argc) out_fmt = parse_output_format(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { fputs(USAGE, stderr); return 0; }
        else if (argv[i][0] != '-' && model_dir.empty()) model_dir = argv[i];
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); fputs(USAGE, stderr); return 1; }
    }

    if (model_dir.empty() || pp_list.empty() || tg_list.empty() || depths.empty() ||
        reps <= 0 || warmup < 0) {
        fputs(USAGE, stderr);
        return 1;
    }
    for (int value : pp_list)
        if (value <= 0) { fputs("--p values must be positive\n", stderr); return 1; }
    for (int value : tg_list)
        if (value <= 0) { fputs("--n values must be positive\n", stderr); return 1; }

    try {
        if (device_index_set) gpu_device_control::apply_device_index(device_index);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    // Derive max_seq from the largest depth + (pp or tg), plus the largest pp
    // size, if the user did not override it with --max-seq.  Both pp and tg are
    // swept over depths, so the cache must hold depth + max(pp, tg) tokens.
    if (max_seq < 0) {
        int max_d = *std::max_element(depths.begin(), depths.end());
        int max_p = *std::max_element(pp_list.begin(), pp_list.end());
        int max_n = *std::max_element(tg_list.begin(), tg_list.end());
        max_seq = std::max({max_d + max_n, max_d + max_p, max_p});
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

    std::vector<BenchRow> rows;

    // ── Prefill (PP) at various KV-cache depths ─────────────────────────────
    // Mirrors the decode sweep: -d pre-fills the cache to `depth` (untimed,
    // chunked) then times processing `pp` prompt tokens.  The timed prefill is
    // also chunked (CHUNK) so full-attention layers never materialize an
    // O(depth²) score matrix; pp <= CHUNK is a single forward (unchanged).
    for (int pp : pp_list) {
      for (int depth : depths) {
        if (depth + pp > max_seq) {
            rows.push_back({test_name("pp", pp, depth), 0, 0, true});
            continue;
        }
        std::vector<double> times;

        for (int r = 0; r < warmup + reps; ++r) {
            model->reset_cache();

            // Pre-fill KV cache to `depth` tokens in chunks (untimed).
            if (depth > 0) {
                constexpr int CHUNK = 512;
                for (int pos = 0; pos < depth; pos += CHUNK) {
                    int sz = std::min(CHUNK, depth - pos);
                    std::vector<int> chunk(sz, bos_id);
                    model->forward(ForwardInput{chunk, pos});
                }
            }

            // Time `pp` prompt tokens processed in chunks from `depth`.
            double t = now_ms();
            constexpr int CHUNK = 512;
            for (int pos = 0; pos < pp; pos += CHUNK) {
                int sz = std::min(CHUNK, pp - pos);
                std::vector<int> chunk(sz, bos_id);
                model->forward(ForwardInput{chunk, depth + pos});
            }
            double dt = now_ms() - t;

            if (r >= warmup) times.push_back(dt);
        }

        Stats s = compute_stats(times, pp);
        rows.push_back({test_name("pp", pp, depth), s.mean_tps, s.sd_tps, false});
      }
    }

    // ── Decode (TG) at various KV-cache depths ──────────────────────────────
    for (int tg : tg_list) {
      for (int depth : depths) {
        if (depth + tg > max_seq) {
            rows.push_back({test_name("tg", tg, depth), 0, 0, true});
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

        Stats s = compute_stats(times, tg);
        rows.push_back({test_name("tg", tg, depth), s.mean_tps, s.sd_tps, false});
      }
    }

    print_results(out_fmt, model_basename(model_dir), GpuEngine::count(), rows);
    printf("\n");
    return 0;
}

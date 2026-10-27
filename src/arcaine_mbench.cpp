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
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <algorithm>

#include <memory>

#include "common/model_interface.hpp"
#include "modeling/flint/model.hpp"
#include "common/registry.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"
#include "bench/diffusion_model_bench.hpp"

// Reads config.json::model_type without constructing the model.
static std::string read_model_type(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f) return "";
    std::string s((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
    auto p = s.find("\"model_type\"");
    if (p == std::string::npos) return "";
    p = s.find(':', p);
    if (p == std::string::npos) return "";
    p = s.find('"', p);
    if (p == std::string::npos) return "";
    auto q = s.find('"', p + 1);
    if (q == std::string::npos) return "";
    return s.substr(p + 1, q - p - 1);
}

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
        "Usage: bench --model <model_dir> [options]\n"
        "  -h, --help    show this help text\n"
        "  -p, --p P,... prefill sizes          (default: 128,512)\n"
        "  -n, --n N,... new-token counts       (default: 128)\n"
        "  -d D,...      KV-cache depths        (default: 0,512,1024,2048)\n"
        "  -r, --r R     timed repetitions      (default: 3)\n"
        "  -w, --w W     warmup runs            (default: 1)\n"
        "  --max-seq N   KvCache capacity       (default: auto)\n"
        "  --device N    run with one visible Level Zero GPU\n"
        "\n"
        "DiffusionGemma models delegate to the block-diffusion driver, which\n"
        "accepts -ds/--kernels/--layers/--experts/--seed/--md/--print-result.\n";

    // DiffusionGemma uses a different arg set (-ds/--experts/...), so detect
    // the architecture up front and hand off before the AR parser rejects it.
    {
        std::string pre_model;
        for (int i = 1; i + 1 < argc; ++i)
            if (!strcmp(argv[i], "--model") || !strcmp(argv[i], "-m"))
                pre_model = argv[i + 1];
        if (pre_model.empty() && argc > 1 && argv[1][0] != '-')
            pre_model = argv[1];
        if (!pre_model.empty() && read_model_type(pre_model) == "diffusion_gemma")
            return arcaine::bench::run_diffusion_bench(argc, argv);
    }

    std::string model_dir;
    std::vector<int> pp_list  = {128, 512};
    std::vector<int> tg_list  = {128};
    std::vector<int> depths   = {0, 512, 1024, 2048};
    std::string device_index;
    int reps    = 3;
    int warmup  = 1;
    int max_seq = -1;  // computed after arg parsing
    bool device_index_set = false;

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

    // Derive max_seq from the largest depth + n, plus the largest pp size,
    // if the user did not override it with --max-seq.
    if (max_seq < 0) {
        int max_d = *std::max_element(depths.begin(), depths.end());
        int max_p = *std::max_element(pp_list.begin(), pp_list.end());
        int max_n = *std::max_element(tg_list.begin(), tg_list.end());
        max_seq = std::max(max_d + max_n, max_p);
    }

    // ── Load ────────────────────────────────────────────────────────────────
    register_builtin_architectures();

    printf("loading model from %s ...\n", model_dir.c_str());
    double t0 = now_ms();
    std::unique_ptr<Model> model = ModelRegistry::instance().create(model_dir, max_seq);
    double load_s = (now_ms() - t0) * 0.001;

    const ModelInfo& info = model->info();
    const char* prefix_env = std::getenv("ARCAINE_FLINT_MBENCH_GPU_CHECKPOINT");
    const bool reuse_prefix = prefix_env && prefix_env[0]=='1' && prefix_env[1]=='\0';
    auto* flint_model = dynamic_cast<FlintModel*>(model.get());
    if (reuse_prefix && (!flint_model || !model->device_generation_enabled())) {
        std::fprintf(stderr,"GPU prefix checkpoints require Flint GPU decode\n");
        return 1;
    }
    const char* attention_ab_env=std::getenv("ARCAINE_FLINT_MBENCH_ATTN_AB");
    const int attention_ab_mode=attention_ab_env?std::atoi(attention_ab_env):0;
    const bool attention_ab=attention_ab_mode!=0;
    if(attention_ab&&!reuse_prefix){
        std::fprintf(stderr,"Flint attention A/B requires GPU prefix checkpoints\n");return 1;
    }
    const char* original_two_pass_env=std::getenv("ARCAINE_FLINT_FLASH_DECODE_TWO_PASS");
    const char* original_global_split_env=std::getenv("ARCAINE_FLINT_DECODE_GLOBAL_ATTN_SPLIT");
    const bool had_two_pass_env=original_two_pass_env!=nullptr,had_global_split_env=original_global_split_env!=nullptr;
    const std::string original_two_pass=original_two_pass_env?original_two_pass_env:"";
    const std::string original_global_split=original_global_split_env?original_global_split_env:"";
    if(attention_ab_mode==2&&((original_two_pass!="1"&&original_two_pass!="auto")||std::atoi(original_global_split.c_str())<256)){
        std::fprintf(stderr,"Three-way A/B requires initial TWO_PASS=auto and GLOBAL_ATTN_SPLIT>=256 for GPU workspace\n");return 1;
    }
    const char* original_native_env=std::getenv("ARCAINE_FLINT_FLASH_DECODE_NATIVE_EXP");
    const bool had_native_env=original_native_env!=nullptr;
    const std::string original_native=original_native_env?original_native_env:"";
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
            if (model->device_generation_enabled()) model->forward_device(ForwardInput{prompt, 0});
            else model->forward(ForwardInput{prompt, 0});
            double dt = now_ms() - t;
            if (r >= warmup) times.push_back(dt);
        }

        char name[32]; snprintf(name, sizeof name, "pp %d", pp);
        print_row(name, "—", compute_stats(times, pp));
    }

    // ── Decode (TG) at various KV-cache depths ──────────────────────────────
    for (int tg : tg_list) {
      for (int depth : depths) {
        char name[32]; snprintf(name, sizeof name, "tg %d", tg);
        char dstr[32]; snprintf(dstr, sizeof dstr, "%d", depth);

        if (depth + tg > max_seq) {
            print_row(name, dstr, {}, /*skipped=*/true);
            continue;
        }

        auto prepare_prefix = [&] {
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
                    if (model->device_generation_enabled()) model->forward_device(ForwardInput{chunk, pos});
                    else model->forward(ForwardInput{chunk, pos});
                }
            }

        };
        if (reuse_prefix) {
            prepare_prefix();
            flint_model->save_decode_prefix();
        }

        for(int variant=0;variant<(attention_ab?(attention_ab_mode==2?3:2):1);++variant){
            if(attention_ab){
                setenv("ARCAINE_FLINT_FLASH_DECODE_NATIVE_EXP",variant==1?"2":"0",1);
                setenv("ARCAINE_FLINT_FLASH_DECODE_TWO_PASS",variant==2?"auto":"0",1);
                setenv("ARCAINE_FLINT_DECODE_GLOBAL_ATTN_SPLIT",variant==2?"256":"16",1);
                flint_model->clear_decode_graphs();
                snprintf(name,sizeof(name),"tg %d %s",tg,variant==2?"twopass":variant==1?"native":"vector");
            }
        std::vector<double> times;

        for (int r = 0; r < warmup + reps; ++r) {
            if (reuse_prefix) flint_model->restore_decode_prefix();
            else prepare_prefix();

            // Time `tg` single-token decode steps.
            // The cache grows from depth to depth+tg during this window,
            // which is realistic for mid-generation latency measurement.
            double t = now_ms();
            if (!model->decode_repeated_device(bos_id, tg, depth))
                for (int step = 0; step < tg; ++step)
                    model->forward(ForwardInput{single, depth + step});
            double dt = now_ms() - t;

            if (r >= warmup) times.push_back(dt);
        }

        print_row(name, dstr, compute_stats(times, tg));
        }
      }
    }

    if(attention_ab){
        if(had_two_pass_env)setenv("ARCAINE_FLINT_FLASH_DECODE_TWO_PASS",original_two_pass.c_str(),1);
        else unsetenv("ARCAINE_FLINT_FLASH_DECODE_TWO_PASS");
        if(had_global_split_env)setenv("ARCAINE_FLINT_DECODE_GLOBAL_ATTN_SPLIT",original_global_split.c_str(),1);
        else unsetenv("ARCAINE_FLINT_DECODE_GLOBAL_ATTN_SPLIT");
        if(had_native_env)setenv("ARCAINE_FLINT_FLASH_DECODE_NATIVE_EXP",original_native.c_str(),1);
        else unsetenv("ARCAINE_FLINT_FLASH_DECODE_NATIVE_EXP");
    }
    printf("\n");
    return 0;
}

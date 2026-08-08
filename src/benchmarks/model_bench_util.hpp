#pragma once
//
// Shared harness for the rewritten arcaine_mbench (llama-bench approach).
// Covers the CLI contract (help text), synthetic payload building, per-cell
// warmup, chunked prefill, timing/statistics, and the markdown result table.
// Model benches own model construction + engine wiring and drive everything
// through these helpers so the methodology stays in one place.
//
//   ./build/arcaine_mbench -m <dir> [options]
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "benchmarks/util.hpp"  // arcaine::bench::split_csv / parse_int_csv

namespace arcaine::bench {

using Clk = std::chrono::high_resolution_clock;
using Ms  = std::chrono::duration<double, std::milli>;
inline double now_ms() { return Ms(Clk::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------------------
// Command line (llama-bench style)
// ---------------------------------------------------------------------------

// A per-bench extra flag that takes csv ints and is repeatable
// (e.g. diffusion's -ds, --denoising-steps).
struct BenchCsvIntFlag {
    const char* long_name;    // e.g. "denoising-steps"
    const char* short_name;   // e.g. "ds"  (nullptr if none)
    std::vector<int>* out;    // appended to
    bool* set;
};

struct BenchArgs {
    std::string model;                 // -m, --model (required; Default: Error)
    std::vector<int> n_prompt;         // -p, --n-prompt   (default {512})
    std::vector<int> n_gen;            // -n, --n-gen      (default {128})
    std::vector<int> n_depth;          // -d, --n-depth    (default {0})
    int reps = 1;                      // -r, --repetitions
    bool no_warmup = false;            // --no-warmup
    int pchunk = 2048;                 // -pchunk, --chunked-prefill-batch (0 = off)
    bool pchunk_set = false;
    std::string device;                // --device, -dev (empty = auto)
    bool device_set = false;
    std::string experts;               // --experts (diffusion-only effect)
    bool experts_set = false;
};

inline std::vector<int> parse_int_csv_safe(const std::string& s, bool& ok) {
    try {
        ok = true;
        return parse_int_csv(s);
    } catch (const std::exception&) {
        ok = false;
        return {};
    }
}

// Base help text (verbatim contract). `extra_section` is appended by benches
// with model-specific flags (e.g. the diffusion -ds block).
inline void print_mbench_usage(const char* prog, const char* extra_section) {
    std::printf("Usage: %s [options]\n", prog);
    std::printf(
        "\n"
        "-m, --model: Path to model.\n"
        "  Default: Error\n"
        "-p, --n-prompt: Prompt tokens.\n"
        "  Default: 512\n"
        "-n, --n-gen: Max tokens to generate.\n"
        "  Default: 128\n"
        "-d, --n-depth: Simulates existing kv cache,\n"
        "  Default: 0\n"
        "  Behavior: Passing -d alone like -d 4096,8192 does a pp512+tg128 on top.\n"
        "-r, --repetitions: Times to run\n"
        "  Default: 1\n"
        "--no-warmup\n"
        "  Default: One warmup run per test at the test's sizes\n"
        "-pchunk, --chunked-prefill-batch: Chunks  input tokens into smaller batches which can speedup prefill.\n"
        "\n"
        "--device\n"
        "--experts\n");
    if (extra_section && *extra_section)
        std::printf("%s\n", extra_section);
}

// Parse argv per the help text. Long args accept '_' -> '-' normalization
// (llama-bench style). Returns:
//   0  success (args filled; defaults applied for unset flags)
//   1  -h/--help shown (usage already printed to stdout)
//  -1  error (message + usage already printed to stderr)
inline int parse_bench_args(int argc, char** argv,
                            const BenchCsvIntFlag* extras, int n_extras,
                            const char* extra_usage, BenchArgs& args) {
    bool bad = false;
    auto fail = [&](const std::string& msg) {
        std::fprintf(stderr, "error: %s\n", msg.c_str());
        print_mbench_usage(argc > 0 ? argv[0] : "arcaine_mbench", extra_usage);
        bad = true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.compare(0, 2, "--") == 0)
            std::replace(a.begin(), a.end(), '_', '-');

        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string(what) + " requires a value");
                return "";
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            print_mbench_usage(argc > 0 ? argv[0] : "arcaine_mbench", extra_usage);
            return 1;
        } else if (a == "-m" || a == "--model") {
            args.model = next("-m, --model");
        } else if (a == "-p" || a == "--n-prompt") {
            bool ok;
            auto v = parse_int_csv_safe(next("-p, --n-prompt"), ok);
            if (!ok) { fail("invalid -p, --n-prompt value"); continue; }
            args.n_prompt.insert(args.n_prompt.end(), v.begin(), v.end());
        } else if (a == "-n" || a == "--n-gen") {
            bool ok;
            auto v = parse_int_csv_safe(next("-n, --n-gen"), ok);
            if (!ok) { fail("invalid -n, --n-gen value"); continue; }
            args.n_gen.insert(args.n_gen.end(), v.begin(), v.end());
        } else if (a == "-d" || a == "--n-depth") {
            bool ok;
            auto v = parse_int_csv_safe(next("-d, --n-depth"), ok);
            if (!ok) { fail("invalid -d, --n-depth value"); continue; }
            args.n_depth.insert(args.n_depth.end(), v.begin(), v.end());
        } else if (a == "-r" || a == "--repetitions") {
            args.reps = std::atoi(next("-r, --repetitions").c_str());
        } else if (a == "--no-warmup") {
            args.no_warmup = true;
        } else if (a == "-pchunk" || a == "--chunked-prefill-batch") {
            args.pchunk = std::atoi(next("-pchunk, --chunked-prefill-batch").c_str());
            args.pchunk_set = true;
        } else if (a == "--device" || a == "-dev") {
            args.device = next("--device");
            args.device_set = true;
        } else if (a == "--experts") {
            args.experts = next("--experts");
            args.experts_set = true;
        } else {
            bool matched = false;
            for (int e = 0; e < n_extras; ++e) {
                if (a == extras[e].long_name ||
                    (extras[e].short_name && a == extras[e].short_name)) {
                    bool ok;
                    auto v = parse_int_csv_safe(next(extras[e].long_name), ok);
                    if (!ok) {
                        fail(std::string("invalid value for ") + a);
                        matched = true;
                        break;
                    }
                    extras[e].out->insert(extras[e].out->end(), v.begin(), v.end());
                    if (extras[e].set) *extras[e].set = true;
                    matched = true;
                    break;
                }
            }
            if (!matched) fail("Unknown argument: " + a);
        }
        if (bad) break;
    }

    // validate
    for (int p : args.n_prompt) if (p < 0) { fail("--n-prompt values must be >= 0"); break; }
    for (int n : args.n_gen)    if (n < 0) { fail("--n-gen values must be >= 0"); break; }
    for (int d : args.n_depth)  if (d < 0) { fail("--n-depth values must be >= 0"); break; }
    if (args.reps < 1)     fail("--repetitions must be >= 1");
    if (args.pchunk < 0)   fail("--chunked-prefill-batch must be >= 0");
    if (args.model.empty()) fail("--model is required");

    if (bad) return -1;

    // llama-bench default fill: only when a flag was never given.
    if (args.n_prompt.empty()) args.n_prompt = {512};
    if (args.n_gen.empty())    args.n_gen = {128};
    if (args.n_depth.empty())  args.n_depth = {0};

    return 0;
}

// ---------------------------------------------------------------------------
// Synthetic payloads (llama-bench test_prompt / test_gen)
// ---------------------------------------------------------------------------

// Prompt payload: BOS on the first token (when the model has a bos id), then
// rand() % n_vocab. Unseeded std::rand (glibc implicit srand(1) sequence).
inline std::vector<int> build_prompt(int n, int bos_id, int n_vocab) {
    std::vector<int> ids((size_t)n);
    for (int i = 0; i < n; ++i)
        ids[i] = (i == 0 && bos_id >= 0) ? bos_id : std::rand() % n_vocab;
    return ids;
}

// llama-bench test_gen input: first decode step BOS (when the model adds BOS),
// then a fresh random token per step — no logit chaining.
inline int next_decode_id(int step, int bos_id, int n_vocab) {
    return (step == 0 && bos_id >= 0) ? bos_id : std::rand() % n_vocab;
}

// Run `ids` through the engine in `pchunk`-token batches (llama-bench -b
// semantics). pchunk <= 0 or >= size => single forward. fn(ids, past_len)
// performs one engine forward.
template <typename Fn>
inline void chunked_prefill(const std::vector<int>& ids, int past_len,
                            int pchunk, Fn&& fn) {
    int total = (int)ids.size();
    if (total == 0) return;
    if (pchunk <= 0 || total <= pchunk) {
        fn(ids, past_len);
        return;
    }
    for (int off = 0; off < total; off += pchunk) {
        int n = std::min(pchunk, total - off);
        std::vector<int> chunk(ids.begin() + off, ids.begin() + off + n);
        fn(chunk, past_len + off);
    }
}

// ---------------------------------------------------------------------------
// Statistics (llama-bench: t/s per sample, then avg / sample stdev)
// ---------------------------------------------------------------------------

struct PpTgStats {
    double mean_ms = 0.0, sd_ms = 0.0;
    double mean_tps = 0.0, sd_tps = 0.0;
    double ms_per_tok = 0.0;
    int    n_toks = 0;
};

inline PpTgStats pp_tg_stats_from_tps(const std::vector<double>& tps, int n_toks) {
    PpTgStats s;
    s.n_toks = n_toks;
    int n = (int)tps.size();
    if (n == 0 || n_toks <= 0) return s;
    double m = 0, sd = 0;
    for (double t : tps) m += t;
    m /= n;
    for (double t : tps) sd += (t - m) * (t - m);
    sd = n > 1 ? std::sqrt(sd / (n - 1)) : 0.0;
    s.mean_tps = m;
    s.sd_tps   = sd;
    s.mean_ms  = 1000.0 * n_toks / m;
    s.sd_ms    = 1000.0 * n_toks * sd / (m * m);
    s.ms_per_tok = s.mean_ms / n_toks;
    return s;
}

inline PpTgStats compute_pp_tg_stats(const std::vector<double>& ms, int n_toks) {
    std::vector<double> tps;
    tps.reserve(ms.size());
    for (double t : ms)
        if (t > 0.0) tps.push_back(n_toks / (t * 0.001));
    return pp_tg_stats_from_tps(tps, n_toks);
}

// llama-bench test-name format: pp512 / tg128 / pp512+tg64, " @ d512" suffix.
inline std::string test_name(int n_prompt, int n_gen, int n_depth) {
    char buf[64];
    if (n_prompt > 0 && n_gen == 0)
        std::snprintf(buf, sizeof buf, "pp%d", n_prompt);
    else if (n_gen > 0 && n_prompt == 0)
        std::snprintf(buf, sizeof buf, "tg%d", n_gen);
    else
        std::snprintf(buf, sizeof buf, "pp%d+tg%d", n_prompt, n_gen);
    if (n_depth > 0) {
        int len = (int)std::strlen(buf);
        std::snprintf(buf + len, sizeof buf - (size_t)len, " @ d%d", n_depth);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Markdown result table (llama-bench shape, single format).
// Performance data only — model/backend metadata is printed in the header
// lines above, not repeated per row. Columns are fixed-width (llama-bench
// style): numeric columns right-aligned so raw output lines up.
// ---------------------------------------------------------------------------

inline std::string fmt_tps(double mean, double sd) {
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.2f ± %.2f", mean, sd);
    return buf;
}

struct MdTable {
    struct Col { const char* name; int width; bool right; };

    std::vector<Col> cols;

    explicit MdTable(std::vector<Col> c = {{"test", 18, false},
                                            {"prefill_t/s", 12, true},
                                            {"decode_t/s", 15, true}})
        : cols(std::move(c)) {}

    void header() const {
        std::printf("|");
        for (const auto& c : cols)
            std::printf(" %*s |", c.right ? c.width : -c.width, c.name);
        std::printf("\n|");
        for (const auto& c : cols) {
            std::printf(" ");
            for (int i = 0; i < c.width; ++i) std::printf("-");
            if (c.right) std::printf(":");
            std::printf(" |");
        }
        std::printf("\n");
    }

    void row(const std::vector<std::string>& vals) const {
        std::printf("|");
        for (size_t i = 0; i < cols.size() && i < vals.size(); ++i) {
            const Col& c = cols[i];
            std::printf(" %*s |", c.right ? c.width : -c.width, vals[i].c_str());
        }
        std::printf("\n");
    }
};

// ---------------------------------------------------------------------------
// AR matrix driver (qwen3_5 / qwen3_5_moe / gemma4_unified)
// ---------------------------------------------------------------------------

// Engine hooks wrapping one AR model's concrete forward/reset_cache.
struct ArEngine {
    std::function<void(const std::vector<int>&, int past_len)> forward;
    std::function<void()> reset_cache;
    int bos_id = -1;
    int n_vocab = 0;
};

inline int compute_max_seq(const BenchArgs& a) {
    int max_seq = 0;
    for (int d : a.n_depth) {
        for (int p : a.n_prompt) if (p > 0) max_seq = std::max(max_seq, d + p);
        for (int n : a.n_gen)    if (n > 0) max_seq = std::max(max_seq, d + n);
    }
    return max_seq > 0 ? max_seq : 2048;
}

// Runs the full llama-bench-style matrix over one loaded AR model:
// for each depth d: pp <p> cells then tg <n> cells; per-cell warmup at the
// test's size; depth fill before each timed rep; -pchunk batching.
// `backend_line` is the detailed header text (may contain '|'); the table
// carries only performance data.
inline int run_ar_bench(const BenchArgs& args, const ArEngine& eng,
                        const std::string& model_desc, const char* backend_line,
                        double load_s) {
    bool have_pp = std::any_of(args.n_prompt.begin(), args.n_prompt.end(),
                               [](int p) { return p > 0; });
    bool have_tg = std::any_of(args.n_gen.begin(), args.n_gen.end(),
                               [](int n) { return n > 0; });
    if (!have_pp && !have_tg) {
        std::fprintf(stderr, "error: no tests to run (-p and -n are both 0)\n");
        return 1;
    }

    std::printf("model   : %s\n", model_desc.c_str());
    std::printf("backend : %s\n", backend_line);
    std::printf("load    : %.1f s\n", load_s);

    MdTable table;
    table.header();

    const int bos = eng.bos_id;
    const int V   = eng.n_vocab;

    for (int d : args.n_depth) {
        // pp cells
        for (int p : args.n_prompt) {
            if (p <= 0) continue;
            if (!args.no_warmup) {
                eng.reset_cache();
                chunked_prefill(build_prompt(p, bos, V), 0, args.pchunk, eng.forward);
            }
            std::vector<double> times;
            for (int r = 0; r < args.reps; ++r) {
                eng.reset_cache();
                if (d > 0)
                    chunked_prefill(build_prompt(d, bos, V), 0, args.pchunk, eng.forward);
                double t0 = now_ms();
                chunked_prefill(build_prompt(p, bos, V), d, args.pchunk, eng.forward);
                times.push_back(now_ms() - t0);
            }
            std::string name = test_name(p, 0, d);
            PpTgStats s = compute_pp_tg_stats(times, p);
            table.row({name, fmt_tps(s.mean_tps, s.sd_tps), "—"});
        }
        // tg cells
        for (int n : args.n_gen) {
            if (n <= 0) continue;
            if (!args.no_warmup) {
                eng.reset_cache();
                eng.forward(std::vector<int>{next_decode_id(0, bos, V)}, 0);
            }
            std::vector<double> times;
            for (int r = 0; r < args.reps; ++r) {
                eng.reset_cache();
                if (d > 0)
                    chunked_prefill(build_prompt(d, bos, V), 0, args.pchunk, eng.forward);
                double t0 = now_ms();
                for (int step = 0; step < n; ++step)
                    eng.forward(std::vector<int>{next_decode_id(step, bos, V)}, d + step);
                times.push_back(now_ms() - t0);
            }
            std::string name = test_name(0, n, d);
            PpTgStats s = compute_pp_tg_stats(times, n);
            table.row({name, "—", fmt_tps(s.mean_tps, s.sd_tps)});
        }
    }
    std::printf("\n");
    return 0;
}

}  // namespace arcaine::bench

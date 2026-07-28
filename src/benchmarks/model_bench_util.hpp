#pragma once
//
// Shared harness for the llama-bench-style prefill/decode model benchmarks.
// Pure timing + formatting helpers (no model logic); each model-owned
// model_bench.cpp drives its concrete engine's forward()/reset_cache() and
// reports through these helpers so output stays consistent across models.
//
//   reset -> prefill synthetic prompt (PP)        -> time forward(past=0)
//   reset -> populate KV to depth in chunks        -> time single-token decode (TG)
//

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "benchmarks/util.hpp"   // arcaine::bench::parse_int_csv

namespace arcaine::bench {

struct PpTgStats {
    double mean_ms, sd_ms;     // raw timing
    double mean_tps, sd_tps;   // derived tokens/sec (per-rep)
    double ms_per_tok;         // mean_ms / n_toks
    int    n_toks;
};

inline PpTgStats compute_pp_tg_stats(const std::vector<double>& ms, int n_toks) {
    int n = (int)ms.size();
    double s = 0, s2 = 0;
    for (double t : ms) { s += t; s2 += t * t; }
    double mean_ms = s / n;
    double sd_ms   = std::sqrt(std::max(0.0, s2 / n - mean_ms * mean_ms));
    double ts = 0, ts2 = 0;
    for (double t : ms) {
        double v = n_toks / (t * 0.001);
        ts += v; ts2 += v * v;
    }
    double mean_tps = ts / n;
    double sd_tps   = std::sqrt(std::max(0.0, ts2 / n - mean_tps * mean_tps));
    return {mean_ms, sd_ms, mean_tps, sd_tps, mean_ms / n_toks, n_toks};
}

inline void print_pp_tg_header() {
    std::printf("\n %-18s %9s   %10s   %7s   %9s   %8s\n",
                "test", "kv-depth", "t/s", "± sd", "ms/tok", "time(s)");
    std::printf(" %-18s %9s   %10s   %7s   %9s   %8s\n",
                "──────────────────", "─────────",
                "──────────", "───────", "─────────", "────────");
}

inline void print_pp_tg_row(const char* test, const char* depth,
                            const PpTgStats& s, bool skipped = false) {
    if (skipped) {
        std::printf(" %-18s %9s   [skipped: depth+tg > max_seq]\n", test, depth);
        return;
    }
    std::printf(" %-18s %9s   %10.2f   %7.2f   %9.3f   %8.3f\n",
                test, depth, s.mean_tps, s.sd_tps, s.ms_per_tok, s.mean_ms * 0.001);
}

}  // namespace arcaine::bench

#include "activation_plan.hpp"
#include "arena.hpp"
#include "../common/gpu/engine.hpp"
#include "../common/gpu/mem_planner.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

static int prefill_chunk_limit(int max_seq_len) {
    const char* e = std::getenv("DIFF_PREFILL_CHUNK");
    int chunk = e ? std::atoi(e) : 2048;
    if (chunk <= 0) return std::max(1, max_seq_len);
    return std::max(1, std::min(chunk, max_seq_len));
}

static size_t planned_activation_peak(const DiffConfig& cfg, int max_seq_len) {
    max_seq_len = std::max(1, max_seq_len);
    int pf_chunk = prefill_chunk_limit(max_seq_len);
    int pf_pastlen = std::max(0, max_seq_len - pf_chunk);

    auto pf = memplan::build_prefill_graph(cfg, pf_chunk, pf_pastlen);
    auto dec = memplan::build_decode_graph(cfg, cfg.canvas_length, max_seq_len);
    size_t pf_peak = memplan::plan(pf);
    size_t dec_peak = memplan::plan(dec);
    return std::max(pf_peak, dec_peak);
}

void reserve_activation_arenas(const DiffConfig& cfg, int max_seq_len) {
    if (diffarena::disabled()) return;
    size_t bytes = planned_activation_peak(cfg, max_seq_len);
    for (int g = 0; g < GpuEngine::count(); ++g)
        diffarena::reserve(g, bytes);
    std::printf("[model] activation arena: %.2f GB/GPU reserved (planner peak-live)\n",
                bytes / (1024.0 * 1024.0 * 1024.0));
}

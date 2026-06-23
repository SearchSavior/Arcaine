#include "profile.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace diffprof {

static std::map<std::string, double> g_times;
static std::map<std::string, long> g_counts;

bool enabled() {
    static bool e = std::getenv("DIFF_PROFILE") != nullptr;
    return e;
}

void add(const char* name, double seconds) {
    g_times[name] += seconds;
    g_counts[name] += 1;
}

void reset() {
    g_times.clear();
    g_counts.clear();
}

std::chrono::steady_clock::time_point tic(sycl::queue& q) {
    if (enabled()) q.wait();
    return std::chrono::steady_clock::now();
}

void toc(sycl::queue& q, const char* name,
         std::chrono::steady_clock::time_point t0) {
    if (!enabled()) return;
    q.wait();
    add(name, std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t0).count());
}

ScopedGpu::ScopedGpu(sycl::queue& queue, const char* nm)
    : q(&queue), name(nm), on(enabled()) {
    if (on) {
        q->wait();
        t0 = std::chrono::steady_clock::now();
    }
}

ScopedGpu::~ScopedGpu() {
    if (on) {
        q->wait();
        double s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
        add(name, s);
    }
}

void report() {
    if (!enabled() || g_times.empty()) return;
    double total = 0;
    for (auto& kv : g_times) total += kv.second;
    std::vector<std::pair<std::string, double>> v(g_times.begin(), g_times.end());
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    std::fprintf(stderr,
                 "\n[profile] forward-pass phase breakdown (DIFF_PROFILE), "
                 "summed profiled GPU time = %.6f s\n", total);
    for (auto& kv : v)
        std::fprintf(stderr, "[profile] %-36s %10.6f s  %5.1f%%  (%ld calls)\n",
                     kv.first.c_str(), kv.second, 100.0 * kv.second / total,
                     g_counts[kv.first]);
}

}  // namespace diffprof

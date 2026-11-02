#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

// Flint sampler (new; independent of the common sampler). Temperature, top-k,
// top-p nucleus filtering, or greedy when temperature <= 0.
inline int flint_sample(const std::vector<float>& logits, float temperature,
                        int top_k, float top_p, uint64_t& rng) {
    if (logits.empty()) return -1;
    if (temperature <= 0.f || top_k == 1) {
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    const int n = (int)logits.size();
    const int k = (top_k > 0 && top_k < n) ? top_k : n;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) {
                          if (logits[a] != logits[b]) return logits[a] > logits[b];
                          return a < b;
                      });
    idx.resize(k);

    const float mx = logits[idx[0]];
    std::vector<float> p(k);
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
        p[i] = std::exp((logits[idx[i]] - mx) / temperature);
        sum += p[i];
    }
    for (float& v : p) v /= sum;

    if (top_p > 0.f && top_p < 1.f) {
        float c = 0.f;
        int keep = 0;
        for (; keep < k; ++keep) {
            c += p[keep];
            if (c >= top_p) {
                ++keep;
                break;
            }
        }
        keep = std::max(keep, 1);
        p.resize(keep);
        idx.resize(keep);
        float s = std::accumulate(p.begin(), p.end(), 0.f);
        for (float& v : p) v /= s;
    }

    // xorshift64* -> uniform [0,1)
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    const uint64_t r = rng * 2685821657736338717ull;
    const float u = (float)(r >> 11) * (1.f / 9007199254740992.f);
    float c = 0.f;
    for (size_t i = 0; i < p.size(); ++i) {
        c += p[i];
        if (u < c) return idx[i];
    }
    return idx.back();
}

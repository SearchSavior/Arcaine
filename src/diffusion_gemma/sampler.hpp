#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstdint>

// Linear temperature schedule. cur_step counts DOWN from N to 1.
inline float diff_temperature(int cur_step, int N, float t_min, float t_max) {
    return t_min + (t_max - t_min) * ((float)cur_step / (float)N);
}

// Entropy-bound acceptance + renoise.
//   accept the lowest-entropy prefix whose cumulative entropy (excluding the
//   current token) stays <= entropy_bound; accepted positions take the denoiser
//   token, rejected positions are renoised with a fresh uniform token.
// Returns the next canvas. `accepted_out` (optional) receives the accept mask.
inline std::vector<int> entropy_bound_accept_renoise(
    const std::vector<int>&   current,
    const std::vector<int>&   denoiser,
    const std::vector<float>& entropy,
    float entropy_bound,
    int vocab_size,
    std::mt19937& rng,
    std::vector<char>* accepted_out = nullptr)
{
    int n = (int)current.size();
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return entropy[a] < entropy[b]; });

    std::vector<char> accept(n, 0);
    float cum = 0.0f;
    for (int j = 0; j < n; ++j) {
        int p = order[j];
        float before = cum;        // sum of entropies of more-confident positions
        cum += entropy[p];
        accept[p] = (before <= entropy_bound) ? 1 : 0;
    }

    std::uniform_int_distribution<int> uni(0, vocab_size - 1);
    std::vector<int> next(n);
    for (int p = 0; p < n; ++p)
        next[p] = accept[p] ? denoiser[p] : uni(rng);

    if (accepted_out) *accepted_out = accept;
    return next;
}

// Stable-and-confident adaptive stopping (batch size 1).
struct DiffStopping {
    int   stability_threshold;
    float confidence_threshold;
    std::vector<int> prev_argmax;   // history of length 1 (threshold==1)
    bool  have_prev = false;

    DiffStopping(int stab, float conf)
        : stability_threshold(stab), confidence_threshold(conf) {}

    void reset() { have_prev = false; prev_argmax.clear(); }

    // Returns true when the canvas is both stable and confident.
    bool update(const std::vector<int>& argmax, const std::vector<float>& entropy) {
        bool stable;
        if (stability_threshold == 0) {
            stable = true;
        } else {
            stable = have_prev && (prev_argmax == argmax);
            prev_argmax = argmax;
            have_prev = true;
        }
        double mean = 0.0;
        for (float e : entropy) mean += e;
        mean /= (double)entropy.size();
        bool confident = mean < confidence_threshold;
        return stable && confident;
    }
};

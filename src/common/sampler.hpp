#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

inline int sample_token(
    const float* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    std::mt19937& rng
) {
    // Greedy for temp == 0
    if (temperature <= 0.0f) {
        return static_cast<int>(
            std::max_element(logits, logits + vocab_size) - logits);
    }

    // Divide by temperature
    std::vector<float> scaled(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
        scaled[i] = logits[i] / temperature;

    // Top-K selection
    int k = std::min(top_k, vocab_size);
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&](int a, int b) { return scaled[a] > scaled[b]; });
    indices.resize(k);

    // Softmax over top-K
    float max_v = scaled[indices[0]];
    std::vector<float> probs(k);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(scaled[indices[i]] - max_v);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;

    // Nucleus (top-p) filtering
    std::vector<int> sorted_k(k);
    std::iota(sorted_k.begin(), sorted_k.end(), 0);
    std::sort(sorted_k.begin(), sorted_k.end(),
        [&](int a, int b) { return probs[a] > probs[b]; });

    float cumsum = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < k; ++i) {
        cumsum += probs[sorted_k[i]];
        ++nucleus_size;
        if (cumsum >= top_p) break;
    }

    // Zero out tokens outside nucleus and renormalize
    std::vector<float> final_probs(nucleus_size);
    std::vector<int>   final_ids(nucleus_size);
    float final_sum = 0.0f;
    for (int i = 0; i < nucleus_size; ++i) {
        int idx = sorted_k[i];
        final_probs[i] = probs[idx];
        final_ids[i]   = indices[idx];
        final_sum += probs[idx];
    }
    for (float& p : final_probs) p /= final_sum;

    std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
    return final_ids[dist(rng)];
}

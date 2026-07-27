#pragma once

#include <random>

namespace arcaine::qwen3_5 {

// Qwen3.5-owned top-k / top-p / greedy sampler. A copy of the project sampler
// kept inside this module so the model owns its sampling policy.
int sample_token(const float* logits, int vocab_size,
                 float temperature, int top_k, float top_p,
                 std::mt19937& rng);

}  // namespace arcaine::qwen3_5

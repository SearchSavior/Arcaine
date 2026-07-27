#pragma once

#include <random>

namespace arcaine::gemma4_unified {

// Gemma4-owned top-k / top-p / greedy sampler. A copy of the project sampler
// kept inside this module so the model owns its sampling policy (no
// cross-model sampler dependency).
int sample_token(const float* logits, int vocab_size,
                 float temperature, int top_k, float top_p,
                 std::mt19937& rng);

}  // namespace arcaine::gemma4_unified

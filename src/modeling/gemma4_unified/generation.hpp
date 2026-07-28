#pragma once
#include <vector>

// Sampling / generation defaults parsed from generation_config.json.
// Architecture-independent — shared by every model implementation.
struct GenerationConfig {
    int              bos_token_id = -1;
    int              pad_token_id = -1;
    std::vector<int> eos_token_ids;
    std::vector<int> suppress_tokens;
    float            temperature = 1.0f;
    int              top_k = 64;
    float            top_p = 0.95f;
};

#pragma once
#include <vector>
#include <array>
#include <string>
#include <cstdint>

struct ImageInput {
    std::vector<float>             pixel_values;    // (max_soft_tokens, 6912) padded
    std::vector<std::array<int,2>> position_ids;    // (max_soft_tokens, 2) — -1 for padding
    int                            num_valid_patches;
    int                            max_patches = 280;
};

// Decode PNG/JPEG and run the full CPU preprocessing pipeline.
ImageInput preprocess_image(const std::string& path,
                             int patch_size,
                             int pooling_kernel,
                             int max_soft_tokens,
                             float rescale_factor,
                             bool do_rescale);

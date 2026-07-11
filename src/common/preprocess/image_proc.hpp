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
    // Qwen VL fields. Gemma leaves these at their defaults.
    std::array<int,3>              grid_thw = {0, 0, 0};
    int                            raw_patches = 0;
    int                            patch_dim = 0;
    int                            modality_type = 1;  // 1=image, 2=video
};

// Decode PNG/JPEG and run the full CPU preprocessing pipeline.
ImageInput preprocess_image(const std::string& path,
                             int patch_size,
                             int pooling_kernel,
                             int max_soft_tokens,
                             float rescale_factor,
                             bool do_rescale);

// Qwen2/3-VL image processor contract used by Qwen3.5: smart resize to a
// multiple of patch_size*merge_size, RGB rescale+normalize, temporal frame
// duplication, and merge-major patch ordering.
ImageInput preprocess_qwen_image(const std::string& path,
                                 int patch_size,
                                 int temporal_patch_size,
                                 int merge_size,
                                 int min_pixels,
                                 int max_pixels,
                                 float rescale_factor,
                                 bool do_rescale,
                                 bool do_normalize,
                                 const std::vector<float>& mean,
                                 const std::vector<float>& stddev);

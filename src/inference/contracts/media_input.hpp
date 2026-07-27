#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace arcaine::inference {

// Neutral, model-agnostic media representation. A project-level media loader
// may decode files into these containers; each model decides how the contained
// samples become embeddings or model tokens (model-specific normalization,
// patching, framing, etc. live inside the owning model module).
//
// The struct is a superset container: a model fills only the fields its
// preprocessing pipeline uses and leaves the rest at defaults.

struct ImageInput {
    std::vector<float>             pixel_values;                 // padded patches
    std::vector<std::array<int,2>> position_ids;                 // (n,2); -1 = pad
    int                            num_valid_patches = 0;
    int                            max_patches       = 280;

    // Vision-language fields used by Qwen-family image processors; Gemma
    // leaves these at their defaults.
    std::array<int,3>              grid_thw       = {0, 0, 0};
    int                            raw_patches    = 0;
    int                            patch_dim      = 0;
    int                            modality_type  = 1;  // 1=image, 2=video
};

struct AudioInput {
    std::vector<float> frames;       // (num_frames, frame_size) raw waveform chunks
    int                num_frames = 0;
};

}  // namespace arcaine::inference

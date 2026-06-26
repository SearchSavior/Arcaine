#pragma once
#include "../../common/gpu/buffer.hpp"
#include "weights.hpp"
#include "config.hpp"
#include "../../common/preprocess/image_proc.hpp"

// Run the GPU vision embedding pipeline.
// Returns device buffer of shape (num_valid_patches, hidden_size).
GpuBuffer<bf16> vision_embedder_forward(
    const VisionWeights& w,
    const ImageInput& img,
    const VisionConfig& cfg,
    int hidden_size
);

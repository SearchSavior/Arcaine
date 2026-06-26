#pragma once
#include "../../common/gpu/buffer.hpp"
#include "weights.hpp"
#include "config.hpp"
#include "../../common/preprocess/audio_proc.hpp"

// Run the GPU audio embedding pipeline.
// Returns device buffer of shape (num_frames, hidden_size).
GpuBuffer<bf16> audio_embedder_forward(
    const AudioWeights& w,
    const AudioInput& audio,
    const AudioConfig& cfg,
    int hidden_size
);

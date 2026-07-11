#pragma once

#include "config.hpp"
#include "weights.hpp"
#include "../../common/gpu/buffer.hpp"
#include "../../common/preprocess/image_proc.hpp"

GpuBuffer<bf16> qwen35_vision_forward(const Qwen35VisionWeights& weights,
                                      const Qwen35VisionConfig& config,
                                      const ImageInput& image);

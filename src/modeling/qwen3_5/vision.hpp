#pragma once

#include "config.hpp"
#include "weights.hpp"
#include "../../runtime/gpu/buffer.hpp"
#include "../../preprocessing/image_processor.hpp"

GpuBuffer<bf16> qwen35_vision_forward(const Qwen35VisionWeights& weights,
                                      const Qwen35VisionConfig& config,
                                      const ImageInput& image);

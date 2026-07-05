#pragma once

#include <string>
#include "../../common/io/quant_loader.hpp"
#include "config.hpp"
#include "weights.hpp"

// Load Qwen3.5-MoE weights from a sharded safetensors source into device
// buffers, stripping the `model.language_model.` checkpoint prefix and folding
// the NVFP4 dst_scale. If max_layers >= 0, only layers [0, max_layers) are
// loaded (single-GPU smoke testing); -1 loads all cfg.num_hidden_layers.
QwenWeights load_qwen_weights(const TensorSource& sf, const QwenConfig& cfg,
                              sycl::queue& q, int max_layers = -1);

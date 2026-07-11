#pragma once

#include "config.hpp"
#include "weights.hpp"
#include "../../common/io/quant_loader.hpp"

Qwen35Weights load_qwen35_weights(
    const ShardedSafetensors& checkpoint,
    const Qwen35Config& config,
    int split_layer,
    int max_layers = -1);

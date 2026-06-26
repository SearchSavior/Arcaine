#pragma once

#include <string>
#include "config.hpp"
#include "weights.hpp"
#include "../../common/gpu/placement.hpp"

DiffWeights load_diffusion_weights(const std::string& model_dir,
                                   const DiffConfig& cfg,
                                   int split_layer,
                                   DiffExpertPlacementMode expert_mode);

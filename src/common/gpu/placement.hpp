#pragma once
#include "../../diffusion_gemma/config.hpp"

#include <string>

// User-facing placement knobs.  Auto preserves the current default behavior.
enum class DiffLayerPlacementMode { Auto, Single, Split };
enum class DiffExpertPlacementMode { Auto, LayerOwner, Shard };

struct DiffPlacementOptions {
    DiffLayerPlacementMode layer_mode = DiffLayerPlacementMode::Auto;
    DiffExpertPlacementMode expert_mode = DiffExpertPlacementMode::Auto;
    int layer_split = -1;  // only used by LayerPlacementMode::Split
};

const char* layer_placement_name(DiffLayerPlacementMode mode);
const char* expert_placement_name(DiffExpertPlacementMode mode);
DiffPlacementOptions resolve_diffusion_placement(const DiffConfig& cfg, DiffPlacementOptions placement);
DiffExpertPlacementMode resolve_expert_placement(DiffExpertPlacementMode mode);

int resolve_diffusion_split_layer(const DiffConfig& cfg, const DiffPlacementOptions& placement);
void print_diffusion_placement(const DiffConfig& cfg, int split_layer, const DiffPlacementOptions& placement);

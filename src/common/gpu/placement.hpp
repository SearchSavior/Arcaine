#pragma once
#include "../../modeling/diffusion_gemma/config.hpp"

#include <string>
#include <utility>
#include <vector>

// User-facing placement knobs.  Auto preserves the current default behavior.
enum class DiffLayerPlacementMode { Auto, Single, Split };
enum class DiffExpertPlacementMode { Auto, LayerOwner, Shard };

struct DiffPlacementOptions {
    DiffLayerPlacementMode layer_mode = DiffLayerPlacementMode::Auto;
    DiffExpertPlacementMode expert_mode = DiffExpertPlacementMode::Auto;
    int layer_split = -1;  // only used by LayerPlacementMode::Split
    // Per-GPU expert counts for Shard mode (--experts ranges:N,N,...).
    // Empty means an even split across all GPU engines.
    std::vector<int> expert_counts;
};

const char* layer_placement_name(DiffLayerPlacementMode mode);
const char* expert_placement_name(DiffExpertPlacementMode mode);
DiffPlacementOptions resolve_diffusion_placement(const DiffConfig& cfg, DiffPlacementOptions placement);
DiffExpertPlacementMode resolve_expert_placement(DiffExpertPlacementMode mode);

int resolve_diffusion_split_layer(const DiffConfig& cfg, const DiffPlacementOptions& placement);
void print_diffusion_placement(const DiffConfig& cfg, int split_layer, const DiffPlacementOptions& placement);

// Parses "48,80" into per-GPU expert counts {48, 80}.
std::vector<int> parse_expert_ranges(const std::string& csv);
// Resolves [first, last) expert bounds per GPU. With empty counts this is the
// even split; otherwise counts must have exactly G entries summing to E.
std::vector<std::pair<int, int>> expert_shard_bounds(int E, int G, const std::vector<int>& counts);

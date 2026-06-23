#include "placement.hpp"
#include "engine.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static size_t bf16_bytes(size_t elements) { return 2 * elements; }

static size_t linear_bytes(size_t out_features, size_t in_features, bool nvfp4) {
    if (!nvfp4) return bf16_bytes(out_features * in_features);
    size_t packed = out_features * ((in_features + 1) / 2);
    size_t scales = out_features * ((in_features + 15) / 16);
    return packed + scales + 2 * sizeof(float);
}

static size_t attention_bytes(const DiffTextConfig& t, bool full) {
    size_t H = t.hidden_size;
    size_t attn;
    if (full) {
        size_t hd = t.global_head_dim;
        attn = (size_t)t.num_attn_heads * hd * H * 2 + (size_t)t.num_global_kv_heads * hd * H;
    } else {
        size_t hd = t.head_dim;
        attn = (size_t)t.num_attn_heads * hd * H * 2 + (size_t)t.num_kv_heads * hd * H * 2;
    }
    return bf16_bytes(attn);
}

static size_t dense_mlp_bytes(const DiffTextConfig& t, bool nvfp4) {
    size_t H = t.hidden_size;
    size_t I = t.intermediate_size;
    return 2 * linear_bytes(I, H, nvfp4) + linear_bytes(H, I, nvfp4);
}

static size_t one_expert_bytes(const DiffTextConfig& t, bool nvfp4) {
    size_t H = t.hidden_size;
    size_t I = t.moe_intermediate_size;
    return 2 * linear_bytes(I, H, nvfp4) + linear_bytes(H, I, nvfp4);
}

static size_t layer_bytes(const DiffTextConfig& t, bool full, bool nvfp4) {
    size_t H = t.hidden_size;
    size_t router = bf16_bytes((size_t)t.num_experts * H);
    size_t layernorms = bf16_bytes(7 * H);
    return attention_bytes(t, full) + dense_mlp_bytes(t, nvfp4) +
           (size_t)t.num_experts * one_expert_bytes(t, nvfp4) +
           router + layernorms;
}

static size_t expert_bytes_per_layer(const DiffTextConfig& t, bool nvfp4) {
    return (size_t)t.num_experts * one_expert_bytes(t, nvfp4);
}

static size_t local_layer_bytes_excluding_experts(const DiffTextConfig& t, bool full, bool nvfp4) {
    size_t H = t.hidden_size;
    size_t router = bf16_bytes((size_t)t.num_experts * H);
    size_t layernorms = bf16_bytes(7 * H);
    return attention_bytes(t, full) + dense_mlp_bytes(t, nvfp4) + router + layernorms;
}

static size_t gpu0_global_bytes(const DiffTextConfig& t) {
    size_t H = t.hidden_size;
    size_t embed = (size_t)t.vocab_size * H;
    size_t final_norm = H;
    size_t self_cond = H + 3 * (size_t)t.intermediate_size * H;
    return bf16_bytes(embed + final_norm + self_cond);
}

} // namespace

const char* layer_placement_name(DiffLayerPlacementMode mode) {
    switch (mode) {
        case DiffLayerPlacementMode::Auto: return "auto";
        case DiffLayerPlacementMode::Single: return "single";
        case DiffLayerPlacementMode::Split: return "split";
    }
    return "unknown";
}

const char* expert_placement_name(DiffExpertPlacementMode mode) {
    switch (mode) {
        case DiffExpertPlacementMode::Auto: return "auto";
        case DiffExpertPlacementMode::LayerOwner: return "layer-owner";
        case DiffExpertPlacementMode::Shard: return "shard";
    }
    return "unknown";
}

DiffPlacementOptions resolve_diffusion_placement(const DiffConfig& cfg, DiffPlacementOptions placement) {
    if (cfg.is_nvfp4_quantized()) {
        if (placement.layer_mode == DiffLayerPlacementMode::Auto) {
            placement.layer_mode = DiffLayerPlacementMode::Single;
            placement.layer_split = -1;
        }
        if (placement.expert_mode == DiffExpertPlacementMode::Auto)
            placement.expert_mode = DiffExpertPlacementMode::LayerOwner;
    }
    return placement;
}

DiffExpertPlacementMode resolve_expert_placement(DiffExpertPlacementMode mode) {
    if (mode != DiffExpertPlacementMode::Auto) return mode;
    return GpuEngine::count() >= 2 ? DiffExpertPlacementMode::Shard : DiffExpertPlacementMode::LayerOwner;
}

int resolve_diffusion_split_layer(const DiffConfig& cfg, const DiffPlacementOptions& placement) {
    int L = cfg.text.num_hidden_layers;
    if (placement.layer_mode == DiffLayerPlacementMode::Single) return L;
    if (placement.layer_mode == DiffLayerPlacementMode::Split) {
        if (placement.layer_split < 0 || placement.layer_split > L)
            throw std::runtime_error("--layers split:N must use N in [0, " + std::to_string(L) + "]");
        if (GpuEngine::count() < 2 && placement.layer_split != L)
            throw std::runtime_error("--layers split:N below layer count requires at least 2 GPU engines");
        return placement.layer_split;
    }
    if (GpuEngine::count() < 2) return L;

    size_t embed_b = 2ull * cfg.text.vocab_size * cfg.text.hidden_size;
    size_t overhead0 = 2 * embed_b;
    std::vector<size_t> lb(L);
    size_t total = 0;
    for (int l = 0; l < L; ++l) {
        lb[l] = layer_bytes(cfg.text, cfg.text.is_full_attention[l], cfg.is_nvfp4_quantized());
        total += lb[l];
    }

    int split_layer = L;
    size_t best_diff = SIZE_MAX, prefix = 0;
    for (int s = 1; s < L; ++s) {
        prefix += lb[s - 1];
        size_t g0 = overhead0 + prefix, g1 = total - prefix;
        size_t d = (g0 > g1) ? g0 - g1 : g1 - g0;
        if (d < best_diff) { best_diff = d; split_layer = s; }
    }
    return split_layer;
}

void print_diffusion_placement(const DiffConfig& cfg, int split_layer, const DiffPlacementOptions& placement) {
    int G = GpuEngine::count();
    int L = cfg.text.num_hidden_layers;
    int E = cfg.text.num_experts;
    DiffExpertPlacementMode expert_mode = resolve_expert_placement(placement.expert_mode);
    std::vector<size_t> local_bytes(G, 0), expert_bytes(G, 0), total_bytes(G, 0);

    local_bytes[0] += gpu0_global_bytes(cfg.text);
    for (int l = 0; l < L; ++l) {
        int owner = (l < split_layer) ? 0 : 1;
        if (owner >= G) owner = 0;
        local_bytes[owner] += local_layer_bytes_excluding_experts(cfg.text, cfg.text.is_full_attention[l], cfg.is_nvfp4_quantized());
    }

    size_t expert_layer = expert_bytes_per_layer(cfg.text, cfg.is_nvfp4_quantized());
    if (expert_mode == DiffExpertPlacementMode::Shard) {
        for (int g = 0; g < G; ++g) {
            int first = g * E / G;
            int last = (g + 1) * E / G;
            expert_bytes[g] = expert_layer * (size_t)(last - first) / (size_t)E * (size_t)L;
        }
    } else {
        for (int l = 0; l < L; ++l) {
            int owner = (l < split_layer) ? 0 : 1;
            if (owner >= G) owner = 0;
            expert_bytes[owner] += expert_layer;
        }
    }
    for (int g = 0; g < G; ++g) total_bytes[g] = local_bytes[g] + expert_bytes[g];

    std::printf("[placement] placement: layers=%s%s experts=%s -> %s\n",
                layer_placement_name(placement.layer_mode),
                placement.layer_mode == DiffLayerPlacementMode::Split ? (":" + std::to_string(placement.layer_split)).c_str() : "",
                expert_placement_name(placement.expert_mode),
                expert_placement_name(expert_mode));
    std::printf("[placement] detected GPU engines: %d\n", G);
    for (int g = 0; g < G; ++g) {
        auto dev = GpuEngine::get(g).queue.get_device();
        std::string name = dev.get_info<sycl::info::device::name>();
        double total_gb = dev.get_info<sycl::info::device::global_mem_size>() / 1e9;
        std::printf("[placement] GPU%d: %s (%.1f GB global memory)\n", g, name.c_str(), total_gb);
    }

    std::printf("[placement] layers:\n");
    if (split_layer > 0)
        std::printf("[placement]   GPU0: layers [0, %d)\n", split_layer);
    if (split_layer < L) {
        int owner = (G >= 2) ? 1 : 0;
        std::printf("[placement]   GPU%d: layers [%d, %d)\n", owner, split_layer, L);
    }
    for (int g = 2; g < G; ++g)
        std::printf("[placement]   GPU%d: no transformer layers\n", g);

    std::printf("[placement] experts:\n");
    if (expert_mode == DiffExpertPlacementMode::Shard) {
        for (int g = 0; g < G; ++g) {
            int first = g * E / G;
            int last = (g + 1) * E / G;
            std::printf("[placement]   GPU%d: experts [%d, %d) for each MoE layer\n", g, first, last);
        }
    } else {
        for (int l = 0; l < L; ++l) {
            int owner = (l < split_layer) ? 0 : 1;
            if (owner >= G) owner = 0;
            std::printf("[placement]   layer %d owner GPU%d: experts [0, %d) local\n", l, owner, E);
        }
    }

    std::printf("[placement] estimated model weights by GPU:\n");
    for (int g = 0; g < G; ++g) {
        std::printf("[placement]   GPU%d: total %.2f GB = layer/local %.2f GB + expert-shard %.2f GB\n",
                    g, total_bytes[g] / 1e9, local_bytes[g] / 1e9, expert_bytes[g] / 1e9);
    }
}

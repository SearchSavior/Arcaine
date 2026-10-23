#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

// Flint (RecursiveSI/flint-int4) configuration.
// Text-only, MLX affine-int4 checkpoint. See docs/flint-int4-inventory.md.
struct FlintConfig {
    std::string model_dir;
    std::string model_type;
    bool awq = false;

    int hidden_size = 2048;
    int intermediate_size = 6144;
    int moe_intermediate_size = 512;
    int shared_expert_intermediate_size = 1024;
    int num_hidden_layers = 49;
    int num_attention_heads = 32;
    int num_key_value_heads = 4;
    int head_dim = 128;
    int num_experts = 256;
    int num_experts_per_tok = 6;
    int vocab_size = 150144;

    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    int sliding_window = 513;

    std::vector<std::string> layer_types;   // "full_attention" | "sliding_attention"
    std::vector<int> no_rope_layer_indices; // == full-attention layers in this checkpoint
    std::vector<int> mlp_only_layers;       // dense MLP layers, [0, 1]

    int bos_token_id = 150000;
    std::vector<int> eos_token_ids;
    int pad_token_id = 150001;

    // generation_config.json
    float temperature = 1.0f;
    int top_k = 20;
    float top_p = 0.95f;

    bool is_global(int layer) const {
        if (layer < 0 || layer >= (int)layer_types.size()) return true;
        return layer_types[layer] == "full_attention";
    }
    bool use_rope(int layer) const {
        for (int i : no_rope_layer_indices)
            if (i == layer) return false;
        return true;
    }
    bool is_dense_mlp(int layer) const {
        for (int i : mlp_only_layers)
            if (i == layer) return true;
        return false;
    }
    int head_dim_effective() const { return head_dim; }
    int q_dim() const { return num_attention_heads * head_dim; }
    int kv_dim() const { return num_key_value_heads * head_dim; }

    static FlintConfig from_dir(const std::string& dir) {
        FlintConfig c;
        c.model_dir = dir;
        std::ifstream f(dir + "/config.json");
        if (!f) throw std::runtime_error("flint: cannot open " + dir + "/config.json");
        nlohmann::json j;
        f >> j;

        c.model_type = j.value("model_type", "");
        if (j.contains("quantization_config") &&
            j["quantization_config"].value("quant_method", "") == "awq") {
            const auto& qc = j.at("quantization_config");
            if (qc.at("bits").get<int>() != 4 ||
                qc.at("group_size").get<int>() != 64 ||
                !qc.at("zero_point").get<bool>() ||
                qc.at("version").get<std::string>() != "gemm")
                throw std::runtime_error("flint: only AWQ GEMM W4A16 group-64 with zero points is supported");
            c.awq = true;
        }
        c.hidden_size = j.value("hidden_size", c.hidden_size);
        c.intermediate_size = j.value("intermediate_size", c.intermediate_size);
        c.moe_intermediate_size = j.value("moe_intermediate_size", c.moe_intermediate_size);
        c.shared_expert_intermediate_size =
            j.value("shared_expert_intermediate_size", c.shared_expert_intermediate_size);
        c.num_hidden_layers = j.value("num_hidden_layers", c.num_hidden_layers);
        c.num_attention_heads = j.value("num_attention_heads", c.num_attention_heads);
        c.num_key_value_heads = j.value("num_key_value_heads", c.num_key_value_heads);
        c.head_dim = j.value("head_dim", c.head_dim);
        c.num_experts = j.value("num_local_experts", j.value("num_experts", c.num_experts));
        c.num_experts_per_tok = j.value("num_experts_per_tok", c.num_experts_per_tok);
        c.vocab_size = j.value("vocab_size", c.vocab_size);
        c.rms_norm_eps = j.value("rms_norm_eps", c.rms_norm_eps);
        c.rope_theta = j.value("rope_theta", c.rope_theta);
        c.sliding_window = j.value("sliding_window", c.sliding_window);
        c.bos_token_id = j.value("bos_token_id", c.bos_token_id);
        c.pad_token_id = j.value("pad_token_id", c.pad_token_id);

        if (j.contains("layer_types") && j["layer_types"].is_array())
            for (auto& v : j["layer_types"]) c.layer_types.push_back(v.get<std::string>());
        if (c.layer_types.empty())
            c.layer_types.assign(c.num_hidden_layers, "full_attention");

        if (j.contains("no_rope_layer_indices") && j["no_rope_layer_indices"].is_array())
            for (auto& v : j["no_rope_layer_indices"])
                c.no_rope_layer_indices.push_back(v.get<int>());
        if (j.contains("mlp_only_layers") && j["mlp_only_layers"].is_array())
            for (auto& v : j["mlp_only_layers"]) c.mlp_only_layers.push_back(v.get<int>());

        if (j.contains("eos_token_id")) {
            if (j["eos_token_id"].is_array())
                for (auto& v : j["eos_token_id"]) c.eos_token_ids.push_back(v.get<int>());
            else
                c.eos_token_ids.push_back(j["eos_token_id"].get<int>());
        }

        if (j.contains("rope_parameters") && j["rope_parameters"].is_object())
            c.rope_theta = j["rope_parameters"].value("rope_theta", c.rope_theta);

        std::ifstream g(dir + "/generation_config.json");
        if (g) {
            nlohmann::json gj;
            g >> gj;
            c.temperature = gj.value("temperature", c.temperature);
            c.top_p = gj.value("top_p", c.top_p);
            if (gj.contains("top_k") && gj["top_k"].is_number_integer())
                c.top_k = gj["top_k"].get<int>();
            if (gj.contains("eos_token_id")) {
                c.eos_token_ids.clear();
                if (gj["eos_token_id"].is_array())
                    for (auto& v : gj["eos_token_id"]) c.eos_token_ids.push_back(v.get<int>());
                else
                    c.eos_token_ids.push_back(gj["eos_token_id"].get<int>());
            }
        }

        if (c.model_type != "flint")
            throw std::runtime_error("flint: unexpected model_type '" + c.model_type + "'");
        if (c.head_dim != 128 || c.num_attention_heads % c.num_key_value_heads != 0)
            throw std::runtime_error("flint: unsupported attention geometry");
        return c;
    }
};

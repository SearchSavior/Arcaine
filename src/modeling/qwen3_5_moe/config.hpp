#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

// Qwen3.5-MoE (text-only) configuration, parsed from config.json +
// generation_config.json. This is the NVFP4 "Qwen-AgentWorld-35B-A3B" text
// model shipped inside a multimodal container, so the checkpoint keys are
// prefixed `model.language_model.` even though config.json is flat (no
// text_config wrapper) — see loader.cpp for the prefix handling.

struct QwenRopeConfig {
    std::string rope_type = "default";
    float       rope_theta = 0.0f;
    float       partial_rotary_factor = 1.0f;   // rotary_dim = head_dim * this
    bool        mrope_interleaved = false;
    std::vector<int> mrope_section;             // vestigial for text-only
};

struct QwenConfig {
    std::string model_dir;
    std::string model_type;          // must be "qwen3_5_moe_text"

    // Core dims
    int hidden_size = 0;
    int vocab_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;     // full-attention Q heads (16)
    int num_key_value_heads = 0;     // full-attention KV heads (2)
    int head_dim = 0;                // 256
    int full_attention_interval = 4;  // full at idx % interval == interval-1
    std::vector<bool> is_full_attention;

    // Gated DeltaNet (linear attention) dims
    int linear_conv_kernel_dim = 4;
    int linear_key_head_dim = 128;
    int linear_num_key_heads = 16;
    int linear_num_value_heads = 32;
    int linear_value_head_dim = 128;
    std::string mamba_ssm_dtype = "float32";   // SSM state dtype at runtime

    // MoE
    int   moe_intermediate_size = 0;            // per-expert intermediate (512)
    int   num_experts = 0;                      // 256
    int   num_experts_per_tok = 0;              // top-8
    int   shared_expert_intermediate_size = 0;  // 512 (always-on shared expert)
    float router_aux_loss_coef = 0.0f;          // 0.001

    // Norms / activation
    float       rms_norm_eps = 1e-6f;
    std::string hidden_act = "silu";

    // Attention output gate (full-attention layers)
    bool        attn_output_gate = false;
    std::string output_gate_type = "swish";    // NOTE: config says "swish"
    bool        attention_bias = false;

    QwenRopeConfig rope;

    bool tie_word_embeddings = true;           // false here -> lm_head is separate
    int   max_position_embeddings = 0;

    // Tokens / generation
    int   bos_token_id = -1;
    int   pad_token_id = -1;
    std::vector<int> eos_token_ids;
    std::vector<int> suppress_tokens;
    float temperature = 1.0f;
    int   top_k = 64;
    float top_p = 0.95f;

    // Quantization (compressed-tensors nvfp4-pack-quantized)
    std::string quant_format;
    int         quant_group_size = 16;

    bool is_full_attn(int layer) const { return is_full_attention.at(layer); }
    int  rotary_dim() const { return (int)(head_dim * rope.partial_rotary_factor); }

    static QwenConfig from_dir(const std::string& dir) {
        QwenConfig cfg;
        cfg.model_dir = dir;

        std::ifstream f(dir + "/config.json");
        if (!f) throw std::runtime_error("Cannot open " + dir + "/config.json");
        auto j = nlohmann::json::parse(f);

        cfg.model_type = j.at("model_type").get<std::string>();
        if (cfg.model_type != "qwen3_5_moe_text")
            throw std::runtime_error("Expected model_type=qwen3_5_moe_text, got " + cfg.model_type);

        auto opt_int = [](const nlohmann::json& o, const char* k, int d) {
            if (!o.contains(k) || o[k].is_null()) return d;
            return o[k].get<int>();
        };
        auto load_eos = [](const nlohmann::json& v) {
            std::vector<int> out;
            if (v.is_array()) for (auto& id : v) out.push_back(id.get<int>());
            else out.push_back(v.get<int>());
            return out;
        };

        cfg.hidden_size        = j.at("hidden_size").get<int>();
        cfg.vocab_size         = j.at("vocab_size").get<int>();
        cfg.num_hidden_layers  = j.at("num_hidden_layers").get<int>();
        cfg.num_attention_heads = j.at("num_attention_heads").get<int>();
        cfg.num_key_value_heads = j.at("num_key_value_heads").get<int>();
        cfg.head_dim           = j.at("head_dim").get<int>();
        cfg.full_attention_interval = j.value("full_attention_interval", 4);

        cfg.is_full_attention.clear();
        for (auto& lt : j.at("layer_types"))
            cfg.is_full_attention.push_back(lt.get<std::string>() == "full_attention");
        if ((int)cfg.is_full_attention.size() != cfg.num_hidden_layers)
            throw std::runtime_error("layer_types length does not match num_hidden_layers");

        cfg.linear_conv_kernel_dim  = j.value("linear_conv_kernel_dim", 4);
        cfg.linear_key_head_dim     = j.value("linear_key_head_dim", 128);
        cfg.linear_num_key_heads    = j.value("linear_num_key_heads", 16);
        cfg.linear_num_value_heads  = j.value("linear_num_value_heads", 32);
        cfg.linear_value_head_dim   = j.value("linear_value_head_dim", 128);
        cfg.mamba_ssm_dtype         = j.value("mamba_ssm_dtype", std::string("float32"));

        cfg.moe_intermediate_size         = j.at("moe_intermediate_size").get<int>();
        cfg.num_experts                  = j.at("num_experts").get<int>();
        cfg.num_experts_per_tok          = j.at("num_experts_per_tok").get<int>();
        cfg.shared_expert_intermediate_size =
            j.value("shared_expert_intermediate_size", cfg.moe_intermediate_size);
        cfg.router_aux_loss_coef = j.value("router_aux_loss_coef", 0.0f);

        cfg.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
        cfg.hidden_act   = j.value("hidden_act", std::string("silu"));

        cfg.attn_output_gate  = j.value("attn_output_gate", false);
        cfg.output_gate_type  = j.value("output_gate_type", std::string("swish"));
        cfg.attention_bias    = j.value("attention_bias", false);

        if (j.contains("rope_parameters")) {
            auto& r = j.at("rope_parameters");
            cfg.rope.rope_type            = r.value("rope_type", std::string("default"));
            cfg.rope.rope_theta           = r.value("rope_theta", 0.0f);
            cfg.rope.partial_rotary_factor = r.value("partial_rotary_factor", 1.0f);
            cfg.rope.mrope_interleaved    = r.value("mrope_interleaved", false);
            if (r.contains("mrope_section") && r["mrope_section"].is_array())
                for (auto& s : r["mrope_section"]) cfg.rope.mrope_section.push_back(s.get<int>());
        }

        cfg.tie_word_embeddings    = j.value("tie_word_embeddings", true);
        cfg.max_position_embeddings = j.value("max_position_embeddings", 0);

        cfg.bos_token_id = opt_int(j, "bos_token_id", -1);
        cfg.pad_token_id = opt_int(j, "pad_token_id", -1);
        if (j.contains("eos_token_id") && !j["eos_token_id"].is_null())
            cfg.eos_token_ids = load_eos(j["eos_token_id"]);

        if (j.contains("quantization_config")) {
            auto& q = j["quantization_config"];
            cfg.quant_format = q.value("format", std::string());
            if (q.contains("config_groups") && q["config_groups"].contains("group_0") &&
                q["config_groups"]["group_0"].contains("weights"))
                cfg.quant_group_size =
                    q["config_groups"]["group_0"]["weights"].value("group_size", 16);
        }

        std::ifstream gf(dir + "/generation_config.json");
        if (gf) {
            auto g = nlohmann::json::parse(gf);
            cfg.bos_token_id = opt_int(g, "bos_token_id", cfg.bos_token_id);
            cfg.pad_token_id = opt_int(g, "pad_token_id", cfg.pad_token_id);
            if (g.contains("eos_token_id") && !g["eos_token_id"].is_null())
                cfg.eos_token_ids = load_eos(g["eos_token_id"]);
            if (g.contains("suppress_tokens") && g["suppress_tokens"].is_array()) {
                cfg.suppress_tokens.clear();
                for (auto& id : g["suppress_tokens"]) cfg.suppress_tokens.push_back(id.get<int>());
            }
            cfg.temperature = g.value("temperature", cfg.temperature);
            cfg.top_k = g.value("top_k", cfg.top_k);
            cfg.top_p = g.value("top_p", cfg.top_p);
        }

        return cfg;
    }
};

#pragma once
// DiffusionGemma 26B-A4B config parsing.  See
// docs/diffusion_gemma/diffusion_gemma_moe_architecture.md for the full spec.
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct DiffRope {
    std::string rope_type;
    float rope_theta = 0.0f;
    float partial_rotary_factor = 1.0f;
};

struct DiffTextConfig {
    int hidden_size = 0, intermediate_size = 0, vocab_size = 0, num_hidden_layers = 0;
    int num_attn_heads = 0, num_kv_heads = 0, num_global_kv_heads = 0;
    int head_dim = 0, global_head_dim = 0;
    int num_experts = 0, top_k_experts = 0, moe_intermediate_size = 0;
    int sliding_window = 0;
    float rms_norm_eps = 1e-6f, final_logit_softcapping = 30.0f;
    int bos_token_id = 2, pad_token_id = 0;
    std::vector<int> eos_token_ids;
    std::vector<bool> is_full_attention;   // per layer
    DiffRope sliding_rope, full_rope;
    std::string use_bidirectional_attention;
};

// Block-diffusion generation parameters (generation_config.json + mixin defaults).
struct DiffGenConfig {
    int   max_new_tokens       = 256;
    int   max_denoising_steps  = 48;
    float entropy_bound        = 0.1f;
    float t_min                = 0.4f;
    float t_max                = 0.8f;
    int   stability_threshold  = 1;
    float confidence_threshold = 0.005f;
    int   pad_token_id         = 0;
    std::vector<int> eos_token_ids;
};

struct DiffConfig {
    DiffTextConfig text;
    DiffGenConfig  gen;
    std::string    model_dir;
    std::string    model_type;
    std::string    quantization_format;
    int canvas_length  = 256;
    int image_token_id = 258880;
    int bos_token_id   = 2;
    std::vector<int> eos_token_ids;   // top-level config eos

    bool is_nvfp4_quantized() const {
        return quantization_format == "nvfp4-pack-quantized";
    }

    bool is_int4_quantized() const {
        return quantization_format == "pack-quantized";
    }

    static DiffConfig from_dir(const std::string& dir) {
        DiffConfig cfg;
        cfg.model_dir = dir;

        std::ifstream f(dir + "/config.json");
        if (!f) throw std::runtime_error("Cannot open " + dir + "/config.json");
        auto j = nlohmann::json::parse(f);

        auto load_ints = [](const nlohmann::json& v) {
            std::vector<int> out;
            if (v.is_array()) for (auto& x : v) out.push_back(x.get<int>());
            else out.push_back(v.get<int>());
            return out;
        };

        cfg.model_type = j.at("model_type").get<std::string>();
        if (cfg.model_type != "diffusion_gemma")
            throw std::runtime_error("Expected model_type=diffusion_gemma, got " + cfg.model_type);
        if (j.contains("quantization_config") && j.at("quantization_config").is_object())
            cfg.quantization_format = j.at("quantization_config").value("format", std::string());

        cfg.canvas_length  = j.value("canvas_length", 256);
        cfg.image_token_id = j.value("image_token_id", 258880);
        if (j.contains("eos_token_id")) cfg.eos_token_ids = load_ints(j.at("eos_token_id"));

        auto& tc = j.at("text_config");
        auto& t = cfg.text;
        t.hidden_size           = tc.at("hidden_size").get<int>();
        t.intermediate_size     = tc.at("intermediate_size").get<int>();
        t.vocab_size            = tc.at("vocab_size").get<int>();
        t.num_hidden_layers     = tc.at("num_hidden_layers").get<int>();
        t.num_attn_heads        = tc.at("num_attention_heads").get<int>();
        t.num_kv_heads          = tc.at("num_key_value_heads").get<int>();
        t.num_global_kv_heads   = tc.at("num_global_key_value_heads").get<int>();
        t.head_dim              = tc.at("head_dim").get<int>();
        t.global_head_dim       = tc.at("global_head_dim").get<int>();
        t.num_experts           = tc.at("num_experts").get<int>();
        t.top_k_experts         = tc.at("top_k_experts").get<int>();
        t.moe_intermediate_size = tc.at("moe_intermediate_size").get<int>();
        t.sliding_window        = tc.at("sliding_window").get<int>();
        t.rms_norm_eps          = tc.at("rms_norm_eps").get<float>();
        t.final_logit_softcapping = tc.at("final_logit_softcapping").get<float>();
        t.bos_token_id          = tc.value("bos_token_id", 2);
        t.pad_token_id          = tc.value("pad_token_id", 0);
        t.eos_token_ids         = load_ints(tc.at("eos_token_id"));
        t.use_bidirectional_attention =
            tc.value("use_bidirectional_attention", std::string("vision"));

        for (auto& lt : tc.at("layer_types"))
            t.is_full_attention.push_back(lt.get<std::string>() == "full_attention");
        if ((int)t.is_full_attention.size() != t.num_hidden_layers)
            throw std::runtime_error("layer_types length != num_hidden_layers");

        auto read_rope = [](const nlohmann::json& r) {
            DiffRope rc;
            rc.rope_type = r.at("rope_type").get<std::string>();
            rc.rope_theta = r.at("rope_theta").get<float>();
            rc.partial_rotary_factor = r.value("partial_rotary_factor", 1.0f);
            return rc;
        };
        auto& rope = tc.at("rope_parameters");
        t.sliding_rope = read_rope(rope.at("sliding_attention"));
        t.full_rope    = read_rope(rope.at("full_attention"));

        cfg.bos_token_id = t.bos_token_id;

        // Generation config
        cfg.gen.eos_token_ids = t.eos_token_ids;
        cfg.gen.pad_token_id  = t.pad_token_id;
        std::ifstream gf(dir + "/generation_config.json");
        if (gf) {
            auto g = nlohmann::json::parse(gf);
            cfg.gen.max_new_tokens      = g.value("max_new_tokens", cfg.gen.max_new_tokens);
            cfg.gen.max_denoising_steps = g.value("max_denoising_steps", cfg.gen.max_denoising_steps);
            cfg.gen.t_min               = g.value("t_min", cfg.gen.t_min);
            cfg.gen.t_max               = g.value("t_max", cfg.gen.t_max);
            cfg.gen.stability_threshold = g.value("stability_threshold", cfg.gen.stability_threshold);
            cfg.gen.confidence_threshold= g.value("confidence_threshold", cfg.gen.confidence_threshold);
            cfg.gen.pad_token_id        = g.value("pad_token_id", cfg.gen.pad_token_id);
            if (g.contains("eos_token_id")) cfg.gen.eos_token_ids = load_ints(g.at("eos_token_id"));
            if (g.contains("sampler_config"))
                cfg.gen.entropy_bound = g.at("sampler_config").value("entropy_bound", cfg.gen.entropy_bound);
        }
        return cfg;
    }
};

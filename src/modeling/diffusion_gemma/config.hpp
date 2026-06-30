#pragma once
// DiffusionGemma 26B-A4B config parsing.  See
// notes/diffusion_gemma/diffusion_gemma_moe_architecture.md for the full spec.
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "../../common/io/gguf.hpp"

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

    static DiffConfig from_gguf(const std::string& gguf_path) {
        GgufFile gg(gguf_path);
        DiffConfig cfg;
        cfg.model_type = "diffusion_gemma";
        cfg.quantization_format = "q8_0";

        auto& t = cfg.text;
        uint32_t u32;  float f32;

        auto need_u32 = [&](const char* k) -> uint32_t {
            if (!gg.get_u32(k, u32)) throw std::runtime_error(std::string("GGUF config: missing ") + k);
            return u32;
        };
        auto need_f32 = [&](const char* k) -> float {
            if (!gg.get_f32(k, f32)) throw std::runtime_error(std::string("GGUF config: missing ") + k);
            return f32;
        };

        t.hidden_size             = need_u32("diffusion-gemma.embedding_length");
        t.intermediate_size       = need_u32("diffusion-gemma.feed_forward_length");
        t.num_hidden_layers       = need_u32("diffusion-gemma.block_count");
        t.num_attn_heads          = need_u32("diffusion-gemma.attention.head_count");
        t.head_dim                = need_u32("diffusion-gemma.attention.key_length_swa");
        t.global_head_dim         = need_u32("diffusion-gemma.attention.key_length");
        t.num_experts             = need_u32("diffusion-gemma.expert_count");
        t.top_k_experts           = need_u32("diffusion-gemma.expert_used_count");
        t.moe_intermediate_size   = need_u32("diffusion-gemma.expert_feed_forward_length");
        t.sliding_window          = need_u32("diffusion-gemma.attention.sliding_window");
        t.rms_norm_eps            = need_f32("diffusion-gemma.attention.layer_norm_rms_epsilon");
        t.final_logit_softcapping = need_f32("diffusion-gemma.final_logit_softcapping");
        t.bos_token_id            = (int)need_u32("tokenizer.ggml.bos_token_id");
        t.pad_token_id            = (int)need_u32("tokenizer.ggml.padding_token_id");

        int32_t eos_id;
        if (!gg.get_i32("tokenizer.ggml.eos_token_id", eos_id))
            throw std::runtime_error("GGUF config: missing tokenizer.ggml.eos_token_id");
        t.eos_token_ids = {eos_id};

        std::vector<std::string> tokens;
        if (!gg.get_str_array("tokenizer.ggml.tokens", tokens))
            throw std::runtime_error("GGUF config: missing tokenizer.ggml.tokens");
        t.vocab_size = (int)tokens.size();

        std::vector<int32_t> kv_heads;
        if (!gg.get_i32_array("diffusion-gemma.attention.head_count_kv", kv_heads))
            throw std::runtime_error("GGUF config: missing head_count_kv");
        if (kv_heads.empty()) throw std::runtime_error("GGUF config: empty head_count_kv");
        t.num_kv_heads = kv_heads[0];
        t.num_global_kv_heads = (kv_heads.size() > 5) ? kv_heads[5] : kv_heads.back();

        std::vector<uint8_t> swp;
        if (!gg.get_bool_array("diffusion-gemma.attention.sliding_window_pattern", swp))
            throw std::runtime_error("GGUF config: missing sliding_window_pattern");
        t.is_full_attention.resize(swp.size());
        for (size_t i = 0; i < swp.size(); ++i) t.is_full_attention[i] = !swp[i];

        t.sliding_rope.rope_theta = need_f32("diffusion-gemma.rope.freq_base_swa");
        t.full_rope.rope_theta    = need_f32("diffusion-gemma.rope.freq_base");
        t.sliding_rope.partial_rotary_factor = 1.0f;
        t.sliding_rope.rope_type = "default";
        t.full_rope.partial_rotary_factor = 0.25f;
        t.full_rope.rope_type = "proportional";
        t.use_bidirectional_attention = "vision";

        cfg.canvas_length = (int)need_u32("diffusion.canvas_length");
        cfg.bos_token_id  = t.bos_token_id;
        cfg.eos_token_ids = t.eos_token_ids;
        cfg.gen.eos_token_ids = t.eos_token_ids;
        cfg.gen.pad_token_id  = t.pad_token_id;

        if (t.hidden_size % 32 != 0)
            throw std::runtime_error("GGUF config: embedding_length not divisible by 32 (Q8_0)");
        if ((int)t.is_full_attention.size() != t.num_hidden_layers)
            throw std::runtime_error("GGUF config: sliding_window_pattern length != block_count");

        return cfg;
    }
};

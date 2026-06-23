#pragma once
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "../common/generation.hpp"   // GenerationConfig

struct RopeConfig {
    std::string rope_type;
    float       rope_theta = 0.0f;
    float       partial_rotary_factor = 1.0f;
};

struct TokenizerConfig {
    std::string bos_token;
    std::string eos_token;
    std::string pad_token;
    std::string sot_token;  // <|turn>
    std::string eot_token;  // <turn|>
    std::string soc_token;  // <|channel>
    std::string eoc_token;  // <channel|>
    std::string boi_token;
    std::string eoi_token;
    std::string image_token;
    std::string boa_token;
    std::string eoa_token;
    std::string audio_token;
    std::string video_token;
    std::string str_token;  // <|tool_response>
};

struct TextConfig {
    std::string model_type;
    int   hidden_size         = 0;
    int   intermediate_size   = 0;
    int   vocab_size          = 0;
    int   num_hidden_layers   = 0;
    int   num_attn_heads      = 0;
    int   num_kv_heads        = 0;
    int   head_dim            = 0;
    int   max_position_embeddings = 0;
    int   global_head_dim     = 0;
    int   num_global_kv_heads = 0;
    int   num_kv_shared_layers = 0;
    bool  attention_k_eq_v    = false;
    bool  attention_bias      = false;
    bool  use_double_wide_mlp = false;
    bool  tie_word_embeddings = true;
    int   sliding_window      = 0;
    int   bos_token_id        = -1;
    int   pad_token_id        = -1;
    std::vector<int> eos_token_ids;
    float attention_dropout   = 0.0f;
    float rms_norm_eps        = 0.0f;
    float final_logit_softcapping = 0.0f;
    std::string hidden_activation;
    std::string use_bidirectional_attention;
    RopeConfig sliding_rope;
    RopeConfig full_rope;
    std::vector<bool> is_full_attention;
};

struct VisionConfig {
    std::string model_type;
    int   patch_size         = 0;
    int   pooling_kernel_size = 0;
    int   num_soft_tokens    = 0;
    int   mm_posemb_size     = 0;
    int   mm_embed_dim       = 0;
    int   model_patch_size   = 0;
    int   output_proj_dims   = 0;
    float rms_norm_eps       = 0.0f;
    float rescale_factor     = 1.0f;
    bool  do_rescale         = true;
};

struct AudioConfig {
    std::string model_type;
    int audio_embed_dim        = 0;
    int audio_samples_per_token = 0;
    int sampling_rate          = 0;
    float padding_value        = 0.0f;
    float rms_norm_eps         = 0.0f;
};

struct ModelConfig {
    TextConfig       text;
    VisionConfig     vision;
    AudioConfig      audio;
    TokenizerConfig  tokenizer;
    GenerationConfig generation;
    std::string      model_dir;
    std::string      model_type;
    std::string      chat_template;

    int boi_token_id   = -1;
    int eoi_token_id   = -1;
    int image_token_id = -1;
    int video_token_id = -1;
    int boa_token_id   = -1;
    int eoa_token_id   = -1;
    int audio_token_id = -1;
    int pad_token_id   = -1;
    int bos_token_id   = -1;

    bool is_eos_token(int token_id) const {
        return std::find(generation.eos_token_ids.begin(),
                         generation.eos_token_ids.end(),
                         token_id) != generation.eos_token_ids.end();
    }

    bool suppress_token(int token_id) const {
        return std::find(generation.suppress_tokens.begin(),
                         generation.suppress_tokens.end(),
                         token_id) != generation.suppress_tokens.end();
    }

    static ModelConfig from_dir(const std::string& dir) {
        ModelConfig cfg;
        cfg.model_dir = dir;

        std::ifstream f(dir + "/config.json");
        if (!f) throw std::runtime_error("Cannot open " + dir + "/config.json");
        auto j = nlohmann::json::parse(f);

        auto load_eos = [](const nlohmann::json& v) {
            std::vector<int> out;
            if (v.is_array()) {
                for (auto& id : v) out.push_back(id.get<int>());
            } else {
                out.push_back(v.get<int>());
            }
            return out;
        };

        cfg.model_type = j.at("model_type").get<std::string>();
        if (cfg.model_type != "gemma4_unified")
            throw std::runtime_error("Expected model_type=gemma4_unified, got " + cfg.model_type);

        cfg.boi_token_id   = j.at("boi_token_id").get<int>();
        cfg.eoi_token_id   = j.at("eoi_token_id").get<int>();
        cfg.image_token_id = j.at("image_token_id").get<int>();
        cfg.video_token_id = j.at("video_token_id").get<int>();
        cfg.boa_token_id   = j.at("boa_token_id").get<int>();
        cfg.eoa_token_id   = j.at("eoa_token_index").get<int>();
        cfg.audio_token_id = j.at("audio_token_id").get<int>();

        auto& tc = j.at("text_config");
        cfg.text.model_type = tc.at("model_type").get<std::string>();
        if (cfg.text.model_type != "gemma4_unified_text")
            throw std::runtime_error("Expected text_config.model_type=gemma4_unified_text");
        cfg.text.hidden_size         = tc.at("hidden_size").get<int>();
        cfg.text.intermediate_size   = tc.at("intermediate_size").get<int>();
        cfg.text.vocab_size          = tc.at("vocab_size").get<int>();
        cfg.text.num_hidden_layers   = tc.at("num_hidden_layers").get<int>();
        cfg.text.num_attn_heads      = tc.at("num_attention_heads").get<int>();
        cfg.text.num_kv_heads        = tc.at("num_key_value_heads").get<int>();
        cfg.text.head_dim            = tc.at("head_dim").get<int>();
        cfg.text.max_position_embeddings = tc.at("max_position_embeddings").get<int>();
        cfg.text.global_head_dim     = tc.at("global_head_dim").get<int>();
        cfg.text.num_global_kv_heads = tc.at("num_global_key_value_heads").get<int>();
        cfg.text.num_kv_shared_layers = tc.value("num_kv_shared_layers", 0);
        cfg.text.attention_k_eq_v    = tc.at("attention_k_eq_v").get<bool>();
        cfg.text.attention_bias      = tc.value("attention_bias", false);
        cfg.text.use_double_wide_mlp = tc.value("use_double_wide_mlp", false);
        cfg.text.tie_word_embeddings = tc.value("tie_word_embeddings", true);
        cfg.text.sliding_window      = tc.at("sliding_window").get<int>();
        cfg.text.bos_token_id        = tc.at("bos_token_id").get<int>();
        cfg.text.pad_token_id        = tc.at("pad_token_id").get<int>();
        cfg.text.eos_token_ids       = load_eos(tc.at("eos_token_id"));
        cfg.text.attention_dropout   = tc.value("attention_dropout", 0.0f);
        cfg.text.rms_norm_eps        = tc.at("rms_norm_eps").get<float>();
        cfg.text.final_logit_softcapping = tc.at("final_logit_softcapping").get<float>();
        cfg.text.hidden_activation   = tc.at("hidden_activation").get<std::string>();
        cfg.text.use_bidirectional_attention =
            tc.value("use_bidirectional_attention", std::string("vision"));

        for (auto& lt : tc.at("layer_types"))
            cfg.text.is_full_attention.push_back(lt.get<std::string>() == "full_attention");
        if ((int)cfg.text.is_full_attention.size() != cfg.text.num_hidden_layers)
            throw std::runtime_error("layer_types length does not match num_hidden_layers");

        auto read_rope = [](const nlohmann::json& r, float default_partial) {
            RopeConfig rc;
            rc.rope_type = r.at("rope_type").get<std::string>();
            rc.rope_theta = r.at("rope_theta").get<float>();
            rc.partial_rotary_factor = r.value("partial_rotary_factor", default_partial);
            return rc;
        };
        auto& rope = tc.at("rope_parameters");
        cfg.text.sliding_rope = read_rope(rope.at("sliding_attention"), 1.0f);
        cfg.text.full_rope    = read_rope(rope.at("full_attention"), 1.0f);
        if (cfg.text.use_bidirectional_attention == "all")
            cfg.text.sliding_window = (cfg.text.sliding_window / 2) + 1;

        auto& vc = j.at("vision_config");
        cfg.vision.model_type = vc.at("model_type").get<std::string>();
        if (cfg.vision.model_type != "gemma4_unified_vision")
            throw std::runtime_error("Expected vision_config.model_type=gemma4_unified_vision");
        cfg.vision.patch_size          = vc.at("patch_size").get<int>();
        cfg.vision.pooling_kernel_size = vc.at("pooling_kernel_size").get<int>();
        cfg.vision.num_soft_tokens     = vc.at("num_soft_tokens").get<int>();
        cfg.vision.mm_posemb_size      = vc.at("mm_posemb_size").get<int>();
        cfg.vision.mm_embed_dim        = vc.at("mm_embed_dim").get<int>();
        cfg.vision.model_patch_size    = vc.at("model_patch_size").get<int>();
        cfg.vision.output_proj_dims    = vc.at("output_proj_dims").get<int>();
        cfg.vision.rms_norm_eps        = vc.at("rms_norm_eps").get<float>();
        if (cfg.vision.model_patch_size != cfg.vision.patch_size * cfg.vision.pooling_kernel_size)
            throw std::runtime_error("vision model_patch_size != patch_size * pooling_kernel_size");

        auto& ac = j.at("audio_config");
        cfg.audio.model_type = ac.at("model_type").get<std::string>();
        if (cfg.audio.model_type != "gemma4_unified_audio")
            throw std::runtime_error("Expected audio_config.model_type=gemma4_unified_audio");
        cfg.audio.audio_embed_dim         = ac.at("audio_embed_dim").get<int>();
        cfg.audio.audio_samples_per_token = ac.at("audio_samples_per_token").get<int>();
        cfg.audio.rms_norm_eps            = ac.at("rms_norm_eps").get<float>();

        cfg.bos_token_id = cfg.text.bos_token_id;
        cfg.pad_token_id = cfg.text.pad_token_id;

        std::ifstream pf(dir + "/processor_config.json");
        if (pf) {
            auto p = nlohmann::json::parse(pf);
            if (p.value("processor_class", std::string()) != "Gemma4UnifiedProcessor")
                throw std::runtime_error("Expected processor_class=Gemma4UnifiedProcessor");
            if (p.contains("image_processor")) {
                auto& ip = p.at("image_processor");
                cfg.vision.patch_size = ip.value("patch_size", cfg.vision.patch_size);
                cfg.vision.pooling_kernel_size =
                    ip.value("pooling_kernel_size", cfg.vision.pooling_kernel_size);
                cfg.vision.num_soft_tokens =
                    ip.value("max_soft_tokens", cfg.vision.num_soft_tokens);
                cfg.vision.rescale_factor = ip.value("rescale_factor", cfg.vision.rescale_factor);
                cfg.vision.do_rescale = ip.value("do_rescale", cfg.vision.do_rescale);
            }
            if (p.contains("feature_extractor")) {
                auto& fe = p.at("feature_extractor");
                cfg.audio.audio_samples_per_token =
                    fe.value("audio_samples_per_token", cfg.audio.audio_samples_per_token);
                cfg.audio.audio_embed_dim = fe.value("feature_size", cfg.audio.audio_embed_dim);
                cfg.audio.sampling_rate = fe.value("sampling_rate", cfg.audio.sampling_rate);
                cfg.audio.padding_value = fe.value("padding_value", cfg.audio.padding_value);
            }
        }

        {
            std::ifstream tf(dir + "/tokenizer_config.json");
            if (!tf) throw std::runtime_error("Cannot open " + dir + "/tokenizer_config.json");
            auto t = nlohmann::json::parse(tf);
            if (t.value("processor_class", std::string()) != "Gemma4UnifiedProcessor")
                throw std::runtime_error("Expected tokenizer processor_class=Gemma4UnifiedProcessor");
            cfg.tokenizer.bos_token   = t.at("bos_token").get<std::string>();
            cfg.tokenizer.eos_token   = t.at("eos_token").get<std::string>();
            cfg.tokenizer.pad_token   = t.at("pad_token").get<std::string>();
            cfg.tokenizer.sot_token   = t.at("sot_token").get<std::string>();
            cfg.tokenizer.eot_token   = t.at("eot_token").get<std::string>();
            cfg.tokenizer.soc_token   = t.at("soc_token").get<std::string>();
            cfg.tokenizer.eoc_token   = t.at("eoc_token").get<std::string>();
            cfg.tokenizer.boi_token   = t.at("boi_token").get<std::string>();
            cfg.tokenizer.eoi_token   = t.at("eoi_token").get<std::string>();
            cfg.tokenizer.image_token = t.at("image_token").get<std::string>();
            cfg.tokenizer.boa_token   = t.at("boa_token").get<std::string>();
            cfg.tokenizer.eoa_token   = t.at("eoa_token").get<std::string>();
            cfg.tokenizer.audio_token = t.at("audio_token").get<std::string>();
            cfg.tokenizer.str_token   = t.at("str_token").get<std::string>();
            if (t.contains("video_token")) {
                cfg.tokenizer.video_token = t.at("video_token").get<std::string>();
            } else if (t.contains("extra_special_tokens") && !t.at("extra_special_tokens").empty()) {
                cfg.tokenizer.video_token = t.at("extra_special_tokens").at(0).get<std::string>();
            }
        }

        {
            std::ifstream cf(dir + "/chat_template.jinja");
            if (!cf) throw std::runtime_error("Cannot open " + dir + "/chat_template.jinja");
            cfg.chat_template.assign(std::istreambuf_iterator<char>(cf),
                                     std::istreambuf_iterator<char>());
        }

        cfg.generation.bos_token_id = cfg.bos_token_id;
        cfg.generation.pad_token_id = cfg.pad_token_id;
        cfg.generation.eos_token_ids = cfg.text.eos_token_ids;
        std::ifstream gf(dir + "/generation_config.json");
        if (gf) {
            auto g = nlohmann::json::parse(gf);
            cfg.generation.bos_token_id = g.value("bos_token_id", cfg.generation.bos_token_id);
            cfg.generation.pad_token_id = g.value("pad_token_id", cfg.generation.pad_token_id);
            if (g.contains("eos_token_id"))
                cfg.generation.eos_token_ids = load_eos(g.at("eos_token_id"));
            if (g.contains("suppress_tokens")) {
                cfg.generation.suppress_tokens.clear();
                for (auto& id : g.at("suppress_tokens"))
                    cfg.generation.suppress_tokens.push_back(id.get<int>());
            }
            cfg.generation.temperature = g.value("temperature", cfg.generation.temperature);
            cfg.generation.top_k = g.value("top_k", cfg.generation.top_k);
            cfg.generation.top_p = g.value("top_p", cfg.generation.top_p);
        }

        if (cfg.text.num_kv_shared_layers != 0)
            throw std::runtime_error("num_kv_shared_layers is not implemented by this C++ backend");
        if (cfg.text.use_double_wide_mlp)
            throw std::runtime_error("use_double_wide_mlp is not implemented by this C++ backend");
        if (cfg.text.attention_bias)
            throw std::runtime_error("attention_bias is not implemented by this C++ backend");
        if (cfg.text.hidden_activation != "gelu_pytorch_tanh")
            throw std::runtime_error("Unsupported hidden_activation: " + cfg.text.hidden_activation);
        if (!cfg.text.tie_word_embeddings)
            throw std::runtime_error("Untied lm_head weights are not implemented by this C++ backend");

        return cfg;
    }
};

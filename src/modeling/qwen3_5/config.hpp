#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

struct Qwen35RopeConfig {
    std::string type;
    float theta = 0.0f;
    float partial_rotary_factor = 0.0f;
    bool mrope_interleaved = false;
    std::vector<int> mrope_section;
};

struct Qwen35TextConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int full_attention_interval = 0;
    std::vector<bool> full_attention;
    int linear_conv_kernel_dim = 0;
    int linear_key_head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_value_head_dim = 0;
    int max_position_embeddings = 0;
    float rms_norm_eps = 0.0f;
    std::string hidden_act;
    std::string mamba_ssm_dtype;
    std::string output_gate_type;
    bool attention_bias = false;
    bool attention_output_gate = false;
    bool tie_word_embeddings = false;
    Qwen35RopeConfig rope;

    bool is_full_attn(int layer) const { return full_attention.at(layer); }
    int rotary_dim() const {
        return static_cast<int>(head_dim * rope.partial_rotary_factor);
    }
};

struct Qwen35VisionConfig {
    int depth = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_heads = 0;
    int out_hidden_size = 0;
    int in_channels = 0;
    int patch_size = 0;
    int temporal_patch_size = 0;
    int spatial_merge_size = 0;
    int num_position_embeddings = 0;
    std::string hidden_act;

    bool do_rescale = false;
    bool do_normalize = false;
    float rescale_factor = 0.0f;
    std::vector<float> image_mean;
    std::vector<float> image_std;
    int min_pixels = 0;
    int max_pixels = 0;
};

struct Qwen35Config {
    std::string model_dir;
    std::string model_type;
    std::string architecture;
    std::string dtype;
    bool language_model_only = false;
    int image_token_id = -1;
    int video_token_id = -1;
    int vision_start_token_id = -1;
    int vision_end_token_id = -1;
    int mtp_num_hidden_layers = 0;
    bool unsloth_fixed_mtp = false;

    Qwen35TextConfig text;
    Qwen35VisionConfig vision;

    std::string quant_method;
    std::string quant_format;
    int nvfp4_group_size = 0;

    int bos_token_id = -1;
    int pad_token_id = -1;
    std::vector<int> eos_token_ids;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 0.0f;

    static Qwen35Config from_dir(const std::string& dir) {
        using json = nlohmann::json;
        auto read_json = [](const std::string& path) {
            std::ifstream file(path);
            if (!file) throw std::runtime_error("Cannot open " + path);
            return json::parse(file);
        };
        auto read_ints = [](const json& value) {
            std::vector<int> out;
            if (value.is_array()) {
                for (const auto& item : value) out.push_back(item.get<int>());
            } else {
                out.push_back(value.get<int>());
            }
            return out;
        };
        auto read_floats = [](const json& value) {
            std::vector<float> out;
            for (const auto& item : value) out.push_back(item.get<float>());
            return out;
        };

        Qwen35Config cfg;
        cfg.model_dir = dir;
        json root = read_json(dir + "/config.json");
        cfg.model_type = root.at("model_type").get<std::string>();
        if (cfg.model_type != "qwen3_5")
            throw std::runtime_error("Expected model_type=qwen3_5, got " + cfg.model_type);
        const auto& architectures = root.at("architectures");
        if (architectures.size() != 1)
            throw std::runtime_error("Expected exactly one Qwen3.5 architecture");
        cfg.architecture = architectures.at(0).get<std::string>();
        if (cfg.architecture != "Qwen3_5ForConditionalGeneration")
            throw std::runtime_error("Unsupported Qwen3.5 architecture: " + cfg.architecture);
        cfg.dtype = root.at("dtype").get<std::string>();
        cfg.language_model_only = root.at("language_model_only").get<bool>();
        if (cfg.language_model_only)
            throw std::runtime_error("This loader expects the multimodal Qwen3.5 checkpoint");
        cfg.image_token_id = root.at("image_token_id").get<int>();
        cfg.video_token_id = root.at("video_token_id").get<int>();
        cfg.vision_start_token_id = root.at("vision_start_token_id").get<int>();
        cfg.vision_end_token_id = root.at("vision_end_token_id").get<int>();
        cfg.mtp_num_hidden_layers = root.at("mtp_num_hidden_layers").get<int>();
        cfg.unsloth_fixed_mtp = root.at("unsloth_fixed_mtp").get<bool>();

        const json& text = root.at("text_config");
        if (text.at("model_type").get<std::string>() != "qwen3_5_text")
            throw std::runtime_error("Expected text_config.model_type=qwen3_5_text");
        cfg.text.hidden_size = text.at("hidden_size").get<int>();
        cfg.text.intermediate_size = text.at("intermediate_size").get<int>();
        cfg.text.vocab_size = text.at("vocab_size").get<int>();
        cfg.text.num_hidden_layers = text.at("num_hidden_layers").get<int>();
        cfg.text.num_attention_heads = text.at("num_attention_heads").get<int>();
        cfg.text.num_key_value_heads = text.at("num_key_value_heads").get<int>();
        cfg.text.head_dim = text.at("head_dim").get<int>();
        cfg.text.full_attention_interval = text.at("full_attention_interval").get<int>();
        for (const auto& layer_type : text.at("layer_types")) {
            std::string type = layer_type.get<std::string>();
            if (type != "full_attention" && type != "linear_attention")
                throw std::runtime_error("Unknown Qwen3.5 layer type: " + type);
            cfg.text.full_attention.push_back(type == "full_attention");
        }
        if (static_cast<int>(cfg.text.full_attention.size()) != cfg.text.num_hidden_layers)
            throw std::runtime_error("layer_types length does not match num_hidden_layers");
        cfg.text.linear_conv_kernel_dim = text.at("linear_conv_kernel_dim").get<int>();
        cfg.text.linear_key_head_dim = text.at("linear_key_head_dim").get<int>();
        cfg.text.linear_num_key_heads = text.at("linear_num_key_heads").get<int>();
        cfg.text.linear_num_value_heads = text.at("linear_num_value_heads").get<int>();
        cfg.text.linear_value_head_dim = text.at("linear_value_head_dim").get<int>();
        cfg.text.max_position_embeddings = text.at("max_position_embeddings").get<int>();
        cfg.text.rms_norm_eps = text.at("rms_norm_eps").get<float>();
        cfg.text.hidden_act = text.at("hidden_act").get<std::string>();
        cfg.text.mamba_ssm_dtype = text.at("mamba_ssm_dtype").get<std::string>();
        cfg.text.output_gate_type = text.at("output_gate_type").get<std::string>();
        cfg.text.attention_bias = text.at("attention_bias").get<bool>();
        cfg.text.attention_output_gate = text.at("attn_output_gate").get<bool>();
        cfg.text.tie_word_embeddings = text.at("tie_word_embeddings").get<bool>();
        const json& rope = text.at("rope_parameters");
        cfg.text.rope.type = rope.at("rope_type").get<std::string>();
        cfg.text.rope.theta = rope.at("rope_theta").get<float>();
        cfg.text.rope.partial_rotary_factor = rope.at("partial_rotary_factor").get<float>();
        cfg.text.rope.mrope_interleaved = rope.at("mrope_interleaved").get<bool>();
        cfg.text.rope.mrope_section = read_ints(rope.at("mrope_section"));

        const json& vision = root.at("vision_config");
        if (vision.at("model_type").get<std::string>() != "qwen3_5_vision")
            throw std::runtime_error("Expected vision_config.model_type=qwen3_5_vision");
        cfg.vision.depth = vision.at("depth").get<int>();
        cfg.vision.hidden_size = vision.at("hidden_size").get<int>();
        cfg.vision.intermediate_size = vision.at("intermediate_size").get<int>();
        cfg.vision.num_heads = vision.at("num_heads").get<int>();
        cfg.vision.out_hidden_size = vision.at("out_hidden_size").get<int>();
        cfg.vision.in_channels = vision.at("in_channels").get<int>();
        cfg.vision.patch_size = vision.at("patch_size").get<int>();
        cfg.vision.temporal_patch_size = vision.at("temporal_patch_size").get<int>();
        cfg.vision.spatial_merge_size = vision.at("spatial_merge_size").get<int>();
        cfg.vision.num_position_embeddings = vision.at("num_position_embeddings").get<int>();
        cfg.vision.hidden_act = vision.at("hidden_act").get<std::string>();

        const json& quant = root.at("quantization_config");
        cfg.quant_method = quant.at("quant_method").get<std::string>();
        cfg.quant_format = quant.at("format").get<std::string>();
        cfg.nvfp4_group_size = quant.at("config_groups").at("group_1")
            .at("weights").at("group_size").get<int>();
        if (cfg.quant_method != "compressed-tensors" ||
            cfg.quant_format != "mixed-precision" || cfg.nvfp4_group_size != 16)
            throw std::runtime_error("Unsupported Qwen3.5 quantization configuration");

        json processor = read_json(dir + "/processor_config.json");
        const json& image = processor.at("image_processor");
        cfg.vision.do_rescale = image.at("do_rescale").get<bool>();
        cfg.vision.do_normalize = image.at("do_normalize").get<bool>();
        cfg.vision.rescale_factor = image.at("rescale_factor").get<float>();
        cfg.vision.image_mean = read_floats(image.at("image_mean"));
        cfg.vision.image_std = read_floats(image.at("image_std"));
        cfg.vision.min_pixels = image.at("size").at("shortest_edge").get<int>();
        cfg.vision.max_pixels = image.at("size").at("longest_edge").get<int>();
        if (image.at("patch_size").get<int>() != cfg.vision.patch_size ||
            image.at("temporal_patch_size").get<int>() != cfg.vision.temporal_patch_size ||
            image.at("merge_size").get<int>() != cfg.vision.spatial_merge_size)
            throw std::runtime_error("Processor and vision patch geometry disagree");

        json generation = read_json(dir + "/generation_config.json");
        cfg.bos_token_id = generation.at("bos_token_id").get<int>();
        cfg.pad_token_id = generation.at("pad_token_id").get<int>();
        cfg.eos_token_ids = read_ints(generation.at("eos_token_id"));
        cfg.temperature = generation.at("temperature").get<float>();
        cfg.top_k = generation.at("top_k").get<int>();
        cfg.top_p = generation.at("top_p").get<float>();
        return cfg;
    }
};

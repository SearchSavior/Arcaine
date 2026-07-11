#include "loader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "../../common/gpu/engine.hpp"

namespace {

bool fused_fp8_projections_enabled() {
    const char* value = std::getenv("ARCAINE_QWEN35_FUSED_FP8_PROJECTIONS");
    if (!value) return true;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
}

class TrackingTensorSource final : public TensorSource {
public:
    explicit TrackingTensorSource(const ShardedSafetensors& source) : source_(source) {}

    const TensorView& get(const std::string& name) const override {
        const TensorView& view = source_.get(name);
        consumed_.insert(name);
        return view;
    }

    bool has(const std::string& name) const override { return source_.has(name); }
    bool consumed(const std::string& name) const { return consumed_.count(name) != 0; }

private:
    const ShardedSafetensors& source_;
    mutable std::unordered_set<std::string> consumed_;
};

void expect_tensor(const TensorSource& source, const std::string& name,
                   const char* dtype, std::vector<int64_t> shape) {
    const TensorView& view = source.get(name);
    if (view.dtype != dtype || view.shape != shape) {
        std::ostringstream message;
        message << "Unexpected tensor metadata for " << name << ": dtype="
                << view.dtype << " shape=(";
        for (size_t i = 0; i < view.shape.size(); ++i) {
            if (i) message << ',';
            message << view.shape[i];
        }
        message << ')';
        throw std::runtime_error(message.str());
    }
}

GpuBuffer<bf16> load_bf16(const TensorSource& source, const std::string& name,
                          std::vector<int64_t> shape, sycl::queue& queue,
                          bool add_one = false) {
    expect_tensor(source, name, "BF16", std::move(shape));
    return add_one ? upload_plus_one(source.get(name), queue, name.c_str())
                   : upload(source.get(name), queue, name.c_str());
}

void expect_fp8(const Fp8Linear& linear, int in, int out, const std::string& name) {
    if (linear.in_features != in || linear.out_features != out)
        throw std::runtime_error("Unexpected FP8 linear shape: " + name);
}

void expect_nvfp4(const Nvfp4Linear& linear, int in, int out, const std::string& name) {
    if (linear.in_features != in || linear.out_features != out)
        throw std::runtime_error("Unexpected NVFP4 linear shape: " + name);
}

Qwen35VisionWeights load_vision(const TensorSource& source,
                                const Qwen35Config& config,
                                sycl::queue& queue) {
    const auto& v = config.vision;
    const std::string prefix = "model.visual.";
    Qwen35VisionWeights weights;
    weights.patch_weight = load_bf16(source, prefix + "patch_embed.proj.weight",
        {v.hidden_size, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size}, queue);
    weights.patch_bias = load_bf16(source, prefix + "patch_embed.proj.bias",
        {v.hidden_size}, queue);
    weights.position_embedding = load_bf16(source, prefix + "pos_embed.weight",
        {v.num_position_embeddings, v.hidden_size}, queue);
    weights.blocks.reserve(v.depth);
    for (int i = 0; i < v.depth; ++i) {
        std::string block = prefix + "blocks." + std::to_string(i) + ".";
        Qwen35VisionBlockWeights w;
        w.norm1_weight = load_bf16(source, block + "norm1.weight", {v.hidden_size}, queue);
        w.norm1_bias = load_bf16(source, block + "norm1.bias", {v.hidden_size}, queue);
        w.norm2_weight = load_bf16(source, block + "norm2.weight", {v.hidden_size}, queue);
        w.norm2_bias = load_bf16(source, block + "norm2.bias", {v.hidden_size}, queue);
        w.qkv_weight = load_bf16(source, block + "attn.qkv.weight",
                                 {3 * v.hidden_size, v.hidden_size}, queue);
        w.qkv_bias = load_bf16(source, block + "attn.qkv.bias", {3 * v.hidden_size}, queue);
        w.proj_weight = load_bf16(source, block + "attn.proj.weight",
                                  {v.hidden_size, v.hidden_size}, queue);
        w.proj_bias = load_bf16(source, block + "attn.proj.bias", {v.hidden_size}, queue);
        w.fc1_weight = load_bf16(source, block + "mlp.linear_fc1.weight",
                                 {v.intermediate_size, v.hidden_size}, queue);
        w.fc1_bias = load_bf16(source, block + "mlp.linear_fc1.bias",
                               {v.intermediate_size}, queue);
        w.fc2_weight = load_bf16(source, block + "mlp.linear_fc2.weight",
                                 {v.hidden_size, v.intermediate_size}, queue);
        w.fc2_bias = load_bf16(source, block + "mlp.linear_fc2.bias",
                               {v.hidden_size}, queue);
        weights.blocks.push_back(std::move(w));
    }
    int merged = v.hidden_size * v.spatial_merge_size * v.spatial_merge_size;
    weights.merger_norm_weight = load_bf16(source, prefix + "merger.norm.weight",
                                            {v.hidden_size}, queue);
    weights.merger_norm_bias = load_bf16(source, prefix + "merger.norm.bias",
                                          {v.hidden_size}, queue);
    weights.merger_fc1_weight = load_bf16(source, prefix + "merger.linear_fc1.weight",
                                           {merged, merged}, queue);
    weights.merger_fc1_bias = load_bf16(source, prefix + "merger.linear_fc1.bias",
                                         {merged}, queue);
    weights.merger_fc2_weight = load_bf16(source, prefix + "merger.linear_fc2.weight",
                                           {v.out_hidden_size, merged}, queue);
    weights.merger_fc2_bias = load_bf16(source, prefix + "merger.linear_fc2.bias",
                                         {v.out_hidden_size}, queue);
    return weights;
}

Qwen35MtpWeights load_mtp(const TensorSource& source,
                          const Qwen35Config& config,
                          sycl::queue& queue) {
    if (config.mtp_num_hidden_layers != 1)
        throw std::runtime_error("Only the checkpoint's one-layer MTP head is supported");
    int H = config.text.hidden_size;
    int I = config.text.intermediate_size;
    int q_out = config.text.num_attention_heads * config.text.head_dim * 2;
    int kv_out = config.text.num_key_value_heads * config.text.head_dim;
    int attn_out = config.text.num_attention_heads * config.text.head_dim;
    Qwen35MtpWeights w;
    w.fc = load_bf16(source, "mtp.fc.weight", {H, 2 * H}, queue);
    w.pre_fc_norm_embedding = load_bf16(source, "mtp.pre_fc_norm_embedding.weight", {H}, queue, true);
    w.pre_fc_norm_hidden = load_bf16(source, "mtp.pre_fc_norm_hidden.weight", {H}, queue, true);
    w.input_layernorm = load_bf16(source, "mtp.layers.0.input_layernorm.weight", {H}, queue, true);
    w.post_attention_layernorm = load_bf16(
        source, "mtp.layers.0.post_attention_layernorm.weight", {H}, queue, true);
    w.q_proj = load_bf16(source, "mtp.layers.0.self_attn.q_proj.weight", {q_out, H}, queue);
    w.k_proj = load_bf16(source, "mtp.layers.0.self_attn.k_proj.weight", {kv_out, H}, queue);
    w.v_proj = load_bf16(source, "mtp.layers.0.self_attn.v_proj.weight", {kv_out, H}, queue);
    w.o_proj = load_bf16(source, "mtp.layers.0.self_attn.o_proj.weight", {H, attn_out}, queue);
    w.q_norm = load_bf16(source, "mtp.layers.0.self_attn.q_norm.weight",
                          {config.text.head_dim}, queue, true);
    w.k_norm = load_bf16(source, "mtp.layers.0.self_attn.k_norm.weight",
                          {config.text.head_dim}, queue, true);
    w.gate_proj = load_bf16(source, "mtp.layers.0.mlp.gate_proj.weight", {I, H}, queue);
    w.up_proj = load_bf16(source, "mtp.layers.0.mlp.up_proj.weight", {I, H}, queue);
    w.down_proj = load_bf16(source, "mtp.layers.0.mlp.down_proj.weight", {H, I}, queue);
    w.norm = load_bf16(source, "mtp.norm.weight", {H}, queue, true);
    return w;
}

}  // namespace

Qwen35Weights load_qwen35_weights(
    const ShardedSafetensors& checkpoint,
    const Qwen35Config& config,
    int split_layer,
    int max_layers) {
    TrackingTensorSource source(checkpoint);
    const auto& c = config.text;
    int total_layers = c.num_hidden_layers;
    if (max_layers <= 0 || max_layers > total_layers) max_layers = total_layers;
    if (split_layer < 0 || split_layer > total_layers) split_layer = total_layers;

    auto& queue0 = GpuEngine::get(0).queue;
    Qwen35Weights weights;
    weights.embed_tokens = load_bf16(
        source, "model.language_model.embed_tokens.weight",
        {c.vocab_size, c.hidden_size}, queue0);
    weights.final_norm = load_bf16(
        source, "model.language_model.norm.weight", {c.hidden_size}, queue0, true);
    weights.lm_head = upload_fp8_linear(source, "lm_head", queue0);
    expect_fp8(weights.lm_head, c.hidden_size, c.vocab_size, "lm_head");

    weights.layers.reserve(max_layers);
    for (int i = 0; i < max_layers; ++i) {
        int gpu = (i < split_layer || GpuEngine::count() < 2) ? 0 : 1;
        auto& queue = GpuEngine::get(gpu).queue;
        std::string layer_prefix = "model.language_model.layers." + std::to_string(i) + ".";
        Qwen35LayerWeights layer;
        layer.index = i;
        layer.gpu = gpu;
        layer.full_attention = c.is_full_attn(i);
        layer.input_layernorm = load_bf16(
            source, layer_prefix + "input_layernorm.weight", {c.hidden_size}, queue, true);
        layer.post_attention_layernorm = load_bf16(
            source, layer_prefix + "post_attention_layernorm.weight", {c.hidden_size}, queue, true);

        if (layer.full_attention) {
            std::string prefix = layer_prefix + "self_attn.";
            Qwen35FullAttentionWeights attention;
            attention.fused_projections = fused_fp8_projections_enabled();
            if (attention.fused_projections)
                attention.qkv_proj = upload_fp8_linear_concat(
                    source, {prefix + "q_proj", prefix + "k_proj",
                             prefix + "v_proj"}, queue);
            else {
                attention.q_proj = upload_fp8_linear(source, prefix + "q_proj", queue);
                attention.k_proj = upload_fp8_linear(source, prefix + "k_proj", queue);
                attention.v_proj = upload_fp8_linear(source, prefix + "v_proj", queue);
            }
            attention.o_proj = upload_fp8_linear(source, prefix + "o_proj", queue);
            int q_out = c.num_attention_heads * c.head_dim * 2;
            int kv_out = c.num_key_value_heads * c.head_dim;
            int attn_out = c.num_attention_heads * c.head_dim;
            if (attention.fused_projections)
                expect_fp8(attention.qkv_proj, c.hidden_size, q_out + 2 * kv_out,
                            prefix + "qkv_proj");
            else {
                expect_fp8(attention.q_proj, c.hidden_size, q_out, prefix + "q_proj");
                expect_fp8(attention.k_proj, c.hidden_size, kv_out, prefix + "k_proj");
                expect_fp8(attention.v_proj, c.hidden_size, kv_out, prefix + "v_proj");
            }
            expect_fp8(attention.o_proj, attn_out, c.hidden_size, prefix + "o_proj");
            attention.q_norm = load_bf16(source, prefix + "q_norm.weight", {c.head_dim}, queue, true);
            attention.k_norm = load_bf16(source, prefix + "k_norm.weight", {c.head_dim}, queue, true);
            attention.k_cache_scale = load_bf16(source, prefix + "k_scale", {1}, queue);
            attention.v_cache_scale = load_bf16(source, prefix + "v_scale", {1}, queue);
            layer.mixer = std::move(attention);
        } else {
            std::string prefix = layer_prefix + "linear_attn.";
            Qwen35LinearAttentionWeights attention;
            int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
            int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
            int conv_dim = 2 * key_dim + value_dim;
            attention.fused_projections = fused_fp8_projections_enabled();
            if (attention.fused_projections)
                attention.in_proj_qkvz = upload_fp8_linear_concat(
                    source, {prefix + "in_proj_qkv", prefix + "in_proj_z"}, queue);
            else {
                attention.in_proj_qkv = upload_fp8_linear(
                    source, prefix + "in_proj_qkv", queue);
                attention.in_proj_z = upload_fp8_linear(
                    source, prefix + "in_proj_z", queue);
            }
            attention.out_proj = upload_fp8_linear(source, prefix + "out_proj", queue);
            if (attention.fused_projections)
                expect_fp8(attention.in_proj_qkvz, c.hidden_size,
                            conv_dim + value_dim, prefix + "in_proj_qkvz");
            else {
                expect_fp8(attention.in_proj_qkv, c.hidden_size, conv_dim,
                            prefix + "in_proj_qkv");
                expect_fp8(attention.in_proj_z, c.hidden_size, value_dim,
                            prefix + "in_proj_z");
            }
            expect_fp8(attention.out_proj, value_dim, c.hidden_size, prefix + "out_proj");
            attention.in_proj_a = load_bf16(source, prefix + "in_proj_a.weight",
                                             {c.linear_num_value_heads, c.hidden_size}, queue);
            attention.in_proj_b = load_bf16(source, prefix + "in_proj_b.weight",
                                             {c.linear_num_value_heads, c.hidden_size}, queue);
            attention.conv1d = load_bf16(source, prefix + "conv1d.weight",
                                          {conv_dim, 1, c.linear_conv_kernel_dim}, queue);
            attention.A_log = load_bf16(source, prefix + "A_log",
                                         {c.linear_num_value_heads}, queue);
            attention.dt_bias = load_bf16(source, prefix + "dt_bias",
                                           {c.linear_num_value_heads}, queue);
            attention.norm = load_bf16(source, prefix + "norm.weight",
                                        {c.linear_value_head_dim}, queue);
            layer.mixer = std::move(attention);
        }

        std::string mlp = layer_prefix + "mlp.";
        if (source.has(mlp + "gate_proj.weight_packed")) {
            layer.mlp.nvfp4 = true;
            Nvfp4Linear gate_up = upload_nvfp4_linear_pair(
                source, mlp + "gate_proj", mlp + "up_proj", queue);
            Nvfp4Linear down = upload_nvfp4_linear(source, mlp + "down_proj", queue);
            expect_nvfp4(gate_up, c.hidden_size, 2 * c.intermediate_size, mlp + "gate_up");
            expect_nvfp4(down, c.intermediate_size, c.hidden_size, mlp + "down_proj");
            layer.mlp.gate_up = std::move(gate_up);
            layer.mlp.down = std::move(down);
        } else {
            layer.mlp.nvfp4 = false;
            Fp8Linear gate_up = upload_fp8_linear_pair(
                source, mlp + "gate_proj", mlp + "up_proj", queue);
            Fp8Linear down = upload_fp8_linear(source, mlp + "down_proj", queue);
            expect_fp8(gate_up, c.hidden_size, 2 * c.intermediate_size, mlp + "gate_up");
            expect_fp8(down, c.intermediate_size, c.hidden_size, mlp + "down_proj");
            layer.mlp.gate_up = std::move(gate_up);
            layer.mlp.down = std::move(down);
        }
        weights.layers.push_back(std::move(layer));
        std::printf("[qwen35-load] layer %d/%d on GPU %d\n", i + 1, max_layers, gpu);
    }

    weights.vision = load_vision(source, config, queue0);
    weights.mtp = load_mtp(source, config, queue0);

    if (max_layers == total_layers) {
        std::vector<std::string> missing;
        for (const std::string& name : checkpoint.names())
            if (!source.consumed(name)) missing.push_back(name);
        if (!missing.empty()) {
            std::sort(missing.begin(), missing.end());
            std::ostringstream message;
            message << "Qwen3.5 checkpoint has " << missing.size()
                    << " unconsumed tensors; first entries:";
            for (size_t i = 0; i < std::min<size_t>(missing.size(), 16); ++i)
                message << "\n  " << missing[i];
            throw std::runtime_error(message.str());
        }
    }
    return weights;
}

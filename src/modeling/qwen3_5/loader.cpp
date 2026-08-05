#include "loader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "../../runtime/gpu/engine.hpp"

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


GpuBuffer<bf16> load_bf16(const TensorSource& source, const std::string& name,
                           std::vector<int64_t> shape, sycl::queue& queue,
                           bool add_one = false) {
    // upload()/upload_plus_one() stage BF16/F16/F32 to BF16 (converting as
    // needed): the NVFP4 checkpoint stores BF16, the AWQ checkpoint F16.
    const TensorView& view = source.get(name);
    if (view.shape != shape ||
        (view.dtype != "BF16" && view.dtype != "F16" && view.dtype != "F32")) {
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
    return add_one ? upload_plus_one(view, queue, name.c_str())
                   : upload(view, queue, name.c_str());
}

// --- compressed-tensors pack-quantized INT4 (AWQ) W4A16 loading -------------
// Checkpoint tensors per projection:
//   weight_packed     I32 [N, K/8]  — 8 nibbles/int32 along K, LSB-first,
//                                     unsigned offset-8
//   weight_scale      F16 [N, G]    — G = K / group_size (group_size=32)
//   weight_zero_point I32 [N/8, G]  — 8 nibbles/int32 along N, LSB-first
// Dequant: w[n,k] = scale[n,g] * (q_u[n,k] - zp_u[n,g]).  Follows the
// gemma4_unified pattern: rebase nibbles to s4 (XOR 0x88) for oneDNN,
// transpose scales to (G, N) BF16, precompute zp_offset = scale*(zp_u-8).

float f16_bits_to_float_local(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

struct StagedInt4 {
    int in_features = 0;
    int out_features = 0;
    int group_size = 0;
    int groups = 0;
    std::vector<uint8_t> packed;   // raw checkpoint bytes, (N, K/2)
    std::vector<bf16> scale_t;     // (G, N)
    std::vector<bf16> corr_t;      // (G, N); empty when no zero_point tensor
};

StagedInt4 stage_int4_linear(const TensorSource& source, const std::string& prefix) {
    const TensorView& packed = source.get(prefix + ".weight_packed");
    if (packed.dtype != "I32" || packed.shape.size() != 2)
        throw std::runtime_error("Expected I32 2D packed weight: " + prefix);
    StagedInt4 staged;
    staged.out_features = (int)packed.shape[0];
    staged.in_features = (int)packed.shape[1] * 8;

    const TensorView& scale = source.get(prefix + ".weight_scale");
    if (scale.shape.size() != 2 || (int)scale.shape[0] != staged.out_features)
        throw std::runtime_error("Unexpected int4 scale shape: " + prefix);
    staged.groups = (int)scale.shape[1];
    if (staged.groups == 0 || staged.in_features % staged.groups != 0)
        throw std::runtime_error("int4 in_features not divisible by groups: " + prefix);
    staged.group_size = staged.in_features / staged.groups;

    auto scale_at = [&](int64_t i) -> float {
        if (scale.dtype == "F32")
            return static_cast<const float*>(scale.data)[i];
        if (scale.dtype == "F16")
            return f16_bits_to_float_local(
                static_cast<const uint16_t*>(scale.data)[i]);
        if (scale.dtype == "BF16")
            return bf16_to_float(static_cast<const uint16_t*>(scale.data)[i]);
        throw std::runtime_error("Unexpected int4 scale dtype for " + prefix +
                                 ": " + scale.dtype);
    };

    const int32_t* zp = nullptr;
    if (source.has(prefix + ".weight_zero_point")) {
        const TensorView& zpv = source.get(prefix + ".weight_zero_point");
        if (zpv.dtype != "I32" || zpv.shape.size() != 2 ||
            (int)zpv.shape[0] != staged.out_features / 8 ||
            (int)zpv.shape[1] != staged.groups)
            throw std::runtime_error("Unexpected zero_point shape: " + prefix);
        zp = static_cast<const int32_t*>(zpv.data);
    }

    // Cross-check the logical shape metadata (also consumes the tensor).
    const TensorView& shape_meta = source.get(prefix + ".weight_shape");
    if (shape_meta.shape.size() != 1 || shape_meta.shape[0] != 2 ||
        shape_meta.dtype != "I64")
        throw std::runtime_error("Unexpected weight_shape metadata: " + prefix);
    const int64_t* dims = static_cast<const int64_t*>(shape_meta.data);
    if (dims[0] != staged.out_features || dims[1] != staged.in_features)
        throw std::runtime_error("weight_shape disagrees with packed dims: " + prefix);

    staged.packed.resize(packed.nbytes);
    std::memcpy(staged.packed.data(), packed.data, packed.nbytes);

    int N = staged.out_features, G = staged.groups;
    staged.scale_t.resize((size_t)G * N);
    if (zp) staged.corr_t.resize((size_t)G * N);
    for (int n = 0; n < N; ++n) {
        int zp_row = n >> 3;
        int zp_shift = (n & 7) * 4;
        for (int g = 0; g < G; ++g) {
            float s = scale_at((size_t)n * G + g);
            staged.scale_t[(size_t)g * N + n] = float_to_bf16(s);
            if (zp) {
                int zp_signed = ((zp[(size_t)zp_row * G + g] >> zp_shift) & 0xF) - 8;
                staged.corr_t[(size_t)g * N + n] =
                    float_to_bf16(s * (float)zp_signed);
            }
        }
    }
    return staged;
}

Int4Linear upload_int4_staged(const std::vector<StagedInt4>& parts,
                              sycl::queue& queue) {
    Int4Linear lin;
    lin.in_features = parts.front().in_features;
    lin.group_size = parts.front().group_size;
    int G = parts.front().groups;
    bool has_zp = !parts.front().corr_t.empty();
    size_t packed_bytes = 0;
    for (const StagedInt4& part : parts) {
        if (part.in_features != lin.in_features || part.groups != G ||
            part.group_size != lin.group_size ||
            !part.corr_t.empty() != has_zp)
            throw std::runtime_error("Incompatible int4 projections in concat");
        packed_bytes += part.packed.size();
        lin.out_features += part.out_features;
    }
    int total_n = lin.out_features;

    static std::vector<uint8_t> staging;
    staging.clear();
    staging.reserve(packed_bytes);
    for (const StagedInt4& part : parts)
        staging.insert(staging.end(), part.packed.begin(), part.packed.end());
    lin.weight_packed = GpuBuffer<uint8_t>(packed_bytes, queue);
    sycl::event copy_done =
        queue.memcpy(lin.weight_packed.data(), staging.data(), packed_bytes);
    uint8_t* dst = lin.weight_packed.data();
    queue.submit([&](sycl::handler& h) {
        h.depends_on(copy_done);
        h.parallel_for(sycl::range<1>(packed_bytes), [=](sycl::id<1> id) {
            dst[id[0]] ^= 0x88;
        });
    }).wait();

    std::vector<bf16> scale_t((size_t)G * total_n);
    std::vector<bf16> corr_t(has_zp ? (size_t)G * total_n : 0);
    size_t offset = 0;
    for (const StagedInt4& part : parts) {
        for (int g = 0; g < G; ++g) {
            std::memcpy(&scale_t[(size_t)g * total_n + offset],
                        &part.scale_t[(size_t)g * part.out_features],
                        (size_t)part.out_features * sizeof(bf16));
            if (has_zp)
                std::memcpy(&corr_t[(size_t)g * total_n + offset],
                            &part.corr_t[(size_t)g * part.out_features],
                            (size_t)part.out_features * sizeof(bf16));
        }
        offset += part.out_features;
    }
    lin.weight_scale = GpuBuffer<bf16>(scale_t.size(), queue);
    lin.weight_scale.upload(scale_t.data(), scale_t.size());
    if (has_zp) {
        lin.zp_offset = GpuBuffer<bf16>(corr_t.size(), queue);
        lin.zp_offset.upload(corr_t.data(), corr_t.size());
    }
    return lin;
}

Int4Linear upload_int4_linear_awq(const TensorSource& source,
                                  const std::string& prefix,
                                  sycl::queue& queue) {
    return upload_int4_staged({stage_int4_linear(source, prefix)}, queue);
}

Int4Linear upload_int4_linear_concat_awq(
    const TensorSource& source,
    std::initializer_list<std::string> prefixes,
    sycl::queue& queue) {
    std::vector<StagedInt4> parts;
    parts.reserve(prefixes.size());
    for (const std::string& prefix : prefixes)
        parts.push_back(stage_int4_linear(source, prefix));
    return upload_int4_staged(parts, queue);
}

void expect_proj(const Qwen35Proj& proj, int in, int out, const std::string& name) {
    int actual_in = std::visit([](const auto& w) { return w.in_features; }, proj);
    int actual_out = std::visit([](const auto& w) { return w.out_features; }, proj);
    if (actual_in != in || actual_out != out)
        throw std::runtime_error("Unexpected projection shape: " + name);
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
    const bool awq_int4 = config.quant_format == "pack-quantized";
    Qwen35Weights weights;
    weights.embed_tokens = load_bf16(
        source, "model.language_model.embed_tokens.weight",
        {c.vocab_size, c.hidden_size}, queue0);
    weights.final_norm = load_bf16(
        source, "model.language_model.norm.weight", {c.hidden_size}, queue0, true);
    if (awq_int4) {
        // lm_head is in the AWQ ignore list: unquantized F16 -> BF16.
        weights.lm_head = load_bf16(source, "lm_head.weight",
                                    {c.vocab_size, c.hidden_size}, queue0);
    } else {
        weights.lm_head = upload_fp8_linear(source, "lm_head", queue0);
        expect_fp8(std::get<Fp8Linear>(weights.lm_head), c.hidden_size,
                   c.vocab_size, "lm_head");
    }

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
            if (awq_int4) {
                if (attention.fused_projections)
                    attention.qkv_proj = upload_int4_linear_concat_awq(
                        source, {prefix + "q_proj", prefix + "k_proj",
                                 prefix + "v_proj"}, queue);
                else {
                    attention.q_proj = upload_int4_linear_awq(source, prefix + "q_proj", queue);
                    attention.k_proj = upload_int4_linear_awq(source, prefix + "k_proj", queue);
                    attention.v_proj = upload_int4_linear_awq(source, prefix + "v_proj", queue);
                }
                attention.o_proj = upload_int4_linear_awq(source, prefix + "o_proj", queue);
            } else {
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
            }
            int q_out = c.num_attention_heads * c.head_dim * 2;
            int kv_out = c.num_key_value_heads * c.head_dim;
            int attn_out = c.num_attention_heads * c.head_dim;
            if (attention.fused_projections)
                expect_proj(attention.qkv_proj, c.hidden_size, q_out + 2 * kv_out,
                            prefix + "qkv_proj");
            else {
                expect_proj(attention.q_proj, c.hidden_size, q_out, prefix + "q_proj");
                expect_proj(attention.k_proj, c.hidden_size, kv_out, prefix + "k_proj");
                expect_proj(attention.v_proj, c.hidden_size, kv_out, prefix + "v_proj");
            }
            expect_proj(attention.o_proj, attn_out, c.hidden_size, prefix + "o_proj");
            attention.q_norm = load_bf16(source, prefix + "q_norm.weight", {c.head_dim}, queue, true);
            attention.k_norm = load_bf16(source, prefix + "k_norm.weight", {c.head_dim}, queue, true);
            // FP8 KV-cache scales only exist in the NVFP4 checkpoint.
            if (source.has(prefix + "k_scale")) {
                attention.k_cache_scale = load_bf16(source, prefix + "k_scale", {1}, queue);
                attention.v_cache_scale = load_bf16(source, prefix + "v_scale", {1}, queue);
            } else {
                attention.k_cache_scale = GpuBuffer<bf16>(1, queue);
                attention.v_cache_scale = GpuBuffer<bf16>(1, queue);
                bf16* k_scale = attention.k_cache_scale.data();
                bf16* v_scale = attention.v_cache_scale.data();
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        k_scale[0] = float_to_bf16(1.0f);
                        v_scale[0] = float_to_bf16(1.0f);
                    });
                });
            }
            layer.mixer = std::move(attention);
        } else {
            std::string prefix = layer_prefix + "linear_attn.";
            Qwen35LinearAttentionWeights attention;
            int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
            int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
            int conv_dim = 2 * key_dim + value_dim;
            attention.fused_projections = fused_fp8_projections_enabled();
            if (awq_int4) {
                if (attention.fused_projections)
                    attention.in_proj_qkvz = upload_int4_linear_concat_awq(
                        source, {prefix + "in_proj_qkv", prefix + "in_proj_z"}, queue);
                else {
                    attention.in_proj_qkv = upload_int4_linear_awq(
                        source, prefix + "in_proj_qkv", queue);
                    attention.in_proj_z = upload_int4_linear_awq(
                        source, prefix + "in_proj_z", queue);
                }
                attention.out_proj = upload_int4_linear_awq(
                    source, prefix + "out_proj", queue);
            } else {
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
            }
            if (attention.fused_projections)
                expect_proj(attention.in_proj_qkvz, c.hidden_size,
                            conv_dim + value_dim, prefix + "in_proj_qkvz");
            else {
                expect_proj(attention.in_proj_qkv, c.hidden_size, conv_dim,
                            prefix + "in_proj_qkv");
                expect_proj(attention.in_proj_z, c.hidden_size, value_dim,
                            prefix + "in_proj_z");
            }
            expect_proj(attention.out_proj, value_dim, c.hidden_size, prefix + "out_proj");
            attention.in_proj_a = load_bf16(source, prefix + "in_proj_a.weight",
                                             {c.linear_num_value_heads, c.hidden_size}, queue);
            attention.in_proj_b = load_bf16(source, prefix + "in_proj_b.weight",
                                             {c.linear_num_value_heads, c.hidden_size}, queue);
            attention.in_proj_ba = GpuBuffer<bf16>(
                (size_t)2 * c.linear_num_value_heads * c.hidden_size, queue);
            queue.memcpy(
                attention.in_proj_ba.data(), attention.in_proj_b.data(),
                (size_t)c.linear_num_value_heads * c.hidden_size * sizeof(bf16));
            queue.memcpy(
                attention.in_proj_ba.data() +
                    (size_t)c.linear_num_value_heads * c.hidden_size,
                attention.in_proj_a.data(),
                (size_t)c.linear_num_value_heads * c.hidden_size * sizeof(bf16));
            attention.conv1d = load_bf16(source, prefix + "conv1d.weight",
                                          {conv_dim, 1, c.linear_conv_kernel_dim}, queue);
            attention.conv1d_time_major = GpuBuffer<bf16>(
                (size_t)conv_dim * c.linear_conv_kernel_dim, queue);
            {
                const bf16* source_weight = attention.conv1d.data();
                bf16* destination_weight = attention.conv1d_time_major.data();
                int kernel = c.linear_conv_kernel_dim;
                queue.submit([&](sycl::handler& handler) {
                    handler.parallel_for(
                        sycl::range<2>((size_t)kernel, (size_t)conv_dim),
                        [=](sycl::id<2> id) {
                            int tap = static_cast<int>(id[0]);
                            int channel = static_cast<int>(id[1]);
                            destination_weight[(size_t)tap * conv_dim + channel] =
                                source_weight[(size_t)channel * kernel + tap];
                        });
                });
            }
            attention.A_log = load_bf16(source, prefix + "A_log",
                                         {c.linear_num_value_heads}, queue);
            attention.dt_bias = load_bf16(source, prefix + "dt_bias",
                                           {c.linear_num_value_heads}, queue);
            attention.norm = load_bf16(source, prefix + "norm.weight",
                                        {c.linear_value_head_dim}, queue);
            layer.mixer = std::move(attention);
        }

        std::string mlp = layer_prefix + "mlp.";
        if (awq_int4) {
            Int4Linear gate_up = upload_int4_linear_concat_awq(
                source, {mlp + "gate_proj", mlp + "up_proj"}, queue);
            Int4Linear down = upload_int4_linear_awq(source, mlp + "down_proj", queue);
            if (gate_up.in_features != c.hidden_size ||
                gate_up.out_features != 2 * c.intermediate_size ||
                down.in_features != c.intermediate_size ||
                down.out_features != c.hidden_size)
                throw std::runtime_error("Unexpected int4 MLP shape in layer " +
                                         std::to_string(i));
            layer.mlp.gate_up = std::move(gate_up);
            layer.mlp.down = std::move(down);
        } else if (source.has(mlp + "gate_proj.weight_packed")) {
            Nvfp4Linear gate_up = upload_nvfp4_linear_pair(
                source, mlp + "gate_proj", mlp + "up_proj", queue);
            Nvfp4Linear down = upload_nvfp4_linear(source, mlp + "down_proj", queue);
            expect_nvfp4(gate_up, c.hidden_size, 2 * c.intermediate_size, mlp + "gate_up");
            expect_nvfp4(down, c.intermediate_size, c.hidden_size, mlp + "down_proj");
            layer.mlp.gate_up = std::move(gate_up);
            layer.mlp.down = std::move(down);
        } else {
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
    // The AWQ checkpoint quantizes the MTP head; it is unused at inference
    // time, so skip it there and exempt its tensors from the consumed check.
    bool mtp_loaded = !awq_int4;
    if (mtp_loaded) weights.mtp = load_mtp(source, config, queue0);

    if (max_layers == total_layers) {
        std::vector<std::string> missing;
        for (const std::string& name : checkpoint.names())
            if (!source.consumed(name) &&
                (mtp_loaded || name.rfind("mtp.", 0) != 0))
                missing.push_back(name);
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

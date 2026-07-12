#include "model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "loader.hpp"
#include "operators.hpp"
#include "vision.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/fp8.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "../../common/kernels/embedding.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/scatter.hpp"
#include "../../common/preprocess/chat_template.hpp"

namespace {

bool qwen35_persistent_io_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_PERSISTENT_IO");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

}  // namespace

Qwen35Model::Qwen35Model(const std::string& model_dir, int max_seq_len)
    : config_(Qwen35Config::from_dir(model_dir)), max_seq_len_(max_seq_len) {
    if (max_seq_len <= 0 || max_seq_len > config_.text.max_position_embeddings)
        throw std::runtime_error("Invalid Qwen3.5 maximum sequence length");
    split_layer_ = GpuEngine::count() >= 2 ? config_.text.num_hidden_layers / 2
                                           : config_.text.num_hidden_layers;
    int max_layers = -1;
    if (const char* value = std::getenv("ARCAINE_QWEN35_MAX_LAYERS")) {
        int parsed = std::atoi(value);
        if (parsed > 0) max_layers = parsed;
    }
    std::printf("[qwen35] %d GPU(s), layer split=%d, NVFP4 DPAS=%s, "
                "XMX attention=%s, ESIMD DeltaNet=%s, "
                "fused Delta decode=%s, fused BA=%s, "
                "persistent IO=%s\n",
                GpuEngine::count(), split_layer_,
                qwen35_nvfp4_dpas_enabled() ? "on" : "off",
                qwen35_xmx_attention_enabled() ? "on" : "off",
                qwen35_esimd_delta_enabled() ? "on" : "off",
                qwen35_fused_esimd_delta_decode_enabled() ? "on" : "off",
                qwen35_fused_ba_projection_enabled() ? "on" : "off",
                qwen35_persistent_io_enabled() ? "on" : "off");
    ShardedSafetensors checkpoint(model_dir);
    weights_ = load_qwen35_weights(checkpoint, config_, split_layer_, max_layers);
    caches_.init(config_, max_seq_len, split_layer_);

    size_t activations = (size_t)max_seq_len * config_.text.hidden_size;
    auto& queue0 = GpuEngine::get(0).queue;
    workspace0_.init(config_, max_seq_len, queue0);
    hidden0_ = GpuBuffer<bf16>(activations, queue0);
    normalized0_ = GpuBuffer<bf16>(activations, queue0);
    sublayer0_ = GpuBuffer<bf16>(activations, queue0);
    if (qwen35_persistent_io_enabled()) {
        token_ids0_ = GpuBuffer<int32_t>(max_seq_len, queue0);
        positions0_ = GpuBuffer<int32_t>((size_t)3 * max_seq_len, queue0);
        logits_bf16_ = GpuBuffer<bf16>(config_.text.vocab_size, queue0);
        logits_f32_ = GpuBuffer<float>(config_.text.vocab_size, queue0);
        transfer_host_.resize(activations);
    }
    if (GpuEngine::count() >= 2) {
        auto& queue1 = GpuEngine::get(1).queue;
        workspace1_.init(config_, max_seq_len, queue1);
        hidden1_ = GpuBuffer<bf16>(activations, queue1);
        normalized1_ = GpuBuffer<bf16>(activations, queue1);
        sublayer1_ = GpuBuffer<bf16>(activations, queue1);
        if (qwen35_persistent_io_enabled())
            positions1_ = GpuBuffer<int32_t>((size_t)3 * max_seq_len, queue1);
    }

    info_.vocab_size = config_.text.vocab_size;
    info_.max_seq_len = max_seq_len;
    info_.bos_token_id = config_.bos_token_id;
    info_.eos_token_ids = config_.eos_token_ids;
    info_.temperature = config_.temperature;
    info_.top_k = config_.top_k;
    info_.top_p = config_.top_p;
    info_.model_dir = model_dir;
    char description[256];
    std::snprintf(description, sizeof(description),
        "qwen3_5 conditional: %d layers, H=%d, %d full/%d linear, vision=%d blocks",
        config_.text.num_hidden_layers, config_.text.hidden_size,
        config_.text.num_hidden_layers / config_.text.full_attention_interval,
        config_.text.num_hidden_layers -
            config_.text.num_hidden_layers / config_.text.full_attention_interval,
        config_.vision.depth);
    info_.description = description;
}

PreparedInput Qwen35Model::prepare_input(
    const std::string& prompt,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& audio_paths,
    const std::string&) {
    if (!audio_paths.empty())
        throw std::runtime_error("Qwen3.5 supports image/video, not audio inputs");
    PreparedInput output;
    std::vector<int> image_token_counts;
    for (const auto& path : image_paths) {
        ImageInput image = preprocess_qwen_image(
            path, config_.vision.patch_size, config_.vision.temporal_patch_size,
            config_.vision.spatial_merge_size, config_.vision.min_pixels,
            config_.vision.max_pixels, config_.vision.rescale_factor,
            config_.vision.do_rescale, config_.vision.do_normalize,
            config_.vision.image_mean, config_.vision.image_std);
        image_token_counts.push_back(image.num_valid_patches);
        output.images.push_back(std::move(image));
    }
    auto built = build_chat_prompt(config_.model_dir, prompt, image_token_counts, {},
                                   true, false);
    output.tokens = std::move(built.tokens);
    output.mm_token_type_ids = std::move(built.mm_token_type_ids);
    return output;
}

std::vector<int32_t> Qwen35Model::build_positions(
    const std::vector<int>& tokens,
    const std::vector<int32_t>* token_types,
    const std::vector<ImageInput>* images,
    int past) {
    int seq = static_cast<int>(tokens.size());
    std::vector<int32_t> positions((size_t)3 * seq);
    if (past > 0) {
        for (int axis = 0; axis < 3; ++axis)
            for (int token = 0; token < seq; ++token)
                positions[(size_t)axis * seq + token] = past + rope_delta_ + token;
        return positions;
    }
    if (!token_types || token_types->empty() || !images || images->empty()) {
        for (int axis = 0; axis < 3; ++axis)
            for (int token = 0; token < seq; ++token)
                positions[(size_t)axis * seq + token] = token;
        rope_delta_ = 0;
        return positions;
    }
    if (static_cast<int>(token_types->size()) != seq)
        throw std::runtime_error("Qwen3.5 multimodal token type length mismatch");

    int current_position = 0;
    size_t image_index = 0;
    int token = 0;
    int maximum = 0;
    while (token < seq) {
        int type = token_types->at(token);
        int end = token + 1;
        while (end < seq && token_types->at(end) == type) ++end;
        if (type == 0) {
            for (int at = token; at < end; ++at) {
                int position = current_position + at - token;
                for (int axis = 0; axis < 3; ++axis)
                    positions[(size_t)axis * seq + at] = position;
                maximum = std::max(maximum, position);
            }
            current_position += end - token;
        } else if (type == 1 || type == 2) {
            if (image_index >= images->size())
                throw std::runtime_error("Qwen3.5 vision group has no matching grid");
            const ImageInput& image = images->at(image_index++);
            int merge = config_.vision.spatial_merge_size;
            int grid_t = image.grid_thw[0];
            int grid_h = image.grid_thw[1] / merge;
            int grid_w = image.grid_thw[2] / merge;
            if (end - token != grid_t * grid_h * grid_w)
                throw std::runtime_error("Qwen3.5 placeholder count does not match vision grid");
            int at = token;
            for (int t = 0; t < grid_t; ++t)
                for (int y = 0; y < grid_h; ++y)
                    for (int x = 0; x < grid_w; ++x, ++at) {
                        positions[at] = current_position + t;
                        positions[(size_t)seq + at] = current_position + y;
                        positions[(size_t)2 * seq + at] = current_position + x;
                        maximum = std::max({maximum, current_position + t,
                                            current_position + y, current_position + x});
                    }
            current_position += std::max(image.grid_thw[1], image.grid_thw[2]) / merge;
        } else {
            throw std::runtime_error("Unsupported Qwen3.5 multimodal token type");
        }
        token = end;
    }
    if (image_index != images->size())
        throw std::runtime_error("Unused Qwen3.5 image grids in prompt");
    rope_delta_ = maximum + 1 - seq;
    return positions;
}

void Qwen35Model::run_layer(
    GpuEngine& context, Qwen35LayerWeights& layer, Qwen35Workspace& workspace,
    Qwen35KvLayerCache& kv, Qwen35DeltaLayerCache& delta, bf16* hidden,
    bf16* normalized, bf16* sublayer, const int32_t* positions,
    int seq, int past) {
    const auto& c = config_.text;
    rms_norm(context.queue, hidden, layer.input_layernorm.data(), normalized,
             seq, c.hidden_size, c.rms_norm_eps);
    if (layer.full_attention) {
        qwen35_full_attention_forward(
            context, std::get<Qwen35FullAttentionWeights>(layer.mixer), kv,
            workspace, normalized, positions, sublayer, seq, past, config_);
    } else {
        qwen35_linear_attention_forward(
            context, std::get<Qwen35LinearAttentionWeights>(layer.mixer), delta,
            workspace, normalized, sublayer, seq, config_);
    }
    add_inplace(context.queue, hidden, sublayer, (size_t)seq * c.hidden_size);
    rms_norm(context.queue, hidden, layer.post_attention_layernorm.data(), normalized,
             seq, c.hidden_size, c.rms_norm_eps);
    qwen35_mlp_forward(context, layer.mlp, workspace, normalized, sublayer, seq, config_);
    add_inplace(context.queue, hidden, sublayer, (size_t)seq * c.hidden_size);
}

std::vector<float> Qwen35Model::forward(const ForwardInput& input) {
    int seq = static_cast<int>(input.token_ids.size());
    if (seq <= 0 || seq > max_seq_len_)
        throw std::runtime_error("Qwen3.5 forward sequence length out of range");
    auto& context0 = GpuEngine::get(0);
    auto& queue0 = context0.queue;
    std::vector<int32_t> host_ids(input.token_ids.begin(), input.token_ids.end());
    GpuBuffer<int32_t> token_ids_local;
    int32_t* token_ids = nullptr;
    if (qwen35_persistent_io_enabled()) {
        token_ids = token_ids0_.data();
        queue0.memcpy(token_ids, host_ids.data(), host_ids.size() * sizeof(int32_t));
    } else {
        token_ids_local = GpuBuffer<int32_t>(seq, queue0);
        token_ids_local.upload(host_ids.data(), host_ids.size());
        token_ids = token_ids_local.data();
    }
    embedding_lookup(queue0, weights_.embed_tokens.data(), token_ids,
                     hidden0_.data(), seq, config_.text.hidden_size, 1.0f);

    if (input.past_len == 0 && input.images && !input.images->empty()) {
        int search_from = 0;
        for (const ImageInput& image : *input.images) {
            GpuBuffer<bf16> features = qwen35_vision_forward(weights_.vision, config_.vision, image);
            std::vector<uint8_t> mask(seq, 0);
            int selected = 0;
            for (int i = search_from; i < seq && selected < image.num_valid_patches; ++i) {
                if (input.token_ids[i] == config_.image_token_id) {
                    mask[i] = 1;
                    ++selected;
                    search_from = i + 1;
                }
            }
            if (selected != image.num_valid_patches)
                throw std::runtime_error("Qwen3.5 image feature/token count mismatch");
            GpuBuffer<uint8_t> mask_device(seq, queue0);
            mask_device.upload(mask.data(), mask.size());
            auto offsets = compute_scatter_offsets(queue0, mask.data(), seq);
            masked_scatter_bf16(queue0, hidden0_.data(), features.data(), offsets.data(),
                                mask_device.data(), seq, config_.text.hidden_size);
        }
    }

    std::vector<int32_t> host_positions = build_positions(
        input.token_ids, input.mm_token_type_ids, input.images, input.past_len);
    GpuBuffer<int32_t> positions0_local;
    int32_t* positions0 = nullptr;
    if (qwen35_persistent_io_enabled()) {
        positions0 = positions0_.data();
        queue0.memcpy(positions0, host_positions.data(),
                      host_positions.size() * sizeof(int32_t));
    } else {
        positions0_local = GpuBuffer<int32_t>(host_positions.size(), queue0);
        positions0_local.upload(host_positions.data(), host_positions.size());
        positions0 = positions0_local.data();
    }

    int loaded_layers = static_cast<int>(weights_.layers.size());
    int first_stage_end = std::min(split_layer_, loaded_layers);
    for (int layer = 0; layer < first_stage_end; ++layer)
        run_layer(context0, weights_.layers[layer], workspace0_, caches_.kv[layer],
                  caches_.delta[layer], hidden0_.data(), normalized0_.data(),
                  sublayer0_.data(), positions0, seq, input.past_len);

    bf16* last_device = hidden0_.data() + (size_t)(seq - 1) * config_.text.hidden_size;
    if (loaded_layers > split_layer_) {
        auto& context1 = GpuEngine::get(1);
        size_t hidden_count = (size_t)seq * config_.text.hidden_size;
        std::vector<bf16> transfer_local;
        bf16* transfer = nullptr;
        if (qwen35_persistent_io_enabled()) {
            transfer = transfer_host_.data();
        } else {
            transfer_local.resize(hidden_count);
            transfer = transfer_local.data();
        }
        queue0.memcpy(transfer, hidden0_.data(), hidden_count * sizeof(bf16)).wait();
        context1.queue.memcpy(hidden1_.data(), transfer,
                              hidden_count * sizeof(bf16)).wait();
        GpuBuffer<int32_t> positions1_local;
        int32_t* positions1 = nullptr;
        if (qwen35_persistent_io_enabled()) {
            positions1 = positions1_.data();
            context1.queue.memcpy(positions1, host_positions.data(),
                                  host_positions.size() * sizeof(int32_t));
        } else {
            positions1_local = GpuBuffer<int32_t>(host_positions.size(), context1.queue);
            positions1_local.upload(host_positions.data(), host_positions.size());
            positions1 = positions1_local.data();
        }
        for (int layer = split_layer_; layer < loaded_layers; ++layer)
            run_layer(context1, weights_.layers[layer], workspace1_, caches_.kv[layer],
                      caches_.delta[layer], hidden1_.data(), normalized1_.data(),
                      sublayer1_.data(), positions1, seq, input.past_len);
        std::vector<bf16> last_local;
        bf16* last = transfer;
        if (!qwen35_persistent_io_enabled()) {
            last_local.resize(config_.text.hidden_size);
            last = last_local.data();
        }
        context1.queue.memcpy(last,
            hidden1_.data() + (size_t)(seq - 1) * config_.text.hidden_size,
            (size_t)config_.text.hidden_size * sizeof(bf16)).wait();
        queue0.memcpy(normalized0_.data(), last,
                      (size_t)config_.text.hidden_size * sizeof(bf16)).wait();
        last_device = normalized0_.data();
    }

    rms_norm(queue0, last_device, weights_.final_norm.data(), normalized0_.data(),
             1, config_.text.hidden_size, config_.text.rms_norm_eps);
    GpuBuffer<bf16> logits_bf16_local;
    bf16* logits_bf16 = nullptr;
    if (qwen35_persistent_io_enabled())
        logits_bf16 = logits_bf16_.data();
    else {
        logits_bf16_local = GpuBuffer<bf16>(config_.text.vocab_size, queue0);
        logits_bf16 = logits_bf16_local.data();
    }
    matmul_fp8(normalized0_.data(), 1, config_.text.hidden_size,
               weights_.lm_head, logits_bf16, context0);
    GpuBuffer<float> logits_f32_local;
    float* logits_f32 = nullptr;
    if (qwen35_persistent_io_enabled())
        logits_f32 = logits_f32_.data();
    else {
        logits_f32_local = GpuBuffer<float>(config_.text.vocab_size, queue0);
        logits_f32 = logits_f32_local.data();
    }
    bf16_to_f32(queue0, logits_bf16, logits_f32, config_.text.vocab_size);
    std::vector<float> logits(config_.text.vocab_size);
    queue0.memcpy(logits.data(), logits_f32,
                  logits.size() * sizeof(float)).wait();
    return logits;
}

void Qwen35Model::reset_cache() {
    caches_.reset();
    rope_delta_ = 0;
}

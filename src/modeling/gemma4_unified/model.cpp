#include "model.hpp"
#include "decoder_layer.hpp"
#include "vision_embedder.hpp"
#include "audio_embedder.hpp"
#include "../../common/preprocess/chat_template.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/kernels/embedding.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "../../common/kernels/scatter.hpp"
#include <cmath>
#include <vector>
#include <optional>
#include <cstdio>
#include <stdexcept>

extern GlobalWeights load_weights(const std::string& model_dir,
                                  const ModelConfig& cfg,
                                  int split_layer);

Gemma4Model::Gemma4Model(const std::string& model_dir, int max_seq_len) {
    cfg_ = ModelConfig::from_dir(model_dir);

    // Split layers evenly across two GPUs; fall back to all-GPU-0 if only one present.
    int L = cfg_.text.num_hidden_layers;
    split_layer_ = (GpuEngine::count() >= 2) ? L / 2 : L;

    std::printf("[model] %d GPU(s) available — split at layer %d\n",
                GpuEngine::count(), split_layer_);

    weights_  = load_weights(model_dir, cfg_, split_layer_);
    kv_cache_ = KvCache(cfg_, max_seq_len, split_layer_);

    // Architecture-independent view of the model for frontends/sampler/chat.
    info_.vocab_size      = cfg_.text.vocab_size;
    info_.max_seq_len     = max_seq_len;
    info_.bos_token_id    = cfg_.bos_token_id;
    info_.eos_token_ids   = cfg_.generation.eos_token_ids;
    info_.suppress_tokens = cfg_.generation.suppress_tokens;
    info_.temperature     = cfg_.generation.temperature;
    info_.top_k           = cfg_.generation.top_k;
    info_.top_p           = cfg_.generation.top_p;
    info_.model_dir       = model_dir;
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "gemma4_unified: %d layers, H=%d, vocab=%d, window=%d",
            cfg_.text.num_hidden_layers, cfg_.text.hidden_size,
            cfg_.text.vocab_size, cfg_.text.sliding_window);
        info_.description = buf;
    }
}

PreparedInput Gemma4Model::prepare_input(
    const std::string&              prompt,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& audio_paths,
    const std::string&              vad_model
) {
    PreparedInput out;
    std::vector<int> image_token_counts;
    std::vector<int> audio_token_counts;

    for (const auto& path : image_paths) {
        std::printf("[model] Processing image: %s\n", path.c_str());
        auto img = preprocess_image(path,
            cfg_.vision.patch_size,
            cfg_.vision.pooling_kernel_size,
            cfg_.vision.num_soft_tokens,
            cfg_.vision.rescale_factor,
            cfg_.vision.do_rescale);
        std::printf("[model] Image: %d valid patches\n", img.num_valid_patches);
        image_token_counts.push_back(img.num_valid_patches);
        out.images.push_back(std::move(img));
    }

    for (const auto& path : audio_paths) {
        std::printf("[model] Processing audio: %s\n", path.c_str());
        auto aud = preprocess_audio_file(path,
            cfg_.audio.audio_samples_per_token,
            cfg_.audio.sampling_rate,
            vad_model);
        std::printf("[model] Audio: %d frames\n", aud.num_frames);
        audio_token_counts.push_back(aud.num_frames);
        out.audio.push_back(std::move(aud));
    }

    // Render the chat through the model's own template metadata.
    auto built = build_chat_prompt(cfg_.model_dir, prompt,
        image_token_counts, audio_token_counts,
        /*add_generation_prompt=*/true, /*enable_thinking=*/false);
    out.tokens            = std::move(built.tokens);
    out.mm_token_type_ids = std::move(built.mm_token_type_ids);
    return out;
}

std::vector<float> Gemma4Model::forward(const ForwardInput& in) {
    return forward_tokens(in.token_ids, in.past_len,
                          in.images, in.audio, in.mm_token_type_ids);
}

std::vector<float> Gemma4Model::forward_tokens(
    const std::vector<int>&        token_ids,
    int                            past_len,
    const std::vector<ImageInput>* images,
    const std::vector<AudioInput>* audio_inputs,
    const std::vector<int32_t>*    mm_token_type_ids
) {
    auto& ctx0 = GpuEngine::get(0);
    auto& q0   = ctx0.queue;

    int seq = (int)token_ids.size();
    int H   = cfg_.text.hidden_size;
    int L   = cfg_.text.num_hidden_layers;
    float embed_scale = std::sqrt(float(H));

    // =========================================================
    // Stage 0 — GPU 0: embedding, modality scatter, layers [0, split_layer_)
    // =========================================================

    // Upload token ids
    GpuBuffer<int32_t> ids_dev(seq, q0);
    {
        std::vector<int32_t> ids32(token_ids.begin(), token_ids.end());
        ids_dev.upload(ids32.data(), seq);
    }

    // Embedding lookup + scale
    GpuBuffer<bf16> hidden((size_t)seq * H, q0);
    embedding_lookup(q0,
        weights_.embed_tokens.data(), ids_dev.data(), hidden.data(),
        seq, H, embed_scale);

    // Vision: scatter vision embeddings into placeholder positions
    if (images && !images->empty()) {
        for (auto& img : *images) {
            auto ve = vision_embedder_forward(
                weights_.vision, img, cfg_.vision, H);

            std::vector<uint8_t> host_mask(seq, 0);
            int count = 0;
            for (int i = 0; i < seq; ++i) {
                if (token_ids[i] == cfg_.image_token_id) {
                    host_mask[i] = 1;
                    ++count;
                }
            }
            if (count != img.num_valid_patches) {
                std::fprintf(stderr,
                    "[warning] image token count %d != num_valid_patches %d\n",
                    count, img.num_valid_patches);
            }

            GpuBuffer<uint8_t> mask_dev(seq, q0);
            mask_dev.upload(host_mask.data(), seq);
            auto offsets = compute_scatter_offsets(q0, host_mask.data(), seq);
            masked_scatter_bf16(q0, hidden.data(), ve.data(),
                                offsets.data(), mask_dev.data(), seq, H);
        }
    }

    // Audio: scatter audio embeddings
    if (audio_inputs && !audio_inputs->empty()) {
        for (auto& aud : *audio_inputs) {
            auto ae = audio_embedder_forward(
                weights_.audio, aud, cfg_.audio, H);

            std::vector<uint8_t> host_mask(seq, 0);
            for (int i = 0; i < seq; ++i) {
                if (token_ids[i] == cfg_.audio_token_id)
                    host_mask[i] = 1;
            }
            GpuBuffer<uint8_t> mask_dev(seq, q0);
            mask_dev.upload(host_mask.data(), seq);
            auto offsets = compute_scatter_offsets(q0, host_mask.data(), seq);
            masked_scatter_bf16(q0, hidden.data(), ae.data(),
                                offsets.data(), mask_dev.data(), seq, H);
        }
    }

    std::vector<int32_t> block_ids_host;
    std::optional<GpuBuffer<int32_t>> block_ids0;
    if (mm_token_type_ids && past_len == 0 &&
        cfg_.text.use_bidirectional_attention == "vision") {
        if ((int)mm_token_type_ids->size() != seq)
            throw std::runtime_error("mm_token_type_ids length does not match token_ids");
        block_ids_host.assign(seq, -1);
        int32_t current_block = -1;
        bool prev_vision = false;
        for (int i = 0; i < seq; ++i) {
            bool is_vision = ((*mm_token_type_ids)[i] == 1 || (*mm_token_type_ids)[i] == 2);
            if (is_vision && !prev_vision)
                ++current_block;
            block_ids_host[i] = is_vision ? current_block : -1;
            prev_vision = is_vision;
        }
        block_ids0.emplace(seq, q0);
        block_ids0->upload(block_ids_host.data(), seq);
    }
    const int32_t* block_ids0_ptr = block_ids0 ? block_ids0->data() : nullptr;

    // Decoder layers on GPU 0
    for (int l = 0; l < split_layer_; ++l) {
        decoder_layer_forward(ctx0, weights_.layers[l], hidden.data(),
                              kv_cache_.layer(l), seq, past_len, cfg_,
                              block_ids0_ptr);
    }

    // =========================================================
    // Stage 1 — GPU 1: layers [split_layer_, L)
    // Active only when split_layer_ < L (i.e. two GPUs present).
    // =========================================================
    if (split_layer_ < L) {
        auto& ctx1 = GpuEngine::get(1);
        auto& q1   = ctx1.queue;

        // Host-staged GPU0→GPU1 transfer of the hidden state.
        // For decode (seq=1): 7.5 KB; for 2048-token prefill: ~15 MB.
        size_t hidden_bytes = (size_t)seq * H * sizeof(bf16);
        std::vector<bf16> staging(seq * H);
        q0.memcpy(staging.data(), hidden.data(), hidden_bytes).wait();

        GpuBuffer<bf16> hidden1((size_t)seq * H, q1);
        q1.memcpy(hidden1.data(), staging.data(), hidden_bytes).wait();

        std::optional<GpuBuffer<int32_t>> block_ids1;
        if (!block_ids_host.empty()) {
            block_ids1.emplace(seq, q1);
            block_ids1->upload(block_ids_host.data(), seq);
        }
        const int32_t* block_ids1_ptr = block_ids1 ? block_ids1->data() : nullptr;

        // Decoder layers on GPU 1
        for (int l = split_layer_; l < L; ++l) {
            decoder_layer_forward(ctx1, weights_.layers[l], hidden1.data(),
                                  kv_cache_.layer(l), seq, past_len, cfg_,
                                  block_ids1_ptr);
        }

        // Transfer last token GPU1→GPU0 for final norm + lm_head (both on GPU 0).
        // Only 1 × H × 2 bytes = 7.5 KB even for large batches.
        std::vector<bf16> last_host(H);
        q1.memcpy(last_host.data(),
                  hidden1.data() + (size_t)(seq - 1) * H,
                  (size_t)H * sizeof(bf16)).wait();

        GpuBuffer<bf16> last(H, q0);
        q0.memcpy(last.data(), last_host.data(), (size_t)H * sizeof(bf16)).wait();

        // Final norm on GPU 0
        rms_norm(q0, last.data(), weights_.final_norm.data(), last.data(),
                 1, H, cfg_.text.rms_norm_eps);

        // lm_head = embed_tokens.T (tied weights, on GPU 0)
        int V = cfg_.text.vocab_size;
        GpuBuffer<bf16> logits_bf16(V, q0);
        matmul_bf16(last.data(), 1, H, weights_.embed_tokens.data(), V,
                    logits_bf16.data(), ctx0);

        GpuBuffer<float> logits_f32(V, q0);
        bf16_to_f32(q0, logits_bf16.data(), logits_f32.data(), V);
        softcap_inplace(q0, logits_f32.data(), V, cfg_.text.final_logit_softcapping);

        std::vector<float> logits(V);
        logits_f32.download(logits.data(), V);
        return logits;
    }

    // =========================================================
    // Single-GPU path: everything ran on GPU 0 above.
    // =========================================================
    GpuBuffer<bf16> last(H, q0);
    q0.memcpy(last.data(), hidden.data() + (size_t)(seq - 1) * H,
              (size_t)H * sizeof(bf16)).wait();

    rms_norm(q0, last.data(), weights_.final_norm.data(), last.data(),
             1, H, cfg_.text.rms_norm_eps);

    int V = cfg_.text.vocab_size;
    GpuBuffer<bf16> logits_bf16(V, q0);
    matmul_bf16(last.data(), 1, H, weights_.embed_tokens.data(), V,
                logits_bf16.data(), ctx0);

    GpuBuffer<float> logits_f32(V, q0);
    bf16_to_f32(q0, logits_bf16.data(), logits_f32.data(), V);
    softcap_inplace(q0, logits_f32.data(), V, cfg_.text.final_logit_softcapping);

    std::vector<float> logits(V);
    logits_f32.download(logits.data(), V);
    return logits;
}

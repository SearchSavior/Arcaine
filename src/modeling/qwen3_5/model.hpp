#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>

#include "cache.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "workspace.hpp"
#include "../../preprocessing/image_processor.hpp"
#include "../../preprocessing/audio_processor.hpp"

// ---------------------------------------------------------------------------
// Model-local interface types (relocated per the model-isolation rule).
// Each autoregressive model owns its copy; the model's session / request_mapper
// / benchmark drive the concrete model through these. DiffusionGemma does not
// use them (its generate() takes a direct prompt + step count).
// ---------------------------------------------------------------------------
struct ForwardInput {
    const std::vector<int>&        token_ids;
    int                            past_len = 0;
    const std::vector<ImageInput>* images            = nullptr;
    const std::vector<AudioInput>* audio             = nullptr;
    const std::vector<int32_t>*    mm_token_type_ids = nullptr;
};

struct PreparedInput {
    std::vector<int>        tokens;
    std::vector<ImageInput> images;
    std::vector<AudioInput> audio;
    std::vector<int32_t>    mm_token_type_ids;  // 0=text, 1=image, 2=video, 3=audio
};

struct ModelInfo {
    int   vocab_size   = 0;
    int   max_seq_len  = 0;
    int   bos_token_id = -1;
    std::vector<int> eos_token_ids;
    std::vector<int> suppress_tokens;
    float temperature  = 1.0f;
    int   top_k        = 64;
    float top_p        = 0.95f;
    std::string model_dir;
    std::string description;

    bool is_eos(int id) const {
        return std::find(eos_token_ids.begin(), eos_token_ids.end(), id)
               != eos_token_ids.end();
    }
};

class Qwen35Model final {
public:
    explicit Qwen35Model(const std::string& model_dir, int max_seq_len = 2048);

    PreparedInput prepare_input(const std::string& prompt,
                                const std::vector<std::string>& image_paths,
                                const std::vector<std::string>& audio_paths,
                                const std::string& vad_model);
    std::vector<float> forward(const ForwardInput& input);
    void reset_cache();
    const ModelInfo& info() const { return info_; }

private:
    std::vector<int32_t> build_positions(const std::vector<int>& tokens,
                                         const std::vector<int32_t>* token_types,
                                         const std::vector<ImageInput>* images,
                                         int past);
    void run_layer(GpuEngine& context, Qwen35LayerWeights& layer,
                   Qwen35Workspace& workspace, Qwen35KvLayerCache& kv,
                   Qwen35DeltaLayerCache& delta, bf16* hidden,
                   bf16* normalized, bf16* sublayer, const int32_t* positions,
                   int seq, int past);

    Qwen35Config config_;
    Qwen35Weights weights_;
    Qwen35Caches caches_;
    int split_layer_ = 0;
    int max_seq_len_ = 0;
    int rope_delta_ = 0;
    Qwen35Workspace workspace0_;
    Qwen35Workspace workspace1_;
    GpuBuffer<bf16> hidden0_;
    GpuBuffer<bf16> normalized0_;
    GpuBuffer<bf16> sublayer0_;
    GpuBuffer<bf16> hidden1_;
    GpuBuffer<bf16> normalized1_;
    GpuBuffer<bf16> sublayer1_;
    GpuBuffer<int32_t> token_ids0_;
    GpuBuffer<int32_t> positions0_;
    GpuBuffer<int32_t> positions1_;
    GpuBuffer<bf16> logits_bf16_;
    GpuBuffer<float> logits_f32_;
    std::vector<bf16> transfer_host_;
    ModelInfo info_;
};

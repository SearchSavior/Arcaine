#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>
#include "config.hpp"
#include "../../runtime/gpu/buffer.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
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
    std::string model_dir;     // for the chat template / tokenizer subprocess
    std::string description;   // one-line banner

    bool is_eos(int id) const {
        return std::find(eos_token_ids.begin(), eos_token_ids.end(), id)
               != eos_token_ids.end();
    }
};

class Gemma4Model {
public:
    explicit Gemma4Model(const std::string& model_dir, int max_seq_len = 2048);

    // --- Model interface ---
    PreparedInput prepare_input(
        const std::string&              prompt,
        const std::vector<std::string>& image_paths,
        const std::vector<std::string>& audio_paths,
        const std::string&              vad_model);
    std::vector<float> forward(const ForwardInput& in);
    void               reset_cache() { kv_cache_.reset(); }
    const ModelInfo&   info() const { return info_; }

    // Full architecture config (concrete-type accessor; not part of Model).
    const ModelConfig& config() const { return cfg_; }

private:
    // Core forward used by the Model::forward override.
    // images/audio are consumed only on the prefill call (past_len == 0).
    std::vector<float> forward_tokens(
        const std::vector<int>&          token_ids,
        int                              past_len,
        const std::vector<ImageInput>*   images,
        const std::vector<AudioInput>*   audio,
        const std::vector<int32_t>*      mm_token_type_ids);

    ModelConfig   cfg_;
    ModelInfo     info_;
    int           split_layer_ = 0;   // layers [0,split) on GPU0; [split,L) on GPU1
    GlobalWeights weights_;
    KvCache       kv_cache_;
};

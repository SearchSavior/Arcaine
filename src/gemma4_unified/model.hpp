#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include "config.hpp"
#include "../common/model.hpp"
#include "../common/gpu/buffer.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
#include "../common/preprocess/image_proc.hpp"
#include "../common/preprocess/audio_proc.hpp"

class Gemma4Model : public Model {
public:
    explicit Gemma4Model(const std::string& model_dir, int max_seq_len = 2048);

    // --- Model interface ---
    PreparedInput prepare_input(
        const std::string&              prompt,
        const std::vector<std::string>& image_paths,
        const std::vector<std::string>& audio_paths,
        const std::string&              vad_model) override;
    std::vector<float> forward(const ForwardInput& in) override;
    void               reset_cache() override { kv_cache_.reset(); }
    const ModelInfo&   info() const override { return info_; }

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

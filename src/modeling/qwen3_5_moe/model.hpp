#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "config.hpp"
#include "weights.hpp"
#include "attention.hpp"
#include "gated_deltanet.hpp"
#include "../../runtime/gpu/buffer.hpp"
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

// Qwen3.5-MoE (text-only) inference model. Registered under config.json's
// model_type "qwen3_5_moe_text" (see registration.cpp / builtin_models.cpp).
//
// 40-layer hybrid decoder: 30x Gated DeltaNet (linear attention) + 10x full
// attention (every 4th layer). Every layer is MoE: 256 routed experts (top-8,
// SwiGLU, NVFP4) + an always-on shared expert with a per-token sigmoid gate.
// lm_head is untied. See notes/qwen_agentworld_35b/ for the full architecture
// spec and the port plan.
class QwenModel {
public:
    explicit QwenModel(const std::string& model_dir, int max_seq_len = 2048);

    // --- Model interface ---
    PreparedInput prepare_input(
        const std::string&              prompt,
        const std::vector<std::string>& image_paths,
        const std::vector<std::string>& audio_paths,
        const std::string&              vad_model);
    std::vector<float> forward(const ForwardInput& in);
    void               reset_cache();
    const ModelInfo&   info() const { return info_; }

    // Concrete-type accessors (not part of Model).
    const QwenConfig&  config()  const { return cfg_; }
    const QwenWeights& weights() const { return weights_; }

private:
    // Core forward used by Model::forward. Text-only: multimodal inputs in
    // ForwardInput are ignored (the model is text-only despite its multimodal
    // container wrapper).
    std::vector<float> forward_tokens(const std::vector<int>& token_ids,
                                      int past_len);

    QwenConfig   cfg_;
    QwenWeights  weights_;
    QwenKvCache            kv_cache_;       // full-attn layers (10)
    QwenLinearAttnCaches   linear_caches_;  // linear-attn layers (30)
    ModelInfo    info_;
};

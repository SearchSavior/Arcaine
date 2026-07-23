#pragma once
#include <string>
#include <vector>
#include "config.hpp"
#include "weights.hpp"
#include "attention.hpp"
#include "gated_deltanet.hpp"
#include "../../common/model_interface.hpp"
#include "../../common/gpu/buffer.hpp"

// Qwen3.5-MoE (text-only) inference model. Subclass of the architecture-
// independent `Model` interface; registered under config.json's model_type
// "qwen3_5_moe_text" (see register.cpp / register_builtins.cpp).
//
// 40-layer hybrid decoder: 30x Gated DeltaNet (linear attention) + 10x full
// attention (every 4th layer). Every layer is MoE: 256 routed experts (top-8,
// SwiGLU, NVFP4) + an always-on shared expert with a per-token sigmoid gate.
// lm_head is untied. See notes/qwen_agentworld_35b/ for the full architecture
// spec and the port plan.
class QwenModel : public Model {
public:
    explicit QwenModel(const std::string& model_dir, int max_seq_len = 2048);

    // --- Model interface ---
    PreparedInput prepare_input(
        const std::string&              prompt,
        const std::vector<std::string>& image_paths,
        const std::vector<std::string>& audio_paths,
        const std::string&              vad_model) override;
    std::vector<float> forward(const ForwardInput& in) override;
    void               reset_cache() override;
    const ModelInfo&   info() const override { return info_; }

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

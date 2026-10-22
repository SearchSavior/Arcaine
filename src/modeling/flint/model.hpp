#pragma once

#include <memory>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "../../common/model_interface.hpp"
#include "config.hpp"
#include "weights.hpp"

class FlintModel final : public Model {
public:
    FlintModel(const std::string& model_dir, int max_seq_len);
    ~FlintModel() override;

    PreparedInput prepare_input(const std::string& prompt,
                                const std::vector<std::string>& image_paths,
                                const std::vector<std::string>& audio_paths,
                                const std::string& vad_model) override;
    std::vector<float> forward(const ForwardInput& in) override;
    void reset_cache() override;
    const ModelInfo& info() const override { return info_; }

private:
    struct KVCache {
        flint::bf16* k = nullptr;
        flint::bf16* v = nullptr;
        int cap = 0;
        bool global = true;
    };

    void step(int token);
    void ensure_batch(int m);
    void prefill_batch(const std::vector<int>& tokens, int pos_base);
    std::vector<float> logits();
public:
    void dump_trace(const std::string& path);
private:

    FlintConfig cfg_;
    flint::FlintWeights w_;
    ModelInfo info_;
    sycl::queue q_;

    int max_seq_ = 0;
    int pos_ = 0;
    float log2base_ = 0.f;

    flint::bf16 *h_ = nullptr, *normed_ = nullptr, *qbuf_ = nullptr,
                *kbuf_ = nullptr, *vbuf_ = nullptr, *attn_ = nullptr,
                *act_ = nullptr, *down_ = nullptr, *gate_ = nullptr,
                *up_ = nullptr;
    float* logits_dev_ = nullptr;
    flint::bf16* trace_ = nullptr;
    int* ids_dev_ = nullptr;
    float* weights_dev_ = nullptr;
    std::vector<KVCache> kv_;

    // Batched-prefill scratch, allocated on first use.
    int batch_cap_ = 0;
    int* b_tokens = nullptr;
    flint::bf16 *b_h = nullptr, *b_normed = nullptr, *b_q = nullptr,
                *b_k = nullptr, *b_v = nullptr, *b_attn = nullptr,
                *b_gate = nullptr, *b_up = nullptr, *b_routed = nullptr,
                *b_shared_act = nullptr, *b_shared_up = nullptr,
                *b_shared_down = nullptr, *b_egate = nullptr, *b_eup = nullptr,
                *b_eact = nullptr;
    int* b_ids = nullptr;
    float* b_weights = nullptr;
    int* b_rows = nullptr;
    int* b_outs = nullptr;
    int8_t* b_xq = nullptr;
    float* b_xs = nullptr;
    int8_t* b_xq2 = nullptr;
    float* b_xs2 = nullptr;
};

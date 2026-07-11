#pragma once

#include <string>
#include <vector>

#include "cache.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "workspace.hpp"
#include "../../common/model_interface.hpp"

class Qwen35Model final : public Model {
public:
    explicit Qwen35Model(const std::string& model_dir, int max_seq_len = 2048);

    PreparedInput prepare_input(const std::string& prompt,
                                const std::vector<std::string>& image_paths,
                                const std::vector<std::string>& audio_paths,
                                const std::string& vad_model) override;
    std::vector<float> forward(const ForwardInput& input) override;
    void reset_cache() override;
    const ModelInfo& info() const override { return info_; }

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

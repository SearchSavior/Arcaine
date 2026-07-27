#pragma once

class QwenModel;  // global (modeling/qwen3_5_moe/model.hpp)

namespace arcaine::qwen3_5_moe {

// Thin ownership wrapper around the model's persistent cache (KV + linear
// attention / DeltaNet recurrent state). The session resets it at the start
// of each generation so each request begins a fresh sequence.
class QwenMoeCache {
public:
    explicit QwenMoeCache(QwenModel& model) : model_(model) {}
    void reset();

private:
    QwenModel& model_;
};

}  // namespace arcaine::qwen3_5_moe

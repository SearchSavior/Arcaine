#pragma once

class Qwen35Model;  // global (modeling/qwen3_5/model.hpp)

namespace arcaine::qwen3_5 {

// Thin ownership wrapper around the model's persistent cache (KV + DeltaNet
// recurrent state). The session resets it at the start of each generation so
// each request begins a fresh sequence.
class Qwen35Cache {
public:
    explicit Qwen35Cache(Qwen35Model& model) : model_(model) {}
    void reset();

private:
    Qwen35Model& model_;
};

}  // namespace arcaine::qwen3_5

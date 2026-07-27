#pragma once

class Gemma4Model;  // global (modeling/gemma4_unified/model.hpp)

namespace arcaine::gemma4_unified {

// Thin ownership wrapper around the model's persistent KV cache. The session
// resets it at the start of each generation so each request begins a fresh
// sequence. (The cache storage remains persistent in the loaded model; the
// session owns the per-request cache behavior.)
class Gemma4Cache {
public:
    explicit Gemma4Cache(Gemma4Model& model) : model_(model) {}
    void reset();

private:
    Gemma4Model& model_;
};

}  // namespace arcaine::gemma4_unified

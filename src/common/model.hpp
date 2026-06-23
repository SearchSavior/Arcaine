#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "preprocess/image_proc.hpp"
#include "preprocess/audio_proc.hpp"

// ---------------------------------------------------------------------------
// Architecture-independent inference interface.
//
// Every model implementation (gemma4_unified/, and any future arch) implements
// `Model`. Frontends (main.cpp, bench.cpp) and the sampler/chat-template layers
// depend only on this header — never on an architecture's private config.
// ---------------------------------------------------------------------------

// One forward call: a contiguous chunk of tokens at absolute offset `past_len`.
// Multimodal inputs are consumed only on the prefill call (past_len == 0).
struct ForwardInput {
    const std::vector<int>&        token_ids;
    int                            past_len = 0;
    const std::vector<ImageInput>* images            = nullptr;
    const std::vector<AudioInput>* audio             = nullptr;
    const std::vector<int32_t>*    mm_token_type_ids = nullptr;
};

// Tokenized prompt plus any multimodal tensors, produced by Model::prepare_input.
struct PreparedInput {
    std::vector<int>        tokens;
    std::vector<ImageInput> images;
    std::vector<AudioInput> audio;
    std::vector<int32_t>    mm_token_type_ids;  // 0=text, 1=image, 2=video, 3=audio
};

// Everything a frontend needs to drive generation without knowing the arch.
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
    std::string description;   // one-line banner, e.g. "48 layers, H=3840, ..."

    bool is_eos(int id) const {
        return std::find(eos_token_ids.begin(), eos_token_ids.end(), id)
               != eos_token_ids.end();
    }
};

class Model {
public:
    virtual ~Model() = default;

    // Apply the chat template to `prompt` and run any multimodal preprocessing
    // for the given media paths. `vad_model` is an optional Silero VAD model
    // path for audio (pass "" to skip). Architectures that are text-only may
    // ignore the media paths.
    virtual PreparedInput prepare_input(
        const std::string&              prompt,
        const std::vector<std::string>& image_paths,
        const std::vector<std::string>& audio_paths,
        const std::string&              vad_model) = 0;

    // Returns float logits (vocab_size,) for the last token position.
    virtual std::vector<float> forward(const ForwardInput& in) = 0;

    // Clear the KV cache so the next forward starts a fresh sequence.
    virtual void reset_cache() = 0;

    virtual const ModelInfo& info() const = 0;
};

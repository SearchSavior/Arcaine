#pragma once
#include <functional>
#include <string>
#include <vector>
#include <optional>
#include <random>
#include "config.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
#include "../../common/gpu/placement.hpp"
#include "../../common/gpu/buffer.hpp"

// Streaming event, fired after every denoising step and once per committed
// canvas (committed=true; `canvas` is then the final block, EOS-truncated).
struct DiffStepEvent {
    int   block       = 0;
    int   cur_step    = 0;      // counts down N..1; 0 on commit
    float temperature = 0.0f;
    float mean_entropy = 0.0f;
    bool  committed   = false;
    const std::vector<int>*   canvas   = nullptr;  // argmax canvas (draft)
    const std::vector<float>* entropy  = nullptr;  // per-position entropy (nats)
    const std::vector<char>*  accepted = nullptr;  // entropy-bound accept mask
};
using DiffStreamCallback = std::function<void(const DiffStepEvent&)>;

// Throughput counters for one generate() call.
struct DiffPerfStats {
    int    prefill_tokens = 0;   // tokens pushed through the encoder
    double prefill_s      = 0.0;
    int    decode_passes  = 0;   // denoiser forward passes (canvas_length each)
    double decode_s       = 0.0;
    int    output_tokens  = 0;   // committed tokens returned

    double prefill_tps()       const { return prefill_s > 0 ? prefill_tokens / prefill_s : 0; }
    double decode_passes_ps()  const { return decode_s  > 0 ? decode_passes  / decode_s  : 0; }
    double effective_tps()     const { return decode_s  > 0 ? output_tokens  / decode_s  : 0; }
    double tokens_per_forward() const { return decode_passes > 0 ? (double)output_tokens / decode_passes : 0; }
};

class DiffusionGemmaModel {
public:
    DiffusionGemmaModel(const std::string& model_dir, int max_seq_len, DiffPlacementOptions placement = {}, bool print_placement = true);

    // Block-diffusion generation. Returns generated token ids (prompt excluded).
    std::vector<int> generate(const std::vector<int>& prompt_ids,
                              int max_new_tokens, int max_denoising_steps,
                              unsigned seed, bool verbose,
                              const DiffStreamCallback& on_step = nullptr);

    const DiffConfig& config() const { return cfg_; }
    const DiffPerfStats& stats() const { return stats_; }   // from the last generate()

    // KV cache footprint (allocated for max_seq_len at construction).
    size_t kv_cache_bytes()           const { return enc_kv_.total_bytes(); }
    size_t kv_cache_bytes_per_token() const { return enc_kv_.bytes_per_token(); }
    int    kv_cache_max_seq()         const { return enc_kv_.max_seq(); }

    // Activation arena capacity retained across all GPUs for liveness-scoped scratch.
    // Each arena is pre-sized from the planner peak-live estimate and grows only on overflow.
    size_t scratch_bytes() const;

private:
    // Encoder pass over `ids` at absolute offset `past_len`; fills enc_kv_.
    // Splits the prompt into prefill chunks (DIFF_PREFILL_CHUNK) to bound the
    // activation arena high-water mark; encode_block does one chunk.
    void encode(const std::vector<int>& ids, int past_len);
    void encode_block(const std::vector<int>& ids, int past_len);

    // One denoising step: decode the canvas, produce per-position argmax /
    // entropy / multinomial sample, and the self-conditioning signal for the
    // next step (left in `soft_next`).
    void decode_step(const std::vector<int>& canvas_ids,
                     const bf16* soft_or_null, int enc_len, float temp,
                     std::vector<int>& argmax,
                     std::vector<float>& entropy,
                     std::vector<int>& denoiser,
                     GpuBuffer<bf16>& soft_next,
                     std::mt19937& rng);

    DiffConfig    cfg_;
    DiffPerfStats stats_;
    int         split_layer_ = 0;
    float       embed_scale_ = 1.0f;
    DiffWeights w_;
    DiffKvCache enc_kv_;
};

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

// Hot-path sampler-kernel AB knobs (defined in model.cpp; default off -> the
// original device kernels).  See device_sampler.hpp / fusions/logits.hpp.
bool diff_use_online_softmax();  // DIFF_ONLINE_SOFTMAX
bool diff_use_gumbel_sample();   // DIFF_GUMBEL_MAX (implies online softmax)
bool diff_use_stop_fix();        // DIFF_STOP_FIX

class DiffusionGemmaModel {
public:
    DiffusionGemmaModel(const std::string& model_dir, int max_seq_len, DiffPlacementOptions placement = {}, bool print_placement = true);

    // Block-diffusion generation. Returns generated token ids (prompt excluded).
    std::vector<int> generate(const std::vector<int>& prompt_ids,
                              int max_new_tokens, int max_denoising_steps,
                              unsigned seed, bool verbose,
                              const DiffStreamCallback& on_step = nullptr,
                              bool ignore_eos = false);

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
    // entropy / multinomial sample, and (when `want_soft_next`) the
    // self-conditioning signal for the next step (left in `soft_next`).  The
    // final scheduled step has no successor, so its `soft_next` is pure waste —
    // callers may pass want_soft_next=false to skip it (and, in exact mode, the
    // vocab-wide probs@embed GEMM that produces it).
    void decode_step(const std::vector<int>& canvas_ids,
                     const bf16* soft_or_null, int enc_len, float temp,
                     std::vector<int>& argmax,
                     std::vector<float>& entropy,
                     std::vector<int>& denoiser,
                     GpuBuffer<bf16>& soft_next,
                     std::mt19937& rng,
                     bool want_soft_next = true,
                     uint64_t rng_seed = 0, uint32_t rng_block = 0, uint32_t rng_step = 0);

    // Device-only denoiser forward (no host round-trip): embed+selfcond,
    // decoder layers, final norm, LM head, fused_logits_head, and the
    // self-conditioning (soft_next) build.  `ids` are the canvas tokens on the
    // device; `u_dev` (seq floats) must already be filled with sampling
    // variates.  Writes argmax / entropy / denoiser to the given device
    // pointers (no D2H).  All heavyweight scratch is arena-scoped internally.
    // On the device-resident denoising loop (default) this is called every
    // step without a host sync; the in-order queue serializes it against the
    // device sampler/stopping kernels that follow.
    void decode_forward(const int32_t* ids, const bf16* soft_or_null,
                        int enc_len, int seq, float temp, const float* u_dev,
                        int32_t* argmax_dev, float* entropy_dev,
                        int32_t* denoiser_dev, GpuBuffer<bf16>& soft_next,
                        bool want_soft_next,
                        uint64_t rng_seed = 0, uint32_t rng_block = 0, uint32_t rng_step = 0);

    // Allocate (or grow) the persistent device buffers backing the
    // device-resident denoising loop, sized for one canvas of `seq` tokens.
    // Idempotent: no-ops when already large enough.
    void ensure_device_buffers(int seq);

    DiffConfig    cfg_;
    DiffPerfStats stats_;
    int         split_layer_ = 0;
    float       embed_scale_ = 1.0f;
    DiffWeights w_;
    DiffKvCache enc_kv_;

    // Persistent device buffers for the device-resident denoising loop (default
    // path; the host-sampler path under DIFF_HOST_SAMPLER does not use these).
    // Allocated once by ensure_device_buffers() and reused across blocks/steps.
    GpuBuffer<int32_t> canvas_dev_;       // input canvas, renoised in place
    GpuBuffer<int32_t> argmax_dev_;       // argmax of logits (committed output)
    GpuBuffer<int32_t> denoiser_dev_;     // multinomial sample (accepted into canvas)
    GpuBuffer<int32_t> argmax_history_dev_; // stability check: (stability_threshold, canvas) ring buffer
    GpuBuffer<float>   entropy_dev_;      // per-position entropy (nats)
    GpuBuffer<float>   u_dev_;            // per-step sampling uniforms
    GpuBuffer<float>   mean_dev_;         // scalar mean-entropy (device stopping)
    GpuBuffer<int32_t> stop_dev_;         // scalar stop flag (device stopping)
    GpuBuffer<char>    accepted_dev_;     // accept mask (callback payload)
    int                dev_buf_seq_ = 0;  // current allocated canvas size
};

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

// ---------------------------------------------------------------------------
// Structured decision reads (the /v1/systemone backend).  One read = one
// denoiser pass over a compiled answer canvas whose single answer slot is
// filled with fresh uniform noise; the slot row is scored at temperature 1
// against the full vocabulary and the allowed label set.  These structs are
// pure token/position data — no HTTP or JSON types cross this boundary.
struct CompiledDecisionTemplate {
    std::vector<int32_t> canvas;      // full canvas incl. closing token + padding
    int                  slot_position = -1;   // canvas row holding the answer label
    std::vector<int32_t> label_ids;    // allowed label token ids at the slot (2..256)
};

struct DecisionReadOptions {
    int      reads = 1;                 // fixed read count (1..32)
    bool     auto_mode = false;         // escalate to auto_max while uncertain
    int      auto_max = 4;              // 1..32
    float    auto_entropy_threshold = 0.5f;  // normalized label entropy in [0,1]
    float    auto_min_label_mass = 0.0f;     // full-vocab label mass floor in [0,1]
    uint64_t stream_seed = 0;           // per-question noise stream (content-derived)
    // GPU work watchdog timeout in seconds. The value 0 turns the watchdog off.
    // If a read makes no GPU progress during the timeout, the process exits.
    // In this way, the model lock is released.
    float    watchdog_s = 0.0f;
};

// One independent read of one answer slot.  logprobs are full-vocabulary
// normalized at temperature 1 (final softcap applied exactly once, FP32).
struct DecisionSlotSample {
    std::vector<float> label_logprobs;  // log p(label_j), one per label_ids entry
    float   log_label_mass = 0.0f;      // logsumexp over the label set
    float   vocab_entropy = 0.0f;       // full-vocabulary entropy (nats)
    int32_t argmax_id = -1;             // global argmax token at the slot row
    float   argmax_logprob = 0.0f;      // its full-vocabulary logprob
};

struct DecisionReadResult {
    std::vector<DecisionSlotSample> reads;
    int    prompt_tokens = 0;           // encoded prompt length (prefilled once)
    int    prefill_calls = 0;           // encoder passes issued (1 per question)
    int    decode_calls  = 0;           // denoiser passes issued (= reads.size())
    double prefill_s = 0.0;
    double decode_s  = 0.0;
};

// Device-side scoring outputs for the structured decode mode (see
// DiffusionGemmaModel::decode_forward).  All pointers are device memory.
struct DecisionScoreTarget {
    const int32_t* rows = nullptr;            // (num_rows) canvas rows to score
    const int32_t* label_ids = nullptr;       // (num_labels)
    int     num_rows = 0;
    int     num_labels = 0;
    float*  label_logprobs = nullptr;         // (num_rows, num_labels) out
    float*  log_label_mass = nullptr;         // (num_rows) out
    float*  vocab_entropy = nullptr;          // (num_rows) out
    int32_t* argmax_id = nullptr;             // (num_rows) out
    float*  argmax_logprob = nullptr;         // (num_rows) out
    int32_t* nonfinite = nullptr;             // (num_rows) out: 1 if raw logit NaN/inf
};

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

    // Structured decision read: prefill `prompt_ids` once, then run
    // `options.reads` (or auto-escalated) independent single-step denoiser
    // passes over `compiled.canvas`, replacing only the answer slot with a
    // fresh rejection-sampled uniform vocab id per read.  Returns the per-read
    // slot scores.  The prompt KV is reset on entry and on exit (including
    // error paths), so no state crosses questions or requests.
    DecisionReadResult read_decisions(const std::vector<int>& prompt_ids,
                                      const CompiledDecisionTemplate& compiled,
                                      const DecisionReadOptions& options);

    // KV cache footprint (allocated for max_seq_len at construction).
    size_t kv_cache_bytes()           const { return enc_kv_.total_bytes(); }
    size_t kv_cache_bytes_per_token() const { return enc_kv_.bytes_per_token(); }
    int    kv_cache_max_seq()         const { return enc_kv_.max_seq(); }

    // Largest canvas width the execution/allocation paths are planned for
    // (the activation arena's decode graph is sized from cfg_.canvas_length).
    // Structured reads must not exceed this without re-planning the arena.
    int    canvas_capacity()          const { return cfg_.canvas_length; }

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
    //
    // When `score_target` is non-null the forward runs in structured-read
    // mode: the whole-layer NVFP4 graph capture is bypassed (its key lacks
    // request dimensions and buffer identities), no sampler/entropy/soft_next
    // work runs, and after the LM head only the rows in `score_target` are
    // scored (full-vocabulary softcap softmax statistics + per-label logprobs
    // at temperature 1).  `temp` and the sampling outputs are ignored.
    void decode_forward(const int32_t* ids, const bf16* soft_or_null,
                        int enc_len, int seq, float temp, const float* u_dev,
                        int32_t* argmax_dev, float* entropy_dev,
                        int32_t* denoiser_dev, GpuBuffer<bf16>& soft_next,
                        bool want_soft_next,
                        uint64_t rng_seed = 0, uint32_t rng_block = 0, uint32_t rng_step = 0,
                        const DecisionScoreTarget* score_target = nullptr);

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

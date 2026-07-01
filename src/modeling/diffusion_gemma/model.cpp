#include "activation_plan.hpp"
#include "model.hpp"
#include "layer.hpp"
#include "loader.hpp"
#include "self_conditioning.hpp"
#include "fusions/logits.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/kernels/embedding.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// DIFF_PERSIST_XFER_STAGE: reuse the host staging buffer across cross-GPU hops
// instead of allocating a fresh std::vector on each call (the split decode path
// hops twice per step).  Default off; identical bytes either way.
static bool persist_xfer_stage_enabled() {
    static bool enabled = [] {
        const char* e = std::getenv("DIFF_PERSIST_XFER_STAGE");
        return e && std::strcmp(e, "0") && std::strcmp(e, "off") &&
               std::strcmp(e, "false") && std::strcmp(e, "no");
    }();
    return enabled;
}

// Cross-GPU copy via host staging (devices have no P2P here).
static void transfer(sycl::queue& src_q, const bf16* src,
                     sycl::queue& dst_q, bf16* dst, size_t n) {
    if (persist_xfer_stage_enabled()) {
        static thread_local std::vector<bf16> stage;
        if (stage.size() < n) stage.resize(n);
        src_q.memcpy(stage.data(), src, n * sizeof(bf16)).wait();
        dst_q.memcpy(dst, stage.data(), n * sizeof(bf16)).wait();
        return;
    }
    std::vector<bf16> stage(n);
    src_q.memcpy(stage.data(), src, n * sizeof(bf16)).wait();
    dst_q.memcpy(dst, stage.data(), n * sizeof(bf16)).wait();
}

// Self-conditioning ("soft_next") signal modes — a per-step hot-path knob for
// the next-step conditioning embedding, selected by DIFF_SOFT_NEXT:
//   exact (default)  soft_next = probs(seq,V) @ embed(V,H).  Faithful, but a
//                    vocab-wide GEMM as expensive as the LM head, plus a seq*V
//                    probs buffer.
//   hard | argmax    gather the argmax token's embedding (seq,H).  Cheapest;
//                    drops both the probs buffer and the GEMM.
//   topk[:K]         top-K weighted embedding (default K=8).  Accuracy-
//                    preserving middle ground: k*H per row, no probs buffer.
// DIFF_Q8_SOFT_NEXT is honored for back-compat with the original Q8-only flag.
enum class SoftNextMode { Exact, Hard, TopK };
struct SoftNextCfg { SoftNextMode mode; int k; };

static SoftNextCfg soft_next_cfg() {
    static SoftNextCfg cfg = [] {
        auto parse = [](const char* e, SoftNextCfg& out) -> bool {
            if (!e) return false;
            if (!std::strcmp(e, "argmax") || !std::strcmp(e, "hard") ||
                !std::strcmp(e, "1") || !std::strcmp(e, "on")) {
                out = {SoftNextMode::Hard, 1}; return true;
            }
            if (!std::strcmp(e, "exact") || !std::strcmp(e, "full") ||
                !std::strcmp(e, "0") || !std::strcmp(e, "off")) {
                out = {SoftNextMode::Exact, 0}; return true;
            }
            if (!std::strncmp(e, "topk", 4)) {
                int k = 8;
                const char* colon = std::strchr(e, ':');
                if (colon) k = std::atoi(colon + 1);
                k = std::max(1, std::min(k, 64));   // bound local-memory top-k storage
                out = {SoftNextMode::TopK, k}; return true;
            }
            return false;
        };
        SoftNextCfg out{SoftNextMode::Exact, 0};
        if (parse(std::getenv("DIFF_SOFT_NEXT"), out) ||
            parse(std::getenv("DIFF_Q8_SOFT_NEXT"), out))
            return out;
        return SoftNextCfg{SoftNextMode::Exact, 0};
    }();
    return cfg;
}

DiffusionGemmaModel::DiffusionGemmaModel(const std::string& model_dir, int max_seq_len, DiffPlacementOptions placement, bool print_placement) {
    cfg_ = DiffConfig::from_dir(model_dir);
    int L = cfg_.text.num_hidden_layers;

    DiffPlacementOptions resolved_placement = resolve_diffusion_placement(cfg_, placement);
    split_layer_ = resolve_diffusion_split_layer(cfg_, resolved_placement);
    DiffExpertPlacementMode resolved_experts = resolve_expert_placement(resolved_placement.expert_mode);
    std::printf("[model] %d GPU(s); %d layers, split at %d; experts=%s\n",
                GpuEngine::count(), L, split_layer_, expert_placement_name(resolved_experts));
    if (print_placement) print_diffusion_placement(cfg_, split_layer_, resolved_placement);

    embed_scale_ = bf16_to_float(float_to_bf16(std::sqrt((float)cfg_.text.hidden_size)));

    w_      = load_diffusion_weights(model_dir, cfg_, split_layer_, resolved_placement.expert_mode);
    enc_kv_ = DiffKvCache(cfg_, max_seq_len, split_layer_);
    reserve_activation_arenas(cfg_, max_seq_len);
}

size_t DiffusionGemmaModel::scratch_bytes() const {
    return diffarena::total_bytes();
}

// ---------------------------------------------------------------------------
// Chunked prefill: the encoder only populates the KV cache (no cross-chunk
// hidden-state dependency — each position's KV depends only on already-cached
// earlier positions, with absolute RoPE), so processing the prompt in chunks is
// bit-identical to one pass but caps the arena high-water mark at chunk size
// instead of prompt length.  The arena is pre-sized from the same chunked
// planner shape, so long prompts do not force activation storage to scale with
// the whole prompt.  Tunable via DIFF_PREFILL_CHUNK (default 2048; <=0 disables).
void DiffusionGemmaModel::encode(const std::vector<int>& ids, int past_len) {
    static const int chunk = [] {
        const char* e = std::getenv("DIFF_PREFILL_CHUNK");
        return e ? std::atoi(e) : 2048;
    }();
    int total = (int)ids.size();
    if (chunk <= 0 || total <= chunk) { encode_block(ids, past_len); return; }
    for (int off = 0; off < total; off += chunk) {
        int cs = std::min(chunk, total - off);
        encode_block(std::vector<int>(ids.begin() + off, ids.begin() + off + cs),
                     past_len + off);
    }
}

void DiffusionGemmaModel::encode_block(const std::vector<int>& ids, int past_len) {
    int seq = (int)ids.size();
    int H = cfg_.text.hidden_size;
    int L = cfg_.text.num_hidden_layers;
    auto& ctx0 = GpuEngine::get(0);
    auto& q0 = ctx0.queue;

    GpuBuffer<int32_t> ids_dev(seq, q0);
    { std::vector<int32_t> tmp(ids.begin(), ids.end()); ids_dev.upload(tmp.data(), seq); }

    auto hidden = diffarena::arena(ctx0.index).alloc<bf16>((size_t)seq * H);
    if (!w_.embed_tokens_q8.empty())
        embedding_lookup_q8_0(q0, w_.embed_tokens_q8, ids_dev.data(), hidden.data(),
                              seq, H, embed_scale_);
    else
        embedding_lookup(q0, w_.embed_tokens.data(), ids_dev.data(), hidden.data(),
                         seq, H, embed_scale_);

    for (int l = 0; l < split_layer_; ++l)
        diff_layer_forward(ctx0, w_.layers[l], hidden.data(), enc_kv_.layer(l),
                           seq, past_len, cfg_.text, /*is_encoder=*/true);

    if (split_layer_ < L) {
        auto& ctx1 = GpuEngine::get(1);
        auto& q1 = ctx1.queue;
        auto hidden1 = diffarena::arena(ctx1.index).alloc<bf16>((size_t)seq * H);
        transfer(q0, hidden.data(), q1, hidden1.data(), (size_t)seq * H);
        for (int l = split_layer_; l < L; ++l)
            diff_layer_forward(ctx1, w_.layers[l], hidden1.data(), enc_kv_.layer(l),
                               seq, past_len, cfg_.text, /*is_encoder=*/true);
        q1.wait();
    }
    q0.wait();
}

// ---------------------------------------------------------------------------
// Persistent device buffers for the device-resident denoising loop.  Sized for
// one canvas (seq <= canvas_length); reallocated only if the canvas grows.
void DiffusionGemmaModel::ensure_device_buffers(int seq) {
    if (dev_buf_seq_ >= seq && !canvas_dev_.empty()) return;
    auto& q0 = GpuEngine::get(0).queue;
    canvas_dev_      = GpuBuffer<int32_t>(seq, q0);
    argmax_dev_      = GpuBuffer<int32_t>(seq, q0);
    denoiser_dev_    = GpuBuffer<int32_t>(seq, q0);
    int hist_slots = std::max(1, cfg_.gen.stability_threshold);
    argmax_history_dev_ = GpuBuffer<int32_t>((size_t)hist_slots * seq, q0);
    entropy_dev_     = GpuBuffer<float>(seq, q0);
    u_dev_           = GpuBuffer<float>(seq, q0);
    mean_dev_        = GpuBuffer<float>(1, q0);
    stop_dev_        = GpuBuffer<int32_t>(1, q0);
    accepted_dev_    = GpuBuffer<char>(seq, q0);
    dev_buf_seq_ = seq;
}

// Device-only denoiser forward: the embed→layers→LM-head→sample→soft_next
// pipeline with no host round-trip.  Extracted from the original decode_step;
// the host-sampler path (decode_step below) wraps this and downloads the
// per-step outputs.  Does not call q0.wait() — callers sync via download()
// (host path) or rely on the in-order queue (device-resident path).
void DiffusionGemmaModel::decode_forward(
    const int32_t* ids, const bf16* soft_or_null, int enc_len, int seq,
    float temp, const float* u_dev,
    int32_t* argmax_dev, float* entropy_dev, int32_t* denoiser_dev,
    GpuBuffer<bf16>& soft_next, bool want_soft_next)
{
    int H = cfg_.text.hidden_size;
    int L = cfg_.text.num_hidden_layers;
    int V = cfg_.text.vocab_size;
    auto& ctx0 = GpuEngine::get(0);
    auto& q0 = ctx0.queue;

    // Embed canvas + self-conditioning.
    auto& ar0 = diffarena::arena(ctx0.index);
    auto hidden = ar0.alloc<bf16>((size_t)seq * H);
    { DIFF_PROF(q0, "embed+selfcond");
      if (!w_.embed_tokens_q8.empty())
          embedding_lookup_q8_0(q0, w_.embed_tokens_q8, ids, hidden.data(),
                                seq, H, embed_scale_);
      else
          embedding_lookup(q0, w_.embed_tokens.data(), ids, hidden.data(),
                           seq, H, embed_scale_);
      self_conditioning_forward(ctx0, w_.self_cond, hidden.data(), soft_or_null,
                                seq, H, cfg_.text.intermediate_size, cfg_.text.rms_norm_eps); }

    // Decoder layers (bidirectional, read-only encoder KV).
    for (int l = 0; l < split_layer_; ++l)
        diff_layer_forward(ctx0, w_.layers[l], hidden.data(), enc_kv_.layer(l),
                           seq, enc_len, cfg_.text, /*is_encoder=*/false);

    if (split_layer_ < L) {
        auto& ctx1 = GpuEngine::get(1);
        auto& q1 = ctx1.queue;
        auto hidden1 = diffarena::arena(ctx1.index).alloc<bf16>((size_t)seq * H);
        { DIFF_PROF(q0, "xfer"); transfer(q0, hidden.data(), q1, hidden1.data(), (size_t)seq * H); }
        for (int l = split_layer_; l < L; ++l)
            diff_layer_forward(ctx1, w_.layers[l], hidden1.data(), enc_kv_.layer(l),
                               seq, enc_len, cfg_.text, /*is_encoder=*/false);
        q1.wait();
        { DIFF_PROF(q0, "xfer"); transfer(q1, hidden1.data(), q0, hidden.data(), (size_t)seq * H); }
    }

    // Final norm + LM head (all canvas positions).
    { DIFF_PROF(q0, "lm.final_norm");
      rms_norm(q0, hidden.data(), w_.final_norm.data(), hidden.data(), seq, H, cfg_.text.rms_norm_eps); }

    auto logits_bf16 = ar0.alloc<bf16>((size_t)seq * V);
    { DIFF_PROF(q0, "lm.logits");
      if (!w_.embed_tokens_q8.empty())
          matmul_q8_0(hidden.data(), seq, H, w_.embed_tokens_q8, logits_bf16.data(), ctx0);
      else
          matmul_bf16(hidden.data(), seq, H, w_.embed_tokens.data(), V, logits_bf16.data(), ctx0); }
    hidden.reset();

    // F1: softcap + temperature + softmax + argmax + entropy + multinomial
    // sample in one kernel.  BF16 probabilities are only materialized when the
    // exact self-conditioning matmul needs them (see SoftNextMode).
    SoftNextCfg sn = soft_next_cfg();
    bool need_probs = want_soft_next && (sn.mode == SoftNextMode::Exact);
    diffarena::Alloc<bf16> probs_bf16;
    if (need_probs) probs_bf16 = ar0.alloc<bf16>((size_t)seq * V);
    { DIFF_PROF(q0, "lm.softmax_sample");
      fused_logits_head(q0, logits_bf16.data(),
                        cfg_.text.final_logit_softcapping, 1.0f / temp,
                        u_dev, need_probs ? probs_bf16.data() : nullptr,
                        argmax_dev, entropy_dev, denoiser_dev, seq, V); }

    // TopK soft_next reads the same logits for its selection — extract before
    // the logits buffer is released.
    GpuBuffer<int32_t> topk_idx_dev;
    GpuBuffer<float>   topk_w_dev;
    if (want_soft_next && sn.mode == SoftNextMode::TopK) {
        topk_idx_dev = GpuBuffer<int32_t>((size_t)seq * sn.k, q0);
        topk_w_dev   = GpuBuffer<float>((size_t)seq * sn.k, q0);
        DIFF_PROF(q0, "lm.topk_select");
        topk_soft_select(q0, logits_bf16.data(),
                         cfg_.text.final_logit_softcapping, 1.0f / temp,
                         sn.k, topk_idx_dev.data(), topk_w_dev.data(), seq, V);
    }
    logits_bf16.reset();

    // soft_next: next-step self-conditioning signal (embed_scale_ folded in).
    // Skipped on the final step, whose signal would never be consumed.
    if (!want_soft_next) { soft_next = GpuBuffer<bf16>(); return; }
    soft_next = GpuBuffer<bf16>((size_t)seq * H, q0);
    { DIFF_PROF(q0, "lm.soft_next");
      switch (sn.mode) {
      case SoftNextMode::Hard:
          // Gather the argmax token embedding.
          if (!w_.embed_tokens_q8.empty())
              embedding_lookup_q8_0(q0, w_.embed_tokens_q8, argmax_dev,
                                    soft_next.data(), seq, H, embed_scale_);
          else
              embedding_lookup(q0, w_.embed_tokens.data(), argmax_dev,
                               soft_next.data(), seq, H, embed_scale_);
          break;
      case SoftNextMode::TopK:
          // Top-k weighted embedding gather.
          if (!w_.embed_tokens_q8.empty())
              weighted_embed_gather_q8_0(q0, w_.embed_tokens_q8, topk_idx_dev.data(),
                                          topk_w_dev.data(), soft_next.data(),
                                          seq, sn.k, H, embed_scale_);
          else
              weighted_embed_gather(q0, w_.embed_tokens.data(), topk_idx_dev.data(),
                                    topk_w_dev.data(), soft_next.data(),
                                    seq, sn.k, H, embed_scale_);
          break;
      case SoftNextMode::Exact:
          // soft_next = probs @ embed, then * embed_scale.
          if (!w_.embed_tokens_q8.empty())
              matmul_q8_0_nn(probs_bf16.data(), seq, V, w_.embed_tokens_q8, H,
                             soft_next.data(), ctx0);
          else
              matmul_bf16_nn(probs_bf16.data(), seq, V, w_.embed_tokens.data(), H,
                             soft_next.data(), ctx0);
          probs_bf16.reset();
          scale_inplace(q0, soft_next.data(), seq * H, embed_scale_);
          break;
      } }
}

// Host-sampler wrapper: the original per-step decode path (kept under
// DIFF_HOST_SAMPLER for A/B).  Uploads the canvas + host-drawn uniforms, runs
// the device-only decode_forward, then downloads argmax / entropy / denoiser
// for the host entropy-bound sampler and host stopping check.  download() syncs
// the in-order queue, so soft_next is also complete on return.
void DiffusionGemmaModel::decode_step(
    const std::vector<int>& canvas_ids, const bf16* soft_or_null, int enc_len,
    float temp, std::vector<int>& argmax, std::vector<float>& entropy,
    std::vector<int>& denoiser, GpuBuffer<bf16>& soft_next, std::mt19937& rng,
    bool want_soft_next)
{
    int seq = (int)canvas_ids.size();
    auto& q0 = GpuEngine::get(0).queue;

    GpuBuffer<int32_t> ids_dev(seq, q0);
    { std::vector<int32_t> tmp(canvas_ids.begin(), canvas_ids.end()); ids_dev.upload(tmp.data(), seq); }
    GpuBuffer<int32_t> amax_dev(seq, q0), deno_dev(seq, q0);
    GpuBuffer<float>   ent_dev(seq, q0);
    std::vector<float> u(seq);
    { std::uniform_real_distribution<float> d(0.0f, 1.0f); for (auto& x : u) x = d(rng); }
    GpuBuffer<float> u_dev(seq, q0); u_dev.upload(u.data(), seq);

    decode_forward(ids_dev.data(), soft_or_null, enc_len, seq, temp, u_dev.data(),
                   amax_dev.data(), ent_dev.data(), deno_dev.data(),
                   soft_next, want_soft_next);

    argmax.resize(seq); entropy.resize(seq); denoiser.resize(seq);
    { DIFF_PROF(q0, "lm.download");
      std::vector<int32_t> a(seq), d(seq);
      amax_dev.download(a.data(), seq); deno_dev.download(d.data(), seq);
      ent_dev.download(entropy.data(), seq);
      for (int i = 0; i < seq; ++i) { argmax[i] = a[i]; denoiser[i] = d[i]; } }
}

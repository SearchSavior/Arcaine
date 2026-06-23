#include "activation_plan.hpp"
#include "model.hpp"
#include "layer.hpp"
#include "loader.hpp"
#include "self_conditioning.hpp"
#include "fusions/logits.hpp"
#include "../common/gpu/engine.hpp"
#include "../common/kernels/embedding.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Cross-GPU copy via host staging (devices have no P2P here).
static void transfer(sycl::queue& src_q, const bf16* src,
                     sycl::queue& dst_q, bf16* dst, size_t n) {
    std::vector<bf16> stage(n);
    src_q.memcpy(stage.data(), src, n * sizeof(bf16)).wait();
    dst_q.memcpy(dst, stage.data(), n * sizeof(bf16)).wait();
}

static bool q8_argmax_soft_next_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_Q8_SOFT_NEXT");
        return env && (!std::strcmp(env, "argmax") || !std::strcmp(env, "hard") ||
                       !std::strcmp(env, "1") || !std::strcmp(env, "on"));
    }();
    return enabled;
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
void DiffusionGemmaModel::decode_step(
    const std::vector<int>& canvas_ids, const bf16* soft_or_null, int enc_len,
    float temp, std::vector<int>& argmax, std::vector<float>& entropy,
    std::vector<int>& denoiser, GpuBuffer<bf16>& soft_next, std::mt19937& rng)
{
    int seq = (int)canvas_ids.size();
    int H = cfg_.text.hidden_size;
    int L = cfg_.text.num_hidden_layers;
    int V = cfg_.text.vocab_size;
    auto& ctx0 = GpuEngine::get(0);
    auto& q0 = ctx0.queue;

    // Embed canvas + self-conditioning.
    GpuBuffer<int32_t> ids_dev(seq, q0);
    { std::vector<int32_t> tmp(canvas_ids.begin(), canvas_ids.end()); ids_dev.upload(tmp.data(), seq); }
    auto& ar0 = diffarena::arena(ctx0.index);
    auto hidden = ar0.alloc<bf16>((size_t)seq * H);
    { DIFF_PROF(q0, "embed+selfcond");
      if (!w_.embed_tokens_q8.empty())
          embedding_lookup_q8_0(q0, w_.embed_tokens_q8, ids_dev.data(), hidden.data(),
                                seq, H, embed_scale_);
      else
          embedding_lookup(q0, w_.embed_tokens.data(), ids_dev.data(), hidden.data(),
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

    // F1: softcap + temperature + softmax + argmax + entropy +
    // multinomial sample in one kernel. Only BF16 probabilities are
    // materialized for the self-conditioning matmul.
    bool q8_argmax_soft_next = !w_.embed_tokens_q8.empty() && q8_argmax_soft_next_enabled();
    diffarena::Alloc<bf16> probs_bf16;
    if (!q8_argmax_soft_next) probs_bf16 = ar0.alloc<bf16>((size_t)seq * V);
    GpuBuffer<int32_t> amax_dev(seq, q0), deno_dev(seq, q0);
    GpuBuffer<float>   ent_dev(seq, q0);
    std::vector<float> u(seq);
    { std::uniform_real_distribution<float> d(0.0f, 1.0f); for (auto& x : u) x = d(rng); }
    GpuBuffer<float> u_dev(seq, q0); u_dev.upload(u.data(), seq);
    { DIFF_PROF(q0, "lm.softmax_sample");
      fused_logits_head(q0, logits_bf16.data(),
                        cfg_.text.final_logit_softcapping, 1.0f / temp,
                        u_dev.data(), q8_argmax_soft_next ? nullptr : probs_bf16.data(),
                        amax_dev.data(), ent_dev.data(), deno_dev.data(), seq, V); }
    logits_bf16.reset();

    argmax.resize(seq); entropy.resize(seq); denoiser.resize(seq);
    { DIFF_PROF(q0, "lm.download");
      std::vector<int32_t> a(seq), d(seq);
      amax_dev.download(a.data(), seq); deno_dev.download(d.data(), seq);
      ent_dev.download(entropy.data(), seq);
      for (int i = 0; i < seq; ++i) { argmax[i] = a[i]; denoiser[i] = d[i]; } }
    // soft_next = probs @ embed * embed_scale   (next-step signal).
    soft_next = GpuBuffer<bf16>((size_t)seq * H, q0);
    { DIFF_PROF(q0, "lm.soft_next");
      if (q8_argmax_soft_next) {
          embedding_lookup_q8_0(q0, w_.embed_tokens_q8, amax_dev.data(),
                                soft_next.data(), seq, H, embed_scale_);
      } else if (!w_.embed_tokens_q8.empty()) {
          matmul_q8_0_nn(probs_bf16.data(), seq, V, w_.embed_tokens_q8, H,
                         soft_next.data(), ctx0);
      } else {
          matmul_bf16_nn(probs_bf16.data(), seq, V, w_.embed_tokens.data(), H, soft_next.data(), ctx0);
      }
      probs_bf16.reset();
      if (!q8_argmax_soft_next)
          scale_inplace(q0, soft_next.data(), seq * H, embed_scale_); }
    q0.wait();
}

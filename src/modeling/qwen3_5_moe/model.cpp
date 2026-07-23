#include "model.hpp"
#include "loader.hpp"
#include "kernels.hpp"
#include "moe.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/io/quant_loader.hpp"
#include "../../common/kernels/embedding.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "../../common/preprocess/chat_template.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

QwenModel::QwenModel(const std::string& model_dir, int max_seq_len) {
    cfg_ = QwenConfig::from_dir(model_dir);

    int n_full = 0, n_linear = 0;
    for (bool b : cfg_.is_full_attention) (b ? n_full : n_linear) += 1;

    std::printf("[qwen] %d GPU(s) available; loading on GPU 0\n", GpuEngine::count());
    auto& q = GpuEngine::get(0).queue;
    std::printf("[qwen] device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    ShardedSafetensors sf(model_dir);
    // ARCAINE_QWEN_MAX_LAYERS (>=1): load only layers [0,N) — lets the registry
    // path and per-layer kernels be exercised against the real checkpoint
    // without the full ~22GB / 40-layer load. Omit (or <=0) to load all layers.
    int max_layers = -1;
    if (const char* env = std::getenv("ARCAINE_QWEN_MAX_LAYERS")) {
        int v = std::atoi(env);
        if (v > 0) {
            max_layers = v;
            std::printf("[qwen] ARCAINE_QWEN_MAX_LAYERS=%d -> loading only layers [0,%d)\n",
                        v, v);
        }
    }
    weights_ = load_qwen_weights(sf, cfg_, q, max_layers);

    // Single-GPU caches: full-attn KV (10 layers) + Gated DeltaNet SSM/conv
    // (30 layers). Allocated per cfg.num_hidden_layers; only the loaded subset
    // [0, weights_.layers.size()) is ever used.
    kv_cache_.init(cfg_, max_seq_len, q);
    linear_caches_.init(cfg_, q);

    info_.vocab_size      = cfg_.vocab_size;
    info_.max_seq_len     = max_seq_len;
    info_.bos_token_id    = cfg_.bos_token_id;
    info_.eos_token_ids   = cfg_.eos_token_ids;
    info_.suppress_tokens = cfg_.suppress_tokens;
    info_.temperature     = cfg_.temperature;
    info_.top_k           = cfg_.top_k;
    info_.top_p           = cfg_.top_p;
    info_.model_dir       = model_dir;
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "qwen3_5_moe_text: %d layers (%d full-attn / %d linear-attn), "
            "H=%d, vocab=%d, experts=%d (top-%d)",
            cfg_.num_hidden_layers, n_full, n_linear,
            cfg_.hidden_size, cfg_.vocab_size,
            cfg_.num_experts, cfg_.num_experts_per_tok);
        info_.description = buf;
    }
    std::printf("[qwen] %s\n", info_.description.c_str());
}

PreparedInput QwenModel::prepare_input(
    const std::string&              prompt,
    const std::vector<std::string>& /*image_paths*/,
    const std::vector<std::string>& /*audio_paths*/,
    const std::string&              /*vad_model*/
) {
    // Text-only model. Vision/audio paths are vestigial (the checkpoint ships
    // inside a multimodal container, but the text model has no visual/audio
    // tensors) and are ignored here.
    PreparedInput out;
    auto built = build_chat_prompt(cfg_.model_dir, prompt,
        /*image_token_counts=*/{}, /*audio_token_counts=*/{},
        /*add_generation_prompt=*/true, /*enable_thinking=*/false);
    out.tokens            = std::move(built.tokens);
    out.mm_token_type_ids = std::move(built.mm_token_type_ids);
    return out;
}

std::vector<float> QwenModel::forward(const ForwardInput& in) {
    // Text-only: ignore in.images / in.audio / in.mm_token_type_ids.
    return forward_tokens(in.token_ids, in.past_len);
}

std::vector<float> QwenModel::forward_tokens(
    const std::vector<int>& token_ids, int past_len) {
    auto& ctx = GpuEngine::get(0);
    auto& q   = ctx.queue;
    int seq   = (int)token_ids.size();
    int H     = cfg_.hidden_size;
    int L     = (int)weights_.layers.size();   // loaded layers (<= cfg_.num_hidden_layers)
    int V     = cfg_.vocab_size;

    // 0. Upload token ids (int32).
    GpuBuffer<int32_t> ids_dev((size_t)seq, q);
    {
        std::vector<int32_t> ids32(token_ids.begin(), token_ids.end());
        ids_dev.upload(ids32.data(), seq);
    }

    // 1. Embedding lookup — Qwen does NOT scale embeddings (scale=1, unlike Gemma).
    GpuBuffer<bf16> hidden((size_t)seq * H, q);
    embedding_lookup(q, weights_.embed_tokens.data(), ids_dev.data(),
                     hidden.data(), seq, H, /*scale=*/1.0f);

    // Per-layer scratch: normed hidden + attention/moe output (pre-residual).
    GpuBuffer<bf16> h_normed((size_t)seq * H, q);
    GpuBuffer<bf16> sub_out((size_t)seq * H, q);

    // 2. 40-layer dispatch. Reference decoder layer (lines 862-894):
    //    residual = h; h = input_layernorm(h); h = attn(h); h = residual + h;
    //    residual = h; h = post_attn_layernorm(h); h = moe(h); h = residual + h.
    // Norm weights are (1+w)-baked at load -> plain rms_norm.
    for (int l = 0; l < L; ++l) {
        auto& layer = weights_.layers[l];

        // attn sub-layer
        rms_norm(q, hidden.data(), layer.input_layernorm.data(),
                h_normed.data(), seq, H, cfg_.rms_norm_eps);
        if (layer.is_full_attention) {
            const auto& a = std::get<QwenFullAttn>(layer.attn);
            qwen_full_attention_forward(ctx, a, kv_cache_.layers[l],
                                        h_normed.data(), sub_out.data(),
                                        seq, past_len, cfg_);
        } else {
            const auto& a = std::get<QwenLinearAttn>(layer.attn);
            qwen_linear_attn_forward(ctx, a, linear_caches_.layers[l],
                                     h_normed.data(), sub_out.data(),
                                     seq, past_len, cfg_);
        }
        add_inplace(q, hidden.data(), sub_out.data(), (size_t)seq * H);

        // moe sub-layer
        rms_norm(q, hidden.data(), layer.post_attention_layernorm.data(),
                h_normed.data(), seq, H, cfg_.rms_norm_eps);
        qwen_moe_forward(ctx, layer.moe, h_normed.data(), sub_out.data(), seq, cfg_);
        add_inplace(q, hidden.data(), sub_out.data(), (size_t)seq * H);
    }

    // 3. Final RMSNorm on the LAST token only (we only need its logits).
    GpuBuffer<bf16> last((size_t)H, q);
    q.memcpy(last.data(), hidden.data() + (size_t)(seq - 1) * H,
             (size_t)H * sizeof(bf16)).wait();
    rms_norm(q, last.data(), weights_.final_norm.data(), last.data(),
             1, H, cfg_.rms_norm_eps);

    // 4. Untied lm_head (separate BF16 weight, NOT embed_tokens). No softcap.
    GpuBuffer<bf16> logits_bf16((size_t)V, q);
    matmul_bf16(last.data(), 1, H, weights_.lm_head.data(), V,
                logits_bf16.data(), ctx);

    // 5. bf16 -> f32 and download.
    GpuBuffer<float> logits_f32((size_t)V, q);
    bf16_to_f32(q, logits_bf16.data(), logits_f32.data(), V);
    std::vector<float> logits((size_t)V);
    logits_f32.download(logits.data(), V);
    return logits;
}

void QwenModel::reset_cache() {
    auto& q = GpuEngine::get(0).queue;
    kv_cache_.reset();            // resets KV `filled` counters (full-attn layers)
    linear_caches_.reset(q);      // zeros SSM + conv1d state (linear-attn layers)
}

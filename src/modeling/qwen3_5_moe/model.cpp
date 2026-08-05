#include <mutex>

#include "model.hpp"
#include "loader.hpp"
#include "kernels.hpp"
#include "moe.hpp"
#include "../../runtime/gpu/engine.hpp"
#include "../../runtime/gpu/ops.hpp"
#include "../../runtime/quantization/quant_loader.hpp"
#include "kernels/embedding.hpp"
#include "kernels/rms_norm.hpp"
#include "kernels/elementwise.hpp"

using namespace qwen35moe_kernels;
#include "../../preprocessing/chat_template.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

// Defined with the forward scratch below; used by the constructor preflight.
namespace qwen_forward_detail { int prefill_segment(); }

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

    // Preflight: fail fast with a clear breakdown when max_seq_len cannot fit
    // alongside the loaded weights, instead of stalling inside a USM alloc at
    // run time (Level Zero over-commit can hang rather than fail).
    // ARCAINE_QWEN_PREFLIGHT=0 disables the check.
    {
        const char* pf = std::getenv("ARCAINE_QWEN_PREFLIGHT");
        if (!pf || std::atoi(pf) != 0) {
            const int S      = max_seq_len;
            // Segmented prefill bounds all O(S) activation scratch at the
            // segment size; only the KV cache scales with max_seq.
            int seg = qwen_forward_detail::prefill_segment();
            const int Sa     = (seg > 0 && S > seg) ? seg : S;
            const int nq     = cfg_.num_attention_heads;
            const int nkv    = cfg_.num_key_value_heads;
            const int hd     = cfg_.head_dim;
            const int n_v    = cfg_.linear_num_value_heads;
            const int d_k    = cfg_.linear_key_head_dim;
            const int d_v    = cfg_.linear_value_head_dim;
            const int kdim   = cfg_.linear_num_key_heads * d_k;
            const int vdim   = n_v * d_v;
            const int cdim   = 2 * kdim + vdim;

            // Worst single forward covers the whole cache (prefill of max_seq).
            const size_t kv_bytes   = (size_t)n_full * S * nkv * hd * 2 * sizeof(bf16);
            const size_t gdn_state  = (size_t)n_linear *
                ((size_t)cdim * (cfg_.linear_conv_kernel_dim - 1) * sizeof(bf16) +
                 (size_t)n_v * d_k * d_v * sizeof(float));
            const size_t attn_scr   = qwen_attn_scratch_bytes(Sa, Sa, nq, nkv, hd, hd);
            const size_t gdn_scr    = qwen_gdn_detail::qwen_gdn_scratch_bytes(
                                          Sa, cdim, n_v, d_k, d_v, vdim);
            const size_t fwd_scr    = 3 * (size_t)Sa * cfg_.hidden_size * sizeof(bf16);
            // 2 GiB: driver/runtime reserve + oneDNN + small buffers not tracked.
            const size_t margin     = (size_t)2 << 30;
            const size_t needed     = kv_bytes + gdn_state + attn_scr + gdn_scr +
                                      fwd_scr + margin;

            const size_t total_mem  = q.get_device().get_info<
                sycl::info::device::global_mem_size>();
            const size_t live       = gpu_buffer_live_bytes().load();
            const size_t free_bytes = total_mem > live ? total_mem - live : 0;
            const double gib = 1.0 / (1 << 30);
            std::printf("[qwen] preflight max_seq=%d (segment=%d): need ~%.2f GiB "
                        "(KV %.2f + GDN state %.2f + attn scratch %.2f + GDN scratch %.2f "
                        "+ fwd %.2f + margin %.2f), est. device free %.2f GiB "
                        "(global %.2f - live %.2f)\n",
                        S, Sa, needed * gib, kv_bytes * gib, gdn_state * gib,
                        attn_scr * gib, gdn_scr * gib, fwd_scr * gib, margin * gib,
                        free_bytes * gib, total_mem * gib, live * gib);
            if (needed > free_bytes) {
                char msg[512];
                std::snprintf(msg, sizeof(msg),
                    "max_seq=%d does not fit: need ~%.1f GiB but only %.1f GiB free. "
                    "Lower --max-seq (see breakdown above; attn scratch shrinks with "
                    "QWEN35_ATTN_TILE). ARCAINE_QWEN_PREFLIGHT=0 skips this check.",
                    S, needed * gib, free_bytes * gib);
                throw std::runtime_error(msg);
            }
        }
    }

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

// Grow-only per-forward scratch. forward_tokens runs once per decode token, so
// per-call USM alloc/free would serialize the pipeline (DPC++ sycl::free
// syncs with the queue). The mutex also covers consumption of the attention
// scratch pointer inside qwen_full_attention_forward (locked there).
namespace qwen_forward_detail {
struct Scratch {
    GpuBuffer<int32_t> ids_dev;
    GpuBuffer<bf16>    hidden;
    GpuBuffer<bf16>    h_normed;
    GpuBuffer<bf16>    sub_out;
    GpuBuffer<bf16>    last;        // [H]
    GpuBuffer<bf16>    logits_bf16; // [V]
    GpuBuffer<float>   logits_f32;  // [V]
    size_t seq_cap = 0;
    size_t v_cap = 0;
};
inline Scratch& scratch()
{
    static Scratch s;
    return s;
}
inline std::mutex& scratch_mutex()
{
    static std::mutex m;
    return m;
}
// QWEN35_PREFILL_SEGMENT: max tokens per prefill segment (llama.cpp-style
// ubatching). Bounds all O(S) activation scratch regardless of total prompt
// length, so pp=112k fits on a 32 GB card. <=0 disables segmentation
// (single-shot prefill; A/B baseline). Default 8192.
inline int prefill_segment()
{
    static int cached = -2;
    if (cached == -2) {
        const char* v = std::getenv("QWEN35_PREFILL_SEGMENT");
        cached = v ? std::atoi(v) : 8192;
    }
    return cached;
}
} // namespace qwen_forward_detail

std::vector<float> QwenModel::forward_tokens(
    const std::vector<int>& token_ids, int past_len) {
    auto& ctx = GpuEngine::get(0);
    auto& q   = ctx.queue;
    int seq   = (int)token_ids.size();
    int H     = cfg_.hidden_size;
    int L     = (int)weights_.layers.size();   // loaded layers (<= cfg_.num_hidden_layers)
    int V     = cfg_.vocab_size;

    // Segmented prefill: cap the per-forward activation rows at the segment
    // size. Decode (seq==1) and prompts shorter than the segment are
    // unaffected. Segment boundaries are exact: full-attn appends KV at the
    // running past offset, GDN carries ssm_state + conv_state between calls.
    int seg = qwen_forward_detail::prefill_segment();
    int rows = (seg > 0 && seq > seg) ? seg : seq;

    std::lock_guard<std::mutex> fwd_lock(qwen_forward_detail::scratch_mutex());
    auto& sc = qwen_forward_detail::scratch();
    if (sc.seq_cap < (size_t)rows) {
        sc.ids_dev  = GpuBuffer<int32_t>((size_t)rows, q);
        sc.hidden   = GpuBuffer<bf16>((size_t)rows * H, q);
        sc.h_normed = GpuBuffer<bf16>((size_t)rows * H, q);
        sc.sub_out  = GpuBuffer<bf16>((size_t)rows * H, q);
        sc.seq_cap  = (size_t)rows;
    }
    if (sc.v_cap < (size_t)V) {
        sc.last        = GpuBuffer<bf16>((size_t)H, q);
        sc.logits_bf16 = GpuBuffer<bf16>((size_t)V, q);
        sc.logits_f32  = GpuBuffer<float>((size_t)V, q);
        sc.v_cap       = (size_t)V;
    }

    // 0. Upload token ids (int32) once; each segment uploads its slice.
    std::vector<int32_t> ids32(token_ids.begin(), token_ids.end());

    // Per-layer scratch: normed hidden + attention/moe output (pre-residual).
    bf16* hidden   = sc.hidden.data();
    bf16* h_normed = sc.h_normed.data();
    bf16* sub_out  = sc.sub_out.data();

    for (int base = 0; base < seq; base += rows) {
        const int n   = std::min(rows, seq - base);
        const int pos = past_len + base;

        sc.ids_dev.upload(ids32.data() + base, n);

        // 1. Embedding lookup — Qwen does NOT scale embeddings (scale=1,
        //    unlike Gemma).
        embedding_lookup(q, weights_.embed_tokens.data(), sc.ids_dev.data(),
                         sc.hidden.data(), n, H, /*scale=*/1.0f);

        // 2. 40-layer dispatch. Reference decoder layer (lines 862-894):
        //    residual = h; h = input_layernorm(h); h = attn(h); h = residual + h;
        //    residual = h; h = post_attn_layernorm(h); h = moe(h); h = residual + h.
        // Norm weights are (1+w)-baked at load -> plain rms_norm.
        for (int l = 0; l < L; ++l) {
            auto& layer = weights_.layers[l];

            // attn sub-layer
            rms_norm(q, hidden, layer.input_layernorm.data(),
                    h_normed, n, H, cfg_.rms_norm_eps);
            if (layer.is_full_attention) {
                const auto& a = std::get<QwenFullAttn>(layer.attn);
                qwen_full_attention_forward(ctx, a, kv_cache_.layers[l],
                                            h_normed, sub_out,
                                            n, pos, cfg_);
            } else {
                const auto& a = std::get<QwenLinearAttn>(layer.attn);
                qwen_linear_attn_forward(ctx, a, linear_caches_.layers[l],
                                         h_normed, sub_out,
                                         n, pos, cfg_);
            }
            add_inplace(q, hidden, sub_out, (size_t)n * H);

            // moe sub-layer
            rms_norm(q, hidden, layer.post_attention_layernorm.data(),
                    h_normed, n, H, cfg_.rms_norm_eps);
            qwen_moe_forward(ctx, layer.moe, h_normed, sub_out, n, cfg_);
            add_inplace(q, hidden, sub_out, (size_t)n * H);
        }
    }

    // The last segment left the final token's hidden state at this row.
    const int last_row = (seq % rows == 0) ? rows - 1 : (seq % rows) - 1;

    // 3. Final RMSNorm on the LAST token only (we only need its logits).
    //    D2D copy on the in-order queue: no wait, the norm is enqueued behind
    //    it and the single pipeline sync happens at the logits download.
    q.memcpy(sc.last.data(), hidden + (size_t)last_row * H,
             (size_t)H * sizeof(bf16));
    rms_norm(q, sc.last.data(), weights_.final_norm.data(), sc.last.data(),
             1, H, cfg_.rms_norm_eps);

    // 4. Untied lm_head (separate BF16 weight, NOT embed_tokens). No softcap.
    matmul_bf16(sc.last.data(), 1, H, weights_.lm_head.data(), V,
                sc.logits_bf16.data(), ctx);

    // 5. bf16 -> f32 and download (the one sync per forward).
    bf16_to_f32(q, sc.logits_bf16.data(), sc.logits_f32.data(), V);
    std::vector<float> logits((size_t)V);
    sc.logits_f32.download(logits.data(), V);

    // QWEN35_LOGIT_FP=<path>: dump the logit vector (f32 binary) for A/B.
    if (const char* fp = std::getenv("QWEN35_LOGIT_FP")) {
        double sum = 0, sumsq = 0;
        for (float x : logits) { sum += x; sumsq += (double)x * x; }
        int am = 0;
        for (int i = 1; i < V; ++i) if (logits[i] > logits[am]) am = i;
        std::printf("[logit-fp] S=%d past=%d sum=%.6f sumsq=%.6f argmax=%d max=%.6f\n",
                    seq, past_len, sum, sumsq, am, logits[am]);
        if (fp[0] == '/' && seq > 1) {  // prefill only; tg decode would overwrite
            if (FILE* f = std::fopen(fp, "wb")) {
                std::fwrite(logits.data(), sizeof(float), logits.size(), f);
                std::fclose(f);
            }
        }
    }
    return logits;
}

void QwenModel::reset_cache() {
    auto& q = GpuEngine::get(0).queue;
    kv_cache_.reset();            // resets KV `filled` counters (full-attn layers)
    linear_caches_.reset(q);      // zeros SSM + conv1d state (linear-attn layers)
}

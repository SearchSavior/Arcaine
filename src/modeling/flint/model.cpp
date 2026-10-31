#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>

#include <minja/chat-template.hpp>

#include "../../common/preprocess/chat_template.hpp"
#include "../../common/preprocess/tokenizer.hpp"
#include "kernels/kernels.hpp"
#include "kernels/prefill.hpp"
#include "kernels/prefill_xmx.hpp"
#include "kernels/awq_xmx.hpp"
#include "kernels/grouped_moe.hpp"
#include "gpu_prefill.hpp"

namespace {
// Mirrors the checkpoint template's flint_text() sanitization.
std::string sanitize_text(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (c != '\0') out.push_back(c);
    auto erase_all = [&](const std::string& needle) {
        for (size_t at = out.find(needle); at != std::string::npos;
             at = out.find(needle, at))
            out.erase(at, needle.size());
    };
    erase_all(" thinking");
    erase_all("</think>");
    static const char* tags[] = {
        "<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|developer|>",
        "<|user|>",          "<|assistant|>",   "<|observation|>"};
    for (const char* tag : tags) {
        const std::string needle(tag);
        const std::string escaped =
            "&lt;" + needle.substr(1, needle.size() - 2) + "&gt;";
        for (size_t at = out.find(needle); at != std::string::npos;
             at = out.find(needle, at + escaped.size()))
            out.replace(at, needle.size(), escaped);
    }
    return out;
}

template <typename T>
T* alloc(sycl::queue& q, size_t n, const char* what) {
    T* p = sycl::malloc_device<T>(n, q);
    if (!p) throw std::runtime_error(std::string("flint: alloc failed: ") + what);
    return p;
}
}  // namespace

FlintModel::FlintModel(const std::string& model_dir, int max_seq_len) {
    cfg_ = FlintConfig::from_dir(model_dir);
    max_seq_ = max_seq_len > 0 ? max_seq_len : 8192;
    log2base_ = std::log2(cfg_.rope_theta);

    q_ = sycl::queue(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                     sycl::property::queue::in_order{});
    std::printf("[flint] device: %s (backend: %s)\n",
                q_.get_device().get_info<sycl::info::device::name>().c_str(),
                q_.get_backend() == sycl::backend::ext_oneapi_level_zero
                    ? "level_zero"
                    : "other");

    int max_layers = -1;
    if (const char* env = std::getenv("ARCAINE_FLINT_MAX_LAYERS")) {
        const int v = std::atoi(env);
        if (v > 0) max_layers = v;
    }
    w_ = flint::load_flint_weights(cfg_, q_, max_layers);

    const int H = cfg_.hidden_size;
    const int QD = cfg_.q_dim();
    const int KV = cfg_.kv_dim();
    h_ = alloc<flint::bf16>(q_, H, "h");
    normed_ = alloc<flint::bf16>(q_, H, "normed");
    qbuf_ = alloc<flint::bf16>(q_, QD, "q");
    kbuf_ = alloc<flint::bf16>(q_, KV, "k");
    vbuf_ = alloc<flint::bf16>(q_, KV, "v");
    attn_ = alloc<flint::bf16>(q_, QD, "attn");
    act_ = alloc<flint::bf16>(q_, (size_t)7 * cfg_.shared_expert_intermediate_size, "act");
    down_ = alloc<flint::bf16>(q_, (size_t)7 * H, "down");
    gate_ = alloc<flint::bf16>(q_, cfg_.intermediate_size, "gate");
    up_ = alloc<flint::bf16>(q_, cfg_.intermediate_size, "up");
    logits_dev_ = alloc<float>(q_, cfg_.vocab_size, "logits");
    trace_enabled_ = std::getenv("ARCAINE_FLINT_TRACE") != nullptr;
    if (trace_enabled_)
        trace_ = alloc<flint::bf16>(q_, (size_t)(cfg_.num_hidden_layers + 1) * H, "trace");
    ids_dev_ = alloc<int>(q_, cfg_.num_experts_per_tok, "ids");
    weights_dev_ = alloc<float>(q_, cfg_.num_experts_per_tok, "route_weights");
    const int decode_parts=std::max({64,flint::gpu_prefill::decode_min_parts(),
                                     flint::gpu_prefill::decode_min_parts(true)});
    attn_partial_ = alloc<float>(q_, (size_t)cfg_.num_attention_heads * decode_parts * 128,
                                 "attn_partial");
    attn_stats_ = alloc<float>(q_, (size_t)cfg_.num_attention_heads * decode_parts * 2,
                               "attn_stats");

    if(flint::flash_decode_twopass::enabled())
        attn_scores_=alloc<float>(q_,size_t(cfg_.num_attention_heads)*max_seq_,"decode_scores");

    kv_.resize(w_.layers.size());
    for (size_t i = 0; i < w_.layers.size(); ++i) {
        const bool global = w_.layers[i].global;
        kv_[i].global = global;
        kv_[i].cap = global ? max_seq_ : std::min(cfg_.sliding_window, max_seq_);
        kv_[i].k = alloc<flint::bf16>(q_, (size_t)cfg_.num_key_value_heads * kv_[i].cap * cfg_.head_dim, "kv_k");
        kv_[i].v = alloc<flint::bf16>(q_, (size_t)cfg_.num_key_value_heads * kv_[i].cap * cfg_.head_dim, "kv_v");
    }

    info_.vocab_size = cfg_.vocab_size;
    info_.max_seq_len = max_seq_;
    info_.bos_token_id = cfg_.bos_token_id;
    info_.eos_token_ids = cfg_.eos_token_ids;
    info_.temperature = cfg_.temperature;
    info_.top_k = cfg_.top_k;
    info_.top_p = cfg_.top_p;
    info_.model_dir = model_dir;
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "flint-int4: %zu layers, H=%d, vocab=%d, experts=%d (top-%d), "
                      "int4 group-64, ctx=%d",
                      w_.layers.size(), cfg_.hidden_size, cfg_.vocab_size,
                      cfg_.num_experts, cfg_.num_experts_per_tok, max_seq_);
        info_.description = buf;
    }
    std::printf("[flint] %s\n", info_.description.c_str());
}

FlintModel::~FlintModel() {
    q_.wait();
    resident_decode_.reset();
    if (w_.arena) sycl::free(w_.arena, q_);
    for (auto& c : kv_) {
        if (c.k) sycl::free(c.k, q_);
        if (c.v) sycl::free(c.v, q_);
    }
    if (h_) sycl::free(h_, q_);
    if (normed_) sycl::free(normed_, q_);
    if (qbuf_) sycl::free(qbuf_, q_);
    if (kbuf_) sycl::free(kbuf_, q_);
    if (vbuf_) sycl::free(vbuf_, q_);
    if (attn_) sycl::free(attn_, q_);
    if (act_) sycl::free(act_, q_);
    if (down_) sycl::free(down_, q_);
    if (gate_) sycl::free(gate_, q_);
    if (up_) sycl::free(up_, q_);
    if (logits_dev_) sycl::free(logits_dev_, q_);
    if (trace_) sycl::free(trace_, q_);
    if (ids_dev_) sycl::free(ids_dev_, q_);
    if (weights_dev_) sycl::free(weights_dev_, q_);
    if (attn_partial_) sycl::free(attn_partial_, q_);
    if (attn_stats_) sycl::free(attn_stats_, q_);
    if (attn_scores_) sycl::free(attn_scores_, q_);
    release_batch();
}

PreparedInput FlintModel::prepare_input(const std::string& prompt,
                                        const std::vector<std::string>&,
                                        const std::vector<std::string>&,
                                        const std::string&) {
    // The checkpoint template expects plain string content and uses Jinja's
    // `replace` filter; minja now implements that filter, so the standard
    // string-message overload renders correctly (and sanitizes the prompt).
    auto built = build_chat_prompt(
        cfg_.model_dir, std::vector<ChatTemplateMessage>{{"user", prompt}},
        /*add_generation_prompt=*/true, /*enable_thinking=*/false);
    PreparedInput out;
    out.tokens = std::move(built.tokens);
    out.mm_token_type_ids = std::move(built.mm_token_type_ids);
    return out;
}

void FlintModel::attend(const flint::bf16* q, const KVCache& c,
                        const flint::bf16* sink, flint::bf16* out, int pos) {
    using namespace flint::kernels;
    const int count = c.global ? pos + 1 : std::min(pos + 1, c.cap);
    const int parts = attention_parts(count);
    if (parts <= 1) {
        attention(q_, q, c.k, c.v, sink, out, cfg_.num_attention_heads,
                  cfg_.num_key_value_heads, c.global, pos, c.cap);
    } else {
        attention_split(q_, q, c.k, c.v, sink, attn_partial_, attn_stats_,
                        cfg_.num_attention_heads, cfg_.num_key_value_heads,
                        c.global, pos, c.cap, parts);
        attention_merge(q_, attn_partial_, attn_stats_, out,
                        cfg_.num_attention_heads, parts);
    }
}

void FlintModel::step(int token) {
    using namespace flint;
    using namespace flint::kernels;
    const int H = cfg_.hidden_size;
    const int heads = cfg_.num_attention_heads;
    const int kvh = cfg_.num_key_value_heads;
    const int top_k = cfg_.num_experts_per_tok;

    embed_i8(q_, w_.embed, token, h_);
    if (trace_enabled_) q_.memcpy(trace_, h_, (size_t)H * sizeof(bf16));
    for (auto& L : w_.layers) {
        const bool rope = cfg_.use_rope(L.index);
        const bool global = L.global;
        const int cap = kv_[L.index].cap;

        rmsnorm(q_, h_, L.in_norm, normed_, H, cfg_.rms_norm_eps);
        qkv_gemv(q_, L.q, L.k, L.v, normed_, qbuf_, kbuf_, vbuf_);
        qkv_post(q_, qbuf_, kbuf_, vbuf_, L.q_norm, L.k_norm, kv_[L.index].k,
                 kv_[L.index].v, heads, kvh, global, pos_, cap, rope, log2base_,
                 cfg_.rms_norm_eps);
        attend(qbuf_, kv_[L.index], L.sink, attn_, pos_);
        q4_gemv(q_, L.o, attn_, h_, 0, h_);

        rmsnorm(q_, h_, L.post_norm, normed_, H, cfg_.rms_norm_eps);
        if (!L.moe) {
            dense_gateup_act(q_, L.gate, L.up, normed_, gate_);
            q4_gemv(q_, L.down, gate_, h_, 0, h_);
        } else {
            router_topk(q_, L.router, normed_, cfg_.num_experts, H, top_k,
                        ids_dev_, weights_dev_);
            expert_gateup(q_, L, H, cfg_.moe_intermediate_size,
                          cfg_.shared_expert_intermediate_size, top_k, normed_,
                          ids_dev_, act_);
            moe_down_combine(q_, L, H, cfg_.moe_intermediate_size,
                             cfg_.shared_expert_intermediate_size, top_k, act_,
                             ids_dev_, weights_dev_, h_);
        }
        if (trace_enabled_) q_.memcpy(trace_ + (size_t)(L.index + 1) * H, h_, (size_t)H * sizeof(bf16));
    }
    ++pos_;
    rmsnorm(q_, h_, w_.final_norm, normed_, H, cfg_.rms_norm_eps);
    lm_head(q_, w_.head, normed_, logits_dev_);
}

void FlintModel::release_batch() {
    auto bfree = [&](auto*& p) { if (p) sycl::free(p, q_); p = nullptr; };
    bfree(b_tokens); bfree(b_h); bfree(b_normed); bfree(b_q); bfree(b_k);
    bfree(b_v); bfree(b_attn); bfree(b_gate); bfree(b_up); bfree(b_routed);
    bfree(b_shared_act); bfree(b_shared_up); bfree(b_shared_down);
    bfree(b_egate); bfree(b_eup); bfree(b_eact); bfree(b_ids); bfree(b_weights);
    bfree(b_rows); bfree(b_outs); bfree(b_xq); bfree(b_xs); bfree(b_xq2); bfree(b_xs2);
    bfree(b_xi);
    bfree(b_grouped_metadata);
    bfree(b_attn_partial); bfree(b_attn_stats);
    batch_cap_ = 0;
}

void FlintModel::ensure_batch(int m) {
    if (m <= batch_cap_ && (!flint::gpu_prefill::enabled() || b_attn_partial)) return;
    int cap = std::max(m, 512);
    if (cap > max_seq_) cap = max_seq_;
    if (m > cap) throw std::runtime_error("flint: prefill chunk exceeds context");
    q_.wait_and_throw();
    release_batch();
    const int H = cfg_.hidden_size;
    const int QD = cfg_.q_dim();
    const int KV = cfg_.kv_dim();
    const int top_k = cfg_.num_experts_per_tok;
    const int maxk = std::max({H, QD, cfg_.intermediate_size});
    b_tokens = alloc<int>(q_, cap, "b_tokens");
    b_h = alloc<flint::bf16>(q_, (size_t)cap * H, "b_h");
    b_normed = alloc<flint::bf16>(q_, (size_t)cap * H, "b_normed");
    b_q = alloc<flint::bf16>(q_, (size_t)cap * QD, "b_q");
    b_k = alloc<flint::bf16>(q_, (size_t)cap * KV, "b_k");
    b_v = alloc<flint::bf16>(q_, (size_t)cap * KV, "b_v");
    b_attn = alloc<flint::bf16>(q_, (size_t)cap * QD, "b_attn");
    b_gate = alloc<flint::bf16>(q_, (size_t)cap * cfg_.intermediate_size, "b_gate");
    b_up = alloc<flint::bf16>(q_, (size_t)cap * cfg_.intermediate_size, "b_up");
    b_routed = alloc<flint::bf16>(q_, (size_t)cap * top_k * H, "b_routed");
    b_shared_act = alloc<flint::bf16>(q_, (size_t)cap * cfg_.shared_expert_intermediate_size, "b_sact");
    b_shared_up = alloc<flint::bf16>(q_, (size_t)cap * cfg_.shared_expert_intermediate_size, "b_sup");
    b_shared_down = alloc<flint::bf16>(q_, (size_t)cap * H, "b_sdown");
    b_egate = alloc<flint::bf16>(q_, (size_t)cap * cfg_.moe_intermediate_size, "b_egate");
    b_eup = alloc<flint::bf16>(q_, (size_t)cap * cfg_.moe_intermediate_size, "b_eup");
    b_eact = alloc<flint::bf16>(q_, (size_t)cap * top_k * cfg_.moe_intermediate_size, "b_eact");
    b_grouped_metadata = alloc<int>(q_, flint::grouped_moe::metadata_size(cfg_.num_experts),
                                    "b_grouped_metadata");
    b_ids = alloc<int>(q_, (size_t)cap * top_k, "b_ids");
    b_weights = alloc<float>(q_, (size_t)cap * top_k, "b_weights");
    b_rows = alloc<int>(q_, (size_t)cap * top_k, "b_rows");
    b_outs = alloc<int>(q_, (size_t)cap * top_k, "b_outs");
    b_xq = alloc<int8_t>(q_, (size_t)cap * maxk, "b_xq");
    b_xs = alloc<float>(q_, (size_t)cap * (maxk / 64), "b_xs");
    b_xq2 = alloc<int8_t>(q_, (size_t)cap * maxk, "b_xq2");
    b_xs2 = alloc<float>(q_, (size_t)cap * (maxk / 64), "b_xs2");
    b_xi = alloc<int>(q_, (size_t)cap * (maxk / 64), "b_xi");
    if (flint::gpu_prefill::enabled()) {
        const size_t rows = (size_t)cap * cfg_.num_attention_heads *
                            flint::gpu_prefill::parts_for(max_seq_);
        b_attn_partial = alloc<float>(q_, rows * 128, "b_attn_partial");
        b_attn_stats = alloc<float>(q_, rows * 2, "b_attn_stats");
    }
    batch_cap_ = cap;
}

void FlintModel::prefill_batch(const std::vector<int>& tokens, int pos_base) {
    using namespace flint;
    using namespace flint::prefill;
    const int M = (int)tokens.size();
    ensure_batch(M);
    const int H = cfg_.hidden_size;
    const int QD = cfg_.q_dim();
    const int KV = cfg_.kv_dim();
    const int heads = cfg_.num_attention_heads;
    const int kvh = cfg_.num_key_value_heads;
    const int top_k = cfg_.num_experts_per_tok;
    const bool int8 = !cfg_.awq && std::getenv("ARCAINE_FLINT_PREFILL_INT8") != nullptr;
    // The int8 prefill weights are repacked into DPAS fragment order at load
    // time (see loader.cpp / prefill_xmx.hpp), so the int8 prefill path is
    // XMX-only: the vector int8 kernel would read the permuted layout and
    // produce garbage.  ARCAINE_FLINT_PREFILL_XMX is therefore implied by the
    // INT8 flag (it still selects the bf16 path when INT8 is unset).
    const bool xmx = int8;
    const bool gpu_tokens = gpu_prefill::enabled();
    const bool dbg = std::getenv("ARCAINE_FLINT_DEBUG_SYNC") != nullptr;
    auto sync = [&](const char* what) {
        if (!dbg) return;
        try { q_.wait_and_throw(); }
        catch (const std::exception& e) {
            std::printf("[flint-debug] %s: %s\n", what, e.what());
            throw;
        }
    };
    int xmx_hits = 0, xmx_miss = 0;
    int awq_hits = 0, awq_miss = 0;
    auto gemm = [&](const QTensor& w, const bf16* A, const int8_t* Aq,
                    const float* xs, const int* ridx, bf16* C, int m, int N,
                    const bf16* res = nullptr, const int* oidx = nullptr,
                    int ostride = 0, int e = 0) {
        if (w.format == WeightFormat::AWQ) {
            if (awq_xmx::gemm(q_, w, A, ridx, C, m, N, res, oidx, ostride, e))
                ++awq_hits;
            else {
                ++awq_miss;
                gemm_awq(q_, w, A, ridx, C, m, N, res, oidx, ostride, e);
            }
            return;
        }
        // b_xi is sized for the widest K the model uses (see ensure_batch).
        if (int8 && xmx && Aq && xs && b_xi) {
            // Aq holds every quantized row of the chunk; expert GEMMs
            // gather their rows through ridx, so xi must cover all of
            // them, not just this call's token count.
            prefill_xmx::group_sums(q_, Aq, b_xi, M, w.k);
            if (prefill_xmx::gemm_q4_xmx(q_, w, Aq, xs, b_xi, ridx, C, m, N,
                                         res, oidx, ostride, e)) {
                ++xmx_hits;
                return;
            }
            ++xmx_miss;
            // No fallback: the weights are in DPAS fragment order, so the
            // vector kernels would read the wrong bytes.
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "flint: gemm_q4_xmx rejected M=%d N=%d K=%d "
                          "out_stride=%d (expert %d); the int8 prefill path is "
                          "XMX-only",
                          m, N, w.k, (ostride ? ostride : N), e);
            throw std::runtime_error(msg);
        }
        if (int8) gemm_q4<true>(q_, w, A, Aq, xs, ridx, C, m, N, res, oidx, ostride, e);
        else gemm_q4<false>(q_, w, A, Aq, xs, ridx, C, m, N, res, oidx, ostride, e);
    };

    q_.memcpy(b_tokens, tokens.data(), M * sizeof(int)).wait();
    embed_batch(q_, w_.embed, b_tokens, b_h, M);

    for (auto& L : w_.layers) {
        const bool rope = cfg_.use_rope(L.index);
        const bool global = L.global;
        const int cap = kv_[L.index].cap;

        rmsnorm_batch(q_, b_h, L.in_norm, b_normed, M, H, cfg_.rms_norm_eps); sync("pre_norm");
        if (int8) quantize_int8(q_, b_normed, b_xq, b_xs, M, H);
        gemm(L.q, b_normed, b_xq, b_xs, nullptr, b_q, M, QD,
                      nullptr, nullptr, 0, 0);
        gemm(L.k, b_normed, b_xq, b_xs, nullptr, b_k, M, KV,
                      nullptr, nullptr, 0, 0);
        gemm(L.v, b_normed, b_xq, b_xs, nullptr, b_v, M, KV,
                      nullptr, nullptr, 0, 0); sync("v_gemm");
        head_norm_rope_batch(q_, b_q, L.q_norm, M, heads, rope, pos_base,
                             log2base_, cfg_.rms_norm_eps); sync("headnorm");
        if (gpu_tokens) {
            // Preserve historical ring slots until every query in the chunk
            // has consumed them. Normalize K in the chunk, attend, then commit.
            head_norm_rope_batch(q_, b_k, L.k_norm, M, kvh, rope, pos_base,
                                 log2base_, cfg_.rms_norm_eps);
            gpu_prefill::attention(q_, b_q, b_k, b_v, kv_[L.index].k, kv_[L.index].v,
                                   L.sink, b_attn, b_attn_partial, b_attn_stats,
                                   M, heads, kvh, global, pos_base, cap);
            gpu_prefill::commit(q_, b_k, b_v, kv_[L.index].k, kv_[L.index].v,
                                M, kvh, global, pos_base, cap);
            sync("gpu_attention_batch");
        } else if (global) {
            kv_finish_batch(q_, b_k, b_v, L.k_norm, kv_[L.index].k,
                            kv_[L.index].v, M, kvh, true, pos_base, cap, rope,
                            log2base_, cfg_.rms_norm_eps);
            for (int t = 0; t < M; ++t)
                attend(b_q + (size_t)t * QD, kv_[L.index], L.sink,
                       b_attn + (size_t)t * QD, pos_base + t);
            sync("attention_global");
        } else {
            // Interleave stores and attention so the 513-slot ring is never
            // overwritten by a later token of the same chunk.
            for (int t = 0; t < M; ++t) {
                kernels::kv_finish(q_, b_k + (size_t)t * KV, b_v + (size_t)t * KV,
                                   L.k_norm, kv_[L.index].k, kv_[L.index].v,
                                   kvh, false, pos_base + t, cap, rope, log2base_,
                                   cfg_.rms_norm_eps);
                attend(b_q + (size_t)t * QD, kv_[L.index], L.sink,
                       b_attn + (size_t)t * QD, pos_base + t);
                sync("attention_sliding");
            }
        }

        if (int8) quantize_int8(q_, b_attn, b_xq, b_xs, M, QD);
        gemm(L.o, b_attn, b_xq, b_xs, nullptr, b_h, M, H, b_h,
                      nullptr, 0, 0); sync("o_proj");

        rmsnorm_batch(q_, b_h, L.post_norm, b_normed, M, H, cfg_.rms_norm_eps); sync("post_norm");
        if (!L.moe) {
            if (int8) quantize_int8(q_, b_normed, b_xq, b_xs, M, H);
            gemm(L.gate, b_normed, b_xq, b_xs, nullptr, b_gate, M,
                          cfg_.intermediate_size, nullptr, nullptr, 0, 0);
            gemm(L.up, b_normed, b_xq, b_xs, nullptr, b_up, M,
                          cfg_.intermediate_size, nullptr, nullptr, 0, 0);
            sync("dense_gateup");
            silu_mul_batch(q_, b_gate, b_up, M * cfg_.intermediate_size);
            if (int8) quantize_int8(q_, b_gate, b_xq, b_xs, M,
                                    cfg_.intermediate_size);
            gemm(L.down, b_gate, b_xq, b_xs, nullptr, b_h, M, H,
                          b_h, nullptr, 0, 0);
            sync("dense_down");
        } else {
            router_batch(q_, L.router, b_normed, M, cfg_.num_experts, H, top_k,
                         b_ids, b_weights);
            sync("router");
            if (cfg_.awq && (grouped_moe::enabled() || gpu_tokens)) {
                if (!grouped_moe::supported(q_, L.egate, L.eup, L.edown))
                    throw std::runtime_error("flint: grouped W4A16 MoE unsupported");
                grouped_moe::run(q_, L.egate, L.eup, L.edown, b_normed, b_ids,
                                 b_eact, b_routed, M, top_k, b_grouped_metadata, b_outs);
                sync("grouped_experts");
            } else {
                std::vector<int> ids((size_t)M * top_k), rows((size_t)M * top_k),
                    outs((size_t)M * top_k);
                q_.memcpy(ids.data(), b_ids, ids.size() * sizeof(int)).wait();
                std::vector<int> count(cfg_.num_experts, 0), off(cfg_.num_experts, 0);
                for (int t = 0; t < M; ++t)
                    for (int r = 0; r < top_k; ++r) ++count[ids[t * top_k + r]];
                int acc = 0;
                for (int e = 0; e < cfg_.num_experts; ++e) {
                    off[e] = acc;
                    acc += count[e];
                }
                std::vector<int> fill = off;
                for (int t = 0; t < M; ++t)
                    for (int r = 0; r < top_k; ++r) {
                        const int e = ids[t * top_k + r];
                        const int at = fill[e]++;
                        rows[at] = t;
                        outs[at] = t * top_k + r;
                    }
                if (M) {
                    q_.memcpy(b_rows, rows.data(), rows.size() * sizeof(int)).wait();
                    q_.memcpy(b_outs, outs.data(), outs.size() * sizeof(int)).wait();
                }
                if (int8) quantize_int8(q_, b_normed, b_xq, b_xs, M, H);
                for (int e = 0; e < cfg_.num_experts; ++e) {
                    const int n = count[e];
                    if (!n) continue;
                    const int* ridx = b_rows + off[e];
                    gemm(L.egate, b_normed, b_xq, b_xs, ridx, b_egate,
                                  n, cfg_.moe_intermediate_size, nullptr, nullptr, 0, e);
                    gemm(L.eup, b_normed, b_xq, b_xs, ridx, b_eup, n,
                                  cfg_.moe_intermediate_size, nullptr, nullptr, 0, e);
                    silu_mul_batch(q_, b_egate, b_eup, n * cfg_.moe_intermediate_size);
                    if (int8) quantize_int8(q_, b_egate, b_xq, b_xs, n,
                                            cfg_.moe_intermediate_size);
                    sync("expert_gateup");
                    const int* oidx = b_outs + off[e];
                    gemm(L.edown, b_egate, b_xq, b_xs, nullptr, b_routed,
                                  n, H, nullptr, oidx, H, e);
                }
            }
            if (int8) quantize_int8(q_, b_normed, b_xq, b_xs, M, H);
            gemm(L.sgate, b_normed, b_xq, b_xs, nullptr,
                          b_shared_act, M, cfg_.shared_expert_intermediate_size,
                          nullptr, nullptr, 0, 0);
            gemm(L.sup, b_normed, b_xq, b_xs, nullptr,
                          b_shared_up, M, cfg_.shared_expert_intermediate_size,
                          nullptr, nullptr, 0, 0);
            silu_mul_batch(q_, b_shared_act, b_shared_up,
                           M * cfg_.shared_expert_intermediate_size);
            if (int8) quantize_int8(q_, b_shared_act, b_xq, b_xs, M,
                                    cfg_.shared_expert_intermediate_size);
            gemm(L.sdown, b_shared_act, b_xq, b_xs, nullptr,
                          b_shared_down, M, H, nullptr, nullptr, 0, 0);
            combine_batch(q_, b_h, b_routed, b_shared_down, b_weights, M, H,
                          top_k);
            sync("combine");
        }
        if (trace_enabled_) q_.memcpy(trace_ + (size_t)(L.index + 1) * H,
                  b_h + (size_t)(M - 1) * H, (size_t)H * sizeof(bf16));
    }

    if (cfg_.awq && std::getenv("ARCAINE_FLINT_AWQ_XMX"))
        std::fprintf(stderr, "[flint-awq-xmx] gemm hits=%d fallback=%d\n",
                     awq_hits, awq_miss);
    if (xmx) {
        q_.wait();
        std::fprintf(stderr, "[flint-xmx] gemm hits=%d fallback=%d\n",
                     xmx_hits, xmx_miss);
    }

    // lm_head only needs the final position.
    kernels::rmsnorm(q_, b_h + (size_t)(M - 1) * H, w_.final_norm, normed_, H,
                     cfg_.rms_norm_eps);
    kernels::lm_head(q_, w_.head, normed_, logits_dev_);
}

void FlintModel::dump_trace(const std::string& path) {
    if (!trace_) throw std::logic_error("set ARCAINE_FLINT_TRACE before model construction");
    const int rows = cfg_.num_hidden_layers + 1;
    const int H = cfg_.hidden_size;
    std::vector<flint::bf16> host((size_t)rows * H);
    q_.memcpy(host.data(), trace_, host.size() * sizeof(flint::bf16)).wait();
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("flint: cannot write trace");
    for (auto v : host) {
        const float x = float(v);
        std::fwrite(&x, sizeof(x), 1, f);
    }
    std::fclose(f);
    std::printf("[flint] trace written: %s (%d x %d f32)\n", path.c_str(), rows, H);
}

std::vector<float> FlintModel::logits() {
    std::vector<float> out(cfg_.vocab_size);
    q_.memcpy(out.data(), logits_dev_, out.size() * sizeof(float)).wait();
    return out;
}

void FlintModel::enqueue_forward(const ForwardInput& in) {
    if (in.past_len == 0) pos_ = 0;
    else if (in.past_len != pos_) pos_ = in.past_len;
    const char* mode = std::getenv("ARCAINE_FLINT_PREFILL");
    const bool batched = ((mode && std::string(mode) == "batched") ||
                          flint::gpu_prefill::enabled()) && in.token_ids.size() > 1;
    if (batched) {
        int at = 0;
        const int total = (int)in.token_ids.size();
        while (at < total) {
            const int n = std::min(512, total - at);
            ensure_batch(n);
            std::vector<int> chunk(in.token_ids.begin() + at,
                                   in.token_ids.begin() + at + n);
            prefill_batch(chunk, pos_);
            pos_ += n;
            at += n;
        }
    } else {
        for (int tok : in.token_ids) step(tok);
    }
}

std::vector<float> FlintModel::forward(const ForwardInput& in) {
    enqueue_forward(in);
    return logits();
}

void FlintModel::forward_device(const ForwardInput& in) {
    if (!device_generation_enabled()) throw std::logic_error("Flint GPU decode is disabled");
    enqueue_forward(in);
    q_.wait_and_throw();
}

void FlintModel::reset_cache() { pos_ = 0; }

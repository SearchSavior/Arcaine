#include "loader.hpp"
#include "loader_awq.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const char* kPrefix = "model.language_model.";

// ---------------------------------------------------------------------------
// Dense projections (AWQ INT4 checkpoint: attention q/k/v/o, linear-attn
// out_proj and the shared expert are unquantized F16 -> BF16 at upload).
// ---------------------------------------------------------------------------
QwenDenseLinear upload_dense_linear(const TensorSource& sf, const std::string& prefix,
                                    sycl::queue& q) {
    const TensorView& tv = sf.get(prefix + ".weight");
    if (tv.shape.size() != 2)
        throw std::runtime_error("Expected 2D dense weight: " + prefix);
    QwenDenseLinear d;
    d.out_features = (int)tv.shape[0];
    d.in_features  = (int)tv.shape[1];
    d.weight = upload(tv, q, (prefix + ".weight").c_str());
    return d;
}

// Fused gate+up dense: concatenates two [N, K] weights along the output dim
// into one [2N, K] linear (gate in rows [0,N), up in [N,2N)).
QwenDenseLinear upload_dense_linear_pair(const TensorSource& sf,
                                         const std::string& first_prefix,
                                         const std::string& second_prefix,
                                         sycl::queue& q) {
    const TensorView& a = sf.get(first_prefix + ".weight");
    const TensorView& b = sf.get(second_prefix + ".weight");
    if (a.shape.size() != 2 || b.shape.size() != 2 || a.shape[1] != b.shape[1] ||
        a.shape[0] != b.shape[0])
        throw std::runtime_error("Dense pair shape mismatch: " + first_prefix + " / " +
                                 second_prefix);
    GpuBuffer<bf16> wa = upload(a, q, (first_prefix + ".weight").c_str());
    GpuBuffer<bf16> wb = upload(b, q, (second_prefix + ".weight").c_str());
    size_t half = (size_t)a.shape[0] * a.shape[1];
    GpuBuffer<bf16> fused(2 * half, q);
    q.memcpy(fused.data(), wa.data(), half * sizeof(bf16));
    q.memcpy(fused.data() + half, wb.data(), half * sizeof(bf16)).wait();
    QwenDenseLinear d;
    d.weight       = std::move(fused);
    d.out_features = (int)(2 * a.shape[0]);
    d.in_features  = (int)a.shape[1];
    return d;
}

}  // namespace

QwenWeights load_qwen_weights(const TensorSource& sf, const QwenConfig& cfg,
                              sycl::queue& q, int max_layers) {
    const int nl = cfg.num_hidden_layers;
    if (max_layers < 0 || max_layers > nl) max_layers = nl;

    // compressed-tensors pack-quantized INT4 (AWQ): only routed experts are
    // quantized; everything else is dense F16. Otherwise expect NVFP4
    // projections throughout (the original qwen3_5_moe_text path).
    const bool awq_int4 = (cfg.quant_format == "pack-quantized");

    QwenWeights w;
    w.embed_tokens = upload(sf.get(std::string(kPrefix) + "embed_tokens.weight"), q,
                            (std::string(kPrefix) + "embed_tokens.weight").c_str());
    w.final_norm = upload_plus_one(sf.get(std::string(kPrefix) + "norm.weight"), q,
                         (std::string(kPrefix) + "norm.weight").c_str());
    // lm_head is top-level (unprefixed) and untied (tie_word_embeddings=false).
    w.lm_head = upload(sf.get("lm_head.weight"), q, "lm_head.weight");

    w.layers.reserve(max_layers);
    for (int i = 0; i < max_layers; ++i) {
        QwenLayer layer;
        layer.layer_idx = i;
        layer.is_full_attention = cfg.is_full_attn(i);
        const std::string lp = std::string(kPrefix) + "layers." + std::to_string(i) + ".";
        const char* tag = layer.is_full_attention ? "full" : "linear";

        layer.input_layernorm =
            upload_plus_one(sf.get(lp + "input_layernorm.weight"), q,
                    (lp + "input_layernorm.weight").c_str());
        layer.post_attention_layernorm =
            upload_plus_one(sf.get(lp + "post_attention_layernorm.weight"), q,
                    (lp + "post_attention_layernorm.weight").c_str());

        if (layer.is_full_attention) {
            QwenFullAttn a;
            const std::string ap = lp + "self_attn.";
            if (awq_int4) {
                a.q_proj = upload_dense_linear(sf, ap + "q_proj", q);
                a.k_proj = upload_dense_linear(sf, ap + "k_proj", q);
                a.v_proj = upload_dense_linear(sf, ap + "v_proj", q);
                a.o_proj = upload_dense_linear(sf, ap + "o_proj", q);
            } else {
                a.q_proj = upload_nvfp4_linear(sf, ap + "q_proj", q);
                a.k_proj = upload_nvfp4_linear(sf, ap + "k_proj", q);
                a.v_proj = upload_nvfp4_linear(sf, ap + "v_proj", q);
                a.o_proj = upload_nvfp4_linear(sf, ap + "o_proj", q);
            }
            a.q_norm = upload_plus_one(sf.get(ap + "q_norm.weight"), q, (ap + "q_norm.weight").c_str());
            a.k_norm = upload_plus_one(sf.get(ap + "k_norm.weight"), q, (ap + "k_norm.weight").c_str());
            layer.attn = std::move(a);
        } else {
            QwenLinearAttn a;
            const std::string ap = lp + "linear_attn.";
            a.in_proj_qkv = upload(sf.get(ap + "in_proj_qkv.weight"), q,
                                   (ap + "in_proj_qkv.weight").c_str());
            a.in_proj_z   = upload(sf.get(ap + "in_proj_z.weight"), q,
                                   (ap + "in_proj_z.weight").c_str());
            a.in_proj_a   = upload(sf.get(ap + "in_proj_a.weight"), q,
                                   (ap + "in_proj_a.weight").c_str());
            a.in_proj_b   = upload(sf.get(ap + "in_proj_b.weight"), q,
                                   (ap + "in_proj_b.weight").c_str());
            a.conv1d      = upload(sf.get(ap + "conv1d.weight"), q,
                                   (ap + "conv1d.weight").c_str());
            // A_log / dt_bias are bare parameters (no .weight suffix).
            a.A_log       = upload(sf.get(ap + "A_log"), q, (ap + "A_log").c_str());
            a.dt_bias     = upload(sf.get(ap + "dt_bias"), q, (ap + "dt_bias").c_str());
            a.norm        = upload(sf.get(ap + "norm.weight"), q, (ap + "norm.weight").c_str());
            a.out_proj    = awq_int4 ? QwenProj(upload_dense_linear(sf, ap + "out_proj", q))
                                     : QwenProj(upload_nvfp4_linear(sf, ap + "out_proj", q));
            layer.attn = std::move(a);
        }

        // MoE: router + 256 routed experts (fused gate/up + down) + shared expert.
        QwenMoE m;
        const std::string mp = lp + "mlp.";
        m.router_gate = upload(sf.get(mp + "gate.weight"), q, (mp + "gate.weight").c_str());
        m.experts_gate_up.reserve(cfg.num_experts);
        m.experts_down.reserve(cfg.num_experts);
        if (awq_int4) {
            // Canonical contiguous raw-u4 expert storage (grouped oneDNN +
            // DPAS fallback + per-expert u4zp reference all read slices).
            m.grouped = qwen_awq::upload_experts_grouped(sf, mp, cfg.num_experts, q);
        } else {
            for (int e = 0; e < cfg.num_experts; ++e) {
                const std::string ep = mp + "experts." + std::to_string(e) + ".";
                m.experts_gate_up.push_back(
                    upload_nvfp4_linear_pair(sf, ep + "gate_proj", ep + "up_proj", q));
                m.experts_down.push_back(upload_nvfp4_linear(sf, ep + "down_proj", q));
            }
        }
        const std::string sep = mp + "shared_expert.";
        if (awq_int4) {
            m.shared_gate_up = upload_dense_linear_pair(sf, sep + "gate_proj",
                                                        sep + "up_proj", q);
            m.shared_down    = upload_dense_linear(sf, sep + "down_proj", q);
        } else {
            m.shared_gate_up = upload_nvfp4_linear_pair(sf, sep + "gate_proj",
                                                        sep + "up_proj", q);
            m.shared_down    = upload_nvfp4_linear(sf, sep + "down_proj", q);
        }
        m.shared_expert_gate =
            upload(sf.get(mp + "shared_expert_gate.weight"), q,
                   (mp + "shared_expert_gate.weight").c_str());
        layer.moe = std::move(m);

        w.layers.push_back(std::move(layer));
        std::printf("[qwen-load] layer %d/%d (%s) done\n", i + 1, max_layers, tag);
    }
    return w;
}

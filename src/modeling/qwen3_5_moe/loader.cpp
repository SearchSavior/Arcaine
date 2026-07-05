#include "loader.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
const char* kPrefix = "model.language_model.";
}  // namespace

QwenWeights load_qwen_weights(const TensorSource& sf, const QwenConfig& cfg,
                              sycl::queue& q, int max_layers) {
    const int nl = cfg.num_hidden_layers;
    if (max_layers < 0 || max_layers > nl) max_layers = nl;

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
            a.q_proj = upload_nvfp4_linear(sf, ap + "q_proj", q);
            a.k_proj = upload_nvfp4_linear(sf, ap + "k_proj", q);
            a.v_proj = upload_nvfp4_linear(sf, ap + "v_proj", q);
            a.o_proj = upload_nvfp4_linear(sf, ap + "o_proj", q);
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
            a.out_proj    = upload_nvfp4_linear(sf, ap + "out_proj", q);
            layer.attn = std::move(a);
        }

        // MoE: router + 256 routed experts (fused gate/up + down) + shared expert.
        QwenMoE m;
        const std::string mp = lp + "mlp.";
        m.router_gate = upload(sf.get(mp + "gate.weight"), q, (mp + "gate.weight").c_str());
        m.experts_gate_up.reserve(cfg.num_experts);
        m.experts_down.reserve(cfg.num_experts);
        for (int e = 0; e < cfg.num_experts; ++e) {
            const std::string ep = mp + "experts." + std::to_string(e) + ".";
            m.experts_gate_up.push_back(
                upload_nvfp4_linear_pair(sf, ep + "gate_proj", ep + "up_proj", q));
            m.experts_down.push_back(upload_nvfp4_linear(sf, ep + "down_proj", q));
        }
        const std::string sep = mp + "shared_expert.";
        m.shared_gate_up = upload_nvfp4_linear_pair(sf, sep + "gate_proj", sep + "up_proj", q);
        m.shared_down    = upload_nvfp4_linear(sf, sep + "down_proj", q);
        m.shared_expert_gate =
            upload(sf.get(mp + "shared_expert_gate.weight"), q,
                   (mp + "shared_expert_gate.weight").c_str());
        layer.moe = std::move(m);

        w.layers.push_back(std::move(layer));
        std::printf("[qwen-load] layer %d/%d (%s) done\n", i + 1, max_layers, tag);
    }
    return w;
}

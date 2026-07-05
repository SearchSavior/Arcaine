// src/qwen_load_smoke.cpp
//
// Phase 5 — Qwen-AgentWorld-35B-A3B-NVFP4 loader smoke test.
//
// Loads the REAL checkpoint through the shared NVFP4/BF16 primitives
// (src/common/io/quant_loader.{hpp,cpp} + src/modeling/qwen3_5_moe/loader.cpp)
// and verifies, without running any kernel:
//   * config.json parsing (dims, MoE, RoPE, layer_types);
//   * checkpoint prefix stripping (model.language_model.*) + tensor resolution;
//   * BF16 uploads (embed / lm_head / norms / linear-attn in_proj_* / router);
//   * NVFP4 linear construction (weight_packed, weight_scale, globals);
//   * scale transpose [N, K/16] -> [K/16, N] (bit-exact vs raw checkpoint bytes);
//   * dst_scale folding = input_global_scale * weight_global_scale (bit-exact);
//   * per-layer struct population for both full-attention and linear-attention
//     layer types, and the 256-expert MoE + shared-expert block.
//
// Usage: qwen_load_smoke [model_dir] [max_layers]
//   model_dir  default /workspace/models/Qwen-AgentWorld-35B-A3B-NVFP4
//   max_layers default 4  (loads layers 0..3 — covers linear + full attention;
//                          the first full-attention layer is idx 3)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include "common/gpu/engine.hpp"
#include "common/io/quant_loader.hpp"
#include "modeling/qwen3_5_moe/config.hpp"
#include "modeling/qwen3_5_moe/loader.hpp"
#include "modeling/qwen3_5_moe/weights.hpp"

static int g_fail = 0;
static void check(bool ok, const std::string& msg) {
    std::fprintf(stderr, "  [%s] %s\n", ok ? " OK " : "FAIL", msg.c_str());
    if (!ok) ++g_fail;
}

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1]
                    : "/workspace/models/Qwen-AgentWorld-35B-A3B-NVFP4";
    int max_layers = (argc > 2) ? std::atoi(argv[2]) : 4;

    std::printf("[smoke] model_dir  = %s\n", dir.c_str());
    std::printf("[smoke] max_layers = %d\n", max_layers);

    // ---- config ----
    QwenConfig cfg = QwenConfig::from_dir(dir);
    std::printf("[smoke] config: type=%s hidden=%d vocab=%d layers=%d heads=%d "
                "kv=%d head_dim=%d experts=%d topk=%d rotary=%d\n",
                cfg.model_type.c_str(), cfg.hidden_size, cfg.vocab_size,
                cfg.num_hidden_layers, cfg.num_attention_heads,
                cfg.num_key_value_heads, cfg.head_dim, cfg.num_experts,
                cfg.num_experts_per_tok, cfg.rotary_dim());
    check(cfg.model_type == "qwen3_5_moe_text", "model_type == qwen3_5_moe_text");
    check(cfg.num_experts == 256, "num_experts == 256");
    check(cfg.num_experts_per_tok == 8, "num_experts_per_tok == 8");
    check(cfg.head_dim == 256, "head_dim == 256");
    check(cfg.rotary_dim() == 64, "rotary_dim == 64 (partial_rotary_factor 0.25)");
    check((int)cfg.is_full_attention.size() == cfg.num_hidden_layers,
          "layer_types length == num_hidden_layers");
    int n_full = 0;
    for (bool b : cfg.is_full_attention) if (b) ++n_full;
    check(n_full == 10, "exactly 10 full-attention layers (idx 3,7,...,39)");
    check(!cfg.tie_word_embeddings, "tie_word_embeddings == false (lm_head separate)");

    // ---- sharded source ----
    ShardedSafetensors sf(dir);
    check(sf.num_tensors() == 124063, "tensor count == 124063");

    auto& q = GpuEngine::get(0).queue;
    std::printf("[smoke] device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- load ----
    QwenWeights w = load_qwen_weights(sf, cfg, q, max_layers);
    int expected_layers = (max_layers < 0 || max_layers > cfg.num_hidden_layers)
                          ? cfg.num_hidden_layers : max_layers;

    // ---- top-level ----
    check((int)w.embed_tokens.count() == cfg.vocab_size * cfg.hidden_size,
          "embed_tokens.count == vocab*hidden");
    check((int)w.final_norm.count() == cfg.hidden_size, "final_norm.count == hidden");
    check((int)w.lm_head.count() == cfg.vocab_size * cfg.hidden_size,
          "lm_head.count == vocab*hidden");
    check((int)w.layers.size() == expected_layers, "layers.size == expected");

    // ---- per-layer ----
    for (int i = 0; i < (int)w.layers.size(); ++i) {
        auto& L = w.layers[i];
        check((int)L.input_layernorm.count() == cfg.hidden_size &&
              (int)L.post_attention_layernorm.count() == cfg.hidden_size,
              "layer " + std::to_string(i) + ": layernorms populated");
        check(L.is_full_attention == cfg.is_full_attn(i),
              "layer " + std::to_string(i) + ": is_full_attention flag matches config");

        if (L.is_full_attention) {
            auto& a = std::get<QwenFullAttn>(L.attn);
            check(!a.q_proj.empty() && !a.k_proj.empty() && !a.v_proj.empty() &&
                  !a.o_proj.empty(),
                  "layer " + std::to_string(i) + " (full): q/k/v/o proj present");
            check(a.q_proj.out_features == 8192 && a.q_proj.in_features == 2048,
                  "layer " + std::to_string(i) + " (full): q_proj [8192,2048]");
            check(a.k_proj.out_features == 512 && a.k_proj.in_features == 2048 &&
                  a.v_proj.out_features == 512 && a.v_proj.in_features == 2048,
                  "layer " + std::to_string(i) + " (full): k/v_proj [512,2048]");
            check(a.o_proj.out_features == 2048 && a.o_proj.in_features == 4096,
                  "layer " + std::to_string(i) + " (full): o_proj [2048,4096]");
            check((int)a.q_norm.count() == cfg.head_dim &&
                  (int)a.k_norm.count() == cfg.head_dim,
                  "layer " + std::to_string(i) + " (full): q/k_norm [256]");
        } else {
            auto& a = std::get<QwenLinearAttn>(L.attn);
            check(!a.in_proj_qkv.empty() && !a.in_proj_z.empty() &&
                  !a.in_proj_a.empty() && !a.in_proj_b.empty(),
                  "layer " + std::to_string(i) + " (linear): in_proj_* present");
            check((int)a.in_proj_qkv.count() == 8192 * 2048,
                  "layer " + std::to_string(i) + " (linear): in_proj_qkv [8192,2048]");
            check((int)a.in_proj_z.count() == 4096 * 2048,
                  "layer " + std::to_string(i) + " (linear): in_proj_z [4096,2048]");
            check((int)a.in_proj_a.count() == 32 * 2048 &&
                  (int)a.in_proj_b.count() == 32 * 2048,
                  "layer " + std::to_string(i) + " (linear): in_proj_a/b [32,2048]");
            check((int)a.A_log.count() == 32 && (int)a.dt_bias.count() == 32,
                  "layer " + std::to_string(i) + " (linear): A_log/dt_bias [32]");
            check((int)a.conv1d.count() == 8192 * 4,
                  "layer " + std::to_string(i) + " (linear): conv1d [8192,1,4]");
            check((int)a.norm.count() == 128,
                  "layer " + std::to_string(i) + " (linear): norm [128]");
            check(!a.out_proj.empty() && a.out_proj.out_features == 2048 &&
                  a.out_proj.in_features == 4096,
                  "layer " + std::to_string(i) + " (linear): out_proj [2048,4096]");
        }

        // MoE block (present on every layer).
        auto& m = L.moe;
        check((int)m.router_gate.count() == 256 * 2048,
              "layer " + std::to_string(i) + ": router gate [256,2048]");
        check((int)m.experts_gate_up.size() == 256 &&
              (int)m.experts_down.size() == 256,
              "layer " + std::to_string(i) + ": 256 routed experts");
        check(m.experts_gate_up[0].out_features == 1024 &&
              m.experts_gate_up[0].in_features == 2048 &&
              m.experts_down[0].out_features == 2048 &&
              m.experts_down[0].in_features == 512,
              "layer " + std::to_string(i) + ": expert[0] gate_up[1024,2048] down[2048,512]");
        check(!m.shared_gate_up.empty() && !m.shared_down.empty(),
              "layer " + std::to_string(i) + ": shared expert present");
        check((int)m.shared_expert_gate.count() == 2048,
              "layer " + std::to_string(i) + ": shared_expert_gate [1,2048]");

        float dst = 0.0f;
        m.experts_gate_up[0].dst_scale.download(&dst, 1);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "layer %d: expert[0] dst_scale>0 (= %.6g)", i, dst);
        check(dst > 0.0f, buf);
    }

    // ---- scale transpose + dst_scale folding (needs the first full-attn layer, idx 3) ----
    if (expected_layers >= 4) {
        const std::string scale_name =
            "model.language_model.layers.3.self_attn.q_proj.weight_scale";
        const TensorView& tv = sf.get(scale_name);
        int out = (int)tv.shape[0];   // 8192
        int grp = (int)tv.shape[1];   // 128
        check(out == 8192 && grp == 128, "q_proj.weight_scale shape [8192,128]");

        const uint8_t* raw = static_cast<const uint8_t*>(tv.data);  // model layout (N, K/16)
        auto& a = std::get<QwenFullAttn>(w.layers[3].attn);
        std::vector<uint8_t> dev((size_t)grp * out);
        a.q_proj.weight_scale.download(dev.data(), dev.size());

        // transposed[g][n] == raw[n*grp + g]
        int ng = 10, gg = 5;
        uint8_t expect = raw[(size_t)ng * grp + gg];
        uint8_t got = dev[(size_t)gg * out + ng];
        char msg[200];
        std::snprintf(msg, sizeof(msg),
            "scale transpose: dev[g=%d][n=%d]=%u == raw[n=%d][g=%d]=%u",
            gg, ng, (unsigned)got, ng, gg, (unsigned)expect);
        check(got == expect, msg);

        // dst_scale == input_global_scale * weight_global_scale (bit-exact)
        float ig = scalar_f32(sf.get(
            "model.language_model.layers.3.self_attn.q_proj.input_global_scale"));
        float wg = scalar_f32(sf.get(
            "model.language_model.layers.3.self_attn.q_proj.weight_global_scale"));
        float dst = 0.0f;
        a.q_proj.dst_scale.download(&dst, 1);
        std::snprintf(msg, sizeof(msg),
            "dst_scale fold: dev=%.6g == ig*wg=%.6g", dst, ig * wg);
        check(dst == ig * wg, msg);
    } else {
        std::printf("[smoke] skipping transpose/dst_scale fold check (need >=4 layers)\n");
    }

    std::printf("[smoke] === %d failure(s) ===\n", g_fail);
    return g_fail ? 1 : 0;
}

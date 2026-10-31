#include "weights.hpp"
#include "config.hpp"
#include "../../common/io/safetensors.hpp"
#include "../../common/io/tensor_reader.hpp"
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cstdio>

// split_layer: layers [0, split_layer) load to GPU 0;
//              layers [split_layer, L) load to GPU 1.
GlobalWeights load_weights(const std::string& model_dir,
                           const ModelConfig& cfg,
                           int split_layer) {
    SafetensorsFile sf(model_dir + "/model.safetensors");
    std::printf("[load_weights] opened safetensors (%zu tensors)\n", sf.num_tensors());

    auto& q0 = GpuEngine::get(0).queue;
    auto& q1 = GpuEngine::get(1).queue;  // same as q0 when only 1 GPU

    GlobalWeights gw;
    gw.layers.resize(cfg.text.num_hidden_layers);

    // Shared tensors always on GPU 0 (embed_tokens used for embedding + lm_head;
    // final_norm + logit ops also stay on GPU 0).
    gw.embed_tokens = upload(sf.get("model.language_model.embed_tokens.weight"), q0);
    gw.final_norm   = upload(sf.get("model.language_model.norm.weight"), q0);

    // Per-layer weights — placed on the GPU that will run that layer.
    for (int l = 0; l < cfg.text.num_hidden_layers; ++l) {
        sycl::queue& ql = (l < split_layer) ? q0 : q1;
        LayerWeights& lw = gw.layers[l];
        lw.is_full = cfg.text.is_full_attention[l];

        std::string pfx = "model.language_model.layers." + std::to_string(l) + ".";

        lw.input_ln     = upload(sf.get(pfx + "input_layernorm.weight"),          ql);
        lw.post_attn_ln = upload(sf.get(pfx + "post_attention_layernorm.weight"),  ql);
        lw.pre_ffn_ln   = upload(sf.get(pfx + "pre_feedforward_layernorm.weight"), ql);
        lw.post_ffn_ln  = upload(sf.get(pfx + "post_feedforward_layernorm.weight"),ql);

        lw.layer_scalar = scalar_as_float(sf.get(pfx + "layer_scalar"));

        lw.ffn.gate_proj = upload(sf.get(pfx + "mlp.gate_proj.weight"), ql);
        lw.ffn.up_proj   = upload(sf.get(pfx + "mlp.up_proj.weight"),   ql);
        lw.ffn.down_proj = upload(sf.get(pfx + "mlp.down_proj.weight"), ql);

        std::string apfx = pfx + "self_attn.";
        if (!lw.is_full) {
            SlidingAttnWeights saw;
            saw.q_proj = upload(sf.get(apfx + "q_proj.weight"), ql);
            saw.k_proj = upload(sf.get(apfx + "k_proj.weight"), ql);
            saw.v_proj = upload(sf.get(apfx + "v_proj.weight"), ql);
            saw.o_proj = upload(sf.get(apfx + "o_proj.weight"), ql);
            saw.q_norm = upload(sf.get(apfx + "q_norm.weight"), ql);
            saw.k_norm = upload(sf.get(apfx + "k_norm.weight"), ql);
            lw.attn    = std::move(saw);
        } else {
            FullAttnWeights faw;
            faw.q_proj = upload(sf.get(apfx + "q_proj.weight"), ql);
            faw.k_proj = upload(sf.get(apfx + "k_proj.weight"), ql);
            faw.o_proj = upload(sf.get(apfx + "o_proj.weight"), ql);
            faw.q_norm = upload(sf.get(apfx + "q_norm.weight"), ql);
            faw.k_norm = upload(sf.get(apfx + "k_norm.weight"), ql);
            lw.attn    = std::move(faw);
        }

        if (l % 8 == 0)
            std::printf("[load_weights] layer %d/%d → GPU %d\n",
                        l, cfg.text.num_hidden_layers, (l < split_layer) ? 0 : 1);
    }

    // Vision and audio embedder weights stay on GPU 0.
    {
        auto& vw = gw.vision;
        vw.patch_ln1_w   = upload(sf.get("model.vision_embedder.patch_ln1.weight"), q0);
        vw.patch_ln1_b   = upload(sf.get("model.vision_embedder.patch_ln1.bias"),   q0);
        vw.patch_dense_w = upload(sf.get("model.vision_embedder.patch_dense.weight"),q0);
        vw.patch_dense_b = upload(sf.get("model.vision_embedder.patch_dense.bias"),  q0);
        vw.patch_ln2_w   = upload(sf.get("model.vision_embedder.patch_ln2.weight"), q0);
        vw.patch_ln2_b   = upload(sf.get("model.vision_embedder.patch_ln2.bias"),   q0);
        vw.pos_embedding = upload(sf.get("model.vision_embedder.pos_embedding"),     q0);
        vw.pos_norm_w    = upload(sf.get("model.vision_embedder.pos_norm.weight"),   q0);
        vw.pos_norm_b    = upload(sf.get("model.vision_embedder.pos_norm.bias"),     q0);
        vw.proj_w        = upload(
            sf.get("model.embed_vision.embedding_projection.weight"), q0);
    }
    {
        gw.audio.proj_w = upload(
            sf.get("model.embed_audio.embedding_projection.weight"), q0);
    }

    std::printf("[load_weights] all weights loaded (split at layer %d)\n", split_layer);
    return gw;
}

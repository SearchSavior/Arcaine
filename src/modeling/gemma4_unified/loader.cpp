#include "weights.hpp"
#include "config.hpp"
#include "../../common/io/quant_loader.hpp"
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/int4.hpp"
#include <vector>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <cstdio>

// Upload packed int4 weights, rebasing compressed-tensors unsigned offset-8
// nibbles to two's-complement s4 (XOR 0x88 per byte) for oneDNN's s4 path. The
// packed byte layout (8 nibbles/int32, LSB-first) is identical for symmetric and
// asymmetric checkpoints, so the rebased bytes feed oneDNN s4 tag::ba directly.
static GpuBuffer<uint8_t> upload_int4_packed_rebased(const void* data, size_t bytes,
                                                     sycl::queue& q) {
    static std::vector<uint8_t> staging;
    if (staging.size() < bytes) staging.resize(bytes);
    std::memcpy(staging.data(), data, bytes);
    GpuBuffer<uint8_t> buf(bytes, q);
    uint8_t* dst = buf.data();
    sycl::event copy_done = q.memcpy(buf.data(), staging.data(), bytes);
    q.submit([&](sycl::handler& h) {
        h.depends_on(copy_done);
        h.parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> id) {
            dst[id[0]] ^= 0x88;
        });
    }).wait();
    return buf;
}

// Read a single-element layer_scalar stored as either F32 (AWQ) or BF16 (BF16
// checkpoint), so the same loader path serves both precisions.
static float read_layer_scalar(const TensorView& tv, const std::string& name) {
    if (tv.dtype == "F32") {
        float f;
        std::memcpy(&f, tv.data, sizeof(float));
        return f;
    }
    if (tv.dtype == "BF16")
        return bf16_to_float(*static_cast<const uint16_t*>(tv.data));
    throw std::runtime_error("Expected F32/BF16 for layer_scalar " + name +
                             ", got " + tv.dtype);
}

// Build an Int4Linear from a compressed-tensors AWQ (pack-quantized) checkpoint
// prefix.  Tensors:
//   weight_packed    I32 [N, K/8]   — 8 nibbles/int32 along K, LSB-first, unsigned offset-8
//   weight_scale     F32 [N, G]     — G = K / group_size (group_size=32 here)
//   weight_zero_point I32 [N/8, G]  — 8 nibbles/int32 along N, LSB-first, unsigned offset-8
//
// Dequant: w[n,k] = scale[n,g] * (q[n,k] - zp[n,g]), g = k / group_size.
// The oneDNN s4 GEMM computes Part1 = scale*q (rebased nibbles).  The asymmetric
// remainder scale*zp is precomputed and stored transposed to (G, N) BF16 in
// weight_zp_corr; matmul_int4 subtracts corr = group_sum(A) @ (scale*zp).
static Int4Linear upload_int4_linear_awq(const TensorSource& sf,
                                         const std::string& prefix,
                                         sycl::queue& q) {
    const TensorView& packed = sf.get(prefix + ".weight_packed");
    if (packed.dtype != "I32")
        throw std::runtime_error("Expected I32 packed weight: " + prefix);
    if (packed.shape.size() != 2)
        throw std::runtime_error("Expected 2D packed weight: " + prefix);

    Int4Linear lin;
    lin.out_features = (int)packed.shape[0];
    lin.in_features  = (int)packed.shape[1] * 8;   // 8 int4 per int32 along K

    const TensorView& scale = sf.get(prefix + ".weight_scale");
    if (scale.dtype != "F32")
        throw std::runtime_error("Expected F32 int4 scale: " + prefix);
    if (scale.shape.size() != 2 || (int)scale.shape[0] != lin.out_features)
        throw std::runtime_error("Unexpected int4 scale shape: " + prefix);
    int G = (int)scale.shape[1];
    if (G == 0 || lin.in_features % G != 0)
        throw std::runtime_error("int4 in_features not divisible by groups: " + prefix);
    lin.group_size = lin.in_features / G;

    lin.weight_packed = upload_int4_packed_rebased(packed.data, packed.nbytes, q);

    const float* sc = static_cast<const float*>(scale.data);
    bool has_zp = sf.has(prefix + ".weight_zero_point");
    const int32_t* zp = nullptr;
    if (has_zp) {
        const TensorView& zpv = sf.get(prefix + ".weight_zero_point");
        if (zpv.dtype != "I32")
            throw std::runtime_error("Expected I32 zero_point: " + prefix);
        if (zpv.shape.size() != 2 || (int)zpv.shape[0] != lin.out_features / 8 ||
            (int)zpv.shape[1] != G)
            throw std::runtime_error("Unexpected zero_point shape: " + prefix);
        zp = static_cast<const int32_t*>(zpv.data);
    }

    // Stage the transposed (G, N) BF16 scale and (G, N) BF16 scale*zp correction.
    size_t cnt = (size_t)G * lin.out_features;
    std::vector<bf16> scale_t(cnt), corr_t(cnt);
    for (int n = 0; n < lin.out_features; ++n) {
        int row    = n >> 3;       // n / 8  (zero_point packed along N)
        int within = n & 7;       // n % 8
        int sh     = within * 4;
        for (int g = 0; g < G; ++g) {
            float s = sc[(size_t)n * G + g];
            scale_t[(size_t)g * lin.out_features + n] = float_to_bf16(s);
            if (has_zp) {
                int32_t word = zp[(size_t)row * G + g];
                int zp_signed = ((word >> sh) & 0xF) - 8;
                corr_t[(size_t)g * lin.out_features + n] =
                    float_to_bf16(s * (float)zp_signed);
            }
        }
    }

    lin.weight_scale = GpuBuffer<bf16>(cnt, q);
    lin.weight_scale.upload(scale_t.data(), cnt);
    if (has_zp) {
        lin.weight_zp_corr = GpuBuffer<bf16>(cnt, q);
        lin.weight_zp_corr.upload(corr_t.data(), cnt);
    }
    return lin;
}

// Load a projection as INT4 (if <prefix>.weight_packed exists) or dense BF16.
static ProjWeight upload_proj(const TensorSource& sf, const std::string& prefix,
                              sycl::queue& q) {
    if (sf.has(prefix + ".weight_packed"))
        return ProjWeight{upload_int4_linear_awq(sf, prefix, q)};
    return ProjWeight{upload(sf.get(prefix + ".weight"), q, prefix.c_str())};
}

// split_layer: layers [0, split_layer) load to GPU 0;
//              layers [split_layer, L) load to GPU 1.
GlobalWeights load_weights(const std::string& model_dir,
                            const ModelConfig& cfg,
                            int split_layer) {
    ShardedSafetensors sf(model_dir);
    std::printf("[load_weights] opened sharded safetensors (%zu tensors)\n",
                sf.num_tensors());

    auto& q0 = GpuEngine::get(0).queue;
    auto& q1 = GpuEngine::get(1).queue;  // same as q0 when only 1 GPU

    GlobalWeights gw;
    gw.layers.resize(cfg.text.num_hidden_layers);

    // Shared tensors always on GPU 0 (embed_tokens used for embedding + lm_head;
    // final_norm + logit ops also stay on GPU 0).
    gw.embed_tokens = upload(sf.get("model.language_model.embed_tokens.weight"), q0, "embed_tokens");
    gw.final_norm   = upload(sf.get("model.language_model.norm.weight"), q0, "final_norm");

    // Per-layer weights — placed on the GPU that will run that layer.
    for (int l = 0; l < cfg.text.num_hidden_layers; ++l) {
        sycl::queue& ql = (l < split_layer) ? q0 : q1;
        LayerWeights& lw = gw.layers[l];
        lw.is_full = cfg.text.is_full_attention[l];

        std::string pfx = "model.language_model.layers." + std::to_string(l) + ".";

        lw.input_ln     = upload(sf.get(pfx + "input_layernorm.weight"),           ql, "input_ln");
        lw.post_attn_ln = upload(sf.get(pfx + "post_attention_layernorm.weight"),  ql, "post_attn_ln");
        lw.pre_ffn_ln   = upload(sf.get(pfx + "pre_feedforward_layernorm.weight"), ql, "pre_ffn_ln");
        lw.post_ffn_ln  = upload(sf.get(pfx + "post_feedforward_layernorm.weight"),ql, "post_ffn_ln");

        {
            const TensorView& tv = sf.get(pfx + "layer_scalar");
            lw.layer_scalar = read_layer_scalar(tv, pfx + "layer_scalar");
        }

        lw.ffn.gate_proj = upload_proj(sf, pfx + "mlp.gate_proj", ql);
        lw.ffn.up_proj   = upload_proj(sf, pfx + "mlp.up_proj",   ql);
        lw.ffn.down_proj = upload_proj(sf, pfx + "mlp.down_proj", ql);

        std::string apfx = pfx + "self_attn.";
        if (!lw.is_full) {
            SlidingAttnWeights saw;
            saw.q_proj = upload_proj(sf, apfx + "q_proj", ql);
            saw.k_proj = upload_proj(sf, apfx + "k_proj", ql);
            saw.v_proj = upload_proj(sf, apfx + "v_proj", ql);
            saw.o_proj = upload_proj(sf, apfx + "o_proj", ql);
            saw.q_norm = upload(sf.get(apfx + "q_norm.weight"), ql, "q_norm");
            saw.k_norm = upload(sf.get(apfx + "k_norm.weight"), ql, "k_norm");
            lw.attn    = std::move(saw);
        } else {
            FullAttnWeights faw;
            faw.q_proj = upload_proj(sf, apfx + "q_proj", ql);
            faw.k_proj = upload_proj(sf, apfx + "k_proj", ql);
            faw.o_proj = upload_proj(sf, apfx + "o_proj", ql);
            faw.q_norm = upload(sf.get(apfx + "q_norm.weight"), ql, "q_norm");
            faw.k_norm = upload(sf.get(apfx + "k_norm.weight"), ql, "k_norm");
            lw.attn    = std::move(faw);
        }

        if (l % 8 == 0)
            std::printf("[load_weights] layer %d/%d → GPU %d\n",
                        l, cfg.text.num_hidden_layers, (l < split_layer) ? 0 : 1);
    }

    // Vision and audio embedder weights stay on GPU 0.
    {
        auto& vw = gw.vision;
        vw.patch_ln1_w   = upload(sf.get("model.vision_embedder.patch_ln1.weight"), q0, "patch_ln1_w");
        vw.patch_ln1_b   = upload(sf.get("model.vision_embedder.patch_ln1.bias"),   q0, "patch_ln1_b");
        vw.patch_dense_w = upload(sf.get("model.vision_embedder.patch_dense.weight"),q0,"patch_dense_w");
        vw.patch_dense_b = upload(sf.get("model.vision_embedder.patch_dense.bias"),  q0,"patch_dense_b");
        vw.patch_ln2_w   = upload(sf.get("model.vision_embedder.patch_ln2.weight"), q0, "patch_ln2_w");
        vw.patch_ln2_b   = upload(sf.get("model.vision_embedder.patch_ln2.bias"),   q0, "patch_ln2_b");
        vw.pos_embedding = upload(sf.get("model.vision_embedder.pos_embedding"),     q0, "pos_embedding");
        vw.pos_norm_w    = upload(sf.get("model.vision_embedder.pos_norm.weight"),   q0,"pos_norm_w");
        vw.pos_norm_b    = upload(sf.get("model.vision_embedder.pos_norm.bias"),     q0,"pos_norm_b");
        vw.proj_w        = upload(
            sf.get("model.embed_vision.embedding_projection.weight"), q0, "vis_proj");
    }
    {
        gw.audio.proj_w = upload(
            sf.get("model.embed_audio.embedding_projection.weight"), q0, "aud_proj");
    }

    std::printf("[load_weights] all weights loaded (split at layer %d)\n", split_layer);
    return gw;
}

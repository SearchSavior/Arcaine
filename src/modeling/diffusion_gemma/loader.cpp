#include "loader.hpp"
#include "../../common/gpu/placement.hpp"
#include "../../common/gpu/expert_parallel.hpp"
#include "../../common/io/quant_loader.hpp"
#include "../../common/io/safetensors.hpp"
#include "../../common/io/gguf.hpp"
#include "../../common/io/tensor_reader.hpp"
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include "fusions/int4_awq.hpp"
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

// TensorSource / ShardedSafetensors and the NVFP4/BF16 upload helpers now
// live in common/io/quant_loader.hpp (shared with other model modules).

class DiffusionGgufSource : public TensorSource {
public:
    explicit DiffusionGgufSource(const std::string& path) : gguf_(path) {
        std::printf("[load] %zu tensors in GGUF file\n", gguf_.num_tensors());
    }

    const TensorView& get(const std::string& name) const override {
        return gguf_.get(map(name));
    }

    bool has(const std::string& name) const override {
        return gguf_.has(map(name));
    }

private:
    std::string map(const std::string& name) const {
        if (name == "model.decoder.embed_tokens.weight") return "token_embd.weight";
        if (name == "model.decoder.norm.weight") return "output_norm.weight";
        if (name == "model.decoder.self_conditioning.pre_norm.weight") return "self_cond_pre_norm.weight";
        if (name == "model.decoder.self_conditioning.gate_proj.weight") return "self_cond_gate.weight";
        if (name == "model.decoder.self_conditioning.up_proj.weight") return "self_cond_up.weight";
        if (name == "model.decoder.self_conditioning.down_proj.weight") return "self_cond_down.weight";

        const std::string dec = "model.decoder.layers.";
        const std::string enc = "model.encoder.language_model.layers.";
        if (name.rfind(dec, 0) == 0) return map_layer(name.substr(dec.size()), false);
        if (name.rfind(enc, 0) == 0) return map_layer(name.substr(enc.size()), true);
        return name;
    }

    static std::string map_layer(const std::string& rest, bool encoder) {
        size_t dot = rest.find('.');
        if (dot == std::string::npos) return rest;
        std::string layer = rest.substr(0, dot);
        std::string suffix = rest.substr(dot + 1);
        std::string p = "blk." + layer + ".";

        if (suffix == "input_layernorm.weight") return p + "attn_norm.weight";
        if (suffix == "post_attention_layernorm.weight") return p + "post_attention_norm.weight";
        if (suffix == "pre_feedforward_layernorm.weight") return p + "ffn_norm.weight";
        if (suffix == "pre_feedforward_layernorm_2.weight") return p + "pre_ffw_norm_2.weight";
        if (suffix == "post_feedforward_layernorm_1.weight") return p + "post_ffw_norm_1.weight";
        if (suffix == "post_feedforward_layernorm_2.weight") return p + "post_ffw_norm_2.weight";
        if (suffix == "post_feedforward_layernorm.weight") return p + "post_ffw_norm.weight";
        if (suffix == "layer_scalar")
            return p + (encoder ? "enc_layer_output_scale.weight" : "layer_output_scale.weight");

        if (suffix == "mlp.gate_proj.weight") return p + "ffn_gate.weight";
        if (suffix == "mlp.up_proj.weight") return p + "ffn_up.weight";
        if (suffix == "mlp.down_proj.weight") return p + "ffn_down.weight";
        if (suffix == "router.proj.weight") return p + "ffn_gate_inp.weight";
        if (suffix == "router.scale") return p + "ffn_gate_inp.scale";
        if (suffix == "router.per_expert_scale") return p + "ffn_down_exps.scale";
        if (suffix == "experts.gate_up_proj") return p + "ffn_gate_up_exps.weight";
        if (suffix == "experts.down_proj") return p + "ffn_down_exps.weight";

        const std::string attn = "self_attn.";
        if (suffix.rfind(attn, 0) == 0) {
            std::string a = suffix.substr(attn.size());
            if (a == "q_proj.weight") return p + "attn_q.weight";
            if (a == "k_proj.weight") return p + "attn_k.weight";
            if (a == "v_proj.weight") return p + "attn_v.weight";
            if (a == "o_proj.weight") return p + "attn_output.weight";
            if (a == "q_norm.weight") return p + "attn_q_norm.weight";
            if (a == "k_norm.weight") return p + "attn_k_norm.weight";
        }
        return p + suffix;
    }

    GgufFile gguf_;
};

// Build a projection weight of whichever quantization the checkpoint provides
// (int4 W4A16 / NVFP4 W4A4 / GGUF Q8_0 / plain BF16). The packer mechanics live
// in common/io/quant_loader; only the kind dispatch is diffusion-specific.
static DiffLinearWeight upload_linear_weight(const TensorSource& sf,
                                             const std::string& prefix,
                                             sycl::queue& q) {
    DiffLinearWeight out;
    if (sf.has(prefix + ".weight_packed")) {
        if (sf.get(prefix + ".weight_packed").dtype == "I32") {
            out.kind = DiffLinearWeight::Kind::INT4;
            out.int4 = upload_int4_linear(sf, prefix, q);
        } else {
            out.kind = DiffLinearWeight::Kind::NVFP4;
            out.nvfp4 = true;
            out.fp4 = upload_nvfp4_linear(sf, prefix, q);
        }
    } else {
        const TensorView& tv = sf.get(prefix + ".weight");
        if (tv.dtype == "Q8_0") {
            out.kind = DiffLinearWeight::Kind::Q8_0;
            out.q8 = upload_q8_linear_view(tv, q, false, (prefix + ".weight").c_str());
        } else {
            out.kind = DiffLinearWeight::Kind::BF16;
            out.bf16 = upload(tv, q, (prefix + ".weight").c_str());
        }
    }
    return out;
}

// split_layer: layers [0,split) -> GPU0, [split,L) -> GPU1.
DiffWeights load_diffusion_weights(const std::string& model_dir,
                                   const DiffConfig& cfg,
                                   int split_layer,
                                   DiffExpertPlacementMode expert_mode) {
    std::unique_ptr<TensorSource> source;
    const char* gguf_path = std::getenv("DIFF_GGUF_Q8_WEIGHTS");
    if (gguf_path && gguf_path[0]) {
        std::printf("[load] using GGUF Q8_0 weights from %s\n", gguf_path);
        source = std::make_unique<DiffusionGgufSource>(gguf_path);
    } else {
        source = std::make_unique<ShardedSafetensors>(model_dir);
    }
    const TensorSource& sf = *source;
    auto& q0 = GpuEngine::get(0).queue;
    auto& q1 = GpuEngine::get(1).queue;

    DiffWeights gw;
    gw.layers.resize(cfg.text.num_hidden_layers);

    // Tied embedding (encoder/decoder embed + lm_head) + final norm + self-cond on GPU0.
    {
        const TensorView& embed = sf.get("model.decoder.embed_tokens.weight");
        if (embed.dtype == "Q8_0")
            gw.embed_tokens_q8 = upload_q8_linear_view(
                embed, q0, true, "model.decoder.embed_tokens.weight");
        else
            gw.embed_tokens = upload(embed, q0, "model.decoder.embed_tokens.weight");
    }
    gw.final_norm   = upload(sf.get("model.decoder.norm.weight"), q0, std::string("model.decoder.norm.weight").c_str());
    gw.self_cond.pre_norm  = upload(sf.get("model.decoder.self_conditioning.pre_norm.weight"), q0, std::string("model.decoder.self_conditioning.pre_norm.weight").c_str());
    if (sf.get("model.decoder.self_conditioning.gate_proj.weight").dtype == "Q8_0") {
        gw.self_cond.gate_proj = upload_linear_weight(
            sf, "model.decoder.self_conditioning.gate_proj", q0);
        gw.self_cond.up_proj = upload_linear_weight(
            sf, "model.decoder.self_conditioning.up_proj", q0);
    } else {
        gw.self_cond.gate_up_proj = upload_bf16_pair(
            sf.get("model.decoder.self_conditioning.gate_proj.weight"),
            sf.get("model.decoder.self_conditioning.up_proj.weight"),
            q0, "model.decoder.self_conditioning.{gate,up}_proj.weight");
    }
    gw.self_cond.down_proj = upload_linear_weight(
        sf, "model.decoder.self_conditioning.down_proj", q0);

    for (int l = 0; l < cfg.text.num_hidden_layers; ++l) {
        int gpu = (l < split_layer) ? 0 : 1;
        sycl::queue& ql = (gpu == 0) ? q0 : q1;
        DiffLayer& lw = gw.layers[l];
        lw.is_full = cfg.text.is_full_attention[l];
        lw.gpu = gpu;

        std::string p = "model.decoder.layers." + std::to_string(l) + ".";

        lw.input_ln      = upload(sf.get(p + "input_layernorm.weight"), ql, std::string(p + "input_layernorm.weight").c_str());
        lw.post_attn_ln  = upload(sf.get(p + "post_attention_layernorm.weight"), ql, std::string(p + "post_attention_layernorm.weight").c_str());
        lw.pre_ffn_ln    = upload(sf.get(p + "pre_feedforward_layernorm.weight"), ql, std::string(p + "pre_feedforward_layernorm.weight").c_str());
        lw.pre_ffn_ln_2  = upload(sf.get(p + "pre_feedforward_layernorm_2.weight"), ql, std::string(p + "pre_feedforward_layernorm_2.weight").c_str());
        lw.post_ffn_ln_1 = upload(sf.get(p + "post_feedforward_layernorm_1.weight"), ql, std::string(p + "post_feedforward_layernorm_1.weight").c_str());
        lw.post_ffn_ln_2 = upload(sf.get(p + "post_feedforward_layernorm_2.weight"), ql, std::string(p + "post_feedforward_layernorm_2.weight").c_str());
        lw.post_ffn_ln   = upload(sf.get(p + "post_feedforward_layernorm.weight"), ql, std::string(p + "post_feedforward_layernorm.weight").c_str());
        lw.dec_layer_scalar = scalar_as_float(sf.get(p + "layer_scalar"));
        lw.enc_layer_scalar = scalar_as_float(
            sf.get("model.encoder.language_model.layers." + std::to_string(l) + ".layer_scalar"));

        // Dense shared MLP. The AWQ checkpoint deliberately leaves this shared
        // path in F16, but it is still part of every INT4 denoiser layer.  Its
        // gate/up weights can be concatenated at load so one same-input GEMM
        // produces both operands (A/B: DIFF_INT4_FUSE_DENSE_GATE_UP).
        bool mlp_nvfp4 = sf.has(p + "mlp.gate_proj.weight_packed") &&
                         sf.get(p + "mlp.gate_proj.weight_packed").dtype != "I32";
        if (mlp_nvfp4) {
            lw.mlp.gate_up_proj_fp4 = upload_nvfp4_linear_pair(
                sf, p + "mlp.gate_proj", p + "mlp.up_proj", ql);
            lw.mlp.down_proj = upload_linear_weight(sf, p + "mlp.down_proj", ql);
        } else {
            bool fuse_awq_gate_up = cfg.is_int4_quantized() &&
                                    diff_int4_fuse_dense_gate_up_enabled() &&
                                    sf.has(p + "mlp.gate_proj.weight") &&
                                    sf.has(p + "mlp.up_proj.weight");
            if (fuse_awq_gate_up) {
                lw.mlp.gate_up_proj_bf16 = upload_bf16_pair(
                    sf.get(p + "mlp.gate_proj.weight"),
                    sf.get(p + "mlp.up_proj.weight"), ql,
                    (p + "mlp.{gate,up}_proj.weight").c_str());
            } else {
                lw.mlp.gate_proj = upload_linear_weight(sf, p + "mlp.gate_proj", ql);
                lw.mlp.up_proj   = upload_linear_weight(sf, p + "mlp.up_proj", ql);
            }
            lw.mlp.down_proj = upload_linear_weight(sf, p + "mlp.down_proj", ql);
        }

        // MoE router is local to the layer owner. Expert weights are sharded
        // across all available GPUs by contiguous expert ranges.
        lw.moe.router_proj      = upload_linear_weight(sf, p + "router.proj", ql);
        lw.moe.router_scale     = upload(sf.get(p + "router.scale"), ql, std::string(p + "router.scale").c_str());
        lw.moe.per_expert_scale = host_floats(sf.get(p + "router.per_expert_scale"));
        lw.moe.per_expert_scale_dev = GpuBuffer<float>(lw.moe.per_expert_scale.size(), ql);
        lw.moe.per_expert_scale_dev.upload(lw.moe.per_expert_scale.data(), lw.moe.per_expert_scale.size());
        {
            int E = cfg.text.num_experts;
            DiffExpertPlacementMode resolved_experts = resolve_expert_placement(expert_mode);
            int G = (resolved_experts == DiffExpertPlacementMode::Shard) ? GpuEngine::count() : 1;
            bool experts_q8     = sf.has(p + "experts.gate_up_proj") &&
                sf.get(p + "experts.gate_up_proj").dtype == "Q8_0";
            bool experts_packed = sf.has(p + "experts.0.gate_proj.weight_packed");
            bool experts_int4   = experts_packed &&
                sf.get(p + "experts.0.gate_proj.weight_packed").dtype == "I32";
            bool experts_nvfp4  = experts_packed && !experts_int4;
            lw.moe.expert_shards.reserve(G);
            for (int eg = 0; eg < G; ++eg) {
                int first = (resolved_experts == DiffExpertPlacementMode::Shard) ? eg * E / G : 0;
                int last  = (resolved_experts == DiffExpertPlacementMode::Shard) ? (eg + 1) * E / G : E;
                if (first == last) continue;
                int shard_gpu = (resolved_experts == DiffExpertPlacementMode::Shard) ? eg : gpu;
                auto& eq = GpuEngine::get(shard_gpu).queue;
                DiffExpertShard shard;
                shard.gpu = shard_gpu;
                shard.first_expert = first;
                shard.num_experts = last - first;
                shard.nvfp4 = experts_nvfp4;
                shard.int4 = experts_int4;
                shard.q8 = experts_q8;
                if (experts_q8) {
                    const TensorView& gate_up = sf.get(p + "experts.gate_up_proj");
                    const TensorView& down = sf.get(p + "experts.down_proj");
                    shard.gate_up_proj_q8_batch = upload_q8_batched_slice(
                        gate_up, first, last, eq, (p + "experts.gate_up_proj").c_str());
                    shard.down_proj_q8_batch = upload_q8_batched_slice(
                        down, first, last, eq, (p + "experts.down_proj").c_str());
                } else if (experts_int4) {
                    shard.gate_up_proj_int4.reserve(shard.num_experts);
                    shard.down_proj_int4.reserve(shard.num_experts);
                    for (int e = first; e < last; ++e) {
                        std::string ep = p + "experts." + std::to_string(e) + ".";
                        shard.gate_up_proj_int4.push_back(upload_int4_linear_pair(
                            sf, ep + "gate_proj", ep + "up_proj", eq));
                        shard.down_proj_int4.push_back(upload_int4_linear(sf, ep + "down_proj", eq));
                    }
                    // Pointer tables are immutable model state.  Build them
                    // once so DIFF_INT4_GROUPED_DPAS_MOE has no per-step host
                    // address-vector upload before launching its grouped Xe2
                    // DPAS kernels.
                    ensure_int4_expert_pointer_tables(shard, GpuEngine::get(shard_gpu));
                } else if (experts_nvfp4) {
                    shard.gate_up_proj_fp4.reserve(shard.num_experts);
                    shard.down_proj_fp4.reserve(shard.num_experts);
                    for (int e = first; e < last; ++e) {
                        std::string ep = p + "experts." + std::to_string(e) + ".";
                        shard.gate_up_proj_fp4.push_back(upload_nvfp4_linear_pair(
                            sf, ep + "gate_proj", ep + "up_proj", eq));
                        shard.down_proj_fp4.push_back(upload_nvfp4_linear(sf, ep + "down_proj", eq));
                    }
                    // Build the persistent raw-weight pointer tables once, at
                    // load (no session is active), so the per-step host->device
                    // upload the gpu-layout MoE path used to do is gone and the
                    // path is SYCL-graph-capturable. See DiffExpertShard::pt_*.
                    ensure_expert_pointer_tables_raw(shard, GpuEngine::get(shard_gpu));
                    // Build the coalesced (xe2 DPAS) pointer tables too: pre-warms
                    // nvfp4_coalesced_weight + the dequant LUT (one-time waits that
                    // cannot happen inside a session) and caches the coalesced
                    // weight ptrs. Makes the xe2 DPAS path capturable.
                    ensure_expert_pointer_tables_coalesced(shard, GpuEngine::get(shard_gpu),
                                                            cfg.text.moe_intermediate_size,
                                                            cfg.text.hidden_size);
                } else {
                    const TensorView& gate_up = sf.get(p + "experts.gate_up_proj");
                    const TensorView& down    = sf.get(p + "experts.down_proj");
                    size_t gate_stride = (size_t)2 * cfg.text.moe_intermediate_size * cfg.text.hidden_size;
                    size_t down_stride = (size_t)cfg.text.hidden_size * cfg.text.moe_intermediate_size;
                    shard.gate_up_proj = upload_bf16_slice(gate_up, (size_t)first * gate_stride,
                        (size_t)shard.num_experts * gate_stride, eq, std::string(p + "experts.gate_up_proj").c_str());
                    shard.down_proj = upload_bf16_slice(down, (size_t)first * down_stride,
                        (size_t)shard.num_experts * down_stride, eq, std::string(p + "experts.down_proj").c_str());
                }
                lw.moe.expert_shards.push_back(std::move(shard));
            }
        }

        // Attention
        std::string a = p + "self_attn.";
        if (!lw.is_full) {
            DiffSlidingAttn s;
            s.q_proj = upload_linear_weight(sf, a + "q_proj", ql);
            s.k_proj = upload_linear_weight(sf, a + "k_proj", ql);
            s.v_proj = upload_linear_weight(sf, a + "v_proj", ql);
            if (s.q_proj.is_int4() && s.k_proj.is_int4() && s.v_proj.is_int4()) {
                s.qkv_proj_int4 = upload_int4_linear_concat(
                    sf, {a + "q_proj", a + "k_proj", a + "v_proj"}, ql,
                    (a + "{q,k,v}_proj").c_str());
            }
            s.o_proj = upload_linear_weight(sf, a + "o_proj", ql);
            s.q_norm = upload(sf.get(a + "q_norm.weight"), ql, std::string(a + "q_norm.weight").c_str());
            s.k_norm = upload(sf.get(a + "k_norm.weight"), ql, std::string(a + "k_norm.weight").c_str());
            lw.attn = std::move(s);
        } else {
            DiffFullAttn fa;
            fa.q_proj = upload_linear_weight(sf, a + "q_proj", ql);
            fa.k_proj = upload_linear_weight(sf, a + "k_proj", ql);
            if (fa.q_proj.is_int4() && fa.k_proj.is_int4()) {
                fa.qk_proj_int4 = upload_int4_linear_concat(
                    sf, {a + "q_proj", a + "k_proj"}, ql,
                    (a + "{q,k}_proj").c_str());
            }
            fa.o_proj = upload_linear_weight(sf, a + "o_proj", ql);
            fa.q_norm = upload(sf.get(a + "q_norm.weight"), ql, std::string(a + "q_norm.weight").c_str());
            fa.k_norm = upload(sf.get(a + "k_norm.weight"), ql, std::string(a + "k_norm.weight").c_str());
            lw.attn = std::move(fa);
        }

        if (l % 5 == 0)
            std::printf("[load] layer %d/%d (%s) -> GPU %d\n",
                        l, cfg.text.num_hidden_layers,
                        lw.is_full ? "full" : "sliding", gpu);
    }
    std::printf("[load] done (split at layer %d)\n", split_layer);
    return gw;
}

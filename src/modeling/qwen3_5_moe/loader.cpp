#include "loader.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const char* kPrefix = "model.language_model.";

float f16_bits_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x03ffu;
            out = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

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

// ---------------------------------------------------------------------------
// compressed-tensors "pack-quantized" int4 W4A16 (AWQ) routed experts.
//   weight_packed     I32 [N, K/8]   8 unsigned nibbles per word along K
//                                    (low nibble first), stored q_u = q + 8
//   weight_scale      F16 [N, G]     per-group scales (G = K / group_size)
//   weight_zero_point I32 [N/8, G]   8 unsigned nibbles per word packed along
//                                    the OUTPUT dim N (zp_u = zp + 8)
//   weight_shape      I64 [2]        original [N, K] (cross-check only)
// True dequant: w = scale * (q_u - zp_u). The packed weights are rebased to
// two's-complement s4 (XOR 0x88, i.e. q_u - 8) for oneDNN; the residual
// scale * (zp_u - 8) is folded into Int4Linear::zp_offset (see int4.hpp).
// ---------------------------------------------------------------------------
GpuBuffer<uint8_t> upload_int4_packed_rebased(const void* a_data, size_t a_bytes,
                                              const void* b_data, size_t b_bytes,
                                              sycl::queue& q) {
    size_t total = a_bytes + b_bytes;
    static std::vector<uint8_t> staging;
    if (staging.size() < total) staging.resize(total);
    std::memcpy(staging.data(), a_data, a_bytes);
    if (b_bytes) std::memcpy(staging.data() + a_bytes, b_data, b_bytes);

    GpuBuffer<uint8_t> buf(total, q);
    uint8_t* dst = buf.data();
    sycl::event copy_done = q.memcpy(buf.data(), staging.data(), total);
    q.submit([&](sycl::handler& h) {
        h.depends_on(copy_done);
        h.parallel_for(sycl::range<1>(total),
                       [=](sycl::id<1> id) { dst[id[0]] ^= 0x88; });
    }).wait();
    return buf;
}

// Host-side per-expert AWQ parameters for one or two (fused) projections.
// Produces oneDNN-layout scales (G, N) BF16 and the zp correction (G, N) BF16.
struct AwqHostParams {
    std::vector<bf16> scales_t;   // [G, N]
    std::vector<bf16> zp_offset;  // [G, N], empty if symmetric (all zp_u == 8)
};

// Read one projection's scale (F16/BF16 [N, G]) into floats, row-major [N, G].
std::vector<float> read_scale_f32(const TensorView& scale, int N, int G,
                                  const std::string& name) {
    if (scale.dtype != "F16" && scale.dtype != "BF16")
        throw std::runtime_error("Expected F16/BF16 int4 scale for " + name +
                                 ", got " + scale.dtype);
    if (scale.shape.size() != 2 || scale.shape[0] != N || scale.shape[1] != G)
        throw std::runtime_error("Unexpected int4 scale shape for " + name);
    std::vector<float> out((size_t)N * G);
    const uint16_t* src = static_cast<const uint16_t*>(scale.data);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = (scale.dtype == "F16") ? f16_bits_to_float(src[i])
                                        : bf16_to_float(src[i]);
    return out;
}

// Read one projection's packed zero-point (I32 [N/8, G], packed along N) into
// unsigned nibbles, row-major [N, G].
std::vector<uint8_t> read_zp_u4(const TensorView& zp, int N, int G,
                                const std::string& name) {
    if (zp.dtype != "I32" || zp.shape.size() != 2 || zp.shape[0] != N / 8 ||
        zp.shape[1] != G)
        throw std::runtime_error("Unexpected int4 zero_point shape for " + name);
    const int32_t* words = static_cast<const int32_t*>(zp.data);
    std::vector<uint8_t> out((size_t)N * G);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g)
            out[(size_t)n * G + g] =
                (uint8_t)(((uint32_t)words[(size_t)(n / 8) * G + g] >> (4 * (n % 8))) & 0xF);
    }
    return out;
}

AwqHostParams build_awq_host_params(const std::vector<std::vector<float>>& scales_f,
                                    const std::vector<std::vector<uint8_t>>& zps_u,
                                    int N_half, int G) {
    // N rows per part, concatenated along the output dim.
    int parts = (int)scales_f.size();
    int N = N_half * parts;
    AwqHostParams p;
    p.scales_t.resize((size_t)G * N);
    p.zp_offset.resize((size_t)G * N);
    bool any_asym = false;
    for (int g = 0; g < G; ++g) {
        for (int n = 0; n < N; ++n) {
            int part = n / N_half, ln = n % N_half;
            float s = scales_f[part][(size_t)ln * G + g];
            int zp  = zps_u[part][(size_t)ln * G + g];
            p.scales_t[(size_t)g * N + n] = float_to_bf16(s);
            if (zp != 8) any_asym = true;
            p.zp_offset[(size_t)g * N + n] = float_to_bf16(s * (float)(zp - 8));
        }
    }
    if (!any_asym) p.zp_offset.clear();
    return p;
}

GpuBuffer<bf16> upload_bf16(const std::vector<bf16>& host, sycl::queue& q) {
    GpuBuffer<bf16> buf(host.size(), q);
    buf.upload(host.data(), host.size());
    return buf;
}

struct AwqTensors {
    const TensorView* packed;
    const TensorView* scale;
    const TensorView* zp;
    int N = 0, K = 0, G = 0;
};

AwqTensors read_awq_tensors(const TensorSource& sf, const std::string& prefix) {
    AwqTensors t;
    t.packed = &sf.get(prefix + ".weight_packed");
    t.scale  = &sf.get(prefix + ".weight_scale");
    t.zp     = &sf.get(prefix + ".weight_zero_point");
    if (t.packed->dtype != "I32" || t.packed->shape.size() != 2)
        throw std::runtime_error("Expected 2D I32 packed weight: " + prefix);
    t.N = (int)t.packed->shape[0];
    t.K = (int)t.packed->shape[1] * 8;
    t.G = (int)t.scale->shape[1];
    if (t.G == 0 || t.K % t.G != 0)
        throw std::runtime_error("int4 K not divisible by groups: " + prefix);
    if (sf.has(prefix + ".weight_shape")) {
        const TensorView& ws = sf.get(prefix + ".weight_shape");
        if (ws.dtype == "I64" && ws.numel() == 2) {
            const int64_t* s = static_cast<const int64_t*>(ws.data);
            if (s[0] != t.N || s[1] != t.K)
                throw std::runtime_error("weight_shape mismatch: " + prefix);
        }
    }
    return t;
}

Int4Linear upload_int4_linear_awq(const TensorSource& sf, const std::string& prefix,
                                  sycl::queue& q) {
    AwqTensors t = read_awq_tensors(sf, prefix);
    Int4Linear lin;
    lin.out_features = t.N;
    lin.in_features  = t.K;
    lin.group_size   = t.K / t.G;
    lin.weight_packed =
        upload_int4_packed_rebased(t.packed->data, t.packed->nbytes, nullptr, 0, q);
    auto scales_f = read_scale_f32(*t.scale, t.N, t.G, prefix + ".weight_scale");
    auto zps_u    = read_zp_u4(*t.zp, t.N, t.G, prefix + ".weight_zero_point");
    AwqHostParams p = build_awq_host_params({scales_f}, {zps_u}, t.N, t.G);
    lin.weight_scale = upload_bf16(p.scales_t, q);
    if (!p.zp_offset.empty()) lin.zp_offset = upload_bf16(p.zp_offset, q);
    return lin;
}

// Fused gate+up AWQ: packed rows concatenate directly ([N,K/8] row-major);
// scales and zp offsets concatenate along the output dim to match
// (gate in outputs [0,N), up in [N,2N)).
Int4Linear upload_int4_linear_pair_awq(const TensorSource& sf,
                                       const std::string& gate_prefix,
                                       const std::string& up_prefix,
                                       sycl::queue& q) {
    AwqTensors g = read_awq_tensors(sf, gate_prefix);
    AwqTensors u = read_awq_tensors(sf, up_prefix);
    if (g.K != u.K || g.G != u.G || g.N != u.N)
        throw std::runtime_error("AWQ pair shape mismatch: " + gate_prefix + " / " +
                                 up_prefix);
    Int4Linear lin;
    lin.out_features = 2 * g.N;
    lin.in_features  = g.K;
    lin.group_size   = g.K / g.G;
    lin.weight_packed = upload_int4_packed_rebased(g.packed->data, g.packed->nbytes,
                                                   u.packed->data, u.packed->nbytes, q);
    auto gs = read_scale_f32(*g.scale, g.N, g.G, gate_prefix + ".weight_scale");
    auto us = read_scale_f32(*u.scale, u.N, u.G, up_prefix + ".weight_scale");
    auto gz = read_zp_u4(*g.zp, g.N, g.G, gate_prefix + ".weight_zero_point");
    auto uz = read_zp_u4(*u.zp, u.N, u.G, up_prefix + ".weight_zero_point");
    AwqHostParams p = build_awq_host_params({gs, us}, {gz, uz}, g.N, g.G);
    lin.weight_scale = upload_bf16(p.scales_t, q);
    if (!p.zp_offset.empty()) lin.zp_offset = upload_bf16(p.zp_offset, q);
    return lin;
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
        for (int e = 0; e < cfg.num_experts; ++e) {
            const std::string ep = mp + "experts." + std::to_string(e) + ".";
            if (awq_int4) {
                m.experts_gate_up.push_back(
                    upload_int4_linear_pair_awq(sf, ep + "gate_proj", ep + "up_proj", q));
                m.experts_down.push_back(upload_int4_linear_awq(sf, ep + "down_proj", q));
            } else {
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

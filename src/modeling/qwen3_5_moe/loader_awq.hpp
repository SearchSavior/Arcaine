#pragma once
// AWQ (compressed-tensors "pack-quantized" int4 W4A16) upload helpers for the
// Qwen3.5-MoE routed experts. Hoisted out of loader.cpp so the kernel bench
// (benchmarks/int4_moe_bench.cpp) can load real expert weights through the
// exact same codepath.
//
//   weight_packed     I32 [N, K/8]   8 unsigned nibbles per word along K
//                                    (low nibble first), stored q_u = q + 8
//   weight_scale      F16 [N, G]     per-group scales (G = K / group_size)
//   weight_zero_point I32 [N/8, G]   8 unsigned nibbles per word packed along
//                                    the OUTPUT dim N (zp_u = zp + 8)
//   weight_shape      I64 [2]        original [N, K] (cross-check only)
// True dequant: w = scale * (q_u - zp_u). The canonical device storage keeps
// the checkpoint's original unsigned nibbles and zero points verbatim (raw
// u4, no XOR-0x88 rebase, no zp_offset precomputation) as one contiguous
// tensor per projection across all experts — directly consumable by oneDNN's
// grouped W4A16 matmul with native u4 zero points and by the custom DPAS
// fallback (see kernels/int4_grouped_onednn.hpp / int4_grouped_moe.hpp).

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../runtime/quantization/quant_loader.hpp" // TensorSource/TensorView
#include "weights.hpp"                                 // QwenInt4ExpertsGrouped

namespace qwen_awq {

inline float f16_bits_to_float(uint16_t h) {
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

// Read one projection's scale (F16/BF16 [N, G]) into floats, row-major [N, G].
inline std::vector<float> read_scale_f32(const TensorView& scale, int N, int G,
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
inline std::vector<uint8_t> read_zp_u4(const TensorView& zp, int N, int G,
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

inline GpuBuffer<bf16> upload_bf16(const std::vector<bf16>& host, sycl::queue& q) {
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

inline AwqTensors read_tensors(const TensorSource& sf, const std::string& prefix) {
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

// Host staging for one projection across all experts (canonical raw layout).
struct GroupedProjHost {
    std::vector<uint8_t> q;       // [E][N][K/2]   raw packed u4, K innermost
    std::vector<bf16>    scales;  // [E][G][N]     BF16
    std::vector<uint8_t> zp;      // [E][G][N/2]   raw u4 packed along N
    bool any_asym = false;
    int N = 0, K = 0, G = 0;
};

// Append one expert's projection (prefix.gate_proj style) to the staging.
inline void append_expert_proj(const TensorSource& sf, const std::string& prefix,
                               GroupedProjHost& p) {
    AwqTensors t = read_tensors(sf, prefix);
    if (p.N == 0) {
        p.N = t.N; p.K = t.K; p.G = t.G;
    } else if (p.N != t.N || p.K != t.K || p.G != t.G) {
        throw std::runtime_error("AWQ expert shape mismatch: " + prefix);
    }
    const size_t q_bytes = (size_t)t.N * t.K / 2;
    const uint8_t* q_src = static_cast<const uint8_t*>(t.packed->data);
    p.q.insert(p.q.end(), q_src, q_src + q_bytes);

    auto scales_f = read_scale_f32(*t.scale, t.N, t.G, prefix + ".weight_scale");
    auto zps_u    = read_zp_u4(*t.zp, t.N, t.G, prefix + ".weight_zero_point");
    size_t base_s = p.scales.size();
    size_t base_z = p.zp.size();
    p.scales.resize(base_s + (size_t)t.G * t.N);
    p.zp.resize(base_z + (size_t)t.G * (t.N / 2));
    for (int g = 0; g < t.G; ++g) {
        for (int n = 0; n < t.N; ++n) {
            p.scales[base_s + (size_t)g * t.N + n] =
                float_to_bf16(scales_f[(size_t)n * t.G + g]);
            int z = zps_u[(size_t)n * t.G + g];
            if (z != 8) p.any_asym = true;
            uint8_t& byte = p.zp[base_z + (size_t)g * (t.N / 2) + n / 2];
            if (n & 1) byte |= uint8_t(z << 4);
            else       byte  = uint8_t(z);
        }
    }
}

// Load all routed experts of one MoE block into canonical contiguous storage:
// gate/up/down as separate [E, N, K]-logical raw u4 tensors + scales + zero
// points (dropped when fully symmetric).
inline QwenInt4ExpertsGrouped upload_experts_grouped(const TensorSource& sf,
                                                     const std::string& moe_prefix,
                                                     int E, sycl::queue& q) {
    GroupedProjHost gate, up, down;
    for (int e = 0; e < E; ++e) {
        const std::string ep = moe_prefix + "experts." + std::to_string(e) + ".";
        append_expert_proj(sf, ep + "gate_proj", gate);
        append_expert_proj(sf, ep + "up_proj", up);
        append_expert_proj(sf, ep + "down_proj", down);
    }
    if (gate.K != up.K || gate.N != up.N || gate.G != up.G || down.N == 0)
        throw std::runtime_error("AWQ expert projection mismatch: " + moe_prefix);

    QwenInt4ExpertsGrouped g;
    g.E = E;
    g.hidden = gate.K;
    g.inter = gate.N;
    g.group_size = gate.K / gate.G;
    if (down.K != gate.N || down.G != gate.N / g.group_size)
        throw std::runtime_error("AWQ down_proj group mismatch: " + moe_prefix);

    auto up_u8 = [&](const std::vector<uint8_t>& h) {
        GpuBuffer<uint8_t> b(h.size(), q);
        b.upload(h.data(), h.size());
        return b;
    };
    g.gate_q = up_u8(gate.q);
    g.up_q   = up_u8(up.q);
    g.down_q = up_u8(down.q);
    g.gate_s = upload_bf16(gate.scales, q);
    g.up_s   = upload_bf16(up.scales, q);
    g.down_s = upload_bf16(down.scales, q);
    if (gate.any_asym) g.gate_zp = up_u8(gate.zp);
    if (up.any_asym)   g.up_zp   = up_u8(up.zp);
    if (down.any_asym) g.down_zp = up_u8(down.zp);
    return g;
}

}  // namespace qwen_awq

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
// True dequant: w = scale * (q_u - zp_u). The packed weights are rebased to
// two's-complement s4 (XOR 0x88, i.e. q_u - 8) for oneDNN; the residual
// scale * (zp_u - 8) is folded into Int4Linear::zp_offset (see int4.hpp).

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../runtime/quantization/int4.hpp"         // Int4Linear, bf16 helpers
#include "../../runtime/quantization/quant_loader.hpp" // TensorSource/TensorView

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

inline GpuBuffer<uint8_t> upload_packed_rebased(const void* a_data, size_t a_bytes,
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
struct HostParams {
    std::vector<bf16> scales_t;   // [G, N]
    std::vector<bf16> zp_offset;  // [G, N], empty if symmetric (all zp_u == 8)
};

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

inline HostParams build_host_params(const std::vector<std::vector<float>>& scales_f,
                                    const std::vector<std::vector<uint8_t>>& zps_u,
                                    int N_half, int G) {
    // N rows per part, concatenated along the output dim.
    int parts = (int)scales_f.size();
    int N = N_half * parts;
    HostParams p;
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

inline Int4Linear upload_linear(const TensorSource& sf, const std::string& prefix,
                                sycl::queue& q) {
    AwqTensors t = read_tensors(sf, prefix);
    Int4Linear lin;
    lin.out_features = t.N;
    lin.in_features  = t.K;
    lin.group_size   = t.K / t.G;
    lin.weight_packed =
        upload_packed_rebased(t.packed->data, t.packed->nbytes, nullptr, 0, q);
    auto scales_f = read_scale_f32(*t.scale, t.N, t.G, prefix + ".weight_scale");
    auto zps_u    = read_zp_u4(*t.zp, t.N, t.G, prefix + ".weight_zero_point");
    HostParams p = build_host_params({scales_f}, {zps_u}, t.N, t.G);
    lin.weight_scale = upload_bf16(p.scales_t, q);
    if (!p.zp_offset.empty()) lin.zp_offset = upload_bf16(p.zp_offset, q);
    return lin;
}

// Fused gate+up AWQ: packed rows concatenate directly ([N,K/8] row-major);
// scales and zp offsets concatenate along the output dim to match
// (gate in outputs [0,N), up in [N,2N)).
inline Int4Linear upload_linear_pair(const TensorSource& sf,
                                     const std::string& gate_prefix,
                                     const std::string& up_prefix,
                                     sycl::queue& q) {
    AwqTensors g = read_tensors(sf, gate_prefix);
    AwqTensors u = read_tensors(sf, up_prefix);
    if (g.K != u.K || g.G != u.G || g.N != u.N)
        throw std::runtime_error("AWQ pair shape mismatch: " + gate_prefix + " / " +
                                 up_prefix);
    Int4Linear lin;
    lin.out_features = 2 * g.N;
    lin.in_features  = g.K;
    lin.group_size   = g.K / g.G;
    lin.weight_packed = upload_packed_rebased(g.packed->data, g.packed->nbytes,
                                              u.packed->data, u.packed->nbytes, q);
    auto gs = read_scale_f32(*g.scale, g.N, g.G, gate_prefix + ".weight_scale");
    auto us = read_scale_f32(*u.scale, u.N, u.G, up_prefix + ".weight_scale");
    auto gz = read_zp_u4(*g.zp, g.N, g.G, gate_prefix + ".weight_zero_point");
    auto uz = read_zp_u4(*u.zp, u.N, u.G, up_prefix + ".weight_zero_point");
    HostParams p = build_host_params({gs, us}, {gz, uz}, g.N, g.G);
    lin.weight_scale = upload_bf16(p.scales_t, q);
    if (!p.zp_offset.empty()) lin.zp_offset = upload_bf16(p.zp_offset, q);
    return lin;
}

}  // namespace qwen_awq

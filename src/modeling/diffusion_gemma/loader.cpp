#include "loader.hpp"
#include "../../common/gpu/placement.hpp"
#include "../../common/gpu/expert_parallel.hpp"
#include "../../common/io/quant_loader.hpp"
#include "../../common/io/safetensors.hpp"
#include "../../common/io/gguf.hpp"
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




static GpuBuffer<bf16> upload_bf16_pair(const TensorView& a, const TensorView& b,
                                         sycl::queue& q, const char* name = "?") {
    if (a.shape != b.shape)
        throw std::runtime_error(std::string("pair shape mismatch for ") + name);
    size_t n = a.numel();
    GpuBuffer<bf16> aa = upload(a, q, name);
    GpuBuffer<bf16> bb = upload(b, q, name);
    std::vector<bf16> staging(2 * n);
    aa.download(staging.data(), n);
    bb.download(staging.data() + n, n);
    GpuBuffer<bf16> out(2 * n, q);
    out.upload(staging.data(), 2 * n);
    return out;
}

static GpuBuffer<bf16> upload_bf16_slice(const TensorView& tv, size_t offset, size_t n,
                                         sycl::queue& q, const char* name = "?") {
    if (tv.dtype != "BF16") throw std::runtime_error("Expected BF16 slice, got " + tv.dtype);
    if (offset + n > tv.numel()) throw std::runtime_error(std::string("slice exceeds tensor: ") + name);
    const bf16* src = static_cast<const bf16*>(tv.data) + offset;
    GpuBuffer<bf16> buf(n, q);
    buf.upload(src, n);
    return buf;
}



static float f16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
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
// compressed-tensors int4 W4A16 ("pack-quantized"): weight_packed is I32 with
// 8 signed int4 per word along K (low-nibble first), so its raw byte stream is
// already oneDNN s4 tag::ba for logical dims (K, N). weight_scale is BF16 or
// F16 (out, K/group_size) and is transposed to (groups, out) for oneDNN; F16
// scales are normalized to BF16 at load (see as_bf16_scale_bits).
// ---------------------------------------------------------------------------
// compressed-tensors int4 checkpoints may store the per-group weight scales as
// F16 instead of BF16 (the model dtype is float16).  oneDNN's s4 weight-scale
// path consumes BF16 scales, so normalize F16 -> BF16 (round-to-nearest-even
// via float) into a host staging buffer and return BF16 bits; BF16 scales are
// returned as-is.  The returned pointer is valid for the lifetime of `storage`.
static const uint16_t* as_bf16_scale_bits(const TensorView& tv,
                                          std::vector<uint16_t>& storage,
                                          const char* name) {
    if (tv.dtype != "BF16" && tv.dtype != "F16")
        throw std::runtime_error(std::string("Expected BF16/F16 int4 scale for ") + name + ", got " + tv.dtype);
    const uint16_t* src = static_cast<const uint16_t*>(tv.data);
    if (tv.dtype == "BF16") return src;
    size_t n = tv.numel();
    storage.resize(n);
    for (size_t i = 0; i < n; ++i) storage[i] = float_to_bf16(f16_to_float(src[i]));
    return storage.data();
}

static GpuBuffer<bf16> upload_int4_scales_transposed(
    const TensorView& tv, int out_features, int groups, sycl::queue& q,
    const char* name = "?") {
    if (tv.dtype != "BF16" && tv.dtype != "F16")
        throw std::runtime_error(std::string("Expected BF16/F16 int4 scale for ") + name + ", got " + tv.dtype);
    if (tv.shape.size() != 2 || tv.shape[0] != out_features || tv.shape[1] != groups)
        throw std::runtime_error(std::string("Unexpected int4 scale shape for ") + name);

    std::vector<uint16_t> scale_storage;
    const uint16_t* src = as_bf16_scale_bits(tv, scale_storage, name);  // (out, groups) BF16 bits
    static bool gpu_transpose = [] {
        const char* e = std::getenv("DIFF_INT4_GPU_SCALE_TRANSPOSE");
        return e && std::strcmp(e, "0") != 0 && std::strcmp(e, "false") != 0 &&
               std::strcmp(e, "FALSE") != 0 && std::strcmp(e, "off") != 0 &&
               std::strcmp(e, "OFF") != 0 && std::strcmp(e, "no") != 0 &&
               std::strcmp(e, "NO") != 0;
    }();
    if (gpu_transpose) {
        size_t count = (size_t)groups * out_features;
        static std::vector<bf16> staging;
        if (staging.size() < count) staging.resize(count);
        std::memcpy(staging.data(), src, count * sizeof(bf16));

        GpuBuffer<bf16> raw(count, q);
        GpuBuffer<bf16> transposed(count, q);
        const bf16* raw_ptr = raw.data();
        bf16* out_ptr = transposed.data();
        sycl::event copy_done = q.memcpy(raw.data(), staging.data(), count * sizeof(bf16));
        q.submit([&](sycl::handler& h) {
            h.depends_on(copy_done);
            h.parallel_for(sycl::range<2>((size_t)out_features, (size_t)groups),
                           [=](sycl::id<2> id) {
                size_t n = id[0];
                size_t g = id[1];
                out_ptr[g * (size_t)out_features + n] = raw_ptr[n * (size_t)groups + g];
            });
        }).wait();
        return transposed;
    }

    std::vector<uint16_t> transposed((size_t)groups * out_features);
    for (int n = 0; n < out_features; ++n)
        for (int g = 0; g < groups; ++g)
            transposed[(size_t)g * out_features + n] = src[(size_t)n * groups + g];

    GpuBuffer<bf16> buf(transposed.size(), q);
    buf.upload(transposed.data(), transposed.size());
    return buf;
}

struct ByteSpan {
    const void* data;
    size_t bytes;
};

static GpuBuffer<uint8_t> upload_int4_packed_rebased(
    const std::vector<ByteSpan>& spans, size_t total_bytes, sycl::queue& q) {
    static std::vector<uint8_t> staging;
    if (staging.size() < total_bytes) staging.resize(total_bytes);

    size_t off = 0;
    for (const ByteSpan& span : spans) {
        std::memcpy(staging.data() + off, span.data, span.bytes);
        off += span.bytes;
    }
    if (off != total_bytes)
        throw std::runtime_error("int4 packed upload byte count mismatch");

    GpuBuffer<uint8_t> buf(total_bytes, q);
    uint8_t* dst = buf.data();
    sycl::event copy_done = q.memcpy(buf.data(), staging.data(), total_bytes);
    q.submit([&](sycl::handler& h) {
        h.depends_on(copy_done);
        h.parallel_for(sycl::range<1>(total_bytes), [=](sycl::id<1> id) {
            dst[id[0]] ^= 0x88;
        });
    }).wait();
    return buf;
}

static GpuBuffer<uint8_t> upload_int4_packed_rebased(
    const void* data, size_t bytes, sycl::queue& q) {
    return upload_int4_packed_rebased(std::vector<ByteSpan>{{data, bytes}}, bytes, q);
}

static Int4Linear upload_int4_linear(const TensorSource& sf,
                                     const std::string& prefix,
                                     sycl::queue& q) {
    const TensorView& packed = sf.get(prefix + ".weight_packed");
    if (packed.dtype != "I32") throw std::runtime_error("Expected I32 packed weight: " + prefix);
    if (packed.shape.size() != 2) throw std::runtime_error("Expected 2D packed weight: " + prefix);

    Int4Linear lin;
    lin.out_features = (int)packed.shape[0];
    lin.in_features  = (int)packed.shape[1] * 8;   // 8 int4 per int32 along K

    const TensorView& scale = sf.get(prefix + ".weight_scale");
    if (scale.shape.size() != 2) throw std::runtime_error("Expected 2D int4 scale: " + prefix);
    int groups = (int)scale.shape[1];
    if (groups == 0 || lin.in_features % groups != 0)
        throw std::runtime_error("int4 in_features not divisible by groups: " + prefix);
    lin.group_size = lin.in_features / groups;

    // compressed-tensors stores symmetric int4 as UNSIGNED nibbles with an
    // implicit zero-point of 8 (dequant = (nibble - 8) * scale). XOR each
    // nibble's sign bit (byte ^ 0x88) to turn it into two's-complement s4, so
    // oneDNN's s4 path consumes it directly with no zero-point argument.
    lin.weight_packed = upload_int4_packed_rebased(packed.data, packed.nbytes, q);
    lin.weight_scale = upload_int4_scales_transposed(
        scale, lin.out_features, groups, q, (prefix + ".weight_scale").c_str());
    return lin;
}

// Fuse attention projection weights along the output dimension. The packed
// bytes stay in oneDNN's raw s4 tag::ba layout: rows are output channels, so a
// plain concatenation of q/k/v rows produces one larger (N_total, K) matrix.
static Int4Linear upload_int4_linear_concat(const TensorSource& sf,
                                            const std::vector<std::string>& prefixes,
                                            sycl::queue& q,
                                            const char* name) {
    if (prefixes.empty())
        throw std::runtime_error(std::string("empty int4 concat: ") + name);

    struct Part {
        const TensorView* packed;
        const TensorView* scale;
        int out_features;
    };
    std::vector<Part> parts;
    parts.reserve(prefixes.size());

    int packed_cols = -1;
    int groups = -1;
    int total_out = 0;
    for (const std::string& prefix : prefixes) {
        const TensorView& packed = sf.get(prefix + ".weight_packed");
        if (packed.dtype != "I32")
            throw std::runtime_error(std::string("Expected I32 packed weight for fused attention: ") + prefix);
        if (packed.shape.size() != 2)
            throw std::runtime_error(std::string("Expected 2D packed weight for fused attention: ") + prefix);

        const TensorView& scale = sf.get(prefix + ".weight_scale");
        if (scale.dtype != "BF16" && scale.dtype != "F16")
            throw std::runtime_error(std::string("Expected BF16/F16 int4 scale for fused attention: ") + prefix);
        if (scale.shape.size() != 2 || scale.shape[0] != packed.shape[0])
            throw std::runtime_error(std::string("Unexpected int4 scale shape for fused attention: ") + prefix);

        int pc = (int)packed.shape[1];
        int g = (int)scale.shape[1];
        if (packed_cols < 0) packed_cols = pc;
        if (groups < 0) groups = g;
        if (pc != packed_cols || g != groups)
            throw std::runtime_error(std::string("fused int4 attention projection shape mismatch: ") + name);

        int out = (int)packed.shape[0];
        parts.push_back({&packed, &scale, out});
        total_out += out;
    }

    Int4Linear lin;
    lin.out_features = total_out;
    lin.in_features = packed_cols * 8;
    if (groups == 0 || lin.in_features % groups != 0)
        throw std::runtime_error(std::string("fused int4 attention invalid group count: ") + name);
    lin.group_size = lin.in_features / groups;

    size_t row_bytes = (size_t)packed_cols * sizeof(int32_t);
    std::vector<ByteSpan> packed_spans;
    packed_spans.reserve(parts.size());
    size_t byte_off = 0;
    for (const Part& part : parts) {
        size_t bytes = (size_t)part.out_features * row_bytes;
        if (part.packed->nbytes != bytes)
            throw std::runtime_error(std::string("fused int4 attention packed byte mismatch: ") + name);
        packed_spans.push_back({part.packed->data, bytes});
        byte_off += bytes;
    }
    // u4 zero-point-8 -> two's-complement s4 (see upload_int4_linear).
    lin.weight_packed = upload_int4_packed_rebased(packed_spans, byte_off, q);

    std::vector<uint16_t> transposed((size_t)groups * total_out);
    std::vector<uint16_t> scale_storage;
    int out_off = 0;
    for (const Part& part : parts) {
        const uint16_t* src = as_bf16_scale_bits(*part.scale, scale_storage, name);
        for (int g = 0; g < groups; ++g)
            for (int n = 0; n < part.out_features; ++n)
                transposed[(size_t)g * total_out + out_off + n] =
                    src[(size_t)n * groups + g];
        out_off += part.out_features;
    }
    lin.weight_scale = GpuBuffer<bf16>(transposed.size(), q);
    lin.weight_scale.upload(transposed.data(), transposed.size());
    return lin;
}

// Fuse gate_proj and up_proj into one int4 weight with out_features = 2*half_out
// (gate block then up block), so the fused output feeds geglu directly.
static Int4Linear upload_int4_linear_pair(const TensorSource& sf,
                                          const std::string& gate_prefix,
                                          const std::string& up_prefix,
                                          sycl::queue& q) {
    const TensorView& gate_packed = sf.get(gate_prefix + ".weight_packed");
    const TensorView& up_packed   = sf.get(up_prefix + ".weight_packed");
    if (gate_packed.dtype != "I32" || up_packed.dtype != "I32")
        throw std::runtime_error("Expected I32 packed gate/up weights: " + gate_prefix);
    if (gate_packed.shape.size() != 2 || up_packed.shape.size() != 2 ||
        gate_packed.shape[0] != up_packed.shape[0] || gate_packed.shape[1] != up_packed.shape[1])
        throw std::runtime_error("int4 gate/up packed shapes differ: " + gate_prefix);

    Int4Linear lin;
    int half_out = (int)gate_packed.shape[0];
    lin.out_features = 2 * half_out;
    lin.in_features  = (int)gate_packed.shape[1] * 8;

    size_t half_bytes = gate_packed.nbytes;  // half_out * (in/2)
    // u4 zero-point-8 -> two's-complement s4 (see upload_int4_linear).
    lin.weight_packed = upload_int4_packed_rebased(
        std::vector<ByteSpan>{{gate_packed.data, half_bytes}, {up_packed.data, half_bytes}},
        2 * half_bytes, q);

    const TensorView& gate_scale = sf.get(gate_prefix + ".weight_scale");
    const TensorView& up_scale   = sf.get(up_prefix + ".weight_scale");
    if ((gate_scale.dtype != "BF16" && gate_scale.dtype != "F16") ||
        (up_scale.dtype != "BF16" && up_scale.dtype != "F16"))
        throw std::runtime_error("Expected BF16/F16 gate/up int4 scales: " + gate_prefix);
    if (gate_scale.shape.size() != 2 || up_scale.shape.size() != 2 ||
        gate_scale.shape[0] != half_out || up_scale.shape[0] != half_out ||
        gate_scale.shape[1] != up_scale.shape[1])
        throw std::runtime_error("int4 gate/up scale shapes differ: " + gate_prefix);
    int groups = (int)gate_scale.shape[1];
    lin.group_size = lin.in_features / groups;

    std::vector<uint16_t> gate_storage, up_storage;
    const uint16_t* gate_s = as_bf16_scale_bits(gate_scale, gate_storage, "gate scale");
    const uint16_t* up_s   = as_bf16_scale_bits(up_scale, up_storage, "up scale");
    std::vector<uint16_t> transposed((size_t)groups * lin.out_features);
    for (int g = 0; g < groups; ++g) {
        for (int n = 0; n < half_out; ++n) {
            transposed[(size_t)g * lin.out_features + n]            = gate_s[(size_t)n * groups + g];
            transposed[(size_t)g * lin.out_features + half_out + n] = up_s[(size_t)n * groups + g];
        }
    }
    lin.weight_scale = GpuBuffer<bf16>(transposed.size(), q);
    lin.weight_scale.upload(transposed.data(), transposed.size());
    return lin;
}

static Q8Linear upload_q8_linear_view(const TensorView& tv, sycl::queue& q,
                                      bool keep_row_scales,
                                      const char* name = "?") {
    if (tv.dtype != "Q8_0")
        throw std::runtime_error(std::string("Expected Q8_0 for ") + name + ", got " + tv.dtype);
    if (tv.shape.size() != 2)
        throw std::runtime_error(std::string("Expected 2D Q8_0 tensor for ") + name);

    Q8Linear lin;
    lin.out_features = (int)tv.shape[0];
    lin.in_features = (int)tv.shape[1];
    if (lin.in_features % 32 != 0)
        throw std::runtime_error(std::string("Q8_0 K not divisible by 32: ") + name);
    int groups = lin.in_features / 32;
    size_t rows = (size_t)lin.out_features;
    size_t qs_count = rows * lin.in_features;
    size_t scale_count = rows * groups;

    static std::vector<int8_t> qs_staging;
    static std::vector<float> scale_rows_staging;
    static std::vector<float> scale_transposed_staging;
    if (qs_staging.size() < qs_count) qs_staging.resize(qs_count);
    if (scale_rows_staging.size() < scale_count) scale_rows_staging.resize(scale_count);
    if (scale_transposed_staging.size() < scale_count) scale_transposed_staging.resize(scale_count);

    const uint8_t* src = static_cast<const uint8_t*>(tv.data);
    constexpr size_t block_bytes = 2 + 32;
    for (int n = 0; n < lin.out_features; ++n) {
        for (int g = 0; g < groups; ++g) {
            const uint8_t* block = src + ((size_t)n * groups + g) * block_bytes;
            uint16_t h;
            std::memcpy(&h, block, sizeof(h));
            float scale = f16_to_float(h);
            scale_rows_staging[(size_t)n * groups + g] = scale;
            scale_transposed_staging[(size_t)g * lin.out_features + n] = scale;
            std::memcpy(qs_staging.data() + (size_t)n * lin.in_features + (size_t)g * 32,
                        block + 2, 32);
        }
    }

    lin.weight_qs = GpuBuffer<int8_t>(qs_count, q);
    lin.weight_qs.upload(qs_staging.data(), qs_count);
    lin.weight_scale = GpuBuffer<float>(scale_count, q);
    lin.weight_scale.upload(scale_transposed_staging.data(), scale_count);
    if (keep_row_scales) {
        lin.weight_scale_rows = GpuBuffer<float>(scale_count, q);
        lin.weight_scale_rows.upload(scale_rows_staging.data(), scale_count);
    }
    return lin;
}

static Q8Linear upload_q8_linear(const TensorSource& sf,
                                 const std::string& prefix,
                                 sycl::queue& q,
                                 bool keep_row_scales = false) {
    return upload_q8_linear_view(sf.get(prefix + ".weight"), q, keep_row_scales,
                                 (prefix + ".weight").c_str());
}

static Q8Linear upload_q8_linear_slice(const TensorView& tv, int expert,
                                       sycl::queue& q, const char* name = "?") {
    if (tv.dtype != "Q8_0")
        throw std::runtime_error(std::string("Expected Q8_0 expert tensor for ") + name);
    if (tv.shape.size() != 3)
        throw std::runtime_error(std::string("Expected 3D Q8_0 expert tensor for ") + name);
    int E = (int)tv.shape[0];
    int N = (int)tv.shape[1];
    int K = (int)tv.shape[2];
    if (expert < 0 || expert >= E)
        throw std::runtime_error(std::string("Q8_0 expert slice out of range for ") + name);
    if (K % 32 != 0)
        throw std::runtime_error(std::string("Q8_0 expert K not divisible by 32 for ") + name);
    size_t bytes_per_expert = (size_t)N * (K / 32) * (2 + 32);
    TensorView slice;
    slice.dtype = "Q8_0";
    slice.shape = {N, K};
    slice.data = static_cast<const uint8_t*>(tv.data) + (size_t)expert * bytes_per_expert;
    slice.nbytes = bytes_per_expert;
    return upload_q8_linear_view(slice, q, false, name);
}

static Q8BatchedLinear upload_q8_batched_slice(const TensorView& tv, int first, int last,
                                               sycl::queue& q, const char* name = "?") {
    if (tv.dtype != "Q8_0")
        throw std::runtime_error(std::string("Expected Q8_0 expert tensor for ") + name);
    if (tv.shape.size() != 3)
        throw std::runtime_error(std::string("Expected 3D Q8_0 expert tensor for ") + name);
    int E = (int)tv.shape[0];
    int N = (int)tv.shape[1];
    int K = (int)tv.shape[2];
    if (first < 0 || last < first || last > E)
        throw std::runtime_error(std::string("Q8_0 expert range out of bounds for ") + name);
    if (K % 32 != 0)
        throw std::runtime_error(std::string("Q8_0 expert K not divisible by 32 for ") + name);

    int B = last - first;
    int groups = K / 32;
    size_t qs_count = (size_t)B * N * K;
    size_t scale_count = (size_t)B * groups * N;
    size_t bytes_per_expert = (size_t)N * groups * (2 + 32);

    static std::vector<int8_t> qs_staging;
    static std::vector<float> scale_staging;
    if (qs_staging.size() < qs_count) qs_staging.resize(qs_count);
    if (scale_staging.size() < scale_count) scale_staging.resize(scale_count);

    const uint8_t* base = static_cast<const uint8_t*>(tv.data);
    constexpr size_t block_bytes = 2 + 32;
    for (int b = 0; b < B; ++b) {
        const uint8_t* expert_base = base + (size_t)(first + b) * bytes_per_expert;
        for (int n = 0; n < N; ++n) {
            for (int g = 0; g < groups; ++g) {
                const uint8_t* block = expert_base + ((size_t)n * groups + g) * block_bytes;
                uint16_t h;
                std::memcpy(&h, block, sizeof(h));
                scale_staging[((size_t)b * groups + g) * N + n] = f16_to_float(h);
                std::memcpy(qs_staging.data() + ((size_t)b * N + n) * K + (size_t)g * 32,
                            block + 2, 32);
            }
        }
    }

    Q8BatchedLinear out;
    out.batch = B;
    out.in_features = K;
    out.out_features = N;
    out.weight_qs = GpuBuffer<int8_t>(qs_count, q);
    out.weight_qs.upload(qs_staging.data(), qs_count);
    out.weight_scale = GpuBuffer<float>(scale_count, q);
    out.weight_scale.upload(scale_staging.data(), scale_count);
    return out;
}

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

static float scalar_bf16(const TensorView& tv) {
    if (tv.dtype == "BF16") return bf16_to_float(*static_cast<const uint16_t*>(tv.data));
    if (tv.dtype == "F16") return f16_to_float(*static_cast<const uint16_t*>(tv.data));
    if (tv.dtype == "F32") return *static_cast<const float*>(tv.data);
    throw std::runtime_error("Expected BF16/F16/F32 scalar, got " + tv.dtype);
}

static std::vector<float> host_floats(const TensorView& tv) {
    size_t n = tv.numel();
    std::vector<float> out(n);
    if (tv.dtype == "BF16") {
        const uint16_t* p = static_cast<const uint16_t*>(tv.data);
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_float(p[i]);
    } else if (tv.dtype == "F16") {
        const uint16_t* p = static_cast<const uint16_t*>(tv.data);
        for (size_t i = 0; i < n; ++i) out[i] = f16_to_float(p[i]);
    } else if (tv.dtype == "F32") {
        const float* p = static_cast<const float*>(tv.data);
        std::memcpy(out.data(), p, n * sizeof(float));
    } else {
        throw std::runtime_error("Expected BF16/F16/F32 vector, got " + tv.dtype);
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
        lw.dec_layer_scalar = scalar_bf16(sf.get(p + "layer_scalar"));
        lw.enc_layer_scalar = scalar_bf16(
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

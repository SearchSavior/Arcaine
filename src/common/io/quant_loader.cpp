#include "quant_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// ShardedSafetensors
// ---------------------------------------------------------------------------
ShardedSafetensors::ShardedSafetensors(const std::string& model_dir) {
    std::ifstream f(model_dir + "/model.safetensors.index.json");
    if (!f) {
        // Single-file checkpoint: no shard index, just model.safetensors.
        // Map every tensor name to that one shard.
        std::string path = model_dir + "/model.safetensors";
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error(
                "Cannot open model.safetensors.index.json or model.safetensors in " + model_dir);
        posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
        close(fd);
        shards_.push_back(std::make_unique<SafetensorsFile>(path));
        for (auto& [name, tv] : shards_[0]->all()) name_to_shard_[name] = 0;
        std::printf("[load] %zu tensors in single safetensors file\n",
                    name_to_shard_.size());
        return;
    }
    auto idx = nlohmann::json::parse(f);
    auto& wm = idx.at("weight_map");

    std::unordered_map<std::string, int> shard_id;
    for (auto& [name, shard] : wm.items()) {
        std::string s = shard.get<std::string>();
        auto it = shard_id.find(s);
        if (it == shard_id.end()) {
            shard_id[s] = (int)shards_.size();
            std::string path = model_dir + "/" + s;
            // Kick off async readahead of the whole shard so the staging
            // memcpys hit warm page cache instead of faulting from disk.
            int fd = open(path.c_str(), O_RDONLY);
            if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED); close(fd); }
            shards_.push_back(std::make_unique<SafetensorsFile>(path));
            it = shard_id.find(s);
        }
        name_to_shard_[name] = it->second;
    }
    std::printf("[load] %zu tensors across %zu shards\n",
                name_to_shard_.size(), shards_.size());
}

const TensorView& ShardedSafetensors::get(const std::string& name) const {
    auto it = name_to_shard_.find(name);
    if (it == name_to_shard_.end())
        throw std::runtime_error("tensor not found: " + name);
    return shards_[it->second]->get(name);
}

bool ShardedSafetensors::has(const std::string& name) const {
    return name_to_shard_.count(name) > 0;
}

std::vector<std::string> ShardedSafetensors::names() const {
    std::vector<std::string> out;
    out.reserve(name_to_shard_.size());
    for (const auto& entry : name_to_shard_) out.push_back(entry.first);
    return out;
}

// Set DIFF_LOAD_TRACE=1 (legacy) or QUANT_LOAD_TRACE=1 for per-tensor
// stall debugging.
static bool g_trace = std::getenv("DIFF_LOAD_TRACE") != nullptr ||
                      std::getenv("QUANT_LOAD_TRACE") != nullptr;

// ---------------------------------------------------------------------------
// SYCL H2D memcpy directly from cold file-backed mmap pages degrades to one
// synchronous 4 KB fault at a time (~10 MB/s).  Stage through a reusable host
// buffer: the CPU memcpy fault path gets kernel readahead (GB/s) and the
// device copy from malloc'd memory runs at full PCIe speed.
// ---------------------------------------------------------------------------
GpuBuffer<bf16> upload(const TensorView& tv, sycl::queue& q, const char* name) {
    size_t n = tv.numel();
    if (g_trace) std::fprintf(stderr, "[trace] alloc+upload %s (%.1f MB)\n",
                              name, n * 2.0 / 1e6);
    static std::vector<bf16> staging;
    if (staging.size() < n) staging.resize(n);
    if (tv.dtype == "BF16") {
        std::memcpy(staging.data(), tv.data, n * sizeof(bf16));
    } else if (tv.dtype == "F32") {
        const float* src = static_cast<const float*>(tv.data);
        for (size_t i = 0; i < n; ++i) staging[i] = float_to_bf16(src[i]);
    } else if (tv.dtype == "F16") {
        const uint16_t* src = static_cast<const uint16_t*>(tv.data);
        for (size_t i = 0; i < n; ++i) staging[i] = float_to_bf16(f16_to_float(src[i]));
    } else {
        throw std::runtime_error(std::string("Expected BF16/F16/F32 for ") + name + ", got " + tv.dtype);
    }
    GpuBuffer<bf16> buf(n, q);
    buf.upload(staging.data(), n);
    if (g_trace) std::fprintf(stderr, "[trace]   done %s\n", name);
    return buf;
}

// (1+w) RMSNorm weight upload: convert to BF16 staging then add 1.0 per
// element before H2D copy. Norm weights are small ([2048] or [256]) so a
// local staging vector is fine (no need for the shared upload() buffer).
GpuBuffer<bf16> upload_plus_one(const TensorView& tv, sycl::queue& q, const char* name) {
    size_t n = tv.numel();
    std::vector<bf16> staging(n);
    if (tv.dtype == "BF16") {
        const bf16* src = static_cast<const bf16*>(tv.data);
        for (size_t i = 0; i < n; ++i)
            staging[i] = float_to_bf16(bf16_to_float(src[i]) + 1.0f);
    } else if (tv.dtype == "F32") {
        const float* src = static_cast<const float*>(tv.data);
        for (size_t i = 0; i < n; ++i)
            staging[i] = float_to_bf16(src[i] + 1.0f);
    } else {
        throw std::runtime_error(std::string("Expected BF16/F32 for +1 norm ") + name + ", got " + tv.dtype);
    }
    GpuBuffer<bf16> buf(n, q);
    buf.upload(staging.data(), n);
    return buf;
}

GpuBuffer<uint8_t> upload_u8(const TensorView& tv, sycl::queue& q, const char* name) {
    if (tv.dtype != "U8" && tv.dtype != "F8_E4M3")
        throw std::runtime_error(std::string("Expected U8/F8_E4M3 for ") + name + ", got " + tv.dtype);
    GpuBuffer<uint8_t> buf(tv.nbytes, q);
    buf.upload(static_cast<const uint8_t*>(tv.data), tv.nbytes);
    return buf;
}

Fp8Linear upload_fp8_linear(const TensorSource& sf, const std::string& prefix,
                            sycl::queue& q) {
    const TensorView& weight = sf.get(prefix + ".weight");
    const TensorView& scale = sf.get(prefix + ".weight_scale");
    if (weight.dtype != "F8_E4M3" || weight.shape.size() != 2)
        throw std::runtime_error("Expected F8_E4M3 [N,K] weight: " + prefix);
    if (scale.dtype != "BF16" || scale.shape.size() != 2 ||
        scale.shape[0] != weight.shape[0] || scale.shape[1] != 1)
        throw std::runtime_error("Expected BF16 [N,1] weight scale: " + prefix);

    Fp8Linear lin;
    lin.out_features = static_cast<int>(weight.shape[0]);
    lin.in_features = static_cast<int>(weight.shape[1]);
    lin.weight = upload_u8(weight, q, (prefix + ".weight").c_str());
    lin.weight_scale = upload(scale, q, (prefix + ".weight_scale").c_str());
    return lin;
}

Fp8Linear upload_fp8_linear_pair(const TensorSource& sf,
                                 const std::string& first_prefix,
                                 const std::string& second_prefix,
                                 sycl::queue& q) {
    return upload_fp8_linear_concat(sf, {first_prefix, second_prefix}, q);
}

Fp8Linear upload_fp8_linear_concat(
    const TensorSource& sf, const std::vector<std::string>& prefixes,
    sycl::queue& q) {
    if (prefixes.empty())
        throw std::runtime_error("FP8 concat requires at least one projection");
    int in = -1;
    int total_out = 0;
    size_t total_weight_bytes = 0;
    std::vector<const TensorView*> weights;
    std::vector<const TensorView*> scales;
    for (const std::string& prefix : prefixes) {
        const TensorView& weight = sf.get(prefix + ".weight");
        const TensorView& scale = sf.get(prefix + ".weight_scale");
        if (weight.dtype != "F8_E4M3" || weight.shape.size() != 2)
            throw std::runtime_error("Expected FP8 [N,K] weight: " + prefix);
        if (scale.dtype != "BF16" || scale.shape.size() != 2 ||
            scale.shape[0] != weight.shape[0] || scale.shape[1] != 1)
            throw std::runtime_error("Expected BF16 [N,1] FP8 scale: " + prefix);
        int current_in = static_cast<int>(weight.shape[1]);
        if (in < 0) in = current_in;
        if (current_in != in)
            throw std::runtime_error("FP8 concat K mismatch: " + prefix);
        total_out += static_cast<int>(weight.shape[0]);
        total_weight_bytes += weight.nbytes;
        weights.push_back(&weight);
        scales.push_back(&scale);
    }
    std::vector<uint8_t> host_weight(total_weight_bytes);
    std::vector<bf16> host_scale(total_out);
    size_t weight_offset = 0;
    size_t scale_offset = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::memcpy(host_weight.data() + weight_offset, weights[i]->data,
                    weights[i]->nbytes);
        size_t outputs = static_cast<size_t>(weights[i]->shape[0]);
        std::memcpy(host_scale.data() + scale_offset, scales[i]->data,
                    outputs * sizeof(bf16));
        weight_offset += weights[i]->nbytes;
        scale_offset += outputs;
    }
    Fp8Linear linear;
    linear.in_features = in;
    linear.out_features = total_out;
    linear.weight = GpuBuffer<uint8_t>(host_weight.size(), q);
    linear.weight.upload(host_weight.data(), host_weight.size());
    linear.weight_scale = GpuBuffer<bf16>(host_scale.size(), q);
    linear.weight_scale.upload(host_scale.data(), host_scale.size());
    return linear;
}

float scalar_f32(const TensorView& tv, const char* name) {
    if (tv.dtype != "F32") throw std::runtime_error(std::string("Expected F32 for ") + name + ", got " + tv.dtype);
    if (tv.nbytes != sizeof(float)) throw std::runtime_error(std::string("Expected scalar F32 for ") + name);
    float out;
    std::memcpy(&out, tv.data, sizeof(float));
    return out;
}

GpuBuffer<uint8_t> upload_nvfp4_scales_transposed(
    const TensorView& tv, int out_features, int groups, sycl::queue& q,
    const char* name) {
    if (tv.dtype != "F8_E4M3")
        throw std::runtime_error(std::string("Expected F8_E4M3 for ") + name + ", got " + tv.dtype);
    if (tv.shape.size() != 2 || tv.shape[0] != out_features || tv.shape[1] != groups)
        throw std::runtime_error(std::string("Unexpected NVFP4 scale shape for ") + name);

    const uint8_t* src = static_cast<const uint8_t*>(tv.data); // model layout: (N, K/16)
    std::vector<uint8_t> transposed((size_t)groups * out_features);
    for (int n = 0; n < out_features; ++n)
        for (int g = 0; g < groups; ++g)
            transposed[(size_t)g * out_features + n] = src[(size_t)n * groups + g];

    GpuBuffer<uint8_t> buf(transposed.size(), q);
    buf.upload(transposed.data(), transposed.size());
    return buf;
}

Nvfp4Linear upload_nvfp4_linear(const TensorSource& sf, const std::string& prefix,
                                 sycl::queue& q) {
    const TensorView& packed = sf.get(prefix + ".weight_packed");
    if (packed.dtype != "U8") throw std::runtime_error("Expected U8 packed weight: " + prefix);
    if (packed.shape.size() != 2) throw std::runtime_error("Expected 2D packed weight: " + prefix);

    Nvfp4Linear lin;
    lin.out_features = (int)packed.shape[0];
    lin.in_features = (int)packed.shape[1] * 2;
    if (lin.in_features % 16 != 0) throw std::runtime_error("NVFP4 K not divisible by 16: " + prefix);
    int groups = lin.in_features / 16;

    lin.weight_packed = upload_u8(packed, q, (prefix + ".weight_packed").c_str());
    lin.weight_scale = upload_nvfp4_scales_transposed(
        sf.get(prefix + ".weight_scale"), lin.out_features, groups, q,
        (prefix + ".weight_scale").c_str());
    lin.input_global_scale = scalar_f32(sf.get(prefix + ".input_global_scale"),
                                        (prefix + ".input_global_scale").c_str());
    lin.weight_global_scale = scalar_f32(sf.get(prefix + ".weight_global_scale"),
                                          (prefix + ".weight_global_scale").c_str());
    float dst_scale = lin.input_global_scale * lin.weight_global_scale;
    lin.dst_scale = GpuBuffer<float>(1, q);
    lin.dst_scale.upload(&dst_scale, 1);
    return lin;
}

Nvfp4Linear upload_nvfp4_linear_pair(const TensorSource& sf,
                                     const std::string& gate_prefix,
                                     const std::string& up_prefix,
                                     sycl::queue& q) {
    const TensorView& gate_packed = sf.get(gate_prefix + ".weight_packed");
    const TensorView& up_packed = sf.get(up_prefix + ".weight_packed");
    if (gate_packed.dtype != "U8" || up_packed.dtype != "U8")
        throw std::runtime_error("Expected U8 packed gate/up weights: " + gate_prefix);
    if (gate_packed.shape.size() != 2 || up_packed.shape.size() != 2 ||
        gate_packed.shape[0] != up_packed.shape[0] || gate_packed.shape[1] != up_packed.shape[1])
        throw std::runtime_error("NVFP4 gate/up packed shapes differ: " + gate_prefix);

    Nvfp4Linear lin;
    int half_out = (int)gate_packed.shape[0];
    int packed_cols = (int)gate_packed.shape[1];
    lin.out_features = 2 * half_out;
    lin.in_features = packed_cols * 2;
    if (lin.in_features % 16 != 0) throw std::runtime_error("NVFP4 K not divisible by 16: " + gate_prefix);
    int groups = lin.in_features / 16;

    const uint8_t* gate_w = static_cast<const uint8_t*>(gate_packed.data);
    const uint8_t* up_w = static_cast<const uint8_t*>(up_packed.data);
    std::vector<uint8_t> packed((size_t)lin.out_features * packed_cols);
    std::memcpy(packed.data(), gate_w, (size_t)half_out * packed_cols);
    std::memcpy(packed.data() + (size_t)half_out * packed_cols, up_w,
                (size_t)half_out * packed_cols);
    lin.weight_packed = GpuBuffer<uint8_t>(packed.size(), q);
    lin.weight_packed.upload(packed.data(), packed.size());

    const TensorView& gate_scale = sf.get(gate_prefix + ".weight_scale");
    const TensorView& up_scale = sf.get(up_prefix + ".weight_scale");
    if (gate_scale.dtype != "F8_E4M3" || up_scale.dtype != "F8_E4M3")
        throw std::runtime_error("Expected F8_E4M3 gate/up scales: " + gate_prefix);
    if (gate_scale.shape.size() != 2 || up_scale.shape.size() != 2 ||
        gate_scale.shape[0] != half_out || up_scale.shape[0] != half_out ||
        gate_scale.shape[1] != groups || up_scale.shape[1] != groups)
        throw std::runtime_error("NVFP4 gate/up scale shapes differ: " + gate_prefix);

    const uint8_t* gate_s = static_cast<const uint8_t*>(gate_scale.data);
    const uint8_t* up_s = static_cast<const uint8_t*>(up_scale.data);
    std::vector<uint8_t> transposed((size_t)groups * lin.out_features);
    for (int g = 0; g < groups; ++g) {
        for (int n = 0; n < half_out; ++n) {
            transposed[(size_t)g * lin.out_features + n] = gate_s[(size_t)n * groups + g];
            transposed[(size_t)g * lin.out_features + half_out + n] = up_s[(size_t)n * groups + g];
        }
    }
    lin.weight_scale = GpuBuffer<uint8_t>(transposed.size(), q);
    lin.weight_scale.upload(transposed.data(), transposed.size());

    lin.input_global_scale = scalar_f32(sf.get(gate_prefix + ".input_global_scale"),
                                        (gate_prefix + ".input_global_scale").c_str());
    float up_input_global = scalar_f32(sf.get(up_prefix + ".input_global_scale"),
                                       (up_prefix + ".input_global_scale").c_str());
    lin.weight_global_scale = scalar_f32(sf.get(gate_prefix + ".weight_global_scale"),
                                          (gate_prefix + ".weight_global_scale").c_str());
    float up_weight_global = scalar_f32(sf.get(up_prefix + ".weight_global_scale"),
                                        (up_prefix + ".weight_global_scale").c_str());
    if (lin.input_global_scale != up_input_global || lin.weight_global_scale != up_weight_global)
        throw std::runtime_error("NVFP4 fused gate/up global scales differ: " + gate_prefix);

    float dst_scale = lin.input_global_scale * lin.weight_global_scale;
    lin.dst_scale = GpuBuffer<float>(1, q);
    lin.dst_scale.upload(&dst_scale, 1);
    return lin;
}

// ---------------------------------------------------------------------------
// int4 W4A16 (compressed-tensors "pack-quantized")
// ---------------------------------------------------------------------------
namespace {

// compressed-tensors int4 checkpoints may store per-group weight scales as F16
// (model dtype float16). oneDNN's s4 weight-scale path consumes BF16, so
// normalize F16 -> BF16 (round-to-nearest-even via float) into host storage and
// return BF16 bits; BF16 scales are returned as-is. The returned pointer is
// valid for the lifetime of `storage`.
const uint16_t* as_bf16_scale_bits(const TensorView& tv,
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

GpuBuffer<bf16> upload_int4_scales_transposed(
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

GpuBuffer<uint8_t> upload_int4_packed_rebased(
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

GpuBuffer<uint8_t> upload_int4_packed_rebased(
    const void* data, size_t bytes, sycl::queue& q) {
    return upload_int4_packed_rebased(std::vector<ByteSpan>{{data, bytes}}, bytes, q);
}

}  // namespace

Int4Linear upload_int4_linear(const TensorSource& sf,
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
Int4Linear upload_int4_linear_concat(const TensorSource& sf,
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
Int4Linear upload_int4_linear_pair(const TensorSource& sf,
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

// ---------------------------------------------------------------------------
// GGUF Q8_0
// ---------------------------------------------------------------------------
Q8Linear upload_q8_linear_view(const TensorView& tv, sycl::queue& q,
                               bool keep_row_scales,
                               const char* name) {
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

Q8Linear upload_q8_linear(const TensorSource& sf,
                          const std::string& prefix,
                          sycl::queue& q,
                          bool keep_row_scales) {
    return upload_q8_linear_view(sf.get(prefix + ".weight"), q, keep_row_scales,
                                 (prefix + ".weight").c_str());
}

Q8Linear upload_q8_linear_slice(const TensorView& tv, int expert,
                                sycl::queue& q, const char* name) {
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

Q8BatchedLinear upload_q8_batched_slice(const TensorView& tv, int first, int last,
                                        sycl::queue& q, const char* name) {
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

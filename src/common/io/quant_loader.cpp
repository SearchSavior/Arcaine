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
        for (size_t i = 0; i < n; ++i) {
            uint16_t h = src[i];
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
            staging[i] = float_to_bf16(f);
        }
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

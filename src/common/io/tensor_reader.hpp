#pragma once

// Generic, model-agnostic checkpoint reader helpers built on top of the shared
// TensorSource + upload primitives in quant_loader.hpp.  Header-only / inline so
// any model loader can use them without extra build wiring.
//
// Everything here was previously duplicated (or nearly so) inside individual
// model loaders (qwen3_5, gemma4_unified, diffusion_gemma).  Behavior is
// preserved; only the location is shared.

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "quant_loader.hpp"   // TensorSource, upload, upload_plus_one, Fp8/Nvfp4Linear
#include "tensor_view.hpp"    // TensorView
#include "../gpu/buffer.hpp"  // GpuBuffer, bf16, bf16_to_float, f16_to_float

// Validate a tensor's dtype and shape exactly, throwing a descriptive error.
inline void expect_tensor(const TensorSource& source, const std::string& name,
                          const char* dtype, std::vector<int64_t> shape) {
    const TensorView& view = source.get(name);
    if (view.dtype != dtype || view.shape != shape) {
        std::ostringstream message;
        message << "Unexpected tensor metadata for " << name << ": dtype="
                << view.dtype << " shape=(";
        for (size_t i = 0; i < view.shape.size(); ++i) {
            if (i) message << ',';
            message << view.shape[i];
        }
        message << ')';
        throw std::runtime_error(message.str());
    }
}

// Validate + upload a tensor as BF16.  When add_one is set the (1+w) RMSNorm
// form is baked in via upload_plus_one().
inline GpuBuffer<bf16> load_bf16(const TensorSource& source, const std::string& name,
                                 std::vector<int64_t> shape, sycl::queue& queue,
                                 bool add_one = false) {
    expect_tensor(source, name, "BF16", std::move(shape));
    return add_one ? upload_plus_one(source.get(name), queue, name.c_str())
                   : upload(source.get(name), queue, name.c_str());
}

// Read a single-element tensor (BF16/F16/F32) as a host float.
inline float scalar_as_float(const TensorView& tv) {
    if (tv.dtype == "BF16") return bf16_to_float(*static_cast<const uint16_t*>(tv.data));
    if (tv.dtype == "F16") return f16_to_float(*static_cast<const uint16_t*>(tv.data));
    if (tv.dtype == "F32") return *static_cast<const float*>(tv.data);
    throw std::runtime_error("Expected BF16/F16/F32 scalar, got " + tv.dtype);
}

// Convert a whole BF16/F16/F32 tensor to a host float vector.
inline std::vector<float> host_floats(const TensorView& tv) {
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

// Concatenate two same-shape BF16 tensors along the leading dimension.
inline GpuBuffer<bf16> upload_bf16_pair(const TensorView& a, const TensorView& b,
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

// Upload a contiguous BF16 sub-range of a tensor starting at element `offset`.
inline GpuBuffer<bf16> upload_bf16_slice(const TensorView& tv, size_t offset, size_t n,
                                         sycl::queue& q, const char* name = "?") {
    if (tv.dtype != "BF16") throw std::runtime_error("Expected BF16 slice, got " + tv.dtype);
    if (offset + n > tv.numel()) throw std::runtime_error(std::string("slice exceeds tensor: ") + name);
    const bf16* src = static_cast<const bf16*>(tv.data) + offset;
    GpuBuffer<bf16> buf(n, q);
    buf.upload(src, n);
    return buf;
}

inline void expect_fp8(const Fp8Linear& linear, int in, int out, const std::string& name) {
    if (linear.in_features != in || linear.out_features != out)
        throw std::runtime_error("Unexpected FP8 linear shape: " + name);
}

inline void expect_nvfp4(const Nvfp4Linear& linear, int in, int out, const std::string& name) {
    if (linear.in_features != in || linear.out_features != out)
        throw std::runtime_error("Unexpected NVFP4 linear shape: " + name);
}

// TensorSource wrapper that records which tensor names were requested, so a
// loader can assert the checkpoint contained no unexpected/unconsumed tensors.
class TrackingTensorSource final : public TensorSource {
public:
    explicit TrackingTensorSource(const TensorSource& source) : source_(source) {}

    const TensorView& get(const std::string& name) const override {
        const TensorView& view = source_.get(name);
        consumed_.insert(name);
        return view;
    }

    bool has(const std::string& name) const override { return source_.has(name); }
    bool consumed(const std::string& name) const { return consumed_.count(name) != 0; }

private:
    const TensorSource& source_;
    mutable std::unordered_set<std::string> consumed_;
};

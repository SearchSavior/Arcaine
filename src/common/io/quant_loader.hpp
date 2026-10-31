#pragma once

// Shared quantized-weight loader primitives: a sharded safetensors tensor
// source plus BF16/U8/scalar uploads and the compressed-tensors NVFP4 (W4A4)
// linear packers. Extracted from modeling/diffusion_gemma so any model module
// (registry- or direct-path) reuses identical NVFP4 mechanics — the kernel +
// Nvfp4Linear struct are unchanged; only the per-model key prefix differs.
//
// All upload helpers stage through a reusable host buffer (see upload()) to
// avoid one-synchronous-4KB-fault-at-a-time H2D copies from cold mmap pages.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/gpu/buffer.hpp"   // GpuBuffer, bf16, float_to_bf16
#include "../../common/gpu/nvfp4.hpp"    // Nvfp4Linear
#include "../../common/gpu/fp8.hpp"      // Fp8Linear
#include "../../common/gpu/int4.hpp"     // Int4Linear
#include "../../common/gpu/q8_0.hpp"     // Q8Linear, Q8BatchedLinear
#include "safetensors.hpp"              // SafetensorsFile
#include "tensor_view.hpp"             // TensorView

// Abstract tensor source: maps a logical tensor name to a mmap'd TensorView.
class TensorSource {
public:
    virtual ~TensorSource() = default;
    virtual const TensorView& get(const std::string& name) const = 0;
    virtual bool has(const std::string& name) const = 0;
};

// Sharded safetensors: reads model.safetensors.index.json (or a single
// model.safetensors) and maps each tensor name to the shard that holds it.
class ShardedSafetensors : public TensorSource {
public:
    explicit ShardedSafetensors(const std::string& model_dir);
    const TensorView& get(const std::string& name) const override;
    bool has(const std::string& name) const override;
    size_t num_tensors() const { return name_to_shard_.size(); }
    std::vector<std::string> names() const;

private:
    std::vector<std::unique_ptr<SafetensorsFile>> shards_;
    std::unordered_map<std::string, int> name_to_shard_;
};

// Stage a BF16/F16/F32 tensor to a BF16 device buffer (F16/F32 converted).
GpuBuffer<bf16>    upload(const TensorView& tv, sycl::queue& q, const char* name = "?");

// Like upload(), but adds 1.0 to every element before staging. Bakes the
// (1+w) form of Qwen3_5MoeRMSNorm into the device weight so the plain
// `rms_norm` kernel (out = x * rsqrt(mean(x^2)+eps) * weight) can be reused
// directly without a +1 branch. Use ONLY for plain (1+w) RMSNorm weights
// (input/post/final layernorm, q_norm, k_norm); the gated linear-attn norm
// keeps `upload` (it scales by w, not 1+w).
GpuBuffer<bf16>    upload_plus_one(const TensorView& tv, sycl::queue& q, const char* name = "?");

// Upload a raw U8 / F8_E4M3 byte buffer verbatim (no conversion).
GpuBuffer<uint8_t> upload_u8(const TensorView& tv, sycl::queue& q, const char* name = "?");

// Build a compressed-tensors float-quantized FP8 linear from
// <prefix>.weight (F8_E4M3 [N,K]) and <prefix>.weight_scale (BF16 [N,1]).
Fp8Linear           upload_fp8_linear(const TensorSource& sf, const std::string& prefix,
                                      sycl::queue& q);

// Concatenate two FP8 projections with identical K into one [2N,K] linear.
Fp8Linear           upload_fp8_linear_pair(const TensorSource& sf,
                                           const std::string& first_prefix,
                                           const std::string& second_prefix,
                                           sycl::queue& q);

Fp8Linear           upload_fp8_linear_concat(
                         const TensorSource& sf,
                         const std::vector<std::string>& prefixes,
                         sycl::queue& q);

// Read a single-element F32 tensor.
float              scalar_f32(const TensorView& tv, const char* name = "?");

// Transpose an F8_E4M3 NVFP4 weight_scale from model layout (N, K/16) to the
// oneDNN layout (K/16, N) expected by the kernel.
GpuBuffer<uint8_t> upload_nvfp4_scales_transposed(const TensorView& tv, int out_features,
                                                  int groups, sycl::queue& q,
                                                  const char* name = "?");

// Build an Nvfp4Linear from a checkpoint prefix that exposes the 4 compressed
// tensors: <prefix>.weight_packed, .weight_scale, .input_global_scale,
// .weight_global_scale. Folds dst_scale = input_global * weight_global.
Nvfp4Linear        upload_nvfp4_linear(const TensorSource& sf, const std::string& prefix,
                                       sycl::queue& q);

// Fused gate+up NVFP4: concatenates two separate gate/up checkpoints into one
// Nvfp4Linear with out_features = 2 * half_out, interleaving the transposed
// scales. Asserts both globals match.
Nvfp4Linear        upload_nvfp4_linear_pair(const TensorSource& sf,
                                            const std::string& gate_prefix,
                                            const std::string& up_prefix,
                                            sycl::queue& q);

// ---------------------------------------------------------------------------
// compressed-tensors int4 W4A16 ("pack-quantized"), shared with all models.
// weight_packed is I32 with 8 signed int4 per word along K (low-nibble first);
// the raw byte stream is oneDNN s4 tag::ba for logical (K, N).  weight_scale is
// BF16/F16 (out, K/group) transposed to (groups, out); F16 scales normalized to
// BF16.  Symmetric u4 zero-point 8 is rebased to two's-complement s4 via a
// per-nibble sign-bit XOR (byte ^ 0x88), so oneDNN consumes it directly.
// ---------------------------------------------------------------------------
Int4Linear         upload_int4_linear(const TensorSource& sf, const std::string& prefix,
                                      sycl::queue& q);

// Fuse gate_proj + up_proj into one int4 weight with out_features = 2*half_out
// (gate block then up block).
Int4Linear         upload_int4_linear_pair(const TensorSource& sf,
                                           const std::string& gate_prefix,
                                           const std::string& up_prefix,
                                           sycl::queue& q);

// Fuse N projections along the output dimension into one (N_total, K) int4
// weight. Rows are output channels, so concatenating packed rows is layout-safe.
Int4Linear         upload_int4_linear_concat(const TensorSource& sf,
                                             const std::vector<std::string>& prefixes,
                                             sycl::queue& q, const char* name);

// ---------------------------------------------------------------------------
// GGUF Q8_0 (W8A16): 32-element blocks of int8 with an F16 scale.
// ---------------------------------------------------------------------------
Q8Linear           upload_q8_linear_view(const TensorView& tv, sycl::queue& q,
                                         bool keep_row_scales, const char* name = "?");

Q8Linear           upload_q8_linear(const TensorSource& sf, const std::string& prefix,
                                    sycl::queue& q, bool keep_row_scales = false);

// Extract expert `expert` from a 3D [E, N, K] Q8_0 tensor.
Q8Linear           upload_q8_linear_slice(const TensorView& tv, int expert,
                                          sycl::queue& q, const char* name = "?");

// Extract experts [first, last) from a 3D [E, N, K] Q8_0 tensor into a batched
// (B, N, K) weight with (B, K/32, N) scales.
Q8BatchedLinear    upload_q8_batched_slice(const TensorView& tv, int first, int last,
                                           sycl::queue& q, const char* name = "?");

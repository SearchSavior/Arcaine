#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <vector>

#include "config.hpp"

namespace flint {

using bf16 = sycl::ext::oneapi::bfloat16;

// MLX affine quantized tensor (group size 64).
// w: packed codes, row-major [n, k/2] bytes (4-bit) — low nibble first.
// s/b: bf16 [n, k/64]; 3-D expert tensors add a leading expert dimension.
enum class WeightFormat { MLX, AWQ, BF16 };

struct QTensor {
    WeightFormat format = WeightFormat::MLX;
    bool decode_vector = false;
    const uint8_t* w = nullptr;
    const bf16* s = nullptr;
    const bf16* b = nullptr;
    const sycl::half* awq_s = nullptr;
    const uint8_t* awq_z = nullptr;
    const bf16* dense = nullptr;
    int n = 0;
    int k = 0;
    int experts = 0;
    size_t stride_w = 0;  // bytes per expert
    size_t stride_s = 0;  // elements (bf16) per expert
    size_t stride_b = 0;

    bool valid() const { return w != nullptr || dense != nullptr; }
};

struct FlintLayer {
    int index = 0;
    bool global = false;  // full attention / no RoPE
    bool moe = false;

    const bf16* in_norm = nullptr;
    const bf16* post_norm = nullptr;
    const bf16* q_norm = nullptr;
    const bf16* k_norm = nullptr;
    const bf16* sink = nullptr;  // [num_attention_heads]
    const bf16* router = nullptr; // [num_experts, hidden]

    QTensor q, k, v, o;
    QTensor gate, up, down;        // dense MLP (moe == false)
    QTensor egate, eup, edown;     // routed experts (moe == true, 3-D)
    QTensor sgate, sup, sdown;     // shared expert (moe == true)
};

struct FlintWeights {
    uint8_t* arena = nullptr;
    size_t arena_bytes = 0;

    QTensor embed;  // 8-bit affine
    QTensor head;   // 4-bit affine
    const bf16* final_norm = nullptr;
    std::vector<FlintLayer> layers;
};

// Loads the MLX-affine checkpoint into one device arena with an independent
// safetensors reader (no shared Arcaine loader code). max_layers < 0 loads all.
FlintWeights load_flint_weights(const FlintConfig& cfg, sycl::queue& q,
                                int max_layers = -1);

}  // namespace flint

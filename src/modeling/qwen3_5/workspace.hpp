#pragma once

#include <algorithm>

#include "config.hpp"
#include "../../runtime/gpu/buffer.hpp"

// Persistent per-device scratch. Buffers are role-based and reused across all
// layers; no layer-forward path allocates device memory.
struct Qwen35Workspace {
    int max_seq_len = 0;
    GpuBuffer<bf16> tmp0;  // gate_up / q_proj / mixed_qkv
    GpuBuffer<bf16> tmp1;  // activation / conv output / K
    GpuBuffer<bf16> tmp2;  // Q
    GpuBuffer<bf16> tmp3;  // gate / repeated K
    GpuBuffer<bf16> tmp4;  // V / DeltaNet core
    GpuBuffer<uint8_t> input_packed;
    GpuBuffer<uint8_t> input_scale;
    GpuBuffer<uint8_t> activation_packed;
    GpuBuffer<uint8_t> activation_scale;
    // INT4 zp-correction rowsum scratch (runtime/quantization/int4.hpp): sized
    // max_seq x max(K/group) with K = intermediate_size (down proj) and
    // group_size = 32. Reused by every INT4 projection; keeps matmul_int4 free
    // of per-call device allocations on the decode path.
    GpuBuffer<bf16> int4_rowsum;
    // INT4 zp-correction GEMM output scratch for small-M (decode/speculative)
    // calls: 16 x max(out_features) with out_features = 2*intermediate_size
    // (fused gate_up). Prefill calls (M > 16) fall back to a per-call heap
    // buffer rather than reserving max_seq rows here.
    static constexpr int int4_corr_rows = 16;
    GpuBuffer<bf16> int4_corr_decode;
    // Decode split-KV attention partials. Sized for head_dim=256, 8 rows,
    // 4 partitions, 64 output dims/partition, and max_kv_slices (default 16).
    // part_out: [key_heads * slices * 4 * 8 * 64] floats; part_state:
    // [key_heads * slices * 8 * 2] floats (m_p, l_p per row).
    static constexpr int decode_max_kv_slices = 16;
    static constexpr int decode_rows = 8;
    static constexpr int decode_partitions = 4;
    static constexpr int decode_part_dims = 64;
    GpuBuffer<float> decode_part_out;
    GpuBuffer<float> decode_part_state;
    // Prefill split-KV (xmx3) partials. Region layout from the kernel:
    // [(tile*query_heads + query_head)*slices + slice][row(128)][dim(256)]
    // floats for part_out, [region][row][2] floats for part_state (m, l in
    // the exp2 domain). Caps: seq <= 1024 (8 tiles of 128), slices <= 8.
    static constexpr int prefill_max_tiles = 8;
    static constexpr int prefill_max_kv_slices = 8;
    static constexpr int prefill_rows_per_tile = 128;
    GpuBuffer<float> prefill_part_out;
    GpuBuffer<float> prefill_part_state;

    void init(const Qwen35Config& config, int max_seq, sycl::queue& queue) {
        max_seq_len = max_seq;
        const auto& c = config.text;
        int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
        int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        int conv_dim = 2 * key_dim + value_dim;
        int q_proj = c.num_attention_heads * c.head_dim * 2;
        size_t s = static_cast<size_t>(max_seq);
        tmp0 = GpuBuffer<bf16>(s * std::max({2 * c.intermediate_size, conv_dim, q_proj}), queue);
        tmp1 = GpuBuffer<bf16>(s * std::max(c.intermediate_size, conv_dim), queue);
        tmp2 = GpuBuffer<bf16>(s * value_dim, queue);
        tmp3 = GpuBuffer<bf16>(s * value_dim, queue);
        tmp4 = GpuBuffer<bf16>(s * value_dim, queue);
        input_packed = GpuBuffer<uint8_t>(s * c.hidden_size / 2, queue);
        input_scale = GpuBuffer<uint8_t>(s * c.hidden_size / 16, queue);
        activation_packed = GpuBuffer<uint8_t>(s * c.intermediate_size / 2, queue);
        activation_scale = GpuBuffer<uint8_t>(s * c.intermediate_size / 16, queue);
        int max_groups = (c.intermediate_size + 31) / 32;  // down proj K = I, gs 32
        int4_rowsum = GpuBuffer<bf16>(s * max_groups, queue);
        int4_corr_decode = GpuBuffer<bf16>(
            (size_t)int4_corr_rows * 2 * c.intermediate_size, queue);
        size_t out_elems = (size_t)c.num_key_value_heads * decode_max_kv_slices *
                           decode_partitions * decode_rows * decode_part_dims;
        size_t state_elems = (size_t)c.num_key_value_heads * decode_max_kv_slices *
                             decode_rows * 2;
        decode_part_out = GpuBuffer<float>(out_elems, queue);
        decode_part_state = GpuBuffer<float>(state_elems, queue);
    }
};

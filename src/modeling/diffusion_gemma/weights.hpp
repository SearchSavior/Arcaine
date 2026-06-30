#pragma once
#include <variant>
#include <vector>
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/nvfp4.hpp"
#include "../../common/gpu/int4.hpp"
#include "../../common/gpu/q8_0.hpp"

// A projection weight that may be plain BF16, NVFP4 (W4A4), int4 W4A16, or
// GGUF Q8_0 W8A16.
struct DiffLinearWeight {
    enum class Kind { BF16, NVFP4, INT4, Q8_0 } kind = Kind::BF16;
    GpuBuffer<bf16> bf16;
    Nvfp4Linear fp4;
    Int4Linear  int4;
    Q8Linear    q8;

    // Retained name for existing NVFP4 call sites.
    bool nvfp4 = false;
    bool is_int4() const { return kind == Kind::INT4; }
    bool is_q8() const { return kind == Kind::Q8_0; }
};

// Sliding-window attention: real v_proj, GQA 16 Q / 8 KV, head_dim 256.
struct DiffSlidingAttn {
    DiffLinearWeight q_proj;  // (4096, 2816)
    DiffLinearWeight k_proj;  // (2048, 2816)
    DiffLinearWeight v_proj;  // (2048, 2816)
    Int4Linear       qkv_proj_int4; // optional fused int4 (q, k, v)
    DiffLinearWeight o_proj;  // (2816, 4096)
    GpuBuffer<bf16>  q_norm;  // (256,)
    GpuBuffer<bf16>  k_norm;  // (256,)
};

// Full attention: K == V (no v_proj), 16 Q / 2 KV, head_dim 512.
struct DiffFullAttn {
    DiffLinearWeight q_proj;  // (8192, 2816)
    DiffLinearWeight k_proj;  // (1024, 2816)
    Int4Linear       qk_proj_int4; // optional fused int4 (q, k)
    DiffLinearWeight o_proj;  // (2816, 8192)
    GpuBuffer<bf16>  q_norm;  // (512,)
    GpuBuffer<bf16>  k_norm;  // (512,)
};

// Dense "shared expert" GeGLU MLP (intermediate 2112).
struct DiffDenseMLP {
    DiffLinearWeight gate_proj;  // BF16 path: (2112, 2816)
    DiffLinearWeight up_proj;    // BF16 path: (2112, 2816)
    DiffLinearWeight down_proj;  // BF16 or NVFP4 equivalent
    Nvfp4Linear gate_up_proj_fp4; // NVFP4 fused gate/up: (2*2112, 2816)
};

struct DiffExpertShard {
    int gpu = 0;
    int first_expert = 0;
    int num_experts = 0;
    bool nvfp4 = false;
    bool int4 = false;
    bool q8 = false;
    GpuBuffer<bf16> gate_up_proj;  // BF16: (num_experts, 2*moe_intermediate, H)
    GpuBuffer<bf16> down_proj;     // BF16: (num_experts, H, moe_intermediate)
    std::vector<Nvfp4Linear> gate_up_proj_fp4; // NVFP4 fused gate/up: local experts, each (2*moe_intermediate, H)
    std::vector<Nvfp4Linear> down_proj_fp4;    // NVFP4: local experts, each (H, moe_intermediate)
    std::vector<Int4Linear>  gate_up_proj_int4; // int4 fused gate/up: local experts, each (2*moe_intermediate, H)
    std::vector<Int4Linear>  down_proj_int4;    // int4: local experts, each (H, moe_intermediate)
    std::vector<Q8Linear>    gate_up_proj_q8;   // Q8_0 fused gate/up: local experts, each (2*moe_intermediate, H)
    std::vector<Q8Linear>    down_proj_q8;      // Q8_0: local experts, each (H, moe_intermediate)
    Q8BatchedLinear          gate_up_proj_q8_batch;
    Q8BatchedLinear          down_proj_q8_batch;
};

// Sparse MoE (128 experts, top-8, per-expert intermediate 704). Router weights
// stay on the owning layer GPU; expert weights are sharded across all GPUs.
struct DiffMoE {
    // Kept for the old local expert helper while expert parallel settles.
    GpuBuffer<bf16>    gate_up_proj;      // optional full expert tensor
    GpuBuffer<bf16>    down_proj;         // optional full expert tensor
    DiffLinearWeight   router_proj;       // (E=128, H=2816)
    GpuBuffer<bf16>    router_scale;      // (H=2816,)  learnable per-channel
    std::vector<float> per_expert_scale;  // (E=128,)   host copy for top-k weighting
    GpuBuffer<float>   per_expert_scale_dev; // (E=128,) device copy for GPU top-k
    std::vector<DiffExpertShard> expert_shards;
};

struct DiffLayer {
    bool is_full = false;
    std::variant<DiffSlidingAttn, DiffFullAttn> attn;
    DiffDenseMLP mlp;
    DiffMoE      moe;
    GpuBuffer<bf16> input_ln;        // (H,)
    GpuBuffer<bf16> post_attn_ln;    // (H,)
    GpuBuffer<bf16> pre_ffn_ln;      // (H,)   dense MLP path
    GpuBuffer<bf16> pre_ffn_ln_2;    // (H,)   MoE path
    GpuBuffer<bf16> post_ffn_ln_1;   // (H,)   dense MLP path
    GpuBuffer<bf16> post_ffn_ln_2;   // (H,)   MoE path
    GpuBuffer<bf16> post_ffn_ln;     // (H,)   combine
    float enc_layer_scalar = 1.0f;   // encoder pass scalar
    float dec_layer_scalar = 1.0f;   // decoder pass scalar
    int   gpu = 0;                   // device this layer runs on
};

// Decoder-only self-conditioning GeGLU FFN (intermediate 2112).
struct DiffSelfCond {
    GpuBuffer<bf16> pre_norm;      // (H,)
    GpuBuffer<bf16> gate_up_proj;  // BF16 fast path: (2*2112, 2816), gate then up
    DiffLinearWeight gate_proj;    // GGUF/quantized path: (2112, 2816)
    DiffLinearWeight up_proj;      // GGUF/quantized path: (2112, 2816)
    DiffLinearWeight down_proj;    // (2816, 2112)
};

struct DiffWeights {
    GpuBuffer<bf16>        embed_tokens;  // (vocab, H) — tied: encoder/decoder embed + lm_head. GPU 0.
    Q8Linear               embed_tokens_q8;
    // Optional transposed (h-major [H,V]) copy of embed_tokens_q8, quantized
    // per-h (group-32 over V). Built only when DIFF_SOFT_NEXT_TN_TABLE is set, so
    // soft_next (probs @ embed) runs on the fast TN matmul path instead of the
    // strided NN kernel. ~+830MB VRAM. GPU 0.
    Q8Linear               embed_tokens_q8_t;
    GpuBuffer<bf16>        final_norm;    // (H,) decoder.norm. GPU 0.
    DiffSelfCond           self_cond;     // GPU 0
    std::vector<DiffLayer> layers;        // [30]
};

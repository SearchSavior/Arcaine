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
    // AWQ checkpoint keeps the shared MLP in BF16/F16.  Optional gate+up row
    // concatenation turns its two same-input GEMMs into one (2*intermediate,N)
    // GEMM. Populated only under DIFF_INT4_FUSE_DENSE_GATE_UP.
    GpuBuffer<bf16> gate_up_proj_bf16;
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

    // Persistent device pointer tables for the NVFP4 gpu-layout expert path
    // (run_shard_nvfp4_gpu_layout). The grouped-GEMM kernels take, per expert,
    // device pointers to that expert's packed weights/scales/dst-scale + the
    // per-expert input_global_scale. These are STABLE across denoising steps
    // (expert weights never move), so they are uploaded ONCE (at load, outside
    // any session) and reused every step -- the per-step host->device upload
    // the path used to do (upload_alloc, with a q.memcpy().wait()) cannot be
    // captured by a SYCL command_graph (the host source vector would dangle at
    // replay time). Built by ensure_expert_pointer_tables_raw() for the default
    // non-coalesced (raw weight_packed) case; the coalesced (xe2) case is built
    // lazily and is not session-capture-safe without a warmup. See
    // nvfp4_wholestep_capture_findings.md.
    GpuBuffer<const uint8_t*> pt_gate_w;
    GpuBuffer<const uint8_t*> pt_gate_s;
    GpuBuffer<const float*>   pt_gate_dst;
    GpuBuffer<float>          pt_gate_input;
    GpuBuffer<const uint8_t*> pt_down_w;
    GpuBuffer<const uint8_t*> pt_down_s;
    GpuBuffer<const float*>   pt_down_dst;
    GpuBuffer<float>          pt_down_input;
    bool                      pt_raw_built = false;
    // Coalesced (xe2 DPAS) variant: same scale/dst/input tables as pt_* above
    // (those point to per-expert scale/dst/input which are layout-independent),
    // but the weight pointers point to the coalesced weight buffers (created
    // once by nvfp4_coalesced_weight, cached on each Nvfp4Linear). Built once
    // at load so the xe2 path's per-step pointer-table upload is eliminated
    // and the path is SYCL-graph-capturable. See pt_raw_built.
    GpuBuffer<const uint8_t*> pt_gate_w_coal;
    GpuBuffer<const uint8_t*> pt_down_w_coal;
    bool                      pt_coal_built = false;

    // Persistent raw pointer tables for the native grouped INT4-AWQ DPAS MoE
    // path.  Int4Linear stores N-major packed s4 rows and [K/group,N] BF16
    // scales; the grouped kernel follows these pointers directly, avoiding a
    // per-denoising-step host upload of the local expert vector.
    GpuBuffer<const uint8_t*> pt_int4_gate_w;
    GpuBuffer<const bf16*>    pt_int4_gate_s;
    GpuBuffer<const uint8_t*> pt_int4_down_w;
    GpuBuffer<const bf16*>    pt_int4_down_s;
    bool                      pt_int4_built = false;
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
    GpuBuffer<bf16>        final_norm;    // (H,) decoder.norm. GPU 0.
    DiffSelfCond           self_cond;     // GPU 0
    std::vector<DiffLayer> layers;        // [30]
};

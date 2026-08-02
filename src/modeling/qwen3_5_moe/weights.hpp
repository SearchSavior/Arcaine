#pragma once

#include <variant>
#include <vector>
#include "../../runtime/gpu/buffer.hpp"
#include "../../runtime/gpu/ops.hpp"
#include "../../runtime/quantization/nvfp4.hpp"

// Qwen3.5-MoE device weights. NVFP4 projections are stored as Nvfp4Linear
// (identical scheme to the block-diffusion MoE: weight_packed U8, weight_scale F8_E4M3
// transposed to [K/16, N], dst_scale = input_global * weight_global). The
// unquantized BF16 params of Gated DeltaNet (in_proj_*, conv1d, A_log, dt_bias,
// norm) and the MoE router / shared_expert_gate stay as GpuBuffer<bf16>.
//
// Projection weights are a per-checkpoint variant: the NVFP4 sibling quantizes
// every Linear to Nvfp4Linear, while the AWQ INT4 checkpoint
// (compressed-tensors pack-quantized, group_size=32, asymmetric zero-points)
// quantizes ONLY the routed experts (QwenInt4ExpertsGrouped, raw u4 + native
// u4 zero points) and keeps attention/out_proj/shared-expert projections
// dense (QwenDenseLinear, F16->BF16 at upload).

// Dense (unquantized) projection uploaded from a plain .weight tensor.
struct QwenDenseLinear {
    GpuBuffer<bf16> weight;        // [out_features, in_features] row-major
    int out_features = 0;
    int in_features  = 0;
};

// A projection weight in whichever representation the checkpoint carries.
using QwenProj = std::variant<Nvfp4Linear, QwenDenseLinear>;

// C(M,N) = A(M,K) @ dequant(W)^T for any supported projection representation.
inline void qwen_matmul_proj(const bf16* A, int M, int K, const QwenProj& W,
                             bf16* C, GpuEngine& ctx) {
    if (const auto* p = std::get_if<Nvfp4Linear>(&W)) {
        matmul_nvfp4(A, M, K, *p, C, ctx);
    } else {
        const auto& d = std::get<QwenDenseLinear>(W);
        matmul_bf16(A, M, K, d.weight.data(), d.out_features, C, ctx);
    }
}

// Canonical AWQ INT4 routed-expert storage: ONE contiguous device tensor per
// projection across all experts, raw unsigned nibbles (q_u, no XOR-0x88
// rebase) and raw unsigned zero points (zp_u, packed along N). True dequant:
//   w = scale * (q_u - zp_u)
// Consumed directly by the oneDNN grouped W4A16 path (weights u4 tag::acb,
// scales/zp mask 7, groups {group_size, 1}) and by the custom DPAS fallback
// (base pointer + per-expert stride). zp buffers are empty when the whole
// projection is symmetric (zp_u == 8 everywhere).
struct QwenInt4ExpertsGrouped {
    int E = 0, hidden = 0, inter = 0, group_size = 0;
    GpuBuffer<uint8_t> gate_q;   // [E, inter, hidden/2]   u4, K innermost
    GpuBuffer<uint8_t> up_q;     // [E, inter, hidden/2]
    GpuBuffer<uint8_t> down_q;   // [E, hidden, inter/2]
    GpuBuffer<bf16>    gate_s;   // [E, hidden/gs, inter]
    GpuBuffer<bf16>    up_s;     // [E, hidden/gs, inter]
    GpuBuffer<bf16>    down_s;   // [E, inter/gs, hidden]
    GpuBuffer<uint8_t> gate_zp;  // [E, hidden/gs, inter/2]  u4 packed along N
    GpuBuffer<uint8_t> up_zp;    // [E, hidden/gs, inter/2]
    GpuBuffer<uint8_t> down_zp;  // [E, inter/gs, hidden/2]

    bool ready() const { return !gate_q.empty(); }
    bool gate_has_zp() const { return !gate_zp.empty(); }
    bool up_has_zp() const { return !up_zp.empty(); }
    bool down_has_zp() const { return !down_zp.empty(); }
};

// Full attention (Qwen3_5MoeAttention): GQA 16:2, head_dim 256, partial RoPE
// (rotary_dim 64), q/k RMSNorm per head, sigmoid/swish output gate taken from
// the 2nd half of q_proj's 8192 outputs.
struct QwenFullAttn {
    QwenProj        q_proj;   // [8192, 2048]  (Q[4096] || output-gate[4096])
    QwenProj        k_proj;   // [512, 2048]   (2 KV heads * 256)
    QwenProj        v_proj;   // [512, 2048]
    QwenProj        o_proj;   // [2048, 4096]
    GpuBuffer<bf16> q_norm;   // [256]
    GpuBuffer<bf16> k_norm;   // [256]
};

// Gated DeltaNet (Qwen3_5MoeGatedDeltaNet), Mamba2-style. All in_proj params are
// BF16/unquantized; only out_proj is NVFP4.
struct QwenLinearAttn {
    GpuBuffer<bf16> in_proj_qkv;  // [8192, 2048]  q[16*128] || k[16*128] || v[32*128]
    GpuBuffer<bf16> in_proj_z;    // [4096, 2048]  gate
    GpuBuffer<bf16> in_proj_a;    // [32, 2048]    A-log decay input
    GpuBuffer<bf16> in_proj_b;    // [32, 2048]    beta input
    GpuBuffer<bf16> conv1d;       // [8192, 1, 4]  depthwise k=4
    GpuBuffer<bf16> A_log;        // [32]
    GpuBuffer<bf16> dt_bias;      // [32]
    GpuBuffer<bf16> norm;         // [128]
    QwenProj        out_proj;     // [2048, 4096]
};

// MoE block: 256 routed experts (SwiGLU, top-8) + always-on shared
// expert (DeepSeekMoE) with a per-token scalar sigmoid gate.
struct QwenMoE {
    GpuBuffer<bf16>            router_gate;          // [256, 2048]
    std::vector<QwenProj>      experts_gate_up;      // 256 x fused [1024, 2048]
    std::vector<QwenProj>      experts_down;         // 256 x [2048, 512]
    QwenProj                   shared_gate_up;       // fused [1024, 2048]
    QwenProj                   shared_down;          // [2048, 512]
    GpuBuffer<bf16>            shared_expert_gate;   // [1, 2048]

    // AWQ INT4 checkpoints only: contiguous per-projection expert tensors
    // (raw u4 + native u4 zero points). experts_gate_up/experts_down stay
    // empty in this mode; the grouped oneDNN path, the DPAS fallback and the
    // per-expert reference path all read slices of this storage.
    QwenInt4ExpertsGrouped     grouped;
};

struct QwenLayer {
    int   layer_idx = 0;
    bool  is_full_attention = false;
    std::variant<QwenFullAttn, QwenLinearAttn> attn;
    QwenMoE          moe;
    GpuBuffer<bf16>  input_layernorm;          // [2048]
    GpuBuffer<bf16>  post_attention_layernorm; // [2048]
};

struct QwenWeights {
    GpuBuffer<bf16>        embed_tokens;  // [vocab, 2048]  (model.language_model.)
    GpuBuffer<bf16>        final_norm;     // [2048]         (model.language_model.norm)
    GpuBuffer<bf16>        lm_head;        // [vocab, 2048]  (top-level, untied)
    std::vector<QwenLayer> layers;
};

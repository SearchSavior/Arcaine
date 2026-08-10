#pragma once

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "cache.hpp"
#include "config.hpp"
#include "kernels.hpp"
#include "weights.hpp"
#include "workspace.hpp"
#include "../../runtime/quantization/fp8.hpp"
#include "../../runtime/quantization/nvfp4.hpp"
#include "../../runtime/gpu/ops.hpp"
#include "runtime/kernels/elementwise.hpp"
#include "runtime/kernels/rms_norm.hpp"

using namespace qwen35_kernels;

inline bool qwen35_nvfp4_dpas_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_NVFP4_DPAS");
        // The current dense Xe2 kernel remains available for explicit A/B
        // measurements, but oneDNN's BMG f4 implementation is materially
        // faster for both M=1 decode and large-M prefill on this checkpoint.
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline bool qwen35_xmx_attention_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_XMX_ATTENTION");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline bool qwen35_splitkv_decode_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_DECODE_XMX_SPLITKV");
        // Flash-style split-KV decode: partition the KV range across S WGs,
        // each keeping flash-softmax partials (m_p, l_p, O_p), combined in a
        // second kernel. The long-KV decode is latency-bound (serial 16-key
        // blocks + 2 barriers each in the general kernel), so S-way scan
        // parallelism hides the latency. Default ON (3x at d8192 decode,
        // neutral at d0); set to 0 to fall back to the general XMX kernel.
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline int qwen35_splitkv_decode_slices() {
    static int slices = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_DECODE_SPLITKV_SLICES");
        // 16 slices measured optimal on B70 (32 saturates; 8 leaves ~15% on
        // the table at d8192). d8192 decode: 6.75 -> 20.59 t/s vs baseline.
        if (!value) return 16;
        int parsed = std::atoi(value);
        return parsed > 0 ? parsed : 16;
    }();
    return slices;
}

inline bool qwen35_esimd_delta_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_ESIMD_DELTA");
        // The scalar/SIMT implementation remains available as the A/B
        // baseline. The ESIMD path is exact at BF16 output precision and keeps
        // the 128x128 recurrent state in registers across the sequence.
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline bool qwen35_fused_esimd_delta_decode_enabled() {
    static bool enabled = [] {
        const char* value =
            std::getenv("ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// Largest batch the per-token fused decode core is used for. Above this the
// chunked path wins, and prefill is far above it. Speculative verify windows
// are a handful of tokens, so the default covers them.
inline int qwen35_fused_decode_max_seq() {
    static int limit = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_FUSED_DECODE_MAX_SEQ");
        if (!value) return 8;
        int parsed = std::atoi(value);
        return parsed > 0 ? parsed : 8;
    }();
    return limit;
}

inline void matmul_proj(const bf16* A, int M, int K, const Qwen35Proj& W,
                        bf16* C, GpuEngine& context) {
    if (const auto* w = std::get_if<Fp8Linear>(&W))
        matmul_fp8(A, M, K, *w, C, context);
    else
        matmul_int4(A, M, K, std::get<Int4Linear>(W), C, context);
}

// Gates the fused [b|a] projection on the M=1 decode path only. The
// multi-token prefill path cannot use it - the fused output interleaves b and
// a per token, which is not the layout its consumers read - and no longer
// consults this flag.
inline bool qwen35_fused_ba_projection_enabled() {
    static bool enabled = [] {
        const char* value =
            std::getenv("ARCAINE_QWEN35_FUSED_BA_PROJECTION");
        if (!value) return true;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline void qwen35_full_attention_forward(
    GpuEngine& context,
    const Qwen35FullAttentionWeights& weights,
    Qwen35KvLayerCache& cache,
    Qwen35Workspace& workspace,
    const bf16* hidden,
    const int32_t* positions,
    bf16* output,
    int seq,
    int past,
    const Qwen35Config& config) {
    const auto& c = config.text;
    auto& queue = context.queue;
    int query_dim = c.num_attention_heads * c.head_dim;
    int key_value_dim = c.num_key_value_heads * c.head_dim;
    if (cache.filled != past)
        throw std::runtime_error("Qwen3.5 KV cache position mismatch");
    if (past + seq > cache.capacity)
        throw std::runtime_error("Qwen3.5 KV cache overflow");

    if (weights.fused_projections) {
        matmul_proj(hidden, seq, c.hidden_size, weights.qkv_proj,
                    workspace.tmp0.data(), context);
        qwen35_split_q_gate_kv(
            queue, workspace.tmp0.data(), workspace.tmp2.data(),
            workspace.tmp3.data(), workspace.tmp1.data(), workspace.tmp4.data(),
            seq, c.num_attention_heads, c.num_key_value_heads, c.head_dim);
    } else {
        matmul_proj(hidden, seq, c.hidden_size, weights.q_proj,
                    workspace.tmp0.data(), context);
        qwen35_split_q_gate(queue, workspace.tmp0.data(), workspace.tmp2.data(),
                            workspace.tmp3.data(), seq, c.num_attention_heads,
                            c.head_dim);
        matmul_proj(hidden, seq, c.hidden_size, weights.k_proj,
                    workspace.tmp1.data(), context);
        matmul_proj(hidden, seq, c.hidden_size, weights.v_proj,
                    workspace.tmp4.data(), context);
    }
    rms_norm(queue, workspace.tmp2.data(), weights.q_norm.data(), workspace.tmp2.data(),
             seq * c.num_attention_heads, c.head_dim, c.rms_norm_eps);
    rms_norm(queue, workspace.tmp1.data(), weights.k_norm.data(), workspace.tmp1.data(),
             seq * c.num_key_value_heads, c.head_dim, c.rms_norm_eps);
    qwen35_apply_mrope(queue, workspace.tmp2.data(), workspace.tmp1.data(), positions,
                       seq, c.num_attention_heads, c.num_key_value_heads,
                       c.head_dim, c.rotary_dim(), c.rope.theta, c.rope.mrope_section);

    size_t offset = (size_t)past * key_value_dim;
    size_t count = (size_t)seq * key_value_dim;
    queue.memcpy(cache.key.data() + offset, workspace.tmp1.data(), count * sizeof(bf16));
    queue.memcpy(cache.value.data() + offset, workspace.tmp4.data(), count * sizeof(bf16));
    cache.filled = past + seq;

    if (qwen35_xmx_attention_enabled()) {
        if (seq == 1 && qwen35_splitkv_decode_enabled() &&
            c.num_attention_heads % c.num_key_value_heads == 0 &&
            c.num_attention_heads / c.num_key_value_heads <= 8) {
            // M=1 decode, split-KV: partition the KV scan across S WGs
            // (flash-attention partial softmax states, combined in a second
            // kernel). Reads query across the whole head group while WGs write
            // disjoint partials to scratch and the combine writes output —
            // query and output MUST be separate buffers: write to tmp4 (free
            // after the KV cache store) then copy back to tmp2.
            int slices = qwen35_splitkv_decode_slices();
            if (slices > Qwen35Workspace::decode_max_kv_slices)
                slices = Qwen35Workspace::decode_max_kv_slices;
            qwen35_xmx_attention_decode_gqa_splitkv(
                queue, workspace.tmp2.data(), cache.key.data(),
                cache.value.data(), workspace.tmp4.data(), past,
                c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                1.0f / std::sqrt((float)c.head_dim),
                workspace.decode_part_out.data(),
                workspace.decode_part_state.data(), slices);
            const size_t qdim = (size_t)seq * query_dim;
            queue.memcpy(workspace.tmp2.data(), workspace.tmp4.data(),
                         qdim * sizeof(bf16));
        } else if (seq > 1) {
            // Prefill: restructured flash-style XMX kernel (128-query tiles,
            // register-resident Q/O per subgroup, SG-local softmax, K via 32B
            // vector loads, V^T staged in SLM). ~4-5x over the original at
            // p2048-p4096 with identical numerics class.
            qwen35_xmx_attention_v2(
                queue, workspace.tmp2.data(), cache.key.data(),
                cache.value.data(), workspace.tmp2.data(), seq, past,
                c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                1.0f / std::sqrt((float)c.head_dim));
        } else {
            qwen35_xmx_attention(
                queue, workspace.tmp2.data(), cache.key.data(),
                cache.value.data(), workspace.tmp2.data(), seq, past,
                c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                1.0f / std::sqrt((float)c.head_dim));
        }
    } else {
        qwen35_online_attention(queue, workspace.tmp2.data(), cache.key.data(),
                                cache.value.data(), workspace.tmp2.data(), seq, past,
                                c.num_attention_heads, c.num_key_value_heads,
                                c.head_dim, 1.0f / std::sqrt((float)c.head_dim));
    }
    mul_sigmoid_inplace(queue, workspace.tmp2.data(), workspace.tmp3.data(),
                        (size_t)seq * query_dim);
    matmul_proj(workspace.tmp2.data(), seq, query_dim, weights.o_proj, output, context);
}

inline void qwen35_linear_attention_forward(
    GpuEngine& context,
    const Qwen35LinearAttentionWeights& weights,
    Qwen35DeltaLayerCache& cache,
    Qwen35Workspace& workspace,
    const bf16* hidden,
    bf16* output,
    int seq,
    const Qwen35Config& config) {
    const auto& c = config.text;
    auto& queue = context.queue;
    int key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
    int value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
    int conv_dim = 2 * key_dim + value_dim;
    int heads = c.linear_num_value_heads;
    size_t head_values = (size_t)seq * heads;

    int projected_stride = conv_dim;
    if (weights.fused_projections) {
        projected_stride = conv_dim + value_dim;
        matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_qkvz,
                    workspace.tmp0.data(), context);
    } else {
        matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_qkv,
                    workspace.tmp0.data(), context);
    }
    // The fused decode core handles one token, so a short batch runs it once
    // per token rather than falling through to the chunked path below. Batch
    // size then stops selecting between two implementations that do not agree
    // numerically: before this, a two-token forward and two one-token forwards
    // over the same tokens produced different logits, which is visible as soon
    // as anything verifies a batch against sequential decoding.
    //
    // Only the recurrent core loops. The projection above is already batched
    // over the whole window, so this re-reads the recurrent state and the small
    // conv/gate tensors, not the weights.
    // qwen35_esimd_delta_enabled() belongs here even though this arm never
    // calls qwen35_recurrent_delta_esimd.
    //
    // Both arms advance the same cache.recurrent_state, and the three
    // recurrent implementations do not agree on its layout:
    //
    //   qwen35_recurrent_delta        state[head*K*V + k*V + v]  [head][key][value]
    //   qwen35_recurrent_delta_esimd  state[head*V*K + v*K + k]  [head][value][key]
    //   qwen35_delta_decode_fused_esimd (below)                  [head][value][key]
    //
    // K and V are both 128 here, so a mismatch is exactly size-compatible:
    // nothing faults, the state is silently transposed between the prefill
    // that wrote it and the decode that reads it.
    //
    // Without this term, ARCAINE_QWEN35_ESIMD_DELTA=0 selects the scalar
    // kernel for prefill while leaving the fused ESIMD core on for decode, and
    // the two corrupt each other across every generation. Measured over 400
    // teacher-forced records, that arm agreed with the default configuration on
    // 33.75% of tokens, against 92% or better for every other kernel flag.
    // An fp64 host reference puts both recurrent kernels within 0.67 bf16 ULP
    // of the true recurrence, so neither was ever at fault - only the pairing.
    // With this term the same arm reports 92.75%, in line with the rest.
    //
    // kernels.hpp already noted that a cache must not switch layouts
    // mid-stream; nothing enforced it. The flag now switches the whole
    // DeltaNet path coherently, which is what a baseline A/B switch should do.
    // The deeper fix is to give the scalar kernel the same [head][value][key]
    // layout so no combination can mix them; this makes the unsafe one
    // unreachable meanwhile.
    if (seq >= 1 && seq <= qwen35_fused_decode_max_seq() &&
        weights.fused_projections && qwen35_fused_esimd_delta_decode_enabled() &&
        qwen35_esimd_delta_enabled()) {
        for (int token = 0; token < seq; ++token) {
            const bf16* token_hidden = hidden + (size_t)token * c.hidden_size;
            const bf16* projected =
                workspace.tmp0.data() + (size_t)token * projected_stride;
            bf16* ba = workspace.tmp1.data() + (size_t)token * 2 * heads;
            bf16* core = workspace.tmp4.data() + (size_t)token * value_dim;
            bf16* gate = workspace.tmp2.data() + (size_t)token * value_dim;

            // Kept at M=1 so a token sees the same projection arithmetic it
            // would have seen decoding alone. These weights are a few hundred
            // KB against the tens of GB a decode step already moves.
            if (qwen35_fused_ba_projection_enabled())
                matmul_bf16(token_hidden, 1, c.hidden_size,
                            weights.in_proj_ba.data(), 2 * heads, ba, context);
            else {
                matmul_bf16(token_hidden, 1, c.hidden_size,
                            weights.in_proj_b.data(), heads, ba, context);
                matmul_bf16(token_hidden, 1, c.hidden_size,
                            weights.in_proj_a.data(), heads, ba + heads, context);
            }
            qwen35_delta_decode_fused_esimd(
                queue, projected, projected_stride,
                weights.conv1d_time_major.data(), cache.conv_state.data(),
                weights.A_log.data(), weights.dt_bias.data(), ba,
                cache.recurrent_state.data(), core, gate,
                c.linear_num_key_heads, heads, c.linear_key_head_dim,
                c.linear_value_head_dim, conv_dim, c.linear_conv_kernel_dim,
                c.rms_norm_eps);
            // Shifts this token into the history before the next reads it. The
            // in-order queue keeps the core and the update interleaved.
            qwen35_update_conv_state_time_major(queue, projected,
                                                projected_stride,
                                                cache.conv_state.data(), conv_dim);
        }
        cache.has_state = true;
        gated_rmsnorm(
            queue, workspace.tmp4.data(), workspace.tmp2.data(),
            weights.norm.data(), workspace.tmp4.data(), seq * heads,
            c.linear_value_head_dim, c.rms_norm_eps);
        matmul_proj(workspace.tmp4.data(), seq, value_dim, weights.out_proj,
                    output, context);
        return;
    }
    qwen35_conv_causal(queue, workspace.tmp0.data(), weights.conv1d.data(),
                       cache.conv_state.data(), workspace.tmp1.data(), seq,
                       conv_dim, c.linear_conv_kernel_dim, cache.has_state,
                       projected_stride);
    qwen35_update_conv_state(queue, workspace.tmp0.data(), cache.conv_state.data(),
                             seq, conv_dim, c.linear_conv_kernel_dim, cache.has_state,
                             projected_stride);
    qwen35_extract_qkv(queue, workspace.tmp1.data(), workspace.tmp2.data(),
                       workspace.tmp3.data(), workspace.tmp4.data(), seq,
                       c.linear_num_key_heads, heads, c.linear_key_head_dim,
                       c.linear_value_head_dim);
    l2norm(queue, workspace.tmp2.data(), workspace.tmp2.data(), seq * heads,
           c.linear_key_head_dim, c.rms_norm_eps);
    l2norm(queue, workspace.tmp3.data(), workspace.tmp3.data(), seq * heads,
           c.linear_key_head_dim, c.rms_norm_eps);
    scale_inplace(queue, workspace.tmp2.data(),
                  (size_t)seq * heads * c.linear_key_head_dim,
                  1.0f / std::sqrt((float)c.linear_key_head_dim));

    // The unfused path reuses tmp0 for z. The fused path keeps z at the tail
    // of each projected row until the recurrent core has consumed q/k/v.
    if (!weights.fused_projections)
        matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_z,
                    workspace.tmp0.data(), context);
    bf16* beta = workspace.tmp1.data();
    bf16* g = workspace.tmp1.data() + head_values;
    // Two matmuls, deliberately, even though a fused [b|a] weight exists.
    //
    // matmul_bf16 writes C(M,N) row-major, and in_proj_ba stacks b's rows then
    // a's, so C(seq, 2*heads) comes back with each token's b and a adjacent:
    // token t occupies [t*2*heads, (t+1)*2*heads). The two consumers below read
    // separate contiguous blocks instead - beta at tmp1 and g at
    // tmp1 + seq*heads, both indexed [token * heads + head] - which describes
    // the same bytes only when seq == 1.
    //
    // Using the fused matmul here therefore fed the DeltaNet recurrence some
    // other token's gate values for every token after the first, on the
    // default configuration, for any prompt longer than the fused-decode
    // guard's 8 tokens. Measured against a corrected engine over 1000 records:
    // top-1 agreement 0.87 and perplexity 191.99 against 186.16.
    //
    // It survived because the decode path above builds ba per token at M=1,
    // where the two layouts coincide, and because scrambled gates perturb the
    // output rather than obviously breaking it.
    //
    // These are hidden_size x heads GEMMs, launch-bound at any sequence
    // length: pp512 measures 664.45 -> 664.20 t/s, inside run-to-run noise.
    matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_b.data(), heads,
                beta, context);
    matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_a.data(), heads,
                g, context);
    sigmoid_inplace(queue, beta, head_values);
    qwen35_compute_g(queue, g, weights.A_log.data(), weights.dt_bias.data(),
                     g, seq, heads);
    if (qwen35_esimd_delta_enabled() && c.linear_key_head_dim == 128 &&
        c.linear_value_head_dim == 128) {
        qwen35_recurrent_delta_esimd_opt(
            queue, workspace.tmp2.data(), workspace.tmp3.data(),
            workspace.tmp4.data(), beta, g, cache.recurrent_state.data(),
            workspace.tmp4.data(), seq, heads, c.linear_key_head_dim,
            c.linear_value_head_dim);
    } else {
        qwen35_recurrent_delta(queue, workspace.tmp2.data(), workspace.tmp3.data(),
                               workspace.tmp4.data(), beta, g,
                               cache.recurrent_state.data(), workspace.tmp4.data(),
                               seq, heads, c.linear_key_head_dim,
                               c.linear_value_head_dim);
    }
    cache.has_state = true;
    bf16* z = workspace.tmp0.data();
    if (weights.fused_projections) {
        qwen35_copy_strided(queue, workspace.tmp0.data(), projected_stride,
                            conv_dim, workspace.tmp1.data(), seq, value_dim);
        z = workspace.tmp1.data();
    }
    gated_rmsnorm(queue, workspace.tmp4.data(), z,
                  weights.norm.data(), workspace.tmp4.data(), seq * heads,
                  c.linear_value_head_dim, c.rms_norm_eps);
    matmul_proj(workspace.tmp4.data(), seq, value_dim, weights.out_proj,
                output, context);
}

inline void qwen35_mlp_forward(
    GpuEngine& context,
    const Qwen35MlpWeights& weights,
    Qwen35Workspace& workspace,
    const bf16* hidden,
    bf16* output,
    int seq,
    const Qwen35Config& config) {
    int H = config.text.hidden_size;
    int I = config.text.intermediate_size;
    auto& queue = context.queue;
    if (std::holds_alternative<Int4Linear>(weights.gate_up)) {
        const auto& gate_up = std::get<Int4Linear>(weights.gate_up);
        const auto& down = std::get<Int4Linear>(weights.down);
        matmul_int4(hidden, seq, H, gate_up, workspace.tmp0.data(), context);
        swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
        matmul_int4(workspace.tmp1.data(), seq, I, down, output, context);
    } else if (std::holds_alternative<Nvfp4Linear>(weights.gate_up)) {
        const auto& gate_up = std::get<Nvfp4Linear>(weights.gate_up);
        const auto& down = std::get<Nvfp4Linear>(weights.down);
        if (qwen35_nvfp4_dpas_enabled()) {
            pack_bf16_to_nvfp4(queue, hidden, workspace.input_packed.data(),
                               workspace.input_scale.data(), seq, H,
                               gate_up.input_global_scale);
            matmul_nvfp4_swiglu_pack_xe2(
                context, workspace.input_packed.data(), workspace.input_scale.data(),
                seq, H, gate_up, down, workspace.activation_packed.data(),
                workspace.activation_scale.data());
            matmul_nvfp4_packed_xe2(
                context, workspace.activation_packed.data(),
                workspace.activation_scale.data(), seq, I, down, output);
        } else {
            matmul_nvfp4(hidden, seq, H, gate_up, workspace.tmp0.data(), context,
                         workspace.input_packed.data(), workspace.input_scale.data());
            swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
            matmul_nvfp4(workspace.tmp1.data(), seq, I, down, output, context,
                         workspace.activation_packed.data(),
                         workspace.activation_scale.data());
        }
    } else {
        const auto& gate_up = std::get<Fp8Linear>(weights.gate_up);
        const auto& down = std::get<Fp8Linear>(weights.down);
        matmul_fp8(hidden, seq, H, gate_up, workspace.tmp0.data(), context);
        swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
        matmul_fp8(workspace.tmp1.data(), seq, I, down, output, context);
    }
}

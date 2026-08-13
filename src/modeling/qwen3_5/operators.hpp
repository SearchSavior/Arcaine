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
#include "runtime/profiling/launch_prof.hpp"

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

// INT4 zp-correction GEMM scratch: workspace-resident for small-M (decode /
// speculative) calls, heap fallback (nullptr) for prefill.
inline bf16* qwen35_int4_corr_scratch(Qwen35Workspace& ws, int seq) {
    return seq <= Qwen35Workspace::int4_corr_rows
               ? ws.int4_corr_decode.data() : nullptr;
}

// ARCAINE_QWEN35_INT4_CORR_FUSED: fold the asymmetric-zp subtract into the
// consumer kernels (split / swiglu / conv / add_inplace) instead of a dedicated
// subtract launch. Bit-identical output (the consumers bf16-round the C - corr
// difference exactly like the subtract kernel). Removes one launch per int4
// projection and the C write-back the subtract performed.
inline bool qwen35_int4_corr_fused_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_INT4_CORR_FUSED");
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// Grow-only per-forward corr buffer (sized seq*max_n, grown to the largest N
// seen; the workspace reserves max_seq-sized buffers but the corr output only
// needs the current forward's rows).
inline bf16* qwen35_int4_corr_work(Qwen35Workspace& ws, int seq,
                                   sycl::queue& queue, int max_n) {
    size_t need = (size_t)seq * max_n;
    if (ws.int4_corr_work_seq < (size_t)seq || ws.int4_corr_work.count() < need) {
        ws.int4_corr_work = GpuBuffer<bf16>(need, queue);
        ws.int4_corr_work_seq = seq;
    }
    return ws.int4_corr_work.data();
}

// Residual-add corr state returned by the operator forwards: when non-null, the
// caller's add_inplace should fold corr (i.e. hidden += C - corr) instead of
// the plain add. corr is (seq, N) with N = the projection's out_features.
struct Qwen35CorrOut {
    const bf16* corr = nullptr;
    int N = 0;
};

inline void matmul_proj(const bf16* A, int M, int K, const Qwen35Proj& W,
                        bf16* C, GpuEngine& context,
                        bf16* int4_rowsum_scratch = nullptr,
                        bf16* int4_corr_scratch = nullptr) {
    if (const auto* w = std::get_if<Fp8Linear>(&W))
        matmul_fp8(A, M, K, *w, C, context);
    else
        matmul_int4(A, M, K, std::get<Int4Linear>(W), C, context,
                    int4_rowsum_scratch, int4_corr_scratch);
}

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

// ARCAINE_QWEN35_FUSED_NORM_ROPE: fold per-head q/k RMS norm + MRoPE into one
// launch (qwen35_norm_rope_fused) instead of three. Bit-exact; A/B off by
// default until measured.
inline bool qwen35_fused_norm_rope_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_FUSED_NORM_ROPE");
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// ARCAINE_QWEN35_SPLIT_WRITE_CACHE: the fused qkv split writes K/V straight
// into the KV cache (at past*kv_dim) instead of tmp1/tmp4 followed by two D2D
// memcpys. Bit-exact (same bytes, same layout). Removes 2 launches per
// full-attention layer.
inline bool qwen35_split_write_cache_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_SPLIT_WRITE_CACHE");
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

// ARCAINE_QWEN35_SPLITKV_FUSED_EPILOGUE: the splitkv combine kernel writes
// O*sigmoid(gate) directly to the query buffer (safe: it runs after the part
// kernels finish reading query) instead of tmp4 + copy-back + mul_sigmoid.
// Bit-exact. Removes 2 launches per decode full-attention layer.
inline bool qwen35_splitkv_fused_epilogue_enabled() {
    static bool enabled = [] {
        const char* value =
            std::getenv("ARCAINE_QWEN35_SPLITKV_FUSED_EPILOGUE");
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline Qwen35CorrOut qwen35_full_attention_forward(
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
        bool cache_write = qwen35_split_write_cache_enabled();
        bf16* cache_key = cache_write ? cache.key.data() : nullptr;
        bf16* cache_value = cache_write ? cache.value.data() : nullptr;
        if (qwen35_int4_corr_fused_enabled() &&
            std::holds_alternative<Int4Linear>(weights.qkv_proj) &&
            std::get<Int4Linear>(weights.qkv_proj).has_zp()) {
            const auto& w = std::get<Int4Linear>(weights.qkv_proj);
            int N = 2 * query_dim + 2 * key_value_dim;
            bf16* corr = qwen35_int4_corr_work(workspace, seq, queue, N);
            matmul_int4_zp_corr(hidden, seq, c.hidden_size, w,
                                workspace.tmp0.data(), context,
                                workspace.int4_rowsum.data(), corr);
            qwen35_split_q_gate_kv(
                queue, workspace.tmp0.data(), workspace.tmp2.data(),
                workspace.tmp3.data(), workspace.tmp1.data(),
                workspace.tmp4.data(), seq, c.num_attention_heads,
                c.num_key_value_heads, c.head_dim, cache_key, cache_value,
                past, corr);
        } else {
            matmul_proj(hidden, seq, c.hidden_size, weights.qkv_proj,
                        workspace.tmp0.data(), context, workspace.int4_rowsum.data(),
                        qwen35_int4_corr_scratch(workspace, seq));
            qwen35_split_q_gate_kv(
                queue, workspace.tmp0.data(), workspace.tmp2.data(),
                workspace.tmp3.data(), workspace.tmp1.data(), workspace.tmp4.data(),
                seq, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                cache_key, cache_value, past, nullptr);
        }
    } else {
        matmul_proj(hidden, seq, c.hidden_size, weights.q_proj,
                    workspace.tmp0.data(), context, workspace.int4_rowsum.data(),
                    qwen35_int4_corr_scratch(workspace, seq));
        qwen35_split_q_gate(queue, workspace.tmp0.data(), workspace.tmp2.data(),
                            workspace.tmp3.data(), seq, c.num_attention_heads,
                            c.head_dim);
        matmul_proj(hidden, seq, c.hidden_size, weights.k_proj,
                    workspace.tmp1.data(), context, workspace.int4_rowsum.data(),
                    qwen35_int4_corr_scratch(workspace, seq));
        matmul_proj(hidden, seq, c.hidden_size, weights.v_proj,
                    workspace.tmp4.data(), context, workspace.int4_rowsum.data(),
                    qwen35_int4_corr_scratch(workspace, seq));
    }
    if (qwen35_fused_norm_rope_enabled()) {
        // One launch: per-head q/k rmsnorm + MRoPE. Bit-exact vs the three
        // separate launches below (same reduction order, same bf16 stores).
        qwen35_norm_rope_fused(
            queue, workspace.tmp2.data(), workspace.tmp1.data(),
            weights.q_norm.data(), weights.k_norm.data(), positions, seq,
            c.num_attention_heads, c.num_key_value_heads, c.head_dim,
            c.rotary_dim(), c.rope.theta, c.rope.mrope_section, c.rms_norm_eps);
    } else {
        rms_norm(queue, workspace.tmp2.data(), weights.q_norm.data(), workspace.tmp2.data(),
                 seq * c.num_attention_heads, c.head_dim, c.rms_norm_eps);
        rms_norm(queue, workspace.tmp1.data(), weights.k_norm.data(), workspace.tmp1.data(),
                 seq * c.num_key_value_heads, c.head_dim, c.rms_norm_eps);
        qwen35_apply_mrope(queue, workspace.tmp2.data(), workspace.tmp1.data(), positions,
                           seq, c.num_attention_heads, c.num_key_value_heads,
                           c.head_dim, c.rotary_dim(), c.rope.theta, c.rope.mrope_section);
    }

    if (!(weights.fused_projections && qwen35_split_write_cache_enabled())) {
        size_t offset = (size_t)past * key_value_dim;
        size_t count = (size_t)seq * key_value_dim;
        auto _ev_k = queue.memcpy(cache.key.data() + offset, workspace.tmp1.data(), count * sizeof(bf16));
        auto _ev_v = queue.memcpy(cache.value.data() + offset, workspace.tmp4.data(), count * sizeof(bf16));
        launchprof::record("cache_k_memcpy", _ev_k);
        launchprof::record("cache_v_memcpy", _ev_v);
    }
    cache.filled = past + seq;

    bool attn_gate_applied = false;
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
            bool fused_epi = qwen35_splitkv_fused_epilogue_enabled();
            if (fused_epi) {
                // Combine runs after all part WGs (in-order queue), so it may
                // write tmp2 (query) directly, applying sigmoid(gate) inline.
                qwen35_xmx_attention_decode_gqa_splitkv(
                    queue, workspace.tmp2.data(), cache.key.data(),
                    cache.value.data(), workspace.tmp2.data(), past,
                    c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                    1.0f / std::sqrt((float)c.head_dim),
                    workspace.decode_part_out.data(),
                    workspace.decode_part_state.data(), slices,
                    workspace.tmp3.data());
                attn_gate_applied = true;
            } else {
                qwen35_xmx_attention_decode_gqa_splitkv(
                    queue, workspace.tmp2.data(), cache.key.data(),
                    cache.value.data(), workspace.tmp4.data(), past,
                    c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                    1.0f / std::sqrt((float)c.head_dim),
                    workspace.decode_part_out.data(),
                    workspace.decode_part_state.data(), slices);
                const size_t qdim = (size_t)seq * query_dim;
                auto _ev_cp = queue.memcpy(workspace.tmp2.data(), workspace.tmp4.data(),
                                           qdim * sizeof(bf16));
                launchprof::record("splitkv_copy_back", _ev_cp);
            }
        } else if (seq > 1) {
            // Prefill: pure-ESIMD streamed K/V mainloop (vllm-xpu
            // chunk_prefill recipe), fp16 dpas, grf_size<256>, KV_BLOCK=32.
            // A/B vs the former SLM-staged v2: 2-4x kernel-level in every
            // cell (q128-q4096, kv up to 131072), bit-exact numerics;
            // e2e -p 512,1024,2048,4096 at d32000: 354/330/361/354 ->
            // 647/798/800/791 t/s.
            qwen35_xmx_attention_v3(
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
    if (!attn_gate_applied)
        mul_sigmoid_inplace(queue, workspace.tmp2.data(), workspace.tmp3.data(),
                            (size_t)seq * query_dim);
    Qwen35CorrOut corr_out;
    if (qwen35_int4_corr_fused_enabled() &&
        std::holds_alternative<Int4Linear>(weights.o_proj) &&
        std::get<Int4Linear>(weights.o_proj).has_zp()) {
        const auto& w = std::get<Int4Linear>(weights.o_proj);
        bf16* corr = qwen35_int4_corr_work(workspace, seq, queue, c.hidden_size);
        matmul_int4_zp_corr(workspace.tmp2.data(), seq, query_dim, w, output,
                            context, workspace.int4_rowsum.data(), corr);
        corr_out = {corr, c.hidden_size};
    } else {
        matmul_proj(workspace.tmp2.data(), seq, query_dim, weights.o_proj, output, context,
                    workspace.int4_rowsum.data(),
                    qwen35_int4_corr_scratch(workspace, seq));
    }
    return corr_out;
}

// out_proj (and MLP down_proj) with optional fused-corr: returns the corr state
// so the caller's residual add can fold the subtract (hidden += C - corr).
// W is either Qwen35Proj (attn/GDN) or the MLP variant; the MLP callers only
// reach this from their Int4Linear branch.
template <typename W>
inline Qwen35CorrOut qwen35_residual_proj(GpuEngine& context, const bf16* core,
                                          int seq, int K, const W& Wt,
                                          bf16* output, Qwen35Workspace& ws,
                                          int out_n) {
    if (qwen35_int4_corr_fused_enabled() &&
        std::holds_alternative<Int4Linear>(Wt) &&
        std::get<Int4Linear>(Wt).has_zp()) {
        const auto& w = std::get<Int4Linear>(Wt);
        bf16* corr = qwen35_int4_corr_work(ws, seq, context.queue, out_n);
        matmul_int4_zp_corr(core, seq, K, w, output, context,
                            ws.int4_rowsum.data(), corr);
        return {corr, out_n};
    }
    if constexpr (std::is_same_v<W, Qwen35Proj>) {
        matmul_proj(core, seq, K, Wt, output, context, ws.int4_rowsum.data(),
                    qwen35_int4_corr_scratch(ws, seq));
    } else {
        matmul_int4(core, seq, K, std::get<Int4Linear>(Wt), output, context,
                    ws.int4_rowsum.data(), qwen35_int4_corr_scratch(ws, seq));
    }
    return {};
}

// ARCAINE_QWEN35_GDN_FUSIONS: chunked-path micro-fusions, each bit-exact:
//   conv_causal + update_conv_state -> one launch
//   l2norm(q) + l2norm(k) + scale(q) -> one launch
//   sigmoid(beta) + compute_g(g)     -> one launch
inline bool qwen35_gdn_fusions_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("ARCAINE_QWEN35_GDN_FUSIONS");
        if (!value) return false;
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "false") != 0 && std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

inline Qwen35CorrOut qwen35_linear_attention_forward(
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
    if (weights.fused_projections)
        projected_stride = conv_dim + value_dim;

    // The fused decode core handles one token, so a short batch runs it once
    // per token rather than falling through to the chunked path below. Batch
    // size then stops selecting between two implementations that do not agree
    // numerically: before this, a two-token forward and two one-token forwards
    // over the same tokens produced different logits, which is visible as soon
    // as anything verifies a batch against sequential decoding.
    //
    // Only the recurrent core loops. The projection below is already batched
    // over the whole window, so this re-reads the recurrent state and the small
    // conv/gate tensors, not the weights.
    if (seq >= 1 && seq <= qwen35_fused_decode_max_seq() &&
        weights.fused_projections && qwen35_fused_esimd_delta_decode_enabled()) {
        // The esimd core consumes the projection directly, so the zp correction
        // stays materialized (matmul_proj -> matmul_int4 with subtract).
        matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_qkvz,
                    workspace.tmp0.data(), context, workspace.int4_rowsum.data(),
                    qwen35_int4_corr_scratch(workspace, seq));
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
        return qwen35_residual_proj(context, workspace.tmp4.data(), seq,
                                    value_dim, weights.out_proj, output,
                                    workspace, c.hidden_size);
    }
    if (weights.fused_projections &&
        qwen35_int4_corr_fused_enabled() &&
        std::holds_alternative<Int4Linear>(weights.in_proj_qkvz) &&
        std::get<Int4Linear>(weights.in_proj_qkvz).has_zp()) {
        const auto& w = std::get<Int4Linear>(weights.in_proj_qkvz);
        int N = conv_dim + value_dim;
        bf16* corr = qwen35_int4_corr_work(workspace, seq, queue, N);
        matmul_int4_zp_corr(hidden, seq, c.hidden_size, w, workspace.tmp0.data(),
                            context, workspace.int4_rowsum.data(), corr);
        qwen35_conv_causal_corr(queue, workspace.tmp0.data(), corr,
                                weights.conv1d.data(), cache.conv_state.data(),
                                workspace.tmp1.data(), seq, conv_dim,
                                c.linear_conv_kernel_dim, cache.has_state,
                                projected_stride);
    } else {
        if (weights.fused_projections)
            matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_qkvz,
                        workspace.tmp0.data(), context,
                        workspace.int4_rowsum.data(),
                        qwen35_int4_corr_scratch(workspace, seq));
        else
            matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_qkv,
                        workspace.tmp0.data(), context,
                        workspace.int4_rowsum.data(),
                        qwen35_int4_corr_scratch(workspace, seq));
        if (qwen35_gdn_fusions_enabled())
            qwen35_conv_causal_state(queue, workspace.tmp0.data(),
                                     weights.conv1d.data(),
                                     cache.conv_state.data(),
                                     workspace.tmp1.data(),
                                     cache.conv_state.data(), seq, conv_dim,
                                     c.linear_conv_kernel_dim, cache.has_state,
                                     projected_stride);
        else {
            qwen35_conv_causal(queue, workspace.tmp0.data(), weights.conv1d.data(),
                               cache.conv_state.data(), workspace.tmp1.data(), seq,
                               conv_dim, c.linear_conv_kernel_dim, cache.has_state,
                               projected_stride);
            qwen35_update_conv_state(queue, workspace.tmp0.data(),
                                     cache.conv_state.data(), seq, conv_dim,
                                     c.linear_conv_kernel_dim, cache.has_state,
                                     projected_stride);
        }
    }
    qwen35_extract_qkv(queue, workspace.tmp1.data(), workspace.tmp2.data(),
                       workspace.tmp3.data(), workspace.tmp4.data(), seq,
                       c.linear_num_key_heads, heads, c.linear_key_head_dim,
                       c.linear_value_head_dim);
    if (qwen35_gdn_fusions_enabled()) {
        qwen35_l2norm_scale_k(queue, workspace.tmp2.data(), workspace.tmp3.data(),
                              workspace.tmp2.data(), workspace.tmp3.data(),
                              seq * heads, c.linear_key_head_dim, c.rms_norm_eps,
                              1.0f / std::sqrt((float)c.linear_key_head_dim));
    } else {
        l2norm(queue, workspace.tmp2.data(), workspace.tmp2.data(), seq * heads,
               c.linear_key_head_dim, c.rms_norm_eps);
        l2norm(queue, workspace.tmp3.data(), workspace.tmp3.data(), seq * heads,
               c.linear_key_head_dim, c.rms_norm_eps);
        scale_inplace(queue, workspace.tmp2.data(),
                      (size_t)seq * heads * c.linear_key_head_dim,
                      1.0f / std::sqrt((float)c.linear_key_head_dim));
    }

    // The unfused path reuses tmp0 for z. The fused path keeps z at the tail
    // of each projected row until the recurrent core has consumed q/k/v.
    if (!weights.fused_projections)
        matmul_proj(hidden, seq, c.hidden_size, weights.in_proj_z,
                    workspace.tmp0.data(), context, workspace.int4_rowsum.data(),
                    qwen35_int4_corr_scratch(workspace, seq));
    bf16* beta = workspace.tmp1.data();
    bf16* g = workspace.tmp1.data() + head_values;
    if (qwen35_fused_ba_projection_enabled())
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_ba.data(),
                    2 * heads, beta, context);
    else {
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_b.data(), heads,
                    beta, context);
        matmul_bf16(hidden, seq, c.hidden_size, weights.in_proj_a.data(), heads,
                    g, context);
    }
    if (qwen35_gdn_fusions_enabled()) {
        qwen35_sigmoid_beta_compute_g(queue, beta, g, weights.A_log.data(),
                                      weights.dt_bias.data(), g, seq, heads);
    } else {
        sigmoid_inplace(queue, beta, head_values);
        qwen35_compute_g(queue, g, weights.A_log.data(), weights.dt_bias.data(),
                         g, seq, heads);
    }
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
    return qwen35_residual_proj(context, workspace.tmp4.data(), seq, value_dim,
                                weights.out_proj, output, workspace,
                                c.hidden_size);
}

inline Qwen35CorrOut qwen35_mlp_forward(
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
        if (qwen35_int4_corr_fused_enabled() && gate_up.has_zp()) {
            bf16* corr = qwen35_int4_corr_work(workspace, seq, queue, 2 * I);
            matmul_int4_zp_corr(hidden, seq, H, gate_up, workspace.tmp0.data(),
                                context, workspace.int4_rowsum.data(), corr);
            qwen35_swiglu_strided_corr(queue, workspace.tmp0.data(), corr,
                                       workspace.tmp1.data(), seq, I);
        } else {
            matmul_int4(hidden, seq, H, gate_up, workspace.tmp0.data(), context,
                        workspace.int4_rowsum.data(),
                        qwen35_int4_corr_scratch(workspace, seq));
            swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
        }
        return qwen35_residual_proj(context, workspace.tmp1.data(), seq, I,
                                    weights.down, output, workspace, H);
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
        return {};
    } else {
        const auto& gate_up = std::get<Fp8Linear>(weights.gate_up);
        const auto& down = std::get<Fp8Linear>(weights.down);
        matmul_fp8(hidden, seq, H, gate_up, workspace.tmp0.data(), context);
        swiglu_strided(queue, workspace.tmp0.data(), workspace.tmp1.data(), seq, I);
        matmul_fp8(workspace.tmp1.data(), seq, I, down, output, context);
        return {};
    }
}

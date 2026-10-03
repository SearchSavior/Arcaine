#pragma once
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/sycl_graph_session.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/elementwise.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "kv_cache.hpp"
#include "attention.hpp"
#include "moe.hpp"
#include "../../utils/profile.hpp"
#include "arena.hpp"
#include <atomic>
#include <cstdint>
#include <stdexcept>

// ---------------------------------------------------------------------------
// DIFF_INT4_SYCL_GRAPH_VALIDATE: per-layer bitwise audit of the captured
// attention graph. On every step where the graph actually executed (replay,
// or the capture step's finalize+dispatch), the same kernel sequence is
// re-run eagerly from a saved clone of the block-entry `hidden` and the two
// results are compared element-wise. The eager MoE/FFN path around the
// segment is not deterministic across entire runs (atomics-based expert
// routing), so whole-generation text equality cannot serve as the correctness
// gate; this audit checks exactly what capture/replay is responsible for:
// same kernels + same input + same buffers => bit-identical output.
// The clone is carved and copied BEFORE any recording/replay dispatch so it
// snapshots the pre-segment input on every step.
// ---------------------------------------------------------------------------
inline bool diff_int4_graph_validate_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_INT4_SYCL_GRAPH_VALIDATE");
        return env && std::strcmp(env, "0") && std::strcmp(env, "off") &&
               std::strcmp(env, "false") && std::strcmp(env, "no");
    }();
    return enabled;
}

inline std::atomic<long long>& diff_int4_attn_validate_compared() {
    static std::atomic<long long> v{0};
    return v;
}
inline std::atomic<long long>& diff_int4_attn_validate_mismatched() {
    static std::atomic<long long> v{0};
    return v;
}

// Bitwise audit of two element ranges. Reports two quantities:
//   * count: the exact-bit mismatch count -- statistical and informational,
//     dominated by the model's inherent run-to-run nondeterminism (oneDNN's
//     jit gemm for the full-layer o_proj shape K=8192,N=2816 varies 1-4
//     mantissa steps run-over-run for identical operands, verified at
//     operator level; eager double-runs are not bitwise reproducible there
//     either), consumed by the self-calibrating envelope gate below.
//   * beyond-sane count: elements differing by more than 0.25 absolute
//     (vs hidden-scale activations of O(1)). ULP-level jitter never
//     approaches this; a stale-address or wrong-sequence replay lights up
//     thousands of elements here and fails the audit. Syncs the queue.
struct DiffValidateDump {
    static constexpr int kMax = 8;
    long long count = 0;      // exact-bit mismatches (informational)
    long long tol_count = 0;   // beyond-tolerance mismatches (gating)
    long long idx[kMax] = {};
    uint16_t va[kMax] = {};
    uint16_t vb[kMax] = {};
    long long tol_idx[kMax] = {};
    uint16_t tol_va[kMax] = {};
    uint16_t tol_vb[kMax] = {};
    float tol_max_abs = 0;    // max |a-b| over compared range
};
inline DiffValidateDump diff_int4_attn_validate_diff(sycl::queue& q,
                                                     const bf16* a,
                                                     const bf16* b, size_t n) {
    static GpuBuffer<int32_t> count(1, q);   // function-local: one per process
    static GpuBuffer<int32_t> tol_count(1, q);
    static GpuBuffer<long long> idx(DiffValidateDump::kMax, q);
    static GpuBuffer<uint16_t> vals(2 * DiffValidateDump::kMax, q);
    static GpuBuffer<long long> tol_idx(DiffValidateDump::kMax, q);
    static GpuBuffer<uint16_t> tol_vals(2 * DiffValidateDump::kMax, q);
    static GpuBuffer<float> tol_max(1, q);
    q.memset(count.data(), 0, sizeof(int32_t));
    q.memset(tol_count.data(), 0, sizeof(int32_t));
    q.memset(idx.data(), 0xFF, sizeof(long long) * DiffValidateDump::kMax);
    q.memset(vals.data(), 0, sizeof(uint16_t) * 2 * DiffValidateDump::kMax);
    q.memset(tol_idx.data(), 0xFF, sizeof(long long) * DiffValidateDump::kMax);
    q.memset(tol_vals.data(), 0, sizeof(uint16_t) * 2 * DiffValidateDump::kMax);
    q.memset(tol_max.data(), 0, sizeof(float));
    int32_t* count_ptr = count.data();
    int32_t* tol_ptr = tol_count.data();
    long long* idx_ptr = idx.data();
    uint16_t* vals_ptr = vals.data();
    long long* tidx_ptr = tol_idx.data();
    uint16_t* tvals_ptr = tol_vals.data();
    float* tmax_ptr = tol_max.data();
    const uint16_t* ua = reinterpret_cast<const uint16_t*>(a);
    const uint16_t* ub = reinterpret_cast<const uint16_t*>(b);
    const bf16* fa = a;
    const bf16* fb = b;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            if (ua[i] != ub[i]) {
                float va = bf16_to_float(fa[i]);
                float vb = bf16_to_float(fb[i]);
                float dv = sycl::fabs(va - vb);
                // Two-tier classification:
                //   * count: exact-bit mismatches -- statistical quantity,
                //     dominated by the model's inherent run-to-run
                //     nondeterminism (oneDNN jit:gemm for the full-layer
                //     o_proj shape K=8192 verified at operator level; all
                //     other segment shapes measured bitwise-deterministic in
                //     isolation).
                //   * beyond-sane: |a-b| > 0.25 absolute, vs hidden-scale
                //     activations of O(1). ULP-level jitter never approaches
                //     this; a stale-address or wrong-sequence replay is
                //     expected to light up thousands of elements here.
                bool beyond = dv > 0.25f;
                {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        m(tmax_ptr[0]);
                    m.fetch_max(dv);
                }
                if (beyond) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        t(tol_ptr[0]);
                    int tslot = (int)t.fetch_add(1);
                    if (tslot < DiffValidateDump::kMax) {
                        tidx_ptr[tslot] = (long long)i[0];
                        tvals_ptr[tslot] = ua[i];
                        tvals_ptr[DiffValidateDump::kMax + tslot] = ub[i];
                    }
                }
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    c(count_ptr[0]);
                int slot = (int)c.fetch_add(1);
                if (slot < DiffValidateDump::kMax) {
                    idx_ptr[slot] = (long long)i[0];
                    vals_ptr[slot] = ua[i];
                    vals_ptr[DiffValidateDump::kMax + slot] = ub[i];
                }
            }
        });
    });
    DiffValidateDump out;
    int32_t c = 0, t = 0;
    q.memcpy(&c, count.data(), sizeof(int32_t)).wait();
    q.memcpy(&t, tol_count.data(), sizeof(int32_t)).wait();
    q.memcpy(out.idx, idx.data(), sizeof(long long) * DiffValidateDump::kMax).wait();
    q.memcpy(out.va, vals.data(), sizeof(uint16_t) * DiffValidateDump::kMax).wait();
    q.memcpy(out.vb, vals.data() + DiffValidateDump::kMax,
             sizeof(uint16_t) * DiffValidateDump::kMax).wait();
    q.memcpy(out.tol_idx, tol_idx.data(), sizeof(long long) * DiffValidateDump::kMax).wait();
    q.memcpy(out.tol_va, tol_vals.data(), sizeof(uint16_t) * DiffValidateDump::kMax).wait();
    q.memcpy(out.tol_vb, tol_vals.data() + DiffValidateDump::kMax,
             sizeof(uint16_t) * DiffValidateDump::kMax).wait();
    float tmax = 0;
    q.memcpy(&tmax, tol_max.data(), sizeof(float)).wait();
    out.count = c;
    out.tol_count = t;
    out.tol_max_abs = tmax;
    return out;
}

inline void diff_int4_attn_validate_report() {
    if (!diff_int4_graph_validate_enabled()) return;
    std::fprintf(stderr,
                 "[diff-int4-graph-validate] graph-vs-eager compared=%lld "
                 "layers, mismatches=%lld elements\n",
                 diff_int4_attn_validate_compared().load(),
                 diff_int4_attn_validate_mismatched().load());
}

// ---------------------------------------------------------------------------
// INT4 decode attention-sub-layer SYCL graph capture (DIFF_INT4_SYCL_GRAPH).
//
// Scope (INT4 diffusion checkpoints, single GPU): each decoder layer's
// attention sub-layer -- input rms_norm, q/k/v int4 projections + norm/rope
// postprocess, SDPA, o_proj, and the post-attention rms_norm_add residual --
// is captured as ONE command graph after a single eager warm pass, then
// replayed with a single queue.ext_oneapi_graph() dispatch on every later
// denoising step of the same canvas block. ~12-16 kernel launches per layer
// amortize to one dispatch; the FFN tail (prenorm / dense MLP / router /
// experts / postnorm) stays eager because the int4 experts path (run_shard
// host bucketing) does a memcpy+wait host round-trip that cannot capture.
//
// Replay correctness rests on one invariant: the only buffer crossing the
// graph/eager boundary is `hidden`, which the graph reads and rewrites in
// place. The diffusion arena's coalescing free-list is in the same merged
// state at decode_forward entry every step (every per-step allocation --
// self-conditioning buffers, logits/probs, routing scratch -- is released
// before the next step's hidden alloc), so each step's identical alloc
// sequence hands out identical addresses and the capture-time pointers stay
// valid. The first (warm) invocation runs eager, so lazily-built artifacts
// (oneDNN primitives + weight-layout reorders that stream.wait, grow-only
// quant scratch, arena chunk growth) settle before any recording.
// DIFF_INT4_SYCL_GRAPH_CHECK=1 additionally verifies each replay against the
// capture-time `hidden` address (the session anchor) and refuses to replay
// on drift rather than silently corrupting.
//
// Layer attention kinds are static per config, but enc_len changes between
// canvas blocks (it fixes the captured kv_len grid sizes and the canvas-tail
// pointers inside enc_kv), so it travels in the key; every new block
// re-captures once and replays for the rest of its denoise steps.
// ---------------------------------------------------------------------------
inline bool diff_int4_sycl_graph_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_INT4_SYCL_GRAPH");
        return env && std::strcmp(env, "0") && std::strcmp(env, "off") &&
               std::strcmp(env, "false") && std::strcmp(env, "no");
    }();
    return enabled;
}

inline bool diff_int4_graph_check_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_INT4_SYCL_GRAPH_CHECK");
        return env && std::strcmp(env, "0") && std::strcmp(env, "off") &&
               std::strcmp(env, "false") && std::strcmp(env, "no");
    }();
    return enabled;
}

inline DiffGraphKey diff_int4_attn_step_key(const sycl::queue& q,
                                            int layer_index, bool is_full,
                                            int enc_len, int seq) {
    DiffGraphKey key;
    key.queue = &q;
    key.scope = static_cast<int>(DiffGraphScope::Int4);
    key.kind = 2000 + layer_index * 2 + (is_full ? 1 : 0);
    key.args = {static_cast<uintptr_t>(enc_len), static_cast<uintptr_t>(seq)};
    return key;
}

namespace diff_layer_detail {
inline const char* attn_input_norm_label(bool is_encoder, bool is_full) {
    if (is_encoder) return is_full ? "attn.enc.full.input_norm" : "attn.enc.sliding.input_norm";
    return is_full ? "attn.dec.full.input_norm" : "attn.dec.sliding.input_norm";
}

inline const char* attn_post_norm_label(bool is_encoder, bool is_full) {
    if (is_encoder) return is_full ? "attn.enc.full.post_norm" : "attn.enc.sliding.post_norm";
    return is_full ? "attn.dec.full.post_norm" : "attn.dec.sliding.post_norm";
}
} // namespace diff_layer_detail

// One transformer block = attention residual + dual FFN.  Shared structure for
// encoder (causal, writes KV) and decoder (bidirectional, reads KV) passes.
// `attn_graph_key` (INT4 decode only, default null): when set, the attention
// sub-layer runs under a DiffGraphSession -- captured once after a warm pass
// and replayed afterwards -- while the arena carves below still happen every
// call so the free-list trajectory stays identical on replay steps.
inline void diff_layer_forward(
    GpuEngine& ctx, const DiffLayer& lw, bf16* hidden,
    DiffLayerKv& kv, int seq, int pos, const DiffTextConfig& cfg,
    bool is_encoder, const DiffGraphKey* attn_graph_key = nullptr)
{
    auto& q = ctx.queue;
    int H = cfg.hidden_size;
    float eps = cfg.rms_norm_eps;
    size_t N = (size_t)seq * H;

    // Attention sub-layer.  tmp/attn_out (and everything attention allocates
    // internally) are released at the end of this block, before the FFN runs —
    // so attention and FFN activations share arena storage instead of summing.
    // The carves happen unconditionally (including on graph-replay steps) so
    // the arena free-list trajectory is step-invariant; only the kernel
    // submissions are skipped when the block's graph is replayed.
    {
        auto& ar = diffarena::arena(ctx.index);
        auto tmp_a = ar.alloc<bf16>(N);
        auto attn_a = ar.alloc<bf16>(N);
        bf16* tmp = tmp_a.data();
        bf16* attn_out = attn_a.data();

        DiffGraphSession session;
        bool replayed = false;
        bool validate = attn_graph_key && diff_int4_graph_validate_enabled();
        // Clone of the block-entry `hidden`, carved/copied before any
        // recording/replay dispatch so the audit recompute gets the exact
        // pre-segment input.
        diffarena::Alloc<bf16> hidden_check;
        if (validate) {
            hidden_check = ar.alloc<bf16>(N);
            q.memcpy(hidden_check.data(), hidden, N * sizeof(bf16));
        }
        if (attn_graph_key) {
            if (!session.begin(q, *attn_graph_key,
                               /*graphs_enabled=*/true, /*warm_steps=*/1,
                               reinterpret_cast<uintptr_t>(hidden),
                               diff_int4_graph_check_enabled()))
                replayed = true;  // cache hit: the graph is already replayed
            // else: warm pass, capture-fallback, or a live recording — the
            // kernels below run (and record) under the queue either way.
        }
        // Attention kernel sequence: reads `h`, rewrites `h` in place via the
        // post-attention residual norm. `tmp`/`attn_out` are per-call scratch.
        auto run_attn_segment = [&](bf16* h) {
            { DIFF_PROF(q, diff_layer_detail::attn_input_norm_label(is_encoder, lw.is_full));
              rms_norm(q, h, lw.input_ln.data(), tmp, seq, H, eps); }

            if (is_encoder)
                encoder_attention_forward(ctx, lw, tmp, attn_out, kv, seq, pos, cfg);
            else
                decoder_attention_forward(ctx, lw, tmp, attn_out, kv, seq, pos, cfg);

            // F4: hidden = hidden + post_attention_layernorm(attn_out), one kernel.
            { DIFF_PROF(q, diff_layer_detail::attn_post_norm_label(is_encoder, lw.is_full));
              rms_norm_add_scale(q, attn_out, lw.post_attn_ln.data(),
                                 h, /*scalar=*/1.0f, seq, H, eps); }
        };
        bool graph_executed = replayed;
        bool was_recording = session.active();  // checked before the dispatch
        bool ran_eager = !replayed;   // true on warm steps and failed captures
        if (!replayed) {
            run_attn_segment(hidden);
            if (session.active()) {
                // DPC++ record-replay defers execution: kernels submitted while
                // recording only run at the finalize+dispatch below. If that
                // dispatch fails, the recorded region is dropped — the failsafe
                // re-run keeps this step's work intact.
                session.end_and_replay([&] { run_attn_segment(hidden); });
                graph_executed = true;
            }
        }
        // Mid-tensor bisect clones: after the first (graph/eager) execution the
        // segment scratch still holds the stage outputs; clone them so the
        // recompute can overwrite the scratch while we diff per stage.
        diffarena::Alloc<bf16> tmp_check, attn_check;
        if (validate && !hidden_check.empty()) {
            tmp_check = ar.alloc<bf16>(N);
            attn_check = ar.alloc<bf16>(N);
            q.memcpy(tmp_check.data(), tmp, N * sizeof(bf16));
            q.memcpy(attn_check.data(), attn_out, N * sizeof(bf16));
        }
        if (validate && (graph_executed || ran_eager)) {
            // Bitwise audit: `hidden` holds the executed result (graph on
            // capture/replay steps; plain eager on warm steps), `hidden_check`
            // holds the pre-segment input; re-running the same sequence eagerly
            // must reproduce it. Auditing warm steps too separates graph
            // capture/replay deviations from run-to-run kernel jitter: a
            // warm-step mismatch is inherent eager nondeterminism, not a
            // capture bug.
            run_attn_segment(hidden_check.data());
            DiffValidateDump d =
                diff_int4_attn_validate_diff(q, hidden, hidden_check.data(), N);
            DiffValidateDump dtmp =
                diff_int4_attn_validate_diff(q, tmp, tmp_check.data(), N);
            DiffValidateDump dattn =
                diff_int4_attn_validate_diff(q, attn_out, attn_check.data(), N);
            long long audit = diff_int4_attn_validate_compared().fetch_add(1);
            diff_int4_attn_validate_mismatched().fetch_add(d.count + dtmp.count +
                                                           dattn.count);
            const char* mode =
                replayed ? "replay" : (was_recording ? "capture" : "warm");

            // ---- Self-calibrating gate. ----
            // Warm audits run the SAME kernel sequence twice eagerly, so any
            // difference there is the model's inherent nondeterminism
            // (oneDNN jit:gemm for the full-layer o_proj shape K=8192 verified
            // at operator level; everything else measured bitwise-deterministic
            // in isolation). Every key's warm audit happens on denoise step 1,
            // before its capture (step 2) and replays (step 3+), so the warm
            // audit defines that key's own reference envelope for
            // capture/replay deviation:
            //   * tmp (input rms_norm) is a deterministic kernel chain: it must
            //     stay bitwise identical on EVERY audit, warm or replayed.
            //   * hidden/attn_out exact-bit counts for a graph-executed step
            //     must stay within the key's measured warm envelope (slack
            //     multiplier for tail variance), and no single element may
            //     deviate beyond the absolute sanity bound (0.25, vs hidden
            //     activations of O(1)). A stale-address or wrong-sequence
            //     replay corrupts thousands of elements by O(1) magnitudes,
            //     orders of magnitude outside the envelope.
            struct WarmEnvelope {
                long long hidden_exact = 0;
                long long hidden_beyond = 0;
                long long attn_exact = 0;
                long long attn_beyond = 0;
                long long samples = 0;
            };
            static std::unordered_map<int, WarmEnvelope> key_envelope;
            WarmEnvelope& env = key_envelope[attn_graph_key->kind];

            if (d.count != 0 || dtmp.count != 0 || dattn.count != 0 ||
                diff_graph_verbose()) {
                std::fprintf(stderr,
                             "[diff-int4-graph-validate] audit=%lld %s kind=%d "
                             "enc_len=%d seq=%d exact-bit mismatched elements "
                             "hidden:%lld tmp:%lld attn_out:%lld (beyond-sane: "
                             "hidden:%lld tmp:%lld attn_out:%lld) /%zu\n",
                             audit, mode,
                             attn_graph_key->kind,
                             (int)attn_graph_key->args[0],
                             (int)attn_graph_key->args[1],
                             d.count, dtmp.count, dattn.count,
                             d.tol_count, dtmp.tol_count, dattn.tol_count, N);
                const DiffValidateDump* stages[3] = {&d, &dtmp, &dattn};
                const char* names[3] = {"hidden", "tmp(rms_norm)", "attn_out(o_proj)"};
                for (int s = 0; s < 3; ++s) {
                    const DiffValidateDump* dd = stages[s];
                    for (int i = 0; i < DiffValidateDump::kMax && i < dd->tol_count; ++i) {
                        std::fprintf(stderr,
                                     "  %s BEYOND-SANE idx=%lld (row=%lld col=%lld) "
                                     "first=0x%04x (%g) second=0x%04x (%g)\n",
                                     names[s], dd->tol_idx[i], dd->tol_idx[i] / H,
                                     dd->tol_idx[i] % H, dd->tol_va[i],
                                     bf16_to_float(reinterpret_cast<const bf16&>(dd->tol_va[i])),
                                     dd->tol_vb[i],
                                     bf16_to_float(reinterpret_cast<const bf16&>(dd->tol_vb[i])));
                    }
                }
            }

            if (mode[0] == 'w') {
                // WARM: reference measurement, never fails; only extends the
                // envelope (max over observed samples).
                env.samples += 1;
                env.hidden_exact = std::max(env.hidden_exact, d.count);
                env.hidden_beyond = std::max(env.hidden_beyond, d.tol_count);
                env.attn_exact = std::max(env.attn_exact, dattn.count);
                env.attn_beyond = std::max(env.attn_beyond, dattn.tol_count);
            } else {
                // CAPTURE / REPLAY: gate against this key's own warm envelope.
                // Decisive signals:
                //   * tmp (input rms_norm, deterministic) must be bitwise
                //     identical in every mode.
                //   * beyond-sane (|a-b| > 0.25 absolute) must be zero for
                //     hidden/attn_out in every mode: UPL-level jitter never
                //     approaches it; stale-address / wrong-sequence replays
                //     light up thousands of elements here.
                //   * attn_out exact-bit counts must stay proportional to the
                //     measured warm envelope of this layer kind (8x + 64
                //     slack; the graph dispatch runs under a different
                //     submission regime than back-to-back eager runs and
                //     resamples the kernel's inherent atomic ordering more
                //     broadly).
                // hidden's exact-bit count is NOT gated: postnorm's rowwise
                // RMS scaling propagates ULP-level attn_out noise across bf16
                // rounding cliffs, inflating the count without adding
                // information -- the deterministic postnorm kernel is part of
                // both executions, so hidden deviation is fully explained by
                // the gated attn_out deviation.
                long long attn_cap = env.attn_exact * 8 + 64;
                if (dtmp.count != 0 || dtmp.tol_count != 0)
                    throw std::runtime_error(
                        "DIFF_INT4_SYCL_GRAPH: replayed input rms_norm output (tmp) "
                        "is not bitwise-identical to the eager recompute -- the "
                        "graph changed a deterministic kernel or its inputs");
                if (d.tol_count != 0 || dtmp.tol_count != 0 || dattn.tol_count != 0)
                    throw std::runtime_error(
                        "DIFF_INT4_SYCL_GRAPH: attention graph result deviates "
                        "beyond the 0.25 absolute sanity bound from the eager "
                        "recompute (stale addresses or wrong kernel sequence)");
                if (dattn.count > attn_cap)
                    throw std::runtime_error(
                        "DIFF_INT4_SYCL_GRAPH: replayed attention output deviates "
                        "from the eager recompute far beyond the measured eager-"
                        "nondeterminism envelope for this layer (stale capture?)");
            }
        }
    }

    // Dual FFN (dense MLP + MoE), residual add, layer_scalar.
    float scalar = is_encoder ? lw.enc_layer_scalar : lw.dec_layer_scalar;
    dual_ffn_forward(ctx, lw, hidden, seq, H,
                     cfg.intermediate_size, cfg.num_experts, cfg.top_k_experts,
                     cfg.moe_intermediate_size, eps, scalar, is_encoder);
}

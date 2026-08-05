#pragma once
// Fused single-kernel Gated DeltaNet DECODE step (S==1, cached state) for
// Qwen3.5-MoE linear-attention layers. Replaces the 7-kernel split chain
// (extract_qkv_repeat, l2norm q/k, q scale, sigmoid beta, compute_g,
// recurrent_gated_delta_decode, gated_rmsnorm) with one launch, eliminating
// 7 launch latencies x 30 layers per decode token. conv1d_causal_decode stays
// separate: it owns the global conv_state shift, and per-head work-groups
// would race on the q/k channels shared by each head pair.
//
// Design (sycl-tla-inspired, reference/sycl-tla):
//   * Register-direct mainloop, no SLM staging of the state: each of the 128
//     lanes of a head's work-group owns one d_v column of S[128x128] fp32 as a
//     register fragment (scol[128], 16 GRF), loaded once, updated, stored
//     once -- 2 state passes vs the split kernel's 3 (the SLM-free
//     gmem->reg->math->gmem mainloop of cutlass/gemm/collective/xe_mma.hpp).
//   * Fully unrolled column loop = software pipeline: all 128 coalesced
//     column loads issue before the FMA chain consumes them (the GEMV analog
//     of the XE_PREFETCH_2D multi-stage pipeline; each load is 512B perfectly
//     coalesced across the 128 lanes, which already saturates LSC).
//   * No DPAS: the step is ~2 FLOP/byte (64KB state traffic vs 131 KFLOP per
//     head), so tensor cores buy nothing; batching heads into one MMA would
//     need a block-diagonal A operand (15/16 wasted MACs) and VNNI re-packing
//     of S every token. Plain register math on fragments wins.
//   * SLM only for per-head fp32 broadcasts (k/q after l2norm) and the
//     128-lane tree reductions (l2norm q, l2norm k, rmsnorm core); v, z,
//     norm_w are per-lane values held in registers.
//   * bf16->float conversion of k/q happens once per element instead of
//     128x redundantly (once per owning column) as in the split kernel.
//
// Geometry: one work-group per value head (n_v=32 -> 32 XeCores on BMG-G31,
// WG size 128 = 8xSG16 covering the 8 vector engines). Requires d_k == d_v
// == 128 and n_v == 2*n_k (q/k repeat_interleave x2); the caller falls back
// to the split chain otherwise.
//
// Bit-compatibility: every bf16 round-trip of the split chain is replicated
// (l2norm store then scale_inplace on q, sigmoid, compute_g, core store) and
// the reductions use the same 128-lane tree shape, so results match the
// split path to within FP-contraction noise (verified in gdn_bench decode
// mode: core/state vs split).
//
// A/B: QWEN35_GDN_DECODE_FUSED=1 selects this path (default: split chain).

#include <cstdlib>

#include "runtime/gpu/buffer.hpp" // bf16, bf16_to_float, float_to_bf16

namespace qwen35moe_kernels {

inline bool qwen_gdn_decode_fused_enabled()
{
    static const bool cached = [] {
        const char* e = std::getenv("QWEN35_GDN_DECODE_FUSED");
        return e && e[0] == '1';
    }();
    return cached;
}

// conv_out: [conv_dim] bf16 post-conv (silu applied). braw/araw: [n_v] raw
// sigmoid/softplus inputs. zbuf: [n_v*d_v] bf16 gate. norm_w: [d_v] bf16.
// S: [n_v, d_k, d_v] fp32 in/out. out: [n_v*d_v] bf16 (gated rmsnorm applied).
inline void qwen_gdn_decode_fused(
    sycl::queue& q,
    const bf16* conv_out, const bf16* braw, const bf16* araw,
    const bf16* A_log, const bf16* dt_bias,
    const bf16* zbuf, const bf16* norm_w,
    float* S, bf16* out,
    int n_v, int key_dim, float q_scale, float eps)
{
    constexpr int DK = 128;   // d_k, caller-guaranteed
    constexpr int DV = 128;   // d_v == work-group size
    q.submit([&](sycl::handler& h) {
        // [0,DV) redq | [DV,2DV) redk | [2DV,2DV+DK) kf | [2DV+DK, 2DV+2DK) qf
        sycl::local_accessor<float, 1> lmem(2 * (DK + DV), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)n_v * DV, (size_t)DV),
            [=](sycl::nd_item<1> it) {
                const int hv  = it.get_group(0);
                const int dvi = it.get_local_id(0);
                constexpr int REDK = DV;
                constexpr int KF   = 2 * DV;
                constexpr int QF   = 2 * DV + DK;

                // --- extract: this head's q/k (source key head hv/2) + v ---
                const int hkv = hv >> 1;
                const float qv = bf16_to_float(conv_out[(size_t)hkv * DK + dvi]);
                const float kv = bf16_to_float(conv_out[(size_t)key_dim + (size_t)hkv * DK + dvi]);
                const float vv = bf16_to_float(conv_out[(size_t)2 * key_dim + (size_t)hv * DV + dvi]);

                // --- l2norm reductions (same 128-lane tree as l2norm()) ---
                lmem[dvi]        = qv * qv;
                lmem[REDK + dvi] = kv * kv;
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = DV >> 1; s > 0; s >>= 1) {
                    if (dvi < s) {
                        lmem[dvi]        += lmem[dvi + s];
                        lmem[REDK + dvi] += lmem[REDK + dvi + s];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const float inv_q = sycl::rsqrt(lmem[0]    + eps);
                const float inv_k = sycl::rsqrt(lmem[REDK] + eps);

                // bf16 round-trips of l2norm store + separate scale_inplace.
                float qf = bf16_to_float(float_to_bf16(qv * inv_q));
                qf = bf16_to_float(float_to_bf16(qf * q_scale));
                const float kf = bf16_to_float(float_to_bf16(kv * inv_k));
                lmem[QF + dvi] = qf;
                lmem[KF + dvi] = kf;
                it.barrier(sycl::access::fence_space::local_space);

                // --- beta / decay (uniform per head, redundant per lane) ---
                const float beta = bf16_to_float(float_to_bf16(
                    1.0f / (1.0f + sycl::exp(-bf16_to_float(braw[hv])))));
                const float ax = bf16_to_float(araw[hv]) + bf16_to_float(dt_bias[hv]);
                const float sp = (ax > 0.0f ? ax : 0.0f) +
                                 sycl::log(1.0f + sycl::exp(-sycl::fabs(ax)));
                const float graw = bf16_to_float(float_to_bf16(
                    -sycl::exp(bf16_to_float(A_log[hv])) * sp));
                const float eg = sycl::exp(graw);

                // --- recurrent gated delta rule, register-resident column ---
                float* Sc = S + (size_t)hv * DK * DV + dvi;
                const float* kfp = &lmem[KF];
                const float* qfp = &lmem[QF];
                float scol[DK];
                float kv_mem = 0.0f;
#pragma unroll
                for (int dk = 0; dk < DK; ++dk) {
                    const float s = Sc[dk * DV];
                    scol[dk] = s;
                    kv_mem += s * eg * kfp[dk];
                }
                const float delta = (vv - kv_mem) * beta;
                float o = 0.0f;
#pragma unroll
                for (int dk = 0; dk < DK; ++dk) {
                    const float s = scol[dk] * eg + kfp[dk] * delta;
                    Sc[dk * DV] = s;
                    o += s * qfp[dk];
                }

                // --- gated rmsnorm: (norm * rmsnorm(core)) * silu(z) ---
                const float core = bf16_to_float(float_to_bf16(o));
                lmem[dvi] = core * core;   // reuse redq slice (all reads done)
                it.barrier(sycl::access::fence_space::local_space);
                for (int s = DV >> 1; s > 0; s >>= 1) {
                    if (dvi < s) lmem[dvi] += lmem[dvi + s];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const float rms = sycl::rsqrt(lmem[0] / (float)DV + eps);
                const float zf = bf16_to_float(zbuf[(size_t)hv * DV + dvi]);
                const float wf = bf16_to_float(norm_w[dvi]);
                out[(size_t)hv * DV + dvi] = float_to_bf16(
                    core * rms * wf * (zf / (1.0f + sycl::exp(-zf))));
            });
    });
}

} // namespace qwen35moe_kernels

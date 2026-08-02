#pragma once
// Hybrid variant of the device chunked gated delta rule (prefill): the two
// chunk-local Gram matmuls (K@K^T, Q@K^T) run as ONE oneDNN batched matmul
// over all (head, chunk) pairs (batch = n_v*nch, fp32 accumulate, fp32 out),
// the decay/beta epilogue + triangular T=(I-A)^-1 solve run as one custom
// kernel with a full work-group per (head, chunk) — both fully parallel
// across chunks, unlike the xmx path which serializes everything behind the
// per-head chunk loop. Only the state-carrying phases stay sequential:
// a persistent kernel with 2 work-groups per head (each owns a dv-half of S,
// fp32 [128,64] in SLM for the whole prefill — no per-chunk fp16 staging or
// global master round-trips) runs v_new / core / S-update with the same
// DPAS-fp16 idiom as gated_delta_chunk_xmx.hpp (which see for the math).
//
// Pipeline (chunk C=64, d_k=d_v=128, B = n_v*nch):
//   1. pad:    Kp,Qp bf16 [B,64,128] contiguous, zero-padded tail chunk
//   2. oneDNN: G1 = Kp@Kp^T, G2 = Qp@Kp^T   fp32 [B,64,64] (shared primitive)
//   3. gram_t: per (h,c) WG — gc cumsum; sA = mask(-beta*decay)*G1 (strict
//      lower); T-solve; G1 <- T+I (in place); G2 <- mask(decay)*G2 (b<=a)
//   4. state:  per (h, dv-half) WG — loop chunks: v_tmp = beta*V - (sw*K)@S,
//      v_new = (T+I)@v_tmp, core = (Q*eg)@S + QK@v_new,
//      S = S*eg_last + (K*dec)^T@v_new (S resident in SLM fp32)
//
// Numerics match the xmx path's rounding points (fp16 operands staged once
// per use site, fp32 state master) with one improvement: S never leaves
// fp32 SLM, so the per-chunk sSb fp16 staging rounding disappears.
//
// Dispatch: QWEN35_GDN_IMPL=hybrid (default stays xmx; see gated_delta_chunk.hpp).
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/engine.hpp"
#include "runtime/quantization/q8_0.hpp" // DPAS builtin declaration / vector operand types.

// Model-local namespace: these kernels are per-model COPIES (see
// AGENTS.md model isolation). Global-scope inline functions with
// identical names in other models would ODR-merge at link time;
// divergent bodies (e.g. tiled attention) then crash at runtime.
namespace qwen35moe_kernels {

namespace qwen_gdn_hybrid_detail {

inline uint16_t f16_bits(float f) {
    return sycl::bit_cast<uint16_t>((sycl::half)f);
}
inline float f16_float(uint16_t b) {
    return (float)sycl::bit_cast<sycl::half>(b);
}

// ---------------------------------------------------------------------------
// 1. Pad q/k into head-major chunk-contiguous scratch (zero-padded tail).
//    dst[(hd*nch + c)*64 + m, dk] = src[((c*64+m)*n_v + hd)*DK + dk] or 0.
// ---------------------------------------------------------------------------
inline void pad_qk(sycl::queue& q, const bf16* src, bf16* dst, int S, int n_v,
                   int nch) {
    const size_t total = (size_t)n_v * nch * 64 * 128;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
            int i   = (int)gid[0];
            int dk  = i & 127;
            int m   = (i >> 7) & 63;
            int bi  = i >> 13;            // hd*nch + c
            int hd  = bi / nch;
            int t   = (bi % nch) * 64 + m;
            dst[i]  = (t < S) ? src[((size_t)t * n_v + hd) * 128 + dk]
                              : bf16{0};
        });
    });
}

// ---------------------------------------------------------------------------
// 2. oneDNN batched Gram matmul: dst[B,64,64] f32 = src[B,64,128] @ Kp^T.
//    One primitive serves KKT (src=Kp) and QKT (src=Qp): same shapes, the
//    weights view is a transposed alias of Kp either way. Cached per (gpu,B).
// ---------------------------------------------------------------------------
struct GramKey {
    int gpu, B;
    bool operator==(const GramKey& o) const { return gpu == o.gpu && B == o.B; }
};
struct GramKeyHash {
    size_t operator()(const GramKey& k) const {
        return std::hash<int>{}(k.gpu) * 0x9e3779b9u ^ std::hash<int>{}(k.B);
    }
};
struct GramEntry {
    dnnl::matmul prim;
    dnnl::memory::desc src_md, wei_md, dst_md;
};
inline std::unordered_map<GramKey, GramEntry, GramKeyHash>& gram_cache() {
    static std::unordered_map<GramKey, GramEntry, GramKeyHash> c;
    return c;
}
inline const GramEntry& gram_entry(GpuEngine& ctx, int B) {
    GramKey key{ctx.index, B};
    auto& c = gram_cache();
    auto it = c.find(key);
    if (it != c.end()) return it->second;
    if (c.size() > 256) c.clear();
    using dt = dnnl::memory::data_type;
    GramEntry e{
        {},
        dnnl::memory::desc({B, 64, 128}, dt::bf16, dnnl::memory::format_tag::abc),
        // [B,128,64] transposed view of a [B,64,128] contiguous buffer.
        dnnl::memory::desc({B, 128, 64}, dt::bf16,
                           dnnl::memory::dims{64 * 128, 1, 128}),
        dnnl::memory::desc({B, 64, 64}, dt::f32, dnnl::memory::format_tag::abc)};
    e.prim = dnnl::matmul(
        dnnl::matmul::primitive_desc(ctx.engine, e.src_md, e.wei_md, e.dst_md));
    return c.emplace(key, std::move(e)).first->second;
}
inline void gram_exec(GpuEngine& ctx, int B, const bf16* src, const bf16* kp,
                      float* dst) {
    const GramEntry& e = gram_entry(ctx, B);
    auto mk = [&](const dnnl::memory::desc& md, void* p) {
        return dnnl::sycl_interop::make_memory(
            md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, p);
    };
    e.prim.execute(ctx.stream,
                   {{DNNL_ARG_SRC, mk(e.src_md, const_cast<bf16*>(src))},
                    {DNNL_ARG_WEIGHTS, mk(e.wei_md, const_cast<bf16*>(kp))},
                    {DNNL_ARG_DST, mk(e.dst_md, dst)}});
}

// ---------------------------------------------------------------------------
// 3. Epilogue + T-solve, one 256-lane WG per (head, chunk). Reads the raw
//    Gram tiles from global, applies the decay/beta masks, solves
//    T = (I-A)^-1 (strictly lower, forward substitution, lanes over columns —
//    same race-free scheme as the scalar device path), and writes T+I back
//    over G1 and the masked QK back over G2 (each tile owned by exactly one
//    WG, so in-place is race-free).
// ---------------------------------------------------------------------------
inline void gram_t(sycl::queue& q, const bf16* beta, const bf16* g, float* g1,
                   float* g2, int S, int n_v, int nch) {
    constexpr int C = 64, WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> sA(sycl::range<1>(C * C), h);
        sycl::local_accessor<float> sT(sycl::range<1>(C * C), h);
        sycl::local_accessor<float> sQK(sycl::range<1>(C * C), h);
        sycl::local_accessor<float> sBeta(sycl::range<1>(C), h);
        sycl::local_accessor<float> sGc(sycl::range<1>(C), h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_v, (size_t)nch * WG),
                              sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) {
                const int hd   = (int)it.get_group(0);
                const int c    = (int)(it.get_group(1));
                const int lane = (int)it.get_local_id(1);
                const int cbase = c * C;
                const int rows  = std::min(C, S - cbase);
                const size_t tile = ((size_t)hd * nch + c) * C * C;

                for (int t = lane; t < C; t += WG) {
                    sBeta[t] = (t < rows)
                        ? bf16_to_float(beta[(size_t)(cbase + t) * n_v + hd])
                        : 0.0f;
                    sGc[t] = (t < rows)
                        ? bf16_to_float(g[(size_t)(cbase + t) * n_v + hd])
                        : 0.0f;
                }
                it.barrier();
                if (lane == 0) {
                    float acc = 0.0f;
                    for (int t = 0; t < C; ++t) { acc += sGc[t]; sGc[t] = acc; }
                }
                it.barrier();

                for (int i = lane; i < C * C; i += WG) {
                    const int a = i >> 6, b = i & 63;
                    const float dm = sycl::exp(sGc[a] - sGc[b]);
                    sA[i]  = (b < a) ? -sBeta[a] * g1[tile + i] * dm : 0.0f;
                    sQK[i] = (b <= a && a < rows) ? g2[tile + i] * dm : 0.0f;
                }
                it.barrier();

                for (int j = lane; j < C; j += WG) sT[j] = 0.0f;
                it.barrier();
                for (int i = 1; i < C; ++i) {
                    for (int j = lane; j < i; j += WG) {
                        float acc = sA[i * C + j];
                        for (int r = j + 1; r < i; ++r)
                            acc += sA[i * C + r] * sT[r * C + j];
                        sT[i * C + j] = acc;
                    }
                    it.barrier();
                }

                // Masked writeback: the solve only defines sT's strict lower
                // triangle (the upper holds uninitialized SLM) — never read it.
                for (int i = lane; i < C * C; i += WG) {
                    const int a = i >> 6, b = i & 63;
                    g1[tile + i] = (b < a) ? sT[i] : (a == b ? 1.0f : 0.0f); // T+I
                    g2[tile + i] = sQK[i];                                 // masked QK
                }
            });
    });
}

// ---------------------------------------------------------------------------
// 4. Persistent state kernel: 2 WGs per head, WG hf owns dv-half
//    [hf*64, hf*64+64) of S (fp32 [128,64] in SLM across ALL chunks). Every
//    phase is dv-column-disjoint, so the halves never communicate. DPAS fp16
//    operands / fp32 accumulate, same idiom as the xmx path.
// ---------------------------------------------------------------------------
inline void state_pass(sycl::queue& q, const bf16* qbuf, const bf16* kbuf,
                       const bf16* vbuf, const bf16* beta, const bf16* g,
                       const float* g1, const float* g2, float* ssm_state,
                       bf16* core, int S, int n_v, int nch) {
    constexpr int C = 64, DK = 128, DV = 128, WG = 256;
    constexpr size_t OFF_SS = 0;                    // fp32 [128][64]
    constexpr size_t OFF_K  = OFF_SS + 32768;       // fp16 [64][128]
    constexpr size_t OFF_TB = OFF_K  + 16384;       // fp16 [64][64]
    constexpr size_t OFF_QK = OFF_TB + 8192;        // fp16 [64][64]
    constexpr size_t OFF_VN = OFF_QK + 8192;        // fp16 [64][64]
    constexpr size_t OFF_W  = OFF_VN + 8192;        // fp16 [64][64]
    constexpr size_t OFF_SM = OFF_W  + 8192;        // fp32 [64] x5
    constexpr size_t ARENA  = OFF_SM + 1280;        // 83,200

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<char> arena(sycl::range<1>(ARENA), h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_v * 2, WG),
                              sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                char* base_ = &arena[0];
                auto dpas = [&](diff_dpas_v8s av, diff_dpas_v8i bv, diff_dpas_v8f acc) -> diff_dpas_v8f {
                    return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, av, bv, acc, kQ8DpasFP16);
                };
                float*   sS    = (float*)(base_ + OFF_SS);
                uint16_t* sK   = (uint16_t*)(base_ + OFF_K);
                uint16_t* sTb  = (uint16_t*)(base_ + OFF_TB);
                uint16_t* sQK  = (uint16_t*)(base_ + OFF_QK);
                uint16_t* sVn  = (uint16_t*)(base_ + OFF_VN);
                uint16_t* sW   = (uint16_t*)(base_ + OFF_W);
                float*   sBeta = (float*)(base_ + OFF_SM);
                float*   sGc   = sBeta + C;
                float*   sEg   = sGc + C;
                float*   sSW   = sEg + C;
                float*   sDec  = sSW + C;

                const int hd   = (int)it.get_group(0) >> 1;
                const int hf   = (int)it.get_group(0) & 1;
                const int lane = (int)it.get_local_id(1);
                const int sg   = lane >> 4;
                const int sl   = lane & 15;
                const int dv0  = hf * 64;
                float* ssm     = ssm_state + (size_t)hd * DK * DV;

                // Load the dv-half of S into SLM (fp32 master, resident for
                // the whole prefill).
                for (int i = lane; i < DK * 64; i += WG) {
                    const int dk = i >> 6, dvc = i & 63;
                    sS[i] = ssm[dk * DV + dv0 + dvc];
                }
                it.barrier();

                for (int c = 0; c < nch; ++c) {
                    const int cbase = c * C;
                    const int rows  = std::min(C, S - cbase);
                    const size_t tile = ((size_t)hd * nch + c) * C * C;

                    // -- stage K (zero-padded, fp16), scalars, T+I, QK, betaV
                    for (int i = lane; i < C * DK; i += WG) {
                        const int t = i >> 7;
                        sK[i] = (t < rows)
                            ? f16_bits(bf16_to_float(kbuf[
                                  ((size_t)(cbase + t) * n_v + hd) * DK + (i & 127)]))
                            : (uint16_t)0;
                    }
                    for (int t = lane; t < C; t += WG) {
                        sBeta[t] = (t < rows)
                            ? bf16_to_float(beta[(size_t)(cbase + t) * n_v + hd])
                            : 0.0f;
                        sGc[t] = (t < rows)
                            ? bf16_to_float(g[(size_t)(cbase + t) * n_v + hd])
                            : 0.0f;
                    }
                    for (int i = lane; i < C * C; i += WG) {
                        sTb[i] = f16_bits(g1[tile + i]);
                        sQK[i] = f16_bits(g2[tile + i]);
                    }
                    it.barrier();
                    if (lane == 0) {
                        float acc = 0.0f;
                        for (int t = 0; t < C; ++t) { acc += sGc[t]; sGc[t] = acc; }
                    }
                    it.barrier();
                    for (int t = lane; t < C; t += WG) {
                        const float eg = sycl::exp(sGc[t]);
                        sEg[t]  = eg;
                        sSW[t]  = eg * sBeta[t];
                        sDec[t] = sycl::exp(sGc[C - 1] - sGc[t]);
                    }
                    for (int i = lane; i < C * 64; i += WG) {
                        const int a = i >> 6, dvc = i & 63;
                        const int ar = a < rows ? a : rows - 1;
                        sW[i] = f16_bits(sBeta[a] * bf16_to_float(vbuf[
                            ((size_t)(cbase + ar) * n_v + hd) * DV + dv0 + dvc]));
                    }
                    it.barrier();

                    // -- v_tmp: sW -= (sw*K) @ S_half    (M=64,N=64,K=128)
                    for (int t = sg; t < 32; t += 16) {
                        const int mt = (t >> 2) * 8, nt = (t & 3) * 16;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < DK; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)f16_bits(sSW[mt + m] *
                                    f16_float(sK[(mt + m) * DK + k0 + sl]));
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)f16_bits(sS[(k0 + 2 * j) * 64 + nt + sl]) |
                                              ((uint32_t)f16_bits(sS[(k0 + 2 * j + 1) * 64 + nt + sl]) << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int m = 0; m < 8; ++m) {
                            const int a = mt + m;
                            sW[a * 64 + nt + sl] = f16_bits(
                                f16_float(sW[a * 64 + nt + sl]) - acc[m]);
                        }
                    }
                    it.barrier();

                    // -- v_new = (T+I) @ v_tmp             (M=64,N=64,K=64)
                    for (int t = sg; t < 32; t += 16) {
                        const int mt = (t >> 2) * 8, nt = (t & 3) * 16;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < C; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)sTb[(mt + m) * C + k0 + sl];
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)sW[(k0 + 2 * j) * 64 + nt + sl] |
                                              ((uint32_t)sW[(k0 + 2 * j + 1) * 64 + nt + sl] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int m = 0; m < 8; ++m)
                            sVn[(mt + m) * 64 + nt + sl] = f16_bits(acc[m]);
                    }
                    it.barrier();

                    // -- core = (Q*eg) @ S_half + QK @ v_new
                    for (int t = sg; t < 32; t += 16) {
                        const int mt = (t >> 2) * 8, nt = (t & 3) * 16;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < DK; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m) {
                                const int a = mt + m;
                                const int ar = a < rows ? a : rows - 1;
                                av[m] = (short)((a < rows)
                                    ? f16_bits(bf16_to_float(qbuf[
                                          ((size_t)(cbase + ar) * n_v + hd) * DK + k0 + sl]) * sEg[a])
                                    : (uint16_t)0);
                            }
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)f16_bits(sS[(k0 + 2 * j) * 64 + nt + sl]) |
                                              ((uint32_t)f16_bits(sS[(k0 + 2 * j + 1) * 64 + nt + sl]) << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int k0 = 0; k0 < C; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)sQK[(mt + m) * C + k0 + sl];
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)sVn[(k0 + 2 * j) * 64 + nt + sl] |
                                              ((uint32_t)sVn[(k0 + 2 * j + 1) * 64 + nt + sl] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int m = 0; m < 8; ++m) {
                            const int a = mt + m;
                            if (a < rows)
                                core[((size_t)(cbase + a) * n_v + hd) * DV + dv0 + nt + sl] =
                                    float_to_bf16(acc[m]);
                        }
                    }
                    it.barrier();

                    // -- S_half = S_half*eg_last + (K*dec)^T @ v_new
                    //    (M=128 dk, N=64 dv, K=64)
                    const float eg_last = sEg[C - 1];
                    for (int t = sg; t < 64; t += 16) {
                        const int mt = (t >> 2) * 8, nt = (t & 3) * 16;
                        diff_dpas_v8f acc = {0,0,0,0,0,0,0,0};
                        for (int k0 = 0; k0 < C; k0 += 16) {
                            diff_dpas_v8s av;
                            diff_dpas_v8i bv;
                            for (int m = 0; m < 8; ++m)
                                av[m] = (short)f16_bits(
                                    f16_float(sK[(k0 + sl) * DK + mt + m]) *
                                    sDec[k0 + sl]);
                            for (int j = 0; j < 8; ++j)
                                bv[j] = (int)((uint32_t)sVn[(k0 + 2 * j) * 64 + nt + sl] |
                                              ((uint32_t)sVn[(k0 + 2 * j + 1) * 64 + nt + sl] << 16));
                            acc = dpas(av, bv, acc);
                        }
                        for (int m = 0; m < 8; ++m) {
                            const int dk = mt + m, dvc = nt + sl;
                            sS[dk * 64 + dvc] = sS[dk * 64 + dvc] * eg_last + acc[m];
                        }
                    }
                    it.barrier();
                }

                // Store the final S half back to the fp32 global master.
                for (int i = lane; i < DK * 64; i += WG) {
                    const int dk = i >> 6, dvc = i & 63;
                    ssm[dk * DV + dv0 + dvc] = sS[i];
                }
            });
    });
}

// Grow-only scratch for the hybrid path (mutex-serialized host reuse).
struct Scratch {
    GpuBuffer<bf16> kp, qp;   // [B*64*128]
    GpuBuffer<float> g1, g2;  // [B*64*64]  (KKT->T+I, QKT->masked QK in place)
    size_t b_cap = 0;
};
inline Scratch& scratch() {
    static Scratch s;
    return s;
}
inline std::mutex& scratch_mutex() {
    static std::mutex m;
    return m;
}

} // namespace qwen_gdn_hybrid_detail

// q,k,v: device bf16 [S, n_v, d] (q,k already l2norm'd, q pre-scaled).
// beta,g: device bf16 [S, n_v]. ssm_state: device fp32 [n_v, d_k, d_v] (in/out).
// core: device bf16 [S, n_v, d_v] (out). Mirrors qwen_gdn_device_chunk_xmx.
inline void qwen_gdn_device_chunk_hybrid(
    GpuEngine& ctx,
    const bf16* qbuf, const bf16* kbuf, const bf16* vbuf,
    const bf16* beta, const bf16* g,
    float* ssm_state, bf16* core,
    int S, int n_v, int d_k, int d_v, int chunk)
{
    if (chunk != 64 || d_k != 128 || d_v != 128)
        throw std::runtime_error("qwen_gdn_device_chunk_hybrid requires chunk=64, d_k=d_v=128");
    if (S <= 0) return;
    auto& q = ctx.queue;
    const int nch = (S + 63) / 64;
    const int B = n_v * nch;

    using namespace qwen_gdn_hybrid_detail;
    std::lock_guard<std::mutex> lock(scratch_mutex());
    Scratch& sc = scratch();
    if (sc.b_cap < (size_t)B) {
        sc.kp = GpuBuffer<bf16>((size_t)B * 64 * 128, q);
        sc.qp = GpuBuffer<bf16>((size_t)B * 64 * 128, q);
        sc.g1 = GpuBuffer<float>((size_t)B * 64 * 64, q);
        sc.g2 = GpuBuffer<float>((size_t)B * 64 * 64, q);
        sc.b_cap = (size_t)B;
    }

    pad_qk(q, kbuf, sc.kp.data(), S, n_v, nch);
    pad_qk(q, qbuf, sc.qp.data(), S, n_v, nch);
    gram_exec(ctx, B, sc.kp.data(), sc.kp.data(), sc.g1.data());  // KKT
    gram_exec(ctx, B, sc.qp.data(), sc.kp.data(), sc.g2.data());  // QKT
    gram_t(q, beta, g, sc.g1.data(), sc.g2.data(), S, n_v, nch);
    state_pass(q, qbuf, kbuf, vbuf, beta, g, sc.g1.data(), sc.g2.data(),
               ssm_state, core, S, n_v, nch);
}

} // namespace qwen35moe_kernels

#include "attention.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/kernels/attention_mask.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include "../../common/kernels/rope.hpp"
#include "attention_kernels.hpp"  // fused norm+rope / softmax leaf kernels + AB knobs
#include "../../common/layers/attention_layout.hpp"  // transpose_q_into, scatter_ctx
#include "linear_dispatch.hpp"
#include "arena.hpp"
#include "../../utils/profile.hpp"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unordered_map>
#include <variant>
#include <stdexcept>

namespace {

enum class AttnProfileKind { EncSliding = 0, EncFull, DecSliding, DecFull };
enum class AttnProfilePhase {
    QProj = 0,
    QNorm,
    KProj,
    KNorm,
    VProj,
    VNorm,
    KToVCopy,
    Rope,
    KVWrite,
    CanvasKVStage,
    QTranspose,
    ScoreMM,
    Softmax,
    ValueMM,
    Scatter,
    OProj,
    Count
};

static AttnProfileKind attn_profile_kind(bool is_encoder, bool is_full) {
    if (is_encoder) return is_full ? AttnProfileKind::EncFull : AttnProfileKind::EncSliding;
    return is_full ? AttnProfileKind::DecFull : AttnProfileKind::DecSliding;
}

static const char* attn_prof(AttnProfileKind kind, AttnProfilePhase phase) {
    static constexpr const char* labels[4][(int)AttnProfilePhase::Count] = {
        {
            "attn.enc.sliding.q_proj",
            "attn.enc.sliding.q_norm",
            "attn.enc.sliding.k_proj",
            "attn.enc.sliding.k_norm",
            "attn.enc.sliding.v_proj",
            "attn.enc.sliding.v_norm",
            "attn.enc.sliding.k_to_v_copy",
            "attn.enc.sliding.rope",
            "attn.enc.sliding.kv_write",
            "attn.enc.sliding.canvas_kv_stage",
            "attn.enc.sliding.q_transpose",
            "attn.enc.sliding.score_mm",
            "attn.enc.sliding.softmax",
            "attn.enc.sliding.value_mm",
            "attn.enc.sliding.scatter",
            "attn.enc.sliding.o_proj",
        },
        {
            "attn.enc.full.q_proj",
            "attn.enc.full.q_norm",
            "attn.enc.full.k_proj",
            "attn.enc.full.k_norm",
            "attn.enc.full.v_proj",
            "attn.enc.full.v_norm",
            "attn.enc.full.k_to_v_copy",
            "attn.enc.full.rope",
            "attn.enc.full.kv_write",
            "attn.enc.full.canvas_kv_stage",
            "attn.enc.full.q_transpose",
            "attn.enc.full.score_mm",
            "attn.enc.full.softmax",
            "attn.enc.full.value_mm",
            "attn.enc.full.scatter",
            "attn.enc.full.o_proj",
        },
        {
            "attn.dec.sliding.q_proj",
            "attn.dec.sliding.q_norm",
            "attn.dec.sliding.k_proj",
            "attn.dec.sliding.k_norm",
            "attn.dec.sliding.v_proj",
            "attn.dec.sliding.v_norm",
            "attn.dec.sliding.k_to_v_copy",
            "attn.dec.sliding.rope",
            "attn.dec.sliding.kv_write",
            "attn.dec.sliding.canvas_kv_stage",
            "attn.dec.sliding.q_transpose",
            "attn.dec.sliding.score_mm",
            "attn.dec.sliding.softmax",
            "attn.dec.sliding.value_mm",
            "attn.dec.sliding.scatter",
            "attn.dec.sliding.o_proj",
        },
        {
            "attn.dec.full.q_proj",
            "attn.dec.full.q_norm",
            "attn.dec.full.k_proj",
            "attn.dec.full.k_norm",
            "attn.dec.full.v_proj",
            "attn.dec.full.v_norm",
            "attn.dec.full.k_to_v_copy",
            "attn.dec.full.rope",
            "attn.dec.full.kv_write",
            "attn.dec.full.canvas_kv_stage",
            "attn.dec.full.q_transpose",
            "attn.dec.full.score_mm",
            "attn.dec.full.softmax",
            "attn.dec.full.value_mm",
            "attn.dec.full.scatter",
            "attn.dec.full.o_proj",
        },
    };
    return labels[(int)kind][(int)phase];
}

static const char* attn_fused_proj_prof(AttnProfileKind kind) {
    static constexpr const char* labels[4] = {
        "attn.enc.sliding.fused_qkv_proj",
        "attn.enc.full.fused_qk_proj",
        "attn.dec.sliding.fused_qkv_proj",
        "attn.dec.full.fused_qk_proj",
    };
    return labels[(int)kind];
}

static const char* attn_fused_split_prof(AttnProfileKind kind) {
    static constexpr const char* labels[4] = {
        "attn.enc.sliding.fused_qkv_post",
        "attn.enc.full.fused_qk_post",
        "attn.dec.sliding.fused_qkv_post",
        "attn.dec.full.fused_qk_post",
    };
    return labels[(int)kind];
}

// Fused int4 attention projection is always on when the fused QKV/QK weights
// were uploaded (loader uploads them whenever the per-layer q/k/v projections
// are int4). No runtime env toggle.
static bool usable_int4_fused_proj(const Int4Linear& W, int in_features,
                                   int out_features) {
    return !W.empty() &&
           W.in_features == in_features && W.out_features == out_features;
}

static const char* onednn_sdpa_mode() {
    static const char* mode = [] {
        const char* e = std::getenv("DIFF_ONEDNN_SDPA");
        if (!e || std::strcmp(e, "0") == 0 || std::strcmp(e, "false") == 0 ||
            std::strcmp(e, "FALSE") == 0 || std::strcmp(e, "off") == 0 ||
            std::strcmp(e, "OFF") == 0 || std::strcmp(e, "no") == 0 ||
            std::strcmp(e, "NO") == 0)
            return "";
        return e;
    }();
    return mode;
}

static bool onednn_sdpa_decoder_enabled() {
    const char* e = onednn_sdpa_mode();
    return std::strcmp(e, "decode") == 0 || std::strcmp(e, "decoder") == 0 ||
           std::strcmp(e, "decode_encoder_full") == 0 ||
           std::strcmp(e, "decoder_encoder_full") == 0 ||
           std::strcmp(e, "all") == 0 ||
           std::strcmp(e, "1") == 0 || std::strcmp(e, "true") == 0 ||
           std::strcmp(e, "TRUE") == 0 || std::strcmp(e, "on") == 0 ||
           std::strcmp(e, "ON") == 0 || std::strcmp(e, "yes") == 0 ||
           std::strcmp(e, "YES") == 0;
}

static bool onednn_sdpa_encoder_full_enabled() {
    const char* e = onednn_sdpa_mode();
    return std::strcmp(e, "all") == 0 ||
           std::strcmp(e, "decode_encoder_full") == 0 ||
           std::strcmp(e, "decoder_encoder_full") == 0 ||
           std::strcmp(e, "encode") == 0 || std::strcmp(e, "encoder") == 0 ||
           std::strcmp(e, "prefill") == 0 ||
           std::strcmp(e, "encoder_all") == 0 ||
           std::strcmp(e, "prefill_all") == 0 ||
           std::strcmp(e, "all_sliding") == 0 ||
           std::strcmp(e, "all_experimental") == 0 ||
           std::strcmp(e, "enc_full") == 0 ||
           std::strcmp(e, "encoder_full") == 0 ||
           std::strcmp(e, "prefill_full") == 0;
}

static bool onednn_sdpa_encoder_sliding_enabled() {
    const char* e = onednn_sdpa_mode();
    return std::strcmp(e, "enc_sliding") == 0 ||
           std::strcmp(e, "encoder_sliding") == 0 ||
           std::strcmp(e, "prefill_sliding") == 0 ||
           std::strcmp(e, "encoder_all") == 0 ||
           std::strcmp(e, "prefill_all") == 0 ||
           std::strcmp(e, "all_sliding") == 0 ||
           std::strcmp(e, "all_experimental") == 0;
}

static void fused_sliding_decoder_kv_cache_postprocess(
    sycl::queue& q, bf16* K_cache, bf16* V_cache, const bf16* kn,
    int seq, int nkv, int hd, int offset,
    float theta, float partial, float eps,
    const float* rope_cos, const float* rope_sin,
    Nvfp4GraphSession* /*session*/ = nullptr) {
    int pair_offset = hd / 2;
    int n_active_pairs = static_cast<int>(partial * hd / 2.0f);
    float freq_denom = static_cast<float>(hd);
    size_t local = std::min(256, hd);
    while (local & (local - 1)) local--;
    int kv_rows = seq * nkv;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)kv_rows * 2 * local, local),
            [=](sycl::nd_item<1> it) {
                int packed = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                bool is_k = packed < kv_rows;
                int row = is_k ? packed : packed - kv_rows;
                int tok = row / nkv;
                int head = row - tok * nkv;
                bf16* xrow = (is_k ? K_cache : V_cache) + ((size_t)tok * nkv + head) * hd;

                float ss = 0.0f;
                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(xrow[d]);
                    ss += v * v;
                }
                float ss_sum = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                float rms_inv = sycl::rsqrt(ss_sum / float(hd) + eps);

                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(xrow[d]) * rms_inv;
                    if (is_k) v *= bf16_to_float(kn[d]);
                    xrow[d] = float_to_bf16(v);
                }

                if (is_k) {
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int pair_i = lid; pair_i < n_active_pairs; pair_i += lsz) {
                        float c, s;
                        if (rope_cos) {
                            int ti = tok * n_active_pairs + pair_i;
                            c = rope_cos[ti];
                            s = rope_sin[ti];
                        } else {
                            float inv_freq = 1.0f / sycl::pow(theta, 2.0f * pair_i / freq_denom);
                            float angle = (float)(offset + tok) * inv_freq;
                            c = sycl::cos(angle);
                            s = sycl::sin(angle);
                        }
                        float x0 = bf16_to_float(xrow[pair_i]);
                        float x1 = bf16_to_float(xrow[pair_i + pair_offset]);
                        xrow[pair_i] = float_to_bf16(x0 * c - x1 * s);
                        xrow[pair_i + pair_offset] = float_to_bf16(x0 * s + x1 * c);
                    }
                }
            });
    });
}

static void fused_full_decoder_kv_cache_postprocess(
    sycl::queue& q, bf16* K_cache, bf16* V_cache, const bf16* kn,
    int seq, int nkv, int hd, int offset,
    float theta, float partial, float eps,
    const float* rope_cos, const float* rope_sin,
    Nvfp4GraphSession* /*session*/ = nullptr) {
    int pair_offset = hd / 2;
    int n_active_pairs = static_cast<int>(partial * hd / 2.0f);
    float freq_denom = static_cast<float>(hd);
    size_t local = std::min(256, hd);
    while (local & (local - 1)) local--;
    int rows = seq * nkv;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)rows * local, local),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                int tok = row / nkv;
                int head = row - tok * nkv;
                bf16* krow = K_cache + ((size_t)tok * nkv + head) * hd;
                bf16* vrow = V_cache + ((size_t)tok * nkv + head) * hd;

                float ss = 0.0f;
                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(krow[d]);
                    ss += v * v;
                }
                float ss_sum = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                float rms_inv = sycl::rsqrt(ss_sum / float(hd) + eps);

                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(krow[d]) * rms_inv;
                    vrow[d] = float_to_bf16(v);
                    krow[d] = float_to_bf16(v * bf16_to_float(kn[d]));
                }

                it.barrier(sycl::access::fence_space::local_space);
                for (int pair_i = lid; pair_i < n_active_pairs; pair_i += lsz) {
                    float c, s;
                    if (rope_cos) {
                        int ti = tok * n_active_pairs + pair_i;
                        c = rope_cos[ti];
                        s = rope_sin[ti];
                    } else {
                        float inv_freq = 1.0f / sycl::pow(theta, 2.0f * pair_i / freq_denom);
                        float angle = (float)(offset + tok) * inv_freq;
                        c = sycl::cos(angle);
                        s = sycl::sin(angle);
                    }
                    float x0 = bf16_to_float(krow[pair_i]);
                    float x1 = bf16_to_float(krow[pair_i + pair_offset]);
                    krow[pair_i] = float_to_bf16(x0 * c - x1 * s);
                    krow[pair_i + pair_offset] = float_to_bf16(x0 * s + x1 * c);
                }
            });
    });
}

static void fused_sliding_qkv_postprocess(
    sycl::queue& q, const bf16* fused,
    bf16* Q, bf16* K, bf16* V,
    const bf16* qn, const bf16* kn,
    int seq, int nq, int nkv, int hd, int offset,
    float theta, float partial, float eps,
    const float* rope_cos, const float* rope_sin,
    Nvfp4GraphSession* /*session*/ = nullptr) {
    int rows = nq + 2 * nkv;
    int q_dim = nq * hd;
    int kv_dim = nkv * hd;
    int total = q_dim + 2 * kv_dim;
    int pair_offset = hd / 2;
    int n_active_pairs = static_cast<int>(partial * hd / 2.0f);
    float freq_denom = static_cast<float>(hd);
    size_t local = std::min(256, hd);
    while (local & (local - 1)) local--;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * rows * local, local),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                int tok = row / rows;
                int r = row - tok * rows;

                bool is_q = r < nq;
                bool is_k = !is_q && r < nq + nkv;
                int head = is_q ? r : (is_k ? r - nq : r - nq - nkv);
                size_t src_base = (size_t)tok * total +
                    (is_q ? (size_t)head * hd :
                     is_k ? (size_t)q_dim + (size_t)head * hd :
                            (size_t)q_dim + (size_t)kv_dim + (size_t)head * hd);

                float ss = 0.0f;
                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(fused[src_base + d]);
                    ss += v * v;
                }
                float ss_sum = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                float rms_inv = sycl::rsqrt(ss_sum / float(hd) + eps);

                bf16* dst = is_q ? Q + ((size_t)tok * nq + head) * hd
                                 : is_k ? K + ((size_t)tok * nkv + head) * hd
                                        : V + ((size_t)tok * nkv + head) * hd;
                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(fused[src_base + d]) * rms_inv;
                    if (is_q) v *= bf16_to_float(qn[d]);
                    else if (is_k) v *= bf16_to_float(kn[d]);
                    dst[d] = float_to_bf16(v);
                }

                if (is_q || is_k) {
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int pair_i = lid; pair_i < n_active_pairs; pair_i += lsz) {
                        float c, s;
                        if (rope_cos) {
                            int ti = tok * n_active_pairs + pair_i;
                            c = rope_cos[ti];
                            s = rope_sin[ti];
                        } else {
                            float inv_freq = 1.0f / sycl::pow(theta, 2.0f * pair_i / freq_denom);
                            float angle = (float)(offset + tok) * inv_freq;
                            c = sycl::cos(angle);
                            s = sycl::sin(angle);
                        }
                        float x0 = bf16_to_float(dst[pair_i]);
                        float x1 = bf16_to_float(dst[pair_i + pair_offset]);
                        dst[pair_i] = float_to_bf16(x0 * c - x1 * s);
                        dst[pair_i + pair_offset] = float_to_bf16(x0 * s + x1 * c);
                    }
                }
            });
    });
}

static void fused_full_qk_postprocess(
    sycl::queue& q, const bf16* fused,
    bf16* Q, bf16* K, bf16* V,
    const bf16* qn, const bf16* kn,
    int seq, int nq, int nkv, int hd, int offset,
    float theta, float partial, float eps,
    const float* rope_cos, const float* rope_sin,
    Nvfp4GraphSession* /*session*/ = nullptr) {
    int rows = nq + nkv;
    int q_dim = nq * hd;
    int kv_dim = nkv * hd;
    int total = q_dim + kv_dim;
    int pair_offset = hd / 2;
    int n_active_pairs = static_cast<int>(partial * hd / 2.0f);
    float freq_denom = static_cast<float>(hd);
    size_t local = std::min(256, hd);
    while (local & (local - 1)) local--;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)seq * rows * local, local),
            [=](sycl::nd_item<1> it) {
                int row = it.get_group(0);
                int lid = it.get_local_id(0);
                int lsz = it.get_local_range(0);
                int tok = row / rows;
                int r = row - tok * rows;

                bool is_q = r < nq;
                int head = is_q ? r : r - nq;
                size_t src_base = (size_t)tok * total +
                    (is_q ? (size_t)head * hd : (size_t)q_dim + (size_t)head * hd);

                float ss = 0.0f;
                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(fused[src_base + d]);
                    ss += v * v;
                }
                float ss_sum = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                float rms_inv = sycl::rsqrt(ss_sum / float(hd) + eps);

                bf16* dst = is_q ? Q + ((size_t)tok * nq + head) * hd
                                 : K + ((size_t)tok * nkv + head) * hd;
                bf16* vdst = V + ((size_t)tok * nkv + head) * hd;
                for (int d = lid; d < hd; d += lsz) {
                    float v = bf16_to_float(fused[src_base + d]) * rms_inv;
                    if (is_q) v *= bf16_to_float(qn[d]);
                    else {
                        vdst[d] = float_to_bf16(v);
                        v *= bf16_to_float(kn[d]);
                    }
                    dst[d] = float_to_bf16(v);
                }

                it.barrier(sycl::access::fence_space::local_space);
                for (int pair_i = lid; pair_i < n_active_pairs; pair_i += lsz) {
                    float c, s;
                    if (rope_cos) {
                        int ti = tok * n_active_pairs + pair_i;
                        c = rope_cos[ti];
                        s = rope_sin[ti];
                    } else {
                        float inv_freq = 1.0f / sycl::pow(theta, 2.0f * pair_i / freq_denom);
                        float angle = (float)(offset + tok) * inv_freq;
                        c = sycl::cos(angle);
                        s = sycl::sin(angle);
                    }
                    float x0 = bf16_to_float(dst[pair_i]);
                    float x1 = bf16_to_float(dst[pair_i + pair_offset]);
                    dst[pair_i] = float_to_bf16(x0 * c - x1 * s);
                    dst[pair_i + pair_offset] = float_to_bf16(x0 * s + x1 * c);
                }
            });
    });
}

// Q/K/V own their storage as scoped arena handles (see arena.hpp).  The caller
// holds the QKV so the buffers stay live across attention; K/V can be released
// the moment they are copied into the cache, Q after the score GEMM.
struct QKV {
    diffarena::Alloc<bf16> Q, K, V;
    int nq, nkv, hd;
};

// Project + per-head Q/K RMSNorm + (scaleless) V norm + RoPE at `offset`.
QKV project_qkv(GpuEngine& ctx, const DiffLayer& lw, const bf16* hidden,
                int seq, const DiffTextConfig& cfg, int offset,
                AttnProfileKind prof,
                Nvfp4GraphSession* session = nullptr) {
    auto& q = ctx.queue;
    int H = cfg.hidden_size;
    int nq = cfg.num_attn_heads;

    const DiffLinearWeight *qpw, *kpw, *vpw;
    const bf16 *qn, *kn;
    int nkv, hd;
    float theta, partial;
    if (!lw.is_full) {
        auto& s = std::get<DiffSlidingAttn>(lw.attn);
        qpw = &s.q_proj; kpw = &s.k_proj; vpw = &s.v_proj;
        qn = s.q_norm.data(); kn = s.k_norm.data();
        nkv = cfg.num_kv_heads; hd = cfg.head_dim;
        theta = cfg.sliding_rope.rope_theta; partial = cfg.sliding_rope.partial_rotary_factor;
    } else {
        auto& fa = std::get<DiffFullAttn>(lw.attn);
        qpw = &fa.q_proj; kpw = &fa.k_proj; vpw = nullptr;
        qn = fa.q_norm.data(); kn = fa.k_norm.data();
        nkv = cfg.num_global_kv_heads; hd = cfg.global_head_dim;
        theta = cfg.full_rope.rope_theta; partial = cfg.full_rope.partial_rotary_factor;
    }

    auto& ar = diffarena::arena(ctx.index);
    QKV out;
    out.nq = nq; out.nkv = nkv; out.hd = hd;
    out.Q = ar.alloc<bf16>((size_t)seq * nq * hd);
    out.K = ar.alloc<bf16>((size_t)seq * nkv * hd);
    out.V = ar.alloc<bf16>((size_t)seq * nkv * hd);
    bf16* Q = out.Q.data(); bf16* K = out.K.data(); bf16* V = out.V.data();

    RopeTable rope = (diff_fused_norm_rope_mode() == RopeKernelMode::Table)
        ? make_rope_table(ctx, seq, offset, hd, theta, partial) : RopeTable{};
    bool used_fused_proj = false;
    bool rope_done = false;
    if (!lw.is_full) {
        const auto& s = std::get<DiffSlidingAttn>(lw.attn);
        int q_dim = nq * hd;
        int kv_dim = nkv * hd;
        int fused_dim = q_dim + 2 * kv_dim;
        if (usable_int4_fused_proj(s.qkv_proj_int4, H, fused_dim)) {
            auto fused = ar.alloc<bf16>((size_t)seq * fused_dim);
            { DIFF_PROF(q, attn_fused_proj_prof(prof));
              matmul_int4(hidden, seq, H, s.qkv_proj_int4, fused.data(), ctx); }
            { DIFF_PROF(q, attn_fused_split_prof(prof));
              fused_sliding_qkv_postprocess(q, fused.data(), Q, K, V,
                                             qn, kn, seq, nq, nkv, hd, offset,
                                             theta, partial, cfg.rms_norm_eps,
                                             rope.cos_data(), rope.sin_data(),
                                             session); }
            used_fused_proj = true;
            rope_done = true;
        }
    } else {
        const auto& fa = std::get<DiffFullAttn>(lw.attn);
        int q_dim = nq * hd;
        int kv_dim = nkv * hd;
        int fused_dim = q_dim + kv_dim;
        if (usable_int4_fused_proj(fa.qk_proj_int4, H, fused_dim)) {
            auto fused = ar.alloc<bf16>((size_t)seq * fused_dim);
            { DIFF_PROF(q, attn_fused_proj_prof(prof));
              matmul_int4(hidden, seq, H, fa.qk_proj_int4, fused.data(), ctx); }
            { DIFF_PROF(q, attn_fused_split_prof(prof));
              fused_full_qk_postprocess(q, fused.data(), Q, K, V,
                                        qn, kn, seq, nq, nkv, hd, offset,
                                        theta, partial, cfg.rms_norm_eps,
                                        rope.cos_data(), rope.sin_data(),
                                        session); }
            used_fused_proj = true;
            rope_done = true;
        }
    }

    if (!used_fused_proj) {
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::QProj));
          matmul_linear_weight(hidden, seq, H, *qpw, nq * hd, Q, ctx); }
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::QNorm));
          rms_norm(q, Q, qn, Q, seq * nq, hd, cfg.rms_norm_eps); }

        if (!lw.is_full) {
            { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KProj));
              matmul_linear_weight(hidden, seq, H, *kpw, nkv * hd, K, ctx); }
            { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KNorm));
              rms_norm(q, K, kn, K, seq * nkv, hd, cfg.rms_norm_eps); }
            { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::VProj));
              matmul_linear_weight(hidden, seq, H, *vpw, nkv * hd, V, ctx); }
            { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::VNorm));
              rms_norm_no_scale(q, V, V, seq * nkv, hd, cfg.rms_norm_eps); }
        } else {
            // K = k_proj; V = v_norm(K) BEFORE k_norm; then K = k_norm(K).
            { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KProj));
              matmul_linear_weight(hidden, seq, H, *kpw, nkv * hd, K, ctx); }
            { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KToVCopy));
              q.memcpy(V, K, (size_t)seq * nkv * hd * sizeof(bf16)); }
            { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::VNorm));
              rms_norm_no_scale(q, V, V, seq * nkv, hd, cfg.rms_norm_eps); }
            { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KNorm));
              rms_norm(q, K, kn, K, seq * nkv, hd, cfg.rms_norm_eps); }
        }
    }

    if (!rope_done) {
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::Rope));
          apply_rope(q, Q, K, seq, offset, nq, nkv, hd, theta, partial,
                     rope.cos_data(), rope.sin_data()); }
    }
    return out;
}

// Decoder-only variant: K/V are materialized directly in the cache's canvas
// region, eliminating the later decoder staging copy/launch. Q still lives in
// the arena because attention consumes it before o_proj.
QKV project_qkv_decoder_direct_cache(
    GpuEngine& ctx, const DiffLayer& lw, const bf16* hidden,
    int seq, const DiffTextConfig& cfg, int offset,
    bf16* K_cache, bf16* V_cache, AttnProfileKind prof,
    Nvfp4GraphSession* session = nullptr) {
    auto& q = ctx.queue;
    int H = cfg.hidden_size;
    int nq = cfg.num_attn_heads;

    const DiffLinearWeight *qpw, *kpw, *vpw;
    const bf16 *qn, *kn;
    int nkv, hd;
    float theta, partial;
    if (!lw.is_full) {
        auto& s = std::get<DiffSlidingAttn>(lw.attn);
        qpw = &s.q_proj; kpw = &s.k_proj; vpw = &s.v_proj;
        qn = s.q_norm.data(); kn = s.k_norm.data();
        nkv = cfg.num_kv_heads; hd = cfg.head_dim;
        theta = cfg.sliding_rope.rope_theta; partial = cfg.sliding_rope.partial_rotary_factor;
    } else {
        auto& fa = std::get<DiffFullAttn>(lw.attn);
        qpw = &fa.q_proj; kpw = &fa.k_proj; vpw = nullptr;
        qn = fa.q_norm.data(); kn = fa.k_norm.data();
        nkv = cfg.num_global_kv_heads; hd = cfg.global_head_dim;
        theta = cfg.full_rope.rope_theta; partial = cfg.full_rope.partial_rotary_factor;
    }

    auto& ar = diffarena::arena(ctx.index);
    QKV out;
    out.nq = nq; out.nkv = nkv; out.hd = hd;
    out.Q = ar.alloc<bf16>((size_t)seq * nq * hd);
    bf16* Q = out.Q.data();

    { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::QProj));
      matmul_linear_weight(hidden, seq, H, *qpw, nq * hd, Q, ctx); }
    RopeTable rope = (diff_fused_norm_rope_mode() == RopeKernelMode::Table)
        ? make_rope_table(ctx, seq, offset, hd, theta, partial) : RopeTable{};
    { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::QNorm));
      if (diff_fused_norm_rope_mode() == RopeKernelMode::Fused) {
          fused_norm_rope_inplace_fused(q, Q, qn, seq, nq, hd, offset, theta, partial,
                                        cfg.rms_norm_eps);
      } else {
          fused_norm_rope_inplace(q, Q, qn, seq, nq, hd, offset, theta, partial,
                                  cfg.rms_norm_eps, rope.cos_data(), rope.sin_data());
      } }

    if (!lw.is_full) {
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KProj));
          matmul_linear_weight(hidden, seq, H, *kpw, nkv * hd, K_cache, ctx); }
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::VProj));
          matmul_linear_weight(hidden, seq, H, *vpw, nkv * hd, V_cache, ctx); }
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KNorm));
          fused_sliding_decoder_kv_cache_postprocess(
              q, K_cache, V_cache, kn, seq, nkv, hd, offset, theta, partial,
              cfg.rms_norm_eps, rope.cos_data(), rope.sin_data(), session); }
    } else {
        // Full attention derives V from raw K before applying K's learned scale.
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KProj));
          matmul_linear_weight(hidden, seq, H, *kpw, nkv * hd, K_cache, ctx); }
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KNorm));
          fused_full_decoder_kv_cache_postprocess(
              q, K_cache, V_cache, kn, seq, nkv, hd, offset, theta, partial,
              cfg.rms_norm_eps, rope.cos_data(), rope.sin_data(), session); }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Fused in-place masked softmax over scores (nq, seq, kv_len), bf16 in/out,
// fp32 math.  The causal/sliding mask is computed inline (same semantics as
// fill_causal_mask); `causal=false` gives the decoder's bidirectional view.
// Replaces: mask buffer + bf16->f32 + oneDNN softmax + f32->bf16.
//
// Positions are absolute: `q_pos0` is the true sequence coordinate of query
// row 0 and `kv_pos0` that of cache column 0.  Callers that tile the queries or
// slice the KV to a sliding-window band pass the band's true coordinates here —
// the mask (and the RoPE already baked into K/V at projection) keep referring to
// real sequence positions, never band-local ones.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GQA attention reading the KV cache directly through strided batched GEMMs.
// Q heads sharing a KV head are contiguous after transpose_q, so the GQA
// expansion folds into M = gqa_ratio * seq with batch = nkv — no expand_kv
// copies.  Cache layout: (kv_len, nkv, hd) row-major.
//
// The queries are tiled (height `tq`) and, for causal layers, each tile only
// touches the KV band it can actually see: positions above the tile's last
// query are dropped (causal upper bound) and, for sliding layers, positions
// older than `sliding_window` before the tile's first query are dropped too.
// This keeps the scores buffer at nq*tq*band instead of the O(nq*seq^2) full
// matrix — the latter is 2 GiB at seq=8192 with 16 heads and was the cliff that
// stalled long prompts.  Band slicing changes *which* K/V columns are read, not
// their coordinates: `q_pos0`/`kv_pos0` stay absolute, and the RoPE baked into
// K/V at projection is untouched.
// ---------------------------------------------------------------------------
diffarena::Alloc<bf16> gqa_attention(
    GpuEngine& ctx,
    const bf16* Q, int seq, int nq, int hd,
    const bf16* Kc, const bf16* Vc, int kv_len, int nkv,
    int past_offset, int sliding_window, bool causal,
    AttnProfileKind prof,
    Nvfp4GraphSession* /*session*/ = nullptr)
{
    auto& q = ctx.queue;
    int g = nq / nkv;

    auto& ar = diffarena::arena(ctx.index);
    auto ctx_tm_a = ar.alloc<bf16>((size_t)seq * nq * hd);  // full output
    bf16* ctx_tm = ctx_tm_a.data();

    // Query-tile height: bound the scores buffer to ~512 MB (bf16) by capping
    // nq*tq*kv_len.  Short prompts collapse to a single full-width tile (tq==seq),
    // so the common path is identical to the untiled version.
    constexpr size_t kScoresBudgetElems = (size_t)256 * 1024 * 1024;
    int tq = (int)std::max<size_t>(1, kScoresBudgetElems / ((size_t)nq * kv_len));
    tq = std::min(tq, seq);

    int kv0 = past_offset - (kv_len - seq);   // absolute position of Kc[0]

    // Hoist the per-tile temps out of the tile loop: Q_hm/ctx_hm are fixed at
    // nq*tq*hd, and scores is bounded by nq*tq*max_kb (kb shrinks across tiles
    // as the causal/sliding band narrows, so the first tile sets the peak).
    // Reusing one arena allocation across tiles avoids per-tile alloc/free churn.
    int max_kb = kv_len;
    if (causal) {
        max_kb = 0;
        for (int qs0 = 0; qs0 < seq; qs0 += tq) {
            int qe0 = std::min(seq, qs0 + tq);
            int q_lo0 = past_offset + qs0;
            int q_hi0 = past_offset + qe0 - 1;
            int ke0 = std::min(kv_len, (q_hi0 - kv0) + 1);
            int ks0 = (sliding_window == INT_MAX)
                   ? 0
                   : std::max(0, (q_lo0 - sliding_window + 1) - kv0);
            int kb0 = ke0 - ks0;
            if (kb0 > max_kb) max_kb = kb0;
        }
        if (max_kb <= 0) max_kb = 1;
    }
    auto Qhm_a   = ar.alloc<bf16>((size_t)nq * tq * hd);
    auto scores_a = ar.alloc<bf16>((size_t)nq * tq * max_kb);
    auto ctxhm_a = ar.alloc<bf16>((size_t)nq * tq * hd);

    for (int qs = 0; qs < seq; qs += tq) {
        int qe = std::min(seq, qs + tq);
        int t  = qe - qs;

        // KV band [ks, ke) (cache indices) this tile is allowed to attend to.
        int ks = 0, ke = kv_len;
        if (causal) {
            int q_lo = past_offset + qs;          // first query, absolute
            int q_hi = past_offset + qe - 1;      // last query, absolute
            ke = std::min(kv_len, (q_hi - kv0) + 1);                 // causal upper
            ks = (sliding_window == INT_MAX)
               ? 0
               : std::max(0, (q_lo - sliding_window + 1) - kv0);     // window lower
        }
        int kb = ke - ks;
        if (kb <= 0) continue;

        bf16* Q_hm   = Qhm_a.data();
        bf16* scores = scores_a.data();
        bf16* ctx_hm = ctxhm_a.data();

        // (t, nq, hd) tile -> (nq, t, hd)
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::QTranspose));
          transpose_q_into(q, Q_hm, Q + (size_t)qs * nq * hd, t, nq, hd); }

        const bf16* Kb = Kc + (size_t)ks * nkv * hd;
        const bf16* Vb = Vc + (size_t)ks * nkv * hd;

        // scores(b, m, n) = Q_hm(b, m, k) @ Kb[n][b][k] — K read in place.
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::ScoreMM));
          matmul_bf16_batched_strided(
            Q_hm, nkv, g * t, hd,
            (dnnl_dim_t)g * t * hd, hd, 1,
            Kb, kb,
            (dnnl_dim_t)hd, 1, (dnnl_dim_t)nkv * hd,
            scores,
            (dnnl_dim_t)g * t * kb, kb, 1,
            ctx); }

        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::Softmax));
          if (softmax_span_mode())
              fused_masked_softmax(q, scores, nq, t, kb,
                                   /*q_pos0=*/past_offset + qs,
                                   /*kv_pos0=*/kv0 + ks,
                                   sliding_window, causal);
          else
              fused_masked_softmax_branchy(q, scores, nq, t, kb,
                                           /*q_pos0=*/past_offset + qs,
                                           /*kv_pos0=*/kv0 + ks,
                                           sliding_window, causal); }

        // ctx(b, m, n) = scores(b, m, k) @ Vb[k][b][n] — V read in place.
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::ValueMM));
          matmul_bf16_batched_strided(
            scores, nkv, g * t, kb,
            (dnnl_dim_t)g * t * kb, kb, 1,
            Vb, hd,
            (dnnl_dim_t)hd, (dnnl_dim_t)nkv * hd, 1,
            ctx_hm,
            (dnnl_dim_t)g * t * hd, hd, 1,
            ctx); }

        // (nq, t, hd) -> (t, nq, hd) into this tile's rows of the output.
        { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::Scatter));
          scatter_ctx(q, ctx_hm, ctx_tm + (size_t)qs * nq * hd, nq, t, hd); }
    }
    return ctx_tm_a;
}

struct SdpaKey {
    int gpu, nq, nkv, seq, kv_len, hd, mask_type;
    bool operator==(const SdpaKey& o) const {
        return gpu == o.gpu && nq == o.nq && nkv == o.nkv && seq == o.seq &&
               kv_len == o.kv_len && hd == o.hd && mask_type == o.mask_type;
    }
};

struct SdpaKeyHash {
    size_t operator()(const SdpaKey& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        auto mix = [&](int v) {
            h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        mix(k.nq); mix(k.nkv); mix(k.seq); mix(k.kv_len); mix(k.hd);
        mix(k.mask_type);
        return h;
    }
};

static dnnl::memory::desc sdpa_q_dst_md(int nq, int seq, int hd) {
    using dt = dnnl::memory::data_type;
    return dnnl::memory::desc(
        {1, nq, seq, hd}, dt::bf16,
        {(dnnl_dim_t)nq * seq * hd, hd, (dnnl_dim_t)nq * hd, 1});
}

static dnnl::memory::desc sdpa_k_md(int nkv, int kv_len, int hd) {
    using dt = dnnl::memory::data_type;
    return dnnl::memory::desc(
        {1, nkv, hd, kv_len}, dt::bf16,
        {(dnnl_dim_t)nkv * kv_len * hd, hd, 1, (dnnl_dim_t)nkv * hd});
}

static dnnl::memory::desc sdpa_v_md(int nkv, int kv_len, int hd) {
    using dt = dnnl::memory::data_type;
    return dnnl::memory::desc(
        {1, nkv, kv_len, hd}, dt::bf16,
        {(dnnl_dim_t)nkv * kv_len * hd, hd, (dnnl_dim_t)nkv * hd, 1});
}

static dnnl::memory::desc sdpa_mask_md(int seq, int kv_len) {
    using dt = dnnl::memory::data_type;
    return dnnl::memory::desc(
        {1, 1, seq, kv_len}, dt::f32,
        {(dnnl_dim_t)seq * kv_len, (dnnl_dim_t)seq * kv_len,
         (dnnl_dim_t)kv_len, 1});
}

static dnnl_status_t create_sdpa_primitive_desc_runtime(
    dnnl_primitive_desc_t* raw_pd,
    dnnl_engine_t engine,
    const_dnnl_memory_desc_t q_md,
    const_dnnl_memory_desc_t k_md,
    const_dnnl_memory_desc_t v_md,
    const_dnnl_memory_desc_t dst_md,
    const_dnnl_memory_desc_t mask_md,
    const_dnnl_memory_desc_t scale_md,
    int mask_type,
    dnnl_dim_t kv_head_number)
{
    using CreateNew = dnnl_status_t (*)(
        dnnl_primitive_desc_t*, dnnl_engine_t,
        const_dnnl_memory_desc_t, const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t, const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t, const_dnnl_memory_desc_t,
        bool, dnnl_dim_t, int, dnnl_alg_kind_t, dnnl_prop_kind_t,
        const_dnnl_primitive_attr_t, const_dnnl_primitive_attr_t,
        const_dnnl_primitive_attr_t);
    using CreateOld = dnnl_status_t (*)(
        dnnl_primitive_desc_t*, dnnl_engine_t,
        const_dnnl_memory_desc_t, const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t, const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t, dnnl_data_type_t,
        bool, dnnl_dim_t, int, dnnl_alg_kind_t,
        const_dnnl_primitive_attr_t, const_dnnl_primitive_attr_t,
        const_dnnl_primitive_attr_t);

    static void* create_new = dlsym(RTLD_DEFAULT,
        "_Z26sdpa_primitive_desc_createPP19dnnl_primitive_descP11dnnl_engine"
        "PK16dnnl_memory_descS6_S6_S6_S6_S6_bli15dnnl_alg_kind_t"
        "16dnnl_prop_kind_tPK19dnnl_primitive_attrSB_SB_");
    static void* create_old = dlsym(RTLD_DEFAULT,
        "_Z26sdpa_primitive_desc_createPP19dnnl_primitive_descP11dnnl_engine"
        "PK16dnnl_memory_descS6_S6_S6_S6_16dnnl_data_type_tbli"
        "15dnnl_alg_kind_tPK19dnnl_primitive_attrSB_SB_");

    if (create_new) {
        return reinterpret_cast<CreateNew>(create_new)(
            raw_pd, engine, q_md, k_md, v_md, dst_md,
            mask_md, scale_md,
            /*invert_scale=*/false, kv_head_number,
            mask_type, dnnl_softmax_accurate,
            dnnl_forward_inference,
            /*attr=*/nullptr, /*kq_attr=*/nullptr, /*vs_attr=*/nullptr);
    }
    if (create_old) {
        return reinterpret_cast<CreateOld>(create_old)(
            raw_pd, engine, q_md, k_md, v_md, dst_md,
            mask_md, dnnl_data_type_undef,
            /*invert_scale=*/false, kv_head_number,
            mask_type, dnnl_softmax_accurate,
            /*attr=*/nullptr, /*kq_attr=*/nullptr, /*vs_attr=*/nullptr);
    }
    return dnnl_unimplemented;
}

diffarena::Alloc<bf16> gqa_attention_sdpa(
    GpuEngine& ctx,
    const bf16* Q, int seq, int nq, int hd,
    const bf16* Kc, const bf16* Vc, int kv_len, int nkv,
    const float* mask, int mask_type,
    AttnProfileKind prof,
    Nvfp4GraphSession* /*session*/ = nullptr)
{
    static std::unordered_map<SdpaKey, dnnl::primitive, SdpaKeyHash> cache;

    auto& ar = diffarena::arena(ctx.index);
    auto ctx_tm_a = ar.alloc<bf16>((size_t)seq * nq * hd);
    bf16* ctx_tm = ctx_tm_a.data();

    SdpaKey key{ctx.index, nq, nkv, seq, kv_len, hd, mask_type};
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto q_md = sdpa_q_dst_md(nq, seq, hd);
        auto k_md = sdpa_k_md(nkv, kv_len, hd);
        auto v_md = sdpa_v_md(nkv, kv_len, hd);
        auto dst_md = sdpa_q_dst_md(nq, seq, hd);
        auto mask_md = mask ? sdpa_mask_md(seq, kv_len) : dnnl::memory::desc();

        dnnl::memory::desc scale_md;
        dnnl_primitive_desc_t raw_pd = nullptr;
        dnnl_status_t st = create_sdpa_primitive_desc_runtime(
            &raw_pd, ctx.engine.get(), q_md.get(), k_md.get(), v_md.get(),
            dst_md.get(), mask ? mask_md.get() : nullptr, scale_md.get(),
            mask_type, (dnnl_dim_t)nkv);
        dnnl::error::wrap_c_api(st,
            "could not create oneDNN SDPA primitive descriptor");

        dnnl::primitive_desc pd(raw_pd);
        cache.emplace(key, dnnl::primitive(pd));
        it = cache.find(key);
    }

    { DIFF_PROF(ctx.queue, attn_prof(prof, AttnProfilePhase::ScoreMM));
      auto q_md = sdpa_q_dst_md(nq, seq, hd);
      auto k_md = sdpa_k_md(nkv, kv_len, hd);
      auto v_md = sdpa_v_md(nkv, kv_len, hd);
      auto dst_md = sdpa_q_dst_md(nq, seq, hd);
      std::unordered_map<int, dnnl::memory> args = {
          {DNNL_ARG_SRC_0, dnnl::sycl_interop::make_memory(
              q_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
              const_cast<bf16*>(Q))},
          {DNNL_ARG_SRC_1, dnnl::sycl_interop::make_memory(
              k_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
              const_cast<bf16*>(Kc))},
          {DNNL_ARG_SRC_2, dnnl::sycl_interop::make_memory(
              v_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
              const_cast<bf16*>(Vc))},
          {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
              dst_md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
              ctx_tm)}
      };
      if (mask) {
          args.emplace(DNNL_ARG_SHIFT, dnnl::sycl_interop::make_memory(
              sdpa_mask_md(seq, kv_len), ctx.engine,
              dnnl::sycl_interop::memory_kind::usm,
              const_cast<float*>(mask)));
      }
      it->second.execute(ctx.stream, args); }

    return ctx_tm_a;
}

}  // namespace

void encoder_attention_forward(
    GpuEngine& ctx, const DiffLayer& lw,
    const bf16* hidden, bf16* out,
    DiffLayerKv& kv, int seq, int past_len,
    const DiffTextConfig& cfg,
    Nvfp4GraphSession* session)
{
    auto& q = ctx.queue;
    int H = cfg.hidden_size;
    AttnProfileKind prof = attn_profile_kind(/*is_encoder=*/true, lw.is_full);
    QKV x = project_qkv(ctx, lw, hidden, seq, cfg, past_len, prof, session);

    // Append K/V to cache at past_len, then release them (the GEMMs read the
    // cache, not these staging buffers).
    size_t row = (size_t)x.nkv * x.hd;
    int end_pos = past_len + seq;
    if (end_pos > kv.max_seq)
        throw std::runtime_error("encoder KV cache capacity exceeded");
    { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::KVWrite));
      q.memcpy(kv.k.data() + (size_t)past_len * row, x.K.data(), (size_t)seq * row * sizeof(bf16));
      q.memcpy(kv.v.data() + (size_t)past_len * row, x.V.data(), (size_t)seq * row * sizeof(bf16)); }
    kv.filled = end_pos;
    x.K.reset(); x.V.reset();

    int win = lw.is_full ? INT_MAX : cfg.sliding_window;
    diffarena::Alloc<float> sdpa_mask;
    diffarena::Alloc<bf16> attn;
    if (lw.is_full && onednn_sdpa_encoder_full_enabled()) {
        // K length may exceed Q length for chunked prefill; bottom-right causal
        // aligns the causal diagonal with the newest query chunk.
        attn = gqa_attention_sdpa(ctx,
            x.Q.data(), seq, x.nq, x.hd,
            kv.k.data(), kv.v.data(), kv.filled, x.nkv,
            /*mask=*/nullptr, /*mask_type=*/3, prof, session);
    } else if (!lw.is_full && onednn_sdpa_encoder_sliding_enabled()) {
        auto& ar = diffarena::arena(ctx.index);
        sdpa_mask = ar.alloc<float>((size_t)seq * kv.filled);
        fill_causal_mask(q, sdpa_mask.data(), seq, kv.filled, past_len, win);
        attn = gqa_attention_sdpa(ctx,
            x.Q.data(), seq, x.nq, x.hd,
            kv.k.data(), kv.v.data(), kv.filled, x.nkv,
            sdpa_mask.data(), /*mask_type=*/1, prof, session);
    } else {
        attn = gqa_attention(ctx,
            x.Q.data(), seq, x.nq, x.hd,
            kv.k.data(), kv.v.data(), kv.filled, x.nkv,
            /*past_offset=*/past_len, win, /*causal=*/true, prof, session);
    }
    x.Q.reset();   // Q consumed by the score GEMM

    const DiffLinearWeight& o_proj = lw.is_full
        ? std::get<DiffFullAttn>(lw.attn).o_proj
        : std::get<DiffSlidingAttn>(lw.attn).o_proj;
    { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::OProj));
      matmul_linear_weight(attn.data(), seq, x.nq * x.hd, o_proj, H, out, ctx); }
}

void decoder_attention_forward(
    GpuEngine& ctx, const DiffLayer& lw,
    const bf16* hidden, bf16* out,
    DiffLayerKv& enc_kv, int seq, int enc_len,
    const DiffTextConfig& cfg,
    Nvfp4GraphSession* session)
{
    auto& q = ctx.queue;
    int H = cfg.hidden_size;
    AttnProfileKind prof = attn_profile_kind(/*is_encoder=*/false, lw.is_full);

    // Stage canvas K/V in the cache's spare region past enc_len (read-only for
    // the cache proper: `filled` stays at enc_len, and the next block's encode
    // overwrites this region with the committed canvas anyway).  The decoder
    // full layers can read [0, enc_len + seq), while sliding layers can
    // right-slice the encoder tail and still include canvas with no concat copy.
    int kv_len = enc_len + seq;
    if (kv_len > enc_kv.max_seq)
        throw std::runtime_error("decoder canvas exceeds KV cache capacity");

    int direct_nkv = lw.is_full ? cfg.num_global_kv_heads : cfg.num_kv_heads;
    int direct_hd  = lw.is_full ? cfg.global_head_dim     : cfg.head_dim;
    size_t row = (size_t)direct_nkv * direct_hd;
    bf16* k_dst = enc_kv.k.data() + (size_t)enc_len * row;
    bf16* v_dst = enc_kv.v.data() + (size_t)enc_len * row;

    // Decoder always materializes K/V directly into the cache canvas (no staging
    // copy/launch).  DIFF_DECODE_KV_DIRECT_CACHE / DIFF_STAGE_DECODER_KV_KERNEL
    // and the staging kernel are gone.
    QKV x = project_qkv_decoder_direct_cache(ctx, lw, hidden, seq, cfg, enc_len,
                                             k_dst, v_dst, prof, session);

    // Bidirectional decoder attention: full layers see all encoder KV + canvas;
    // sliding layers see the last sliding_window-1 encoder positions + canvas
    // (the canvas's own first position fills the window's last slot).
    // Canvas K/V was staged at enc_len, so the right-sliced encoder tail and the
    // canvas rows are still one contiguous cache span.
    int kv0 = lw.is_full ? 0 : std::max(0, enc_len - cfg.sliding_window + 1);
    const bf16* Kc = enc_kv.k.data() + (size_t)kv0 * row;
    const bf16* Vc = enc_kv.v.data() + (size_t)kv0 * row;
    auto attn = onednn_sdpa_decoder_enabled()
        ? gqa_attention_sdpa(ctx,
            x.Q.data(), seq, x.nq, x.hd, Kc, Vc, kv_len - kv0, x.nkv,
            /*mask=*/nullptr, /*mask_type=*/0, prof, session)
        : gqa_attention(ctx,
            x.Q.data(), seq, x.nq, x.hd, Kc, Vc, kv_len - kv0, x.nkv,
            /*past_offset=*/0, /*sliding_window=*/INT_MAX, /*causal=*/false,
            prof, session);
    x.Q.reset();

    const DiffLinearWeight& o_proj = lw.is_full
        ? std::get<DiffFullAttn>(lw.attn).o_proj
        : std::get<DiffSlidingAttn>(lw.attn).o_proj;
    { DIFF_PROF(q, attn_prof(prof, AttnProfilePhase::OProj));
      matmul_linear_weight(attn.data(), seq, x.nq * x.hd, o_proj, H, out, ctx); }
}

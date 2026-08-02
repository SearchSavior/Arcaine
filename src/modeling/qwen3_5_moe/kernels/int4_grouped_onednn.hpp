#pragma once
// oneDNN experimental grouped W4A16 routed-experts path for the Qwen3.5-MoE
// AWQ INT4 checkpoint (QWEN35_MOE_INT4_IMPL=grouped). Primary MoE execution:
// custom kernels keep routing/gather/combine; the three expert GEMMs run as
// oneDNN grouped matmuls over the canonical contiguous raw-u4 expert tensors
// (QwenInt4ExpertsGrouped):
//
//   up           = grouped_matmul(src_g, up_q)                          [bf16]
//   intermediate = grouped_matmul(src_g, gate_q,
//                      post_ops = { swish,
//                                   binary_mul(up, grouped [pairs,I]),
//                                   binary_mul(rw, dense [pairs,1]) })  [bf16]
//   pair_out     = grouped_matmul(intermediate, down_q)                 [bf16]
//   out[t]       = sum_slots pair_out[inv_pos[t*top_k + s]]  (custom combine)
//
// Validated on oneDNN 3.13 grouped_gemm:micro (BMG): weights u4 tag::acb (=
// [E][N][K] nibbles, K innermost), scales/zero-points mask 7 with groups
// {group_size, 1}, NATIVE u4 zero points (no XOR rebase / rowsum correction),
// fpmath bf16 apply_to_int. Grouped offsets are s32 cumulative END offsets —
// exactly offsets[1..E] produced by qwen_int4_grouped_build_routes.
//
// Primitives bake in the total row count (pairs), so they are cached per
// (gpu, E, K, N, gs, pairs, kind, zp). The cache is hard-capped (clear-on-
// overflow): distinct prefill lengths would otherwise grow it unboundedly.
//
// Also provides qwen_int4_matmul_u4zp: the per-expert 2D oneDNN matmul with
// native u4 zero points used by the retained host-orchestrated reference
// path (same canonical storage, expert slices).
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/engine.hpp"
#include "../weights.hpp"
#include "int4_grouped_moe.hpp" // qwen_int4_grouped_build_routes

// Model-local namespace: these kernels are per-model COPIES (see
// AGENTS.md model isolation). Global-scope inline functions with
// identical names in other models would ODR-merge at link time;
// divergent bodies (e.g. tiled attention) then crash at runtime.
namespace qwen35moe_kernels {

// ---------------------------------------------------------------------------
// Per-expert 2D matmul with native u4 zero points (reference path).
// wq: [N, K/2] raw u4 (K innermost = oneDNN {K,N} u4 tag::ba), ws: [G,N] BF16,
// wzp: [G, N/2] raw u4 packed along N (nullptr = symmetric, zp_u == 8).
// ---------------------------------------------------------------------------
namespace qwen_u4zp_detail {
struct Key {
    int gpu, M, K, N, gs;
    bool zp;
    bool operator==(const Key& o) const {
        return gpu == o.gpu && M == o.M && K == o.K && N == o.N && gs == o.gs &&
               zp == o.zp;
    }
};
struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.gs) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.zp) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
struct Entry {
    dnnl::matmul prim;
    dnnl::memory::desc src_md, wei_md, dst_md, sc_md, zp_md;
};
inline std::unordered_map<Key, Entry, KeyHash>& cache() {
    static std::unordered_map<Key, Entry, KeyHash> c;
    return c;
}
} // namespace qwen_u4zp_detail

inline void qwen_int4_matmul_u4zp(GpuEngine& ctx, const bf16* A, int M, int K,
                                  int N, int gs, const uint8_t* wq,
                                  const bf16* ws, const uint8_t* wzp, bf16* C) {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const int G = K / gs;
    qwen_u4zp_detail::Key key{ctx.index, M, K, N, gs, wzp != nullptr};
    auto& c = qwen_u4zp_detail::cache();
    auto it = c.find(key);
    if (it == c.end()) {
        if (c.size() > 256) c.clear();
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {gs, 1}, dt::bf16);
        if (wzp)
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {gs, 1},
                                 dt::u4);
        attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, /*apply_to_int=*/true);
        qwen_u4zp_detail::Entry e{
            {},
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            dnnl::memory::desc({K, N}, dt::u4, tag::ba),
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab),
            dnnl::memory::desc({G, N}, dt::bf16, tag::ab),
            dnnl::memory::desc({G, N}, dt::u4, tag::ab)};
        e.prim = dnnl::matmul(dnnl::matmul::primitive_desc(
            ctx.engine, e.src_md, e.wei_md, e.dst_md, attr));
        it = c.emplace(key, std::move(e)).first;
    }
    const qwen_u4zp_detail::Entry& e = it->second;
    auto mk = [&](const dnnl::memory::desc& md, void* p) {
        return dnnl::sycl_interop::make_memory(
            md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, p);
    };
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, mk(e.src_md, const_cast<bf16*>(A))},
        {DNNL_ARG_WEIGHTS, mk(e.wei_md, const_cast<uint8_t*>(wq))},
        {DNNL_ARG_DST, mk(e.dst_md, C)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, mk(e.sc_md, const_cast<bf16*>(ws))},
    };
    if (wzp)
        args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] =
            mk(e.zp_md, const_cast<uint8_t*>(wzp));
    e.prim.execute(ctx.stream, args);
}

// ---------------------------------------------------------------------------
// Grouped-path device kernels: gather into expert-major order, indirect
// top-k combine.
// ---------------------------------------------------------------------------

// src[p, :] = hidden[tokens[p] / top_k, :]; rw[p] = wgt[tokens[p]];
// inv_pos[tokens[p]] = p (inverse permutation for the combine).
inline void qwen_int4_grouped_gather(sycl::queue& q, const bf16* hidden, int H,
                                     int top_k, const int32_t* tokens,
                                     const float* wgt, int pairs, bf16* src,
                                     bf16* rw, int32_t* inv_pos) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)pairs), [=](sycl::id<1> id) {
            int p = (int)id[0];
            int pair = tokens[p];
            rw[p] = float_to_bf16(wgt[pair]);
            inv_pos[pair] = p;
        });
    });
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)pairs, (size_t)H),
                       [=](sycl::id<2> id) {
            int p = (int)id[0], d = (int)id[1];
            int tok = tokens[p] / top_k;
            src[(size_t)p * H + d] = hidden[(size_t)tok * H + d];
        });
    });
}

// out[t, d] = sum_s down_out[inv_pos[t*top_k + s], d]
// fp32 accumulation, single bf16 rounding — matches qwen_int4_grouped_combine.
inline void qwen_int4_grouped_combine_indirect(sycl::queue& q,
                                               const bf16* down_out,
                                               const int32_t* inv_pos, int S,
                                               int top_k, int H, bf16* out) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)S, (size_t)H), [=](sycl::id<2> id) {
            int t = (int)id[0], d = (int)id[1];
            float acc = 0.0f;
            for (int s = 0; s < top_k; ++s) {
                int p = inv_pos[(size_t)t * top_k + s];
                acc += bf16_to_float(down_out[(size_t)p * H + d]);
            }
            out[(size_t)t * H + d] = float_to_bf16(acc);
        });
    });
}

// ---------------------------------------------------------------------------
// Grouped matmul primitive cache + execution.
// ---------------------------------------------------------------------------
namespace qwen_grouped_detail {
enum Kind { kPlain = 0, kGatePostOps = 1 };
struct Key {
    int gpu, E, K, N, gs, total, kind;
    bool zp;
    bool operator==(const Key& o) const {
        return gpu == o.gpu && E == o.E && K == o.K && N == o.N && gs == o.gs &&
               total == o.total && kind == o.kind && zp == o.zp;
    }
};
struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = std::hash<int>{}(k.gpu);
        h ^= std::hash<int>{}(k.E) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.gs) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.total) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.kind) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.zp) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
struct Entry {
    dnnl::matmul prim;
    dnnl::memory::desc src_md, wei_md, dst_md, sc_md, zp_md;
    dnnl::memory::desc po_grouped_md, po_dense_md;
};
inline std::unordered_map<Key, Entry, KeyHash>& cache() {
    static std::unordered_map<Key, Entry, KeyHash> c;
    return c;
}

inline const Entry& entry(GpuEngine& ctx, int E, int K, int N, int gs,
                          int total, int kind, bool has_zp) {
    Key key{ctx.index, E, K, N, gs, total, kind, has_zp};
    auto& c = cache();
    auto it = c.find(key);
    if (it != c.end()) return it->second;
    if (c.size() > 256) c.clear();

    using dt = dnnl::memory::data_type;
    Entry e;
    e.src_md = dnnl::memory::desc::grouped({total, K}, dt::bf16, 0, E);
    e.wei_md = dnnl::memory::desc({E, K, N}, dt::u4,
                                  dnnl::memory::format_tag::acb);
    e.dst_md = dnnl::memory::desc::grouped({total, N}, dt::bf16, 0, E);
    e.sc_md = dnnl::memory::desc({E, K / gs, N}, dt::bf16,
                                 dnnl::memory::format_tag::abc);
    e.zp_md = dnnl::memory::desc({E, K / gs, N}, dt::u4,
                                 dnnl::memory::format_tag::abc);

    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, 7, {gs, 1}, dt::bf16);
    if (has_zp) attr.set_zero_points(DNNL_ARG_WEIGHTS, 7, {gs, 1}, dt::u4);
    attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, /*apply_to_int=*/true);
    if (kind == kGatePostOps) {
        e.po_grouped_md = dnnl::memory::desc::grouped({total, N}, dt::bf16, 0, E);
        e.po_dense_md = dnnl::memory::desc({total, 1}, dt::bf16,
                                           dnnl::memory::format_tag::ab);
        dnnl::post_ops po;
        po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
        po.append_binary(dnnl::algorithm::binary_mul, e.po_grouped_md);
        po.append_binary(dnnl::algorithm::binary_mul, e.po_dense_md);
        attr.set_post_ops(po);
    }
    e.prim = dnnl::matmul(dnnl::matmul::primitive_desc(ctx.engine, e.src_md,
                                                       e.wei_md, e.dst_md, attr));
    return c.emplace(key, std::move(e)).first->second;
}

inline dnnl::memory mk_grouped(GpuEngine& ctx, const dnnl::memory::desc& md,
                               void* vals, int32_t* offs_ends) {
    return dnnl::sycl_interop::make_memory(
        md, ctx.engine, dnnl::sycl_interop::memory_kind::usm,
        std::vector<void*>{vals, (void*)offs_ends});
}
inline dnnl::memory mk_plain(GpuEngine& ctx, const dnnl::memory::desc& md,
                             void* vals) {
    return dnnl::sycl_interop::make_memory(
        md, ctx.engine, dnnl::sycl_interop::memory_kind::usm, vals);
}

inline void exec(GpuEngine& ctx, int E, int K, int N, int gs, int total,
                 int kind, bf16* src, const uint8_t* wq, const bf16* ws,
                 const uint8_t* wzp, bf16* dst, int32_t* offs_ends,
                 bf16* po_grouped = nullptr, bf16* po_dense = nullptr) {
    const Entry& e = entry(ctx, E, K, N, gs, total, kind, wzp != nullptr);
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, mk_grouped(ctx, e.src_md, src, offs_ends)},
        {DNNL_ARG_WEIGHTS, mk_plain(ctx, e.wei_md, const_cast<uint8_t*>(wq))},
        {DNNL_ARG_DST, mk_grouped(ctx, e.dst_md, dst, offs_ends)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
         mk_plain(ctx, e.sc_md, const_cast<bf16*>(ws))},
    };
    if (wzp)
        args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] =
            mk_plain(ctx, e.zp_md, const_cast<uint8_t*>(wzp));
    if (kind == kGatePostOps) {
        args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1] =
            mk_grouped(ctx, e.po_grouped_md, po_grouped, offs_ends);
        args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1] =
            mk_plain(ctx, e.po_dense_md, po_dense);
    }
    e.prim.execute(ctx.stream, args);
}
} // namespace qwen_grouped_detail

// ---------------------------------------------------------------------------
// Full grouped routed-experts forward. idx/wgt are DEVICE [S*top_k] buffers
// produced by qwen_moe_router_topk (kernels/router_topk.hpp) — routing never
// leaves the device. Scratch buffers are cached grow-only; a mutex serializes
// host-side reuse across sessions.
// ---------------------------------------------------------------------------
namespace qwen_moe_grouped_detail {
struct Scratch {
    GpuBuffer<int32_t> offsets;   // [E+1]   (build_routes layout)
    GpuBuffer<int32_t> offs_ends; // [E]     aligned cumulative ends (oneDNN)
    GpuBuffer<int32_t> tokens;    // [pairs]
    GpuBuffer<int32_t> inv_pos;   // [pairs]
    GpuBuffer<bf16>    src;       // [pairs, H]
    GpuBuffer<bf16>    rw;        // [pairs]     (dense [pairs,1] post-op)
    GpuBuffer<bf16>    up;        // [pairs, I]
    GpuBuffer<bf16>    inter;     // [pairs, I]
    GpuBuffer<bf16>    pair_out;  // [pairs, H]
    size_t pairs_cap = 0;
    int    experts_cap = 0;
};
inline Scratch& scratch() {
    static Scratch s;
    return s;
}
inline std::mutex& scratch_mutex() {
    static std::mutex m;
    return m;
}
} // namespace qwen_moe_grouped_detail

inline void qwen_routed_experts_forward_grouped(
    GpuEngine& ctx, const QwenInt4ExpertsGrouped& w,
    const bf16* hidden,                 // device [S, H]
    const int32_t* idx,                 // device [S*top_k]
    const float* wgt,                   // device [S*top_k]
    bf16* out,                          // device [S, H]
    int S, int top_k) {
    auto& q = ctx.queue;
    const int H = w.hidden, I = w.inter, E = w.E, gs = w.group_size;
    const int pairs = S * top_k;
    // oneDNN grouped_gemm:micro loses the device in the post-op (gate) kernel
    // when total rows < number of groups (reproduced: G=256, total=24). Pad
    // total to >= E with unused tail rows appended to the last expert's
    // segment; GEMM rows are independent and pad-row outputs are never read
    // (combine only touches inv_pos[0..pairs)). Side benefit: every batch
    // with pairs <= E shares one cached primitive (total == E).
    const int total = std::max(pairs, E);

    std::lock_guard<std::mutex> lock(qwen_moe_grouped_detail::scratch_mutex());
    auto& sc = qwen_moe_grouped_detail::scratch();
    if (sc.pairs_cap < (size_t)total) {
        sc.tokens   = GpuBuffer<int32_t>((size_t)total, q);
        sc.inv_pos  = GpuBuffer<int32_t>((size_t)total, q);
        sc.src      = GpuBuffer<bf16>((size_t)total * H, q);
        sc.rw       = GpuBuffer<bf16>((size_t)total, q);
        sc.up       = GpuBuffer<bf16>((size_t)total * I, q);
        sc.inter    = GpuBuffer<bf16>((size_t)total * I, q);
        sc.pair_out = GpuBuffer<bf16>((size_t)total * H, q);
        sc.pairs_cap = (size_t)total;
    }
    if (sc.experts_cap < E) {
        sc.offsets = GpuBuffer<int32_t>((size_t)E + 1, q);
        sc.offs_ends = GpuBuffer<int32_t>((size_t)E, q);
        sc.experts_cap = E;
    }

    qwen_int4_grouped_build_routes(q, idx, E, pairs,
                                   sc.offsets.data(), sc.tokens.data());
    qwen_int4_grouped_gather(q, hidden, H, top_k, sc.tokens.data(),
                             wgt, pairs, sc.src.data(), sc.rw.data(),
                             sc.inv_pos.data());

    // oneDNN grouped offsets: s32 cumulative ENDS per group == offsets[1..E]
    // in an aligned buffer (offsets.data()+1 is only 4B-aligned), with the
    // last end extended to the padded total (pad rows belong to expert E-1).
    int32_t* offs_ends = sc.offs_ends.data();
    const int32_t* offsets = sc.offsets.data();
    q.parallel_for(sycl::range<1>((size_t)E), [=](sycl::id<1> id) {
        int i = (int)id[0];
        offs_ends[i] = (i == E - 1) ? total : offsets[i + 1];
    });

    qwen_grouped_detail::exec(ctx, E, H, I, gs, total,
                              qwen_grouped_detail::kPlain, sc.src.data(),
                              w.up_q.data(), w.up_s.data(),
                              w.up_has_zp() ? w.up_zp.data() : nullptr,
                              sc.up.data(), offs_ends);
    qwen_grouped_detail::exec(ctx, E, H, I, gs, total,
                              qwen_grouped_detail::kGatePostOps, sc.src.data(),
                              w.gate_q.data(), w.gate_s.data(),
                              w.gate_has_zp() ? w.gate_zp.data() : nullptr,
                              sc.inter.data(), offs_ends, sc.up.data(),
                              sc.rw.data());
    qwen_grouped_detail::exec(ctx, E, I, H, gs, total,
                              qwen_grouped_detail::kPlain, sc.inter.data(),
                              w.down_q.data(), w.down_s.data(),
                              w.down_has_zp() ? w.down_zp.data() : nullptr,
                              sc.pair_out.data(), offs_ends);

    qwen_int4_grouped_combine_indirect(q, sc.pair_out.data(), sc.inv_pos.data(),
                                       S, top_k, H, out);
}

} // namespace qwen35moe_kernels

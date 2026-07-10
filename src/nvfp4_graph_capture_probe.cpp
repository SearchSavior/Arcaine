// =============================================================================
// Phase 0 probe — Is dnnl::matmul::execute() capture-safe inside a SYCL
// command_graph recording on this backend (oneDNN 3.12 SYCL interop, Intel GPU)?
//
// This is the BLOCKING gate for whole-step SYCL graph capture of the
// DiffusionGemma denoising loop. The attention projections and the MoE
// grouped-GEMMs both bottom out in oneDNN matmul primitives executed on the
// GpuEngine queue via `primitive.execute(ctx.stream, ...)` (ctx.stream is a
// oneDNN stream backed by the in-order SYCL queue). If those primitives cannot
// be captured by `command_graph::begin_recording(q)` and replayed bit-exactly,
// then a per-step session that sweeps them in would either throw (recoverable)
// or, worse, silently drop the matmul from the graph and produce zeroed/wrong
// outputs (NOT recoverable by the existing per-kernel catch-and-fallback).
//
// Method (per matmul variant):
//   1. Warm the primitive cache (and, if DIFF_NVFP4_WEIGHT_LAYOUT=any, the
//      coalesced weight) with one direct call -- discarded. This guarantees the
//      recorded graph contains ONLY `matmul.execute`, not first-time JIT/reorder.
//   2. CONTROL: call the matmul twice DIRECTLY (no graph), diff -> establishes
//      the oneDNN non-determinism floor (expected bit-exact on Intel GPU).
//   3. GRAPH: open command_graph, begin_recording, call the matmul once,
//      end_recording, finalize, q.ext_oneapi_graph(exec). Download + diff vs
//      the direct reference.
//   4. REPLAY: zero the baked-in DST buffer, replay the SAME executable, diff
//      vs reference -- confirms replay reproduces the result at the baked-in
//      USM address (this is what a warm session relies on every denoising step).
//
// Verdict:
//   PASS  iff no exception during capture/finalize/replay AND graph output is
//         bit-exact vs direct (within the control floor) AND replay2 bit-exact.
//   FAIL  iff any exception, OR silent divergence (graph output != direct).
//
// Env knobs (AB-testable, per AGENTS.md):
//   GC_M / GC_K / GC_N      single matmul dims (defaults 16 / 64 / 64)
//   GC_B                    batched batch count  (default 4)
//   GC_SKIP_SINGLE / GC_SKIP_BATCHED =1 to skip a path
//   DIFF_NVFP4_WEIGHT_LAYOUT=any  exercises the reorder path too
// =============================================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph/command_graph.hpp>

#include "common/gpu/buffer.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/nvfp4.hpp"
#include "common/gpu/ops.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using bf16 = uint16_t;

static int env_int(const char* name, int def) {
    if (const char* e = std::getenv(name)) { int v = std::atoi(e); return v > 0 ? v : def; }
    return def;
}
static bool env_skip(const char* name) {
    if (const char* e = std::getenv(name)) return std::atoi(e) == 1;
    return false;
}

struct DiffResult { double max_abs; size_t mismatched; size_t n; };

static DiffResult diff_buffers(const std::vector<bf16>& a,
                                const std::vector<bf16>& b) {
    DiffResult r{0.0, 0, std::min(a.size(), b.size())};
    for (size_t i = 0; i < r.n; ++i) {
        double d = std::fabs((double)bf16_to_float(a[i]) -
                             (double)bf16_to_float(b[i]));
        if (d > r.max_abs) r.max_abs = d;
        if (a[i] != b[i]) ++r.mismatched;
    }
    return r;
}

static void print_diff(const char* tag, const DiffResult& r) {
    std::printf("  %-28s max_abs=%.6g mismatched=%zu/%zu (%.4f%%)\n",
                tag, r.max_abs, r.mismatched, r.n,
                r.n ? 100.0 * r.mismatched / r.n : 0.0);
}

// -----------------------------------------------------------------------------
// Single matmul: matmul_nvfp4_packed (the q/k/v/o-proj and dense-MLP GEMM).
// -----------------------------------------------------------------------------
static bool probe_single(GpuEngine& ctx, int M, int K, int N) {
    auto& q = ctx.queue;
    int G = K / 16;
    std::printf("[single] M=%d K=%d N=%d (layout=%s)\n", M, K, N,
                nvfp4_weight_layout() == Nvfp4WeightLayout::Any ? "any" : "raw");

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> nib(0, 15), sc(40, 110);
    std::vector<uint8_t> hA((size_t)M * K / 2), hAs((size_t)M * G),
                         hW((size_t)K * N / 2), hWs((size_t)G * N);
    for (auto& b : hA)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& b : hW)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& b : hAs) b = (uint8_t)sc(rng);
    for (auto& b : hWs) b = (uint8_t)sc(rng);
    float dst_scale = 1.0f;

    GpuBuffer<uint8_t> A((size_t)M * K / 2, q);   A.upload(hA.data(), A.count());
    GpuBuffer<uint8_t> As((size_t)M * G, q);      As.upload(hAs.data(), As.count());
    GpuBuffer<uint8_t> W((size_t)K * N / 2, q);   W.upload(hW.data(), W.count());
    GpuBuffer<uint8_t> Ws((size_t)G * N, q);      Ws.upload(hWs.data(), Ws.count());
    GpuBuffer<float>   Ds(1, q);                  Ds.upload(&dst_scale, 1);

    Nvfp4Linear Wlin;
    Wlin.in_features = K; Wlin.out_features = N; Wlin.input_global_scale = 1.0f;
    Wlin.weight_packed = std::move(W);
    Wlin.weight_scale  = std::move(Ws);
    Wlin.dst_scale     = std::move(Ds);

    // 1) Warm primitive cache (+ coalesced weight if layout=any). Discarded.
    {
        GpuBuffer<bf16> Cw((size_t)M * N, q);
        matmul_nvfp4_packed(A.data(), As.data(), M, K, Wlin, Cw.data(), ctx);
        q.wait();
    }

    // 2) CONTROL: two direct calls, diff -> non-determinism floor.
    std::vector<bf16> ref((size_t)M * N), ref2((size_t)M * N);
    {
        GpuBuffer<bf16> C0((size_t)M * N, q);
        matmul_nvfp4_packed(A.data(), As.data(), M, K, Wlin, C0.data(), ctx);
        q.wait(); C0.download(ref.data(), ref.size());
    }
    {
        GpuBuffer<bf16> C1((size_t)M * N, q);
        matmul_nvfp4_packed(A.data(), As.data(), M, K, Wlin, C1.data(), ctx);
        q.wait(); C1.download(ref2.data(), ref2.size());
    }
    DiffResult ctrl = diff_buffers(ref, ref2);
    print_diff("control direct-vs-direct", ctrl);

    // Sanity: reference must be non-constant (matmul actually ran).
    {
        double mn = 1e30, mx = -1e30; size_t nz = 0;
        for (auto v : ref) { float f = bf16_to_float(v);
            mn = std::min(mn, (double)f); mx = std::max(mx, (double)f);
            if (f != 0.0f) ++nz; }
        std::printf("  ref range=[%.4g, %.4g] nonzero=%zu/%zu\n",
                    mn, mx, nz, ref.size());
        if (mn == mx)
            std::printf("  WARNING: reference is constant -- matmul may not have run\n");
    }

    // 3) GRAPH: capture matmul inside a command_graph, finalize, replay.
    std::vector<bf16> gout((size_t)M * N);
    try {
        using Mod = sycl::ext::oneapi::experimental::command_graph<>;
        using Exe = sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>;
        Mod graph(q);
        graph.begin_recording(q);
        GpuBuffer<bf16> Cg((size_t)M * N, q);   // DST pointer baked into the graph
        matmul_nvfp4_packed(A.data(), As.data(), M, K, Wlin, Cg.data(), ctx);
        graph.end_recording(q);
        Exe exec = graph.finalize();
        q.ext_oneapi_graph(exec);
        q.wait();
        Cg.download(gout.data(), gout.size());
        DiffResult g = diff_buffers(ref, gout);
        print_diff("graph-vs-ref", g);

        // 4) REPLAY stability: zero the baked-in DST, replay the same exec.
        Cg.zero();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::vector<bf16> gout2((size_t)M * N);
        Cg.download(gout2.data(), gout2.size());
        DiffResult r2 = diff_buffers(ref, gout2);
        print_diff("replay2-vs-ref", r2);

        // 5) INPUT-MUTATION: overwrite the baked-in SRC buffer with DIFFERENT
        // data, recompute a direct reference, then replay the SAME exec.
        // If the matmul node is truly captured and re-executes, gout3 must
        // match the NEW reference. If capture silently dropped the matmul
        // (ran it once eagerly at finalize), gout3 would reflect the OLD
        // input or stay zero -- this is the decisive test against the SYCL
        // Graph docs' "scheduled immediately as part of graph finalize" caveat.
        for (auto& b : hA) b = (uint8_t)(nib(rng) | (nib(rng) << 4));
        for (auto& b : hAs) b = (uint8_t)sc(rng);
        A.upload(hA.data(), A.count());
        As.upload(hAs.data(), As.count());
        std::vector<bf16> ref_new((size_t)M * N);
        {
            GpuBuffer<bf16> Cn((size_t)M * N, q);
            matmul_nvfp4_packed(A.data(), As.data(), M, K, Wlin, Cn.data(), ctx);
            q.wait(); Cn.download(ref_new.data(), ref_new.size());
        }
        Cg.zero();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::vector<bf16> gout3((size_t)M * N);
        Cg.download(gout3.data(), gout3.size());
        DiffResult r3 = diff_buffers(ref_new, gout3);
        print_diff("replay3-vs-NEW-ref", r3);
        DiffResult r3_old = diff_buffers(ref, gout3);
        print_diff("replay3-vs-OLD-ref(stale?)", r3_old);

        bool bit_exact = (g.mismatched == 0 && r2.mismatched == 0 &&
                          r3.mismatched == 0 && ctrl.mismatched == 0);
        bool ok = bit_exact && (g.max_abs == 0.0 && r2.max_abs == 0.0 &&
                                r3.max_abs == 0.0);
        std::printf("[single] %s\n", ok ? "PASS (bit-exact capture+replay+re-exec)"
                                       : "FAIL (divergence -- see above)");
        return ok;
    } catch (const std::exception& e) {
        std::printf("[single] FAIL: exception during capture/finalize/replay: %s\n",
                    e.what());
        return false;
    }
}

// -----------------------------------------------------------------------------
// Batched matmul: matmul_nvfp4_packed_batched (the expert grouped-GEMM path).
// Weights fed per-expert concatenated -> [B,K,N] tag::acb (K-contiguous).
// -----------------------------------------------------------------------------
static bool probe_batched(GpuEngine& ctx, int B, int M, int K, int N) {
    auto& q = ctx.queue;
    int G = K / 16;
    std::printf("[batched] B=%d M=%d K=%d N=%d\n", B, M, K, N);

    std::mt19937 rng(5678);
    std::uniform_int_distribution<int> nib(0, 15), sc(40, 110);
    std::vector<uint8_t> hA((size_t)B * M * K / 2), hAs((size_t)B * M * G),
                         hW((size_t)B * K * N / 2), hWs((size_t)B * G * N);
    for (auto& b : hA)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& b : hW)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& b : hAs) b = (uint8_t)sc(rng);
    for (auto& b : hWs) b = (uint8_t)sc(rng);
    float dst_scale = 1.0f;

    GpuBuffer<uint8_t> A((size_t)B * M * K / 2, q);  A.upload(hA.data(), A.count());
    GpuBuffer<uint8_t> As((size_t)B * M * G, q);     As.upload(hAs.data(), As.count());
    GpuBuffer<uint8_t> W((size_t)B * K * N / 2, q);  W.upload(hW.data(), W.count());
    GpuBuffer<uint8_t> Ws((size_t)B * G * N, q);     Ws.upload(hWs.data(), Ws.count());
    GpuBuffer<float>   Ds(1, q);                      Ds.upload(&dst_scale, 1);

    // Warm.
    {
        GpuBuffer<bf16> Cw((size_t)B * M * N, q);
        matmul_nvfp4_packed_batched(A.data(), As.data(), B, M, K,
                                    W.data(), Ws.data(), Ds.data(), N,
                                    Cw.data(), ctx);
        q.wait();
    }

    // CONTROL.
    std::vector<bf16> ref((size_t)B * M * N), ref2((size_t)B * M * N);
    {
        GpuBuffer<bf16> C0((size_t)B * M * N, q);
        matmul_nvfp4_packed_batched(A.data(), As.data(), B, M, K,
                                    W.data(), Ws.data(), Ds.data(), N,
                                    C0.data(), ctx);
        q.wait(); C0.download(ref.data(), ref.size());
    }
    {
        GpuBuffer<bf16> C1((size_t)B * M * N, q);
        matmul_nvfp4_packed_batched(A.data(), As.data(), B, M, K,
                                    W.data(), Ws.data(), Ds.data(), N,
                                    C1.data(), ctx);
        q.wait(); C1.download(ref2.data(), ref2.size());
    }
    print_diff("control direct-vs-direct", diff_buffers(ref, ref2));
    {
        double mn = 1e30, mx = -1e30; size_t nz = 0;
        for (auto v : ref) { float f = bf16_to_float(v);
            mn = std::min(mn, (double)f); mx = std::max(mx, (double)f);
            if (f != 0.0f) ++nz; }
        std::printf("  ref range=[%.4g, %.4g] nonzero=%zu/%zu\n",
                    mn, mx, nz, ref.size());
        if (mn == mx)
            std::printf("  WARNING: reference is constant -- matmul may not have run\n");
    }

    std::vector<bf16> gout((size_t)B * M * N);
    try {
        using Mod = sycl::ext::oneapi::experimental::command_graph<>;
        using Exe = sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>;
        Mod graph(q);
        graph.begin_recording(q);
        GpuBuffer<bf16> Cg((size_t)B * M * N, q);
        matmul_nvfp4_packed_batched(A.data(), As.data(), B, M, K,
                                    W.data(), Ws.data(), Ds.data(), N,
                                    Cg.data(), ctx);
        graph.end_recording(q);
        Exe exec = graph.finalize();
        q.ext_oneapi_graph(exec);
        q.wait();
        Cg.download(gout.data(), gout.size());
        DiffResult g = diff_buffers(ref, gout);
        print_diff("graph-vs-ref", g);

        Cg.zero();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::vector<bf16> gout2((size_t)B * M * N);
        Cg.download(gout2.data(), gout2.size());
        DiffResult r2 = diff_buffers(ref, gout2);
        print_diff("replay2-vs-ref", r2);

        // 5) INPUT-MUTATION: overwrite SRC with different data, recompute a
        // direct NEW reference, replay the same exec, diff vs NEW ref.
        for (auto& b : hA)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
        for (auto& b : hAs) b = (uint8_t)sc(rng);
        A.upload(hA.data(), A.count());
        As.upload(hAs.data(), As.count());
        std::vector<bf16> ref_new((size_t)B * M * N);
        {
            GpuBuffer<bf16> Cn((size_t)B * M * N, q);
            matmul_nvfp4_packed_batched(A.data(), As.data(), B, M, K,
                                        W.data(), Ws.data(), Ds.data(), N,
                                        Cn.data(), ctx);
            q.wait(); Cn.download(ref_new.data(), ref_new.size());
        }
        Cg.zero();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::vector<bf16> gout3((size_t)B * M * N);
        Cg.download(gout3.data(), gout3.size());
        DiffResult r3 = diff_buffers(ref_new, gout3);
        print_diff("replay3-vs-NEW-ref", r3);

        bool ok = (g.mismatched == 0 && r2.mismatched == 0 && r3.mismatched == 0 &&
                   g.max_abs == 0.0 && r2.max_abs == 0.0 && r3.max_abs == 0.0);
        std::printf("[batched] %s\n", ok ? "PASS (bit-exact capture+replay+re-exec)"
                                         : "FAIL (divergence -- see above)");
        return ok;
    } catch (const std::exception& e) {
        std::printf("[batched] FAIL: exception during capture/finalize/replay: %s\n",
                    e.what());
        return false;
    }
}

// -----------------------------------------------------------------------------
// BF16 batched-strided matmul: matmul_bf16_batched_strided (the decoder
// attention score + value GEMMs in the default gqa_attention path, since
// DIFF_ONEDNN_SDPA is off by default). Same oneDNN `matmul` primitive family
// as the nvfp4 matmuls but bf16 with explicit strides -- a distinct primitive
// desc, so it gets its own capture verdict. Uses contiguous strides here
// (representative of the strided-batched primitive path the attention uses).
// -----------------------------------------------------------------------------
static bool probe_bf16_bs(GpuEngine& ctx, int B, int M, int K, int N) {
    auto& q = ctx.queue;
    std::printf("[bf16_bs] B=%d M=%d K=%d N=%d\n", B, M, K, N);

    std::mt19937 rng(9090);
    std::uniform_int_distribution<int> bv(0, 65535);
    std::vector<bf16> hA((size_t)B * M * K), hW((size_t)B * K * N);
    for (auto& b : hA) b = (bf16)bv(rng);
    for (auto& b : hW) b = (bf16)bv(rng);

    GpuBuffer<bf16> A((size_t)B * M * K, q); A.upload(hA.data(), A.count());
    GpuBuffer<bf16> W((size_t)B * K * N, q); W.upload(hW.data(), W.count());

    dnnl_dim_t as0 = (dnnl_dim_t)M * K, as1 = (dnnl_dim_t)K, as2 = 1;
    dnnl_dim_t ws0 = (dnnl_dim_t)K * N, ws1 = (dnnl_dim_t)N, ws2 = 1;
    dnnl_dim_t cs0 = (dnnl_dim_t)M * N, cs1 = (dnnl_dim_t)N, cs2 = 1;

    // Warm.
    {
        GpuBuffer<bf16> Cw((size_t)B * M * N, q);
        matmul_bf16_batched_strided(A.data(), B, M, K, as0, as1, as2,
                                    W.data(), N, ws0, ws1, ws2,
                                    Cw.data(), cs0, cs1, cs2, ctx);
        q.wait();
    }

    // CONTROL.
    std::vector<bf16> ref((size_t)B * M * N), ref2((size_t)B * M * N);
    {
        GpuBuffer<bf16> C0((size_t)B * M * N, q);
        matmul_bf16_batched_strided(A.data(), B, M, K, as0, as1, as2,
                                    W.data(), N, ws0, ws1, ws2,
                                    C0.data(), cs0, cs1, cs2, ctx);
        q.wait(); C0.download(ref.data(), ref.size());
    }
    {
        GpuBuffer<bf16> C1((size_t)B * M * N, q);
        matmul_bf16_batched_strided(A.data(), B, M, K, as0, as1, as2,
                                    W.data(), N, ws0, ws1, ws2,
                                    C1.data(), cs0, cs1, cs2, ctx);
        q.wait(); C1.download(ref2.data(), ref2.size());
    }
    print_diff("control direct-vs-direct", diff_buffers(ref, ref2));
    {
        double mn = 1e30, mx = -1e30; size_t nz = 0;
        for (auto v : ref) { float f = bf16_to_float(v);
            mn = std::min(mn, (double)f); mx = std::max(mx, (double)f);
            if (f != 0.0f) ++nz; }
        std::printf("  ref range=[%.4g, %.4g] nonzero=%zu/%zu\n",
                    mn, mx, nz, ref.size());
        if (mn == mx)
            std::printf("  WARNING: reference is constant -- matmul may not have run\n");
    }

    std::vector<bf16> gout((size_t)B * M * N);
    try {
        using Mod = sycl::ext::oneapi::experimental::command_graph<>;
        using Exe = sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>;
        Mod graph(q);
        graph.begin_recording(q);
        GpuBuffer<bf16> Cg((size_t)B * M * N, q);
        matmul_bf16_batched_strided(A.data(), B, M, K, as0, as1, as2,
                                    W.data(), N, ws0, ws1, ws2,
                                    Cg.data(), cs0, cs1, cs2, ctx);
        graph.end_recording(q);
        Exe exec = graph.finalize();
        q.ext_oneapi_graph(exec);
        q.wait();
        Cg.download(gout.data(), gout.size());
        DiffResult g = diff_buffers(ref, gout);
        print_diff("graph-vs-ref", g);

        Cg.zero();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::vector<bf16> gout2((size_t)B * M * N);
        Cg.download(gout2.data(), gout2.size());
        DiffResult r2 = diff_buffers(ref, gout2);
        print_diff("replay2-vs-ref", r2);

        // 5) INPUT-MUTATION: overwrite SRC with different data, recompute a
        // direct NEW reference, replay the same exec, diff vs NEW ref.
        for (auto& b : hA) b = (bf16)bv(rng);
        A.upload(hA.data(), A.count());
        std::vector<bf16> ref_new((size_t)B * M * N);
        {
            GpuBuffer<bf16> Cn((size_t)B * M * N, q);
            matmul_bf16_batched_strided(A.data(), B, M, K, as0, as1, as2,
                                        W.data(), N, ws0, ws1, ws2,
                                        Cn.data(), cs0, cs1, cs2, ctx);
            q.wait(); Cn.download(ref_new.data(), ref_new.size());
        }
        Cg.zero();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::vector<bf16> gout3((size_t)B * M * N);
        Cg.download(gout3.data(), gout3.size());
        DiffResult r3 = diff_buffers(ref_new, gout3);
        print_diff("replay3-vs-NEW-ref", r3);

        bool ok = (g.mismatched == 0 && r2.mismatched == 0 && r3.mismatched == 0 &&
                   g.max_abs == 0.0 && r2.max_abs == 0.0 && r3.max_abs == 0.0);
        std::printf("[bf16_bs] %s\n", ok ? "PASS (bit-exact capture+replay+re-exec)"
                                        : "FAIL (divergence -- see above)");
        return ok;
    } catch (const std::exception& e) {
        std::printf("[bf16_bs] FAIL: exception during capture/finalize/replay: %s\n",
                    e.what());
        return false;
    }
}

// -----------------------------------------------------------------------------
// Session-backstop test: validates the queue-keyed active-session registry in
// nvfp4.hpp. matmul_nvfp4() packs its bf16 input via pack_bf16_to_nvfp4(), which
// calls nvfp4_sycl_graph_submit() with session=nullptr. Without the registry
// backstop, calling this inside an active Nvfp4GraphSession recording would
// trigger a nested begin_recording() -> throw (graph_impl.cpp:2238) and poison
// the per-kernel cache. With the backstop, the pack stands down to a bare
// submit() that the session captures, the oneDNN matmul auto-captures, and the
// whole step finalizes + replays bit-exactly with ZERO new fallbacks.
// Requires DIFF_NVFP4_SYCL_GRAPH=1 (set in main before any nvfp4 call).
// -----------------------------------------------------------------------------
static bool probe_session_backstop(GpuEngine& ctx, int M, int K, int N) {
    auto& q = ctx.queue;
    int G = K / 16;
    std::printf("[session-backstop] M=%d K=%d N=%d (matmul_nvfp4 pack path in a session)\n",
                M, K, N);

    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> nib(0, 15), sc(40, 110);
    std::vector<uint8_t> hW((size_t)K * N / 2), hWs((size_t)G * N);
    for (auto& b : hW)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& b : hWs) b = (uint8_t)sc(rng);
    float dst_scale = 1.0f;
    std::vector<bf16> hA((size_t)M * K);
    for (auto& b : hA) b = (bf16)(0x3C00 + (rng() & 0x3FF));  // ~1..2 bf16

    GpuBuffer<uint8_t> W((size_t)K * N / 2, q);   W.upload(hW.data(), W.count());
    GpuBuffer<uint8_t> Ws((size_t)G * N, q);     Ws.upload(hWs.data(), Ws.count());
    GpuBuffer<float>   Ds(1, q);                  Ds.upload(&dst_scale, 1);
    GpuBuffer<bf16>    A((size_t)M * K, q);       A.upload(hA.data(), A.count());

    Nvfp4Linear Wlin;
    Wlin.in_features = K; Wlin.out_features = N; Wlin.input_global_scale = 1.0f;
    Wlin.weight_packed = std::move(W);
    Wlin.weight_scale  = std::move(Ws);
    Wlin.dst_scale     = std::move(Ds);

    // Direct reference (no graph, no session).
    std::vector<bf16> ref((size_t)M * N);
    {
        GpuBuffer<bf16> C0((size_t)M * N, q);
        matmul_nvfp4(A.data(), M, K, Wlin, C0.data(), ctx);
        q.wait(); C0.download(ref.data(), ref.size());
    }

    size_t fb_before = nvfp4_sycl_graph_cache().fallbacks;
    size_t cap_before = nvfp4_sycl_graph_cache().captures;

    std::vector<bf16> gout((size_t)M * N);
    try {
        Nvfp4GraphSession session;
        Nvfp4SyclGraphKey key = nvfp4_step_key(q, /*layer_index=*/0,
                                               /*is_sliding=*/true, /*denoise_step=*/0);
        if (!session.begin(q, key)) {
            std::printf("[session-backstop] FAIL: begin() reported cache hit on a "
                        "fresh key (unexpected)\n");
            return false;
        }
        if (!session.active()) {
            std::printf("[session-backstop] FAIL: session not recording after begin()\n");
            return false;
        }
        // This call internally pack_bf16_to_nvfp4 -> nvfp4_sycl_graph_submit(session=nullptr).
        // The registry backstop must stand it down; a nested begin_recording would throw.
        GpuBuffer<bf16> Cg((size_t)M * N, q);
        matmul_nvfp4(A.data(), M, K, Wlin, Cg.data(), ctx);
        session.end_and_replay();
        q.wait();
        Cg.download(gout.data(), gout.size());

        size_t fb_after = nvfp4_sycl_graph_cache().fallbacks;
        size_t cap_after = nvfp4_sycl_graph_cache().captures;
        DiffResult g = diff_buffers(ref, gout);
        print_diff("session-graph-vs-ref", g);
        std::printf("  per-kernel cache: captures +=%zu fallbacks +=%zu "
                    "(fallbacks must be 0 -- no poisoning)\n",
                    cap_after - cap_before, fb_after - fb_before);

        // Replay the cached session graph (cache hit) and re-verify.
        Nvfp4GraphSession s2;
        bool need_rec = s2.begin(q, key);
        if (need_rec) {
            std::printf("[session-backstop] FAIL: second begin() should have hit the "
                        "cached session graph\n");
            return false;
        }
        q.wait();
        // The cached session graph wrote to Cg's baked-in pointer on replay.
        // Re-download (Cg still holds the replay result).
        std::vector<bf16> gout2((size_t)M * N);
        Cg.download(gout2.data(), gout2.size());
        DiffResult r2 = diff_buffers(ref, gout2);
        print_diff("session-replay-vs-ref", r2);

        bool ok = (g.mismatched == 0 && r2.mismatched == 0 &&
                   (fb_after - fb_before) == 0 &&
                   g.max_abs == 0.0 && r2.max_abs == 0.0);
        std::printf("[session-backstop] %s\n",
                    ok ? "PASS (registry backstop: no throw, no poisoning, bit-exact)"
                       : "FAIL (see above)");
        return ok;
    } catch (const std::exception& e) {
        std::printf("[session-backstop] FAIL: exception: %s\n", e.what());
        return false;
    }
}

int main() {
    // The session-backstop test needs the SYCL-graph path enabled so that
    // Nvfp4GraphSession actually records (nvfp4_sycl_graph_enabled() == true).
    // Set it before any nvfp4 call so the static flag caches the right value.
    // This does not affect the oneDNN-only probes above (they use explicit
    // command_graph objects, not nvfp4_sycl_graph_submit).
    if (!std::getenv("DIFF_NVFP4_SYCL_GRAPH"))
        setenv("DIFF_NVFP4_SYCL_GRAPH", "1", 1);
    GpuEngine& ctx = GpuEngine::get(0);
    std::printf("=== Phase 0: oneDNN matmul SYCL graph-capture probe ===\n");
    std::printf("oneDNN device: %s\n",
                dnnl::sycl_interop::get_device(ctx.engine)
                    .get_info<sycl::info::device::name>().c_str());

    int M = env_int("GC_M", 16), K = env_int("GC_K", 64), N = env_int("GC_N", 64);
    int B = env_int("GC_B", 4);
    if (K % 16 != 0) { std::fprintf(stderr, "K must be divisible by 16\n"); return 2; }

    bool single_ok = true, batched_ok = true, bf16_ok = true;
    if (!env_skip("GC_SKIP_SINGLE"))
        single_ok = probe_single(ctx, M, K, N);
    if (!env_skip("GC_SKIP_BATCHED"))
        batched_ok = probe_batched(ctx, B, M, K, N);
    if (!env_skip("GC_SKIP_BF16")) {
        int bM = env_int("GC_BF_M", 512), bK = env_int("GC_BF_K", 256),
            bN = env_int("GC_BF_N", 512), bB = env_int("GC_BF_B", 8);
        bf16_ok = probe_bf16_bs(ctx, bB, bM, bK, bN);
    }

    bool backstop_ok = true;
    if (!env_skip("GC_SKIP_BACKSTOP")) {
        int sM = env_int("GC_SB_M", 64), sK = env_int("GC_SB_K", 256),
            sN = env_int("GC_SB_N", 128);
        backstop_ok = probe_session_backstop(ctx, sM, sK, sN);
    }

    std::printf("\n=== PHASE 0 VERDICT ===\n");
    std::printf("  single  matmul_nvfp4_packed        : %s\n",
                single_ok ? "CAPTURE-SAFE" : "NOT capture-safe");
    std::printf("  batched matmul_nvfp4_packed_batched: %s\n",
                batched_ok ? "CAPTURE-SAFE" : "NOT capture-safe");
    std::printf("  bf16-bs matmul_bf16_batched_strided: %s\n",
                bf16_ok ? "CAPTURE-SAFE" : "NOT capture-safe");
    std::printf("  session-backstop (registry)        : %s\n",
                backstop_ok ? "OK (no nested-throw / no poisoning)"
                            : "FAIL");
    bool pass = single_ok && batched_ok && bf16_ok && backstop_ok;
    std::printf("  => %s: oneDNN matmul %s live inside a SYCL command_graph session.\n",
                pass ? "PASS" : "FAIL",
                pass ? "CAN" : "CANNOT");
    return pass ? 0 : 1;
}

// =============================================================================
// Probe — Is the xe2 DPAS grouped-rows NVFP4 kernel
// (matmul_nvfp4_grouped_rows_xe2, __spirv_SubgroupMatrixMultiplyAccumulateINTEL +
// local_accessor + nd_range + reqd_sub_group_size(16)) capture-safe inside a
// SYCL command_graph on BMG G31?
//
// Phase 0 verified oneDNN USM matmuls + the scalar custom kernel capture+replay
// bit-exact. The xe2 DPAS kernel was NEVER captured in a graph before — it is a
// raw q.submit, not wrapped in nvfp4_sycl_graph_submit. The whole-step session
// capture (B5) forces this kernel and produced JUNK output; this probe isolates
// whether the DPAS kernel itself is the capture failure.
//
// Method (mirrors Phase 0 probe_single):
//   1. WARM: call the kernel once directly (warms the dequant LUT + coalesced
//      weight cache + JIT). Discarded.
//   2. CONTROL: two direct calls, diff -> determinism floor.
//   3. GRAPH: command_graph(begin_recording) -> kernel -> end -> finalize ->
//      replay. Diff vs direct reference.
//   4. REPLAY2: zero DST, replay same exec, diff vs reference (replay reproduces
//      at the baked-in USM address).
//   5. REPLAY3 (DECISIVE): overwrite SRC (A_packed/A_scale) with DIFFERENT data,
//      recompute a NEW direct reference, replay the SAME exec, diff vs NEW ref
//      AND vs OLD ref. If output == NEW ref -> node RE-EXECUTES (capture OK).
//      If output == OLD ref -> silently cached/eager-once (capture FAIL).
//      If output is garbage -> something else is broken (possibly the kernel
//      itself, independent of graphs).
//
// Verdict:
//   PASS iff no exception + graph/replay2/replay3-vs-NEW-ref all bit-exact
//        (within control floor) + replay3-vs-OLD-ref mismatches ~100%.
//   FAIL iff any exception, OR silent-drop (replay3 == OLD ref), OR divergence.
//
// The scalar sibling (matmul_nvfp4_grouped_rows_custom) is also probed as a
// control: if it captures fine but xe2 doesn't, the DPAS intrinsics/local_accessor
// are the capture blocker. If BOTH fail, something about the grouped-rows shape
// or the pointer-table indirection breaks capture.
//
// Env knobs (AB-testable, per AGENTS.md):
//   XE2_M / XE2_K / XE2_N / XE2_E   dims (defaults 64 / 256 / 128 / 4)
//     M=max_rows, K=hidden, N=out, E=num_experts. K,N must be %16==0.
//   XE2_SKIP_XE2=1 / XE2_SKIP_SCALAR=1  skip a sub-probe
//   XE2_SEED                       RNG seed (default 1234)
// =============================================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph/command_graph.hpp>

#include "common/gpu/buffer.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/nvfp4.hpp"

#include <algorithm>
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
    std::printf("  %-32s max_abs=%.6g mismatched=%zu/%zu (%.4f%%)\n",
                tag, r.max_abs, r.mismatched, r.n,
                r.n ? 100.0 * r.mismatched / r.n : 0.0);
}

// Sanity: reference must be non-constant (kernel actually ran).
static bool ref_is_nontrivial(const std::vector<bf16>& ref) {
    double mn = 1e30, mx = -1e30; size_t nz = 0;
    for (auto v : ref) { float f = bf16_to_float(v);
        mn = std::min(mn, (double)f); mx = std::max(mx, (double)f);
        if (f != 0.0f) ++nz; }
    std::printf("  ref range=[%.4g, %.4g] nonzero=%zu/%zu\n",
                mn, mx, nz, ref.size());
    if (mn == mx) {
        std::printf("  WARNING: reference is constant -- kernel may not have run\n");
        return false;
    }
    return true;
}

// Build per-expert NVFP4 weights + coalesced pointer tables for the grouped
// kernel. Returns device pointer arrays the grouped kernels read. Mirrors what
// run_shard_nvfp4_gpu_layout does, but with synthetic weights.
struct GroupedWeights {
    int E, K, N;
    std::vector<Nvfp4Linear> linears;       // per-expert packed weights
    GpuBuffer<const uint8_t*> w_raw_dev;    // RAW packed weight ptr table (scalar kernel)
    GpuBuffer<const uint8_t*> w_coal_dev;  // coalesced weight ptr table (xe2 kernel)
    GpuBuffer<const uint8_t*> w_scale_dev; // weight scale ptr table (shared)
    GpuBuffer<const float*>   dst_scale_dev; // dst scale ptr table (shared)

    GroupedWeights(GpuEngine& ctx, int E_, int K_, int N_, std::mt19937& rng)
        : E(E_), K(K_), N(N_),
          w_raw_dev((size_t)E, ctx.queue),
          w_coal_dev((size_t)E, ctx.queue),
          w_scale_dev((size_t)E, ctx.queue),
          dst_scale_dev((size_t)E, ctx.queue),
          linears(E) {
        int G = K / 16;
        std::uniform_int_distribution<int> nib(0, 15), sc(10, 40);
        std::vector<const uint8_t*> w_raw(E), w_coal(E), w_scale(E);
        std::vector<const float*> dst_scale(E);
        for (int e = 0; e < E; ++e) {
            std::vector<uint8_t> hW((size_t)K * N / 2), hWs((size_t)G * N);
            for (auto& b : hW)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
            for (auto& b : hWs) b = (uint8_t)sc(rng);
            GpuBuffer<uint8_t> W((size_t)K * N / 2, ctx.queue);
            W.upload(hW.data(), W.count());
            GpuBuffer<uint8_t> Ws((size_t)G * N, ctx.queue);
            Ws.upload(hWs.data(), Ws.count());
            float dst_sc = 1.0f;
            GpuBuffer<float> Ds(1, ctx.queue);
            Ds.upload(&dst_sc, 1);

            linears[e].in_features = K;
            linears[e].out_features = N;
            linears[e].input_global_scale = 1.0f;
            linears[e].weight_packed = std::move(W);
            linears[e].weight_scale  = std::move(Ws);
            linears[e].dst_scale     = std::move(Ds);

             w_raw[e]  = linears[e].weight_packed.data();
             // Pre-warm coalesced weight (caches on the linear).
             w_coal[e] = nvfp4_coalesced_weight(linears[e], K, N, ctx);
             // TLB flush: after a large alloc/free cycle, the GPU's TLB may
             // still map the coalesced weight's address to a stale (freed)
             // buffer. A device→host memcpy of the full buffer forces the
             // driver to establish the correct page mapping.
             if (std::getenv("XE2_COAL_TLB_FLUSH")) {
                 std::vector<uint8_t> tmp((size_t)N * K / 2);
                 ctx.queue.memcpy(tmp.data(), w_coal[e], (size_t)N * K / 2).wait();
             }
             w_scale[e] = linears[e].weight_scale.data();
             dst_scale[e] = linears[e].dst_scale.data();
        }
        ctx.queue.wait();
        w_raw_dev.upload(w_raw.data(), E);
        w_coal_dev.upload(w_coal.data(), E);
        w_scale_dev.upload(w_scale.data(), E);
        dst_scale_dev.upload(dst_scale.data(), E);
        ctx.queue.wait();
    }
};

// Build synthetic A_packed/A_scale + row_expert assignment.
struct GroupedInput {
    int M, K, E;
    GpuBuffer<uint8_t> A_packed;   // [M, K/2] f4_e2m1
    GpuBuffer<uint8_t> A_scale;    // [M, K/16] f8_e4m3
    GpuBuffer<int32_t> row_expert; // [M]

    GroupedInput(GpuEngine& ctx, int M_, int K_, int E_, std::mt19937& rng)
        : M(M_), K(K_), E(E_),
          A_packed((size_t)M * K / 2, ctx.queue),
          A_scale((size_t)M * (K / 16), ctx.queue),
          row_expert((size_t)M, ctx.queue) {
        std::uniform_int_distribution<int> nib(0, 15), sc(40, 110), ex(0, E - 1);
        std::vector<uint8_t> hA((size_t)M * K / 2), hAs((size_t)M * (K / 16));
        std::vector<int32_t> hre((size_t)M);
        for (auto& b : hA)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
        for (auto& b : hAs) b = (uint8_t)sc(rng);
        for (auto& v : hre) v = ex(rng);
        A_packed.upload(hA.data(), A_packed.count());
        A_scale.upload(hAs.data(), A_scale.count());
        row_expert.upload(hre.data(), row_expert.count());
    }

    // Overwrite A with new random data (for replay3 input-mutation test).
    void reseed(GpuEngine& ctx, std::mt19937& rng) {
        std::uniform_int_distribution<int> nib(0, 15), sc(10, 40);
        std::vector<uint8_t> hA((size_t)M * K / 2), hAs((size_t)M * (K / 16));
        for (auto& b : hA)  b = (uint8_t)(nib(rng) | (nib(rng) << 4));
        for (auto& b : hAs) b = (uint8_t)sc(rng);
        A_packed.upload(hA.data(), A_packed.count());
        A_scale.upload(hAs.data(), A_scale.count());
        ctx.queue.wait();
    }
};

// =============================================================================
// Sub-probe: test one grouped-rows kernel (xe2 or scalar) for capture safety.
// =============================================================================
// `kernel_kind`: 0 = xe2 DPAS (matmul_nvfp4_grouped_rows_xe2),
//                1 = scalar  (matmul_nvfp4_grouped_rows_custom).
static bool probe_grouped_kernel(GpuEngine& ctx, const char* tag,
                                 int kernel_kind, int M, int K, int N, int E,
                                 unsigned seed) {
    auto& q = ctx.queue;
    std::printf("[%s] M=%d K=%d N=%d E=%d (seed=%u)\n", tag, M, K, N, E, seed);
    if (K % 16 != 0 || N % 16 != 0) {
        std::printf("[%s] FAIL: K and N must be divisible by 16\n", tag);
        return false;
    }

    std::mt19937 rng(seed);
    std::printf("[%s] building weights + coalesced tables...\n", tag);
    GroupedWeights gw(ctx, E, K, N, rng);
    GroupedInput gi(ctx, M, K, E, rng);
    std::printf("[%s] warming dequant LUT + kernel cache...\n", tag);
    (void)nvfp4_dequant_lut(ctx);

    // Lambda to run the chosen kernel into a fresh DST buffer.
    // Scalar reads RAW packed weights; xe2 reads COALESCED weights. The two
    // kernels interpret the weight buffer with different layouts, so each must
    // get the matching pointer table.
    auto run_kernel = [&](bf16* C) {
        if (kernel_kind == 0) {
            matmul_nvfp4_grouped_rows_xe2(ctx, gi.A_packed.data(),
                gi.A_scale.data(), K, gi.row_expert.data(), M,
                gw.w_coal_dev.data(), gw.w_scale_dev.data(), gw.dst_scale_dev.data(),
                N, C);
        } else {
            matmul_nvfp4_grouped_rows_custom(q, gi.A_packed.data(),
                gi.A_scale.data(), K, gi.row_expert.data(), M,
                gw.w_raw_dev.data(), gw.w_scale_dev.data(), gw.dst_scale_dev.data(),
                N, C);
        }
    };

    // 1) WARM: one direct call (discarded).
    {
        GpuBuffer<bf16> Cw((size_t)M * N, q);
        run_kernel(Cw.data());
        q.wait();
    }

    // 2) CONTROL: two direct calls -> determinism floor.
    std::vector<bf16> ref((size_t)M * N), ref2((size_t)M * N);
    {
        GpuBuffer<bf16> C0((size_t)M * N, q);
        run_kernel(C0.data());
        q.wait(); C0.download(ref.data(), ref.size());
    }
    {
        GpuBuffer<bf16> C1((size_t)M * N, q);
        run_kernel(C1.data());
        q.wait(); C1.download(ref2.data(), ref2.size());
    }
    DiffResult ctrl = diff_buffers(ref, ref2);
    print_diff("control direct-vs-direct", ctrl);
    if (!ref_is_nontrivial(ref)) {
        std::printf("[%s] FAIL: reference is trivial (kernel did not produce output)\n", tag);
        return false;
    }

    // 3) GRAPH: capture the kernel inside a command_graph, finalize, replay.
    GpuBuffer<bf16> Cg((size_t)M * N, q);  // DST pointer baked into the graph
    std::vector<bf16> gout((size_t)M * N);
    bool ok = true;
    try {
        using Mod = sycl::ext::oneapi::experimental::command_graph<>;
        using Exe = sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>;
        Mod graph(q, sycl::property_list{
            sycl::ext::oneapi::experimental::property::graph::assume_buffer_outlives_graph{}});
        graph.begin_recording(q);
        run_kernel(Cg.data());
        graph.end_recording(q);
        Exe exec = graph.finalize();
        q.ext_oneapi_graph(exec);
        q.wait();
        Cg.download(gout.data(), gout.size());
        DiffResult g = diff_buffers(ref, gout);
        print_diff("graph-vs-ref", g);
        ok = ok && (g.mismatched == 0);

        // 4) REPLAY2: zero DST, replay same exec, diff vs ref.
        Cg.zero();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::vector<bf16> gout2((size_t)M * N);
        Cg.download(gout2.data(), gout2.size());
        DiffResult r2 = diff_buffers(ref, gout2);
        print_diff("replay2-vs-ref", r2);
        ok = ok && (r2.mismatched == 0);

        // 5) REPLAY3 (DECISIVE): mutate SRC, recompute NEW direct ref, replay
        //    same exec, diff vs NEW ref AND vs OLD ref.
        std::mt19937 rng2(seed + 777);
        gi.reseed(ctx, rng2);
        std::vector<bf16> newref((size_t)M * N);
        {
            GpuBuffer<bf16> Cn((size_t)M * N, q);
            run_kernel(Cn.data());
            q.wait(); Cn.download(newref.data(), newref.size());
        }
        if (!ref_is_nontrivial(newref)) {
            std::printf("  WARNING: NEW ref is trivial -- mutation did not change output?\n");
        }
        // Diff OLD ref vs NEW ref: must differ (~100%) to prove the mutation
        // actually changed the computation.
        DiffResult old_vs_new = diff_buffers(ref, newref);
        print_diff("old-ref-vs-new-ref", old_vs_new);

        Cg.zero();
        q.ext_oneapi_graph(exec);
        q.wait();
        std::vector<bf16> gout3((size_t)M * N);
        Cg.download(gout3.data(), gout3.size());
        DiffResult r3_new = diff_buffers(newref, gout3);
        print_diff("replay3-vs-NEW-ref", r3_new);
        DiffResult r3_old = diff_buffers(ref, gout3);
        print_diff("replay3-vs-OLD-ref(stale?)", r3_old);

        // PASS requires: replay3 matches NEW ref (node re-executes), NOT OLD ref.
        // The mutation must have actually changed the output (old-vs-new differs)
        // and replay3 must follow the NEW data (not the stale OLD ref). We do not
        // require 100% mismatch on replay3-vs-OLD: rows that land on the same
        // expert under both seeds produce the same result, so a few elements can
        // legitimately match. The decisive signal is replay3-vs-NEW-ref==0 AND
        // old-vs-new != 0 (mutation was meaningful) AND replay3-vs-OLD != 0
        // (not pure stale-once).
        bool re_executes = (r3_new.mismatched == 0) &&
                           (old_vs_new.mismatched > 0) &&
                           (r3_old.mismatched > 0);
        if (!re_executes) {
            if (r3_old.mismatched == 0 && old_vs_new.mismatched > 0)
                std::printf("  => SILENT-DROP: replay3 == OLD ref (node NOT re-executing)\n");
            else if (old_vs_new.mismatched == 0)
                std::printf("  => INCONCLUSIVE: mutation did not change the output\n");
            else
                std::printf("  => DIVERGENCE: replay3 matches NEW ref but not bit-exact\n");
        }
        ok = ok && re_executes;
    } catch (const std::exception& e) {
        std::printf("[%s] FAIL: exception during capture/replay: %s\n", tag, e.what());
        return false;
    }

    std::printf("[%s] %s\n", tag, ok ? "PASS (capture+replay+re-exec bit-exact)"
                                    : "FAIL (see above)");
    return ok;
}

int main() {
    int M  = env_int("XE2_M", 64);
    int K  = env_int("XE2_K", 256);
    int N  = env_int("XE2_N", 128);
    int E  = env_int("XE2_E", 4);
    unsigned seed = (unsigned)env_int("XE2_SEED", 1234);
    if (K % 16 != 0 || N % 16 != 0) {
        std::fprintf(stderr, "XE2_K and XE2_N must be divisible by 16\n");
        return 2;
    }
    if (M % 8 != 0) {
        std::printf("[main] rounding M=%d up to multiple of 8 (xe2 m-tiles)\n", M);
        M = ((M + 7) / 8) * 8;
    }

    auto& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    std::printf("[main] device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("[main] M=%d K=%d N=%d E=%d seed=%u\n", M, K, N, E, seed);

    // XE2_POLLUTE controls what runs before the xe2 sub-probe to isolate the
    // state-pollution root cause:
    //   none       (default): scalar sub-probe first (the original failing order)
    //   xe2-twice  : xe2 sub-probe, then xe2 again (no scalar)
    //   weights    : create+destroy scalar GroupedWeights (no kernel), then xe2
    //   kernel     : run scalar kernel with its own weights, then xe2
    //   coal-only  : create+destroy only coalesced weights (no packed, no kernel)
    std::string pollute = std::getenv("XE2_POLLUTE") ? std::getenv("XE2_POLLUTE") : "none";
    std::printf("[main] pollute mode: %s\n", pollute.c_str());

    bool all_ok = true;

     if (pollute == "xe2-twice-sync") {
         std::printf("\n=== XE2 DPAS run 1 ===\n");
         all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
         q.wait();
         {
             GpuBuffer<uint8_t> dummy(1, q);
             uint8_t z = 0;
             q.memcpy(dummy.data(), &z, 1).wait();
             std::printf("[sync] dummy memcpy+wait between runs\n");
         }
         q.wait();
         std::printf("\n=== XE2 DPAS run 2 (after sync memcpy) ===\n");
         all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
     } else if (pollute == "xe2-twice") {
         std::printf("\n=== XE2 DPAS run 1 ===\n");
         all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
         q.wait();
         std::printf("\n=== XE2 DPAS run 2 (after xe2 weight recycling) ===\n");
         all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
     } else if (pollute == "xe2-keepalive") {
        // Run xe2 but keep the first run's GroupedWeights alive during the
        // second run. If this PASSES, the issue is GPU memory recycling (free
        // + realloc at the same address). If it still FAILS, it's something else.
        std::printf("\n=== XE2 DPAS run 1 (keepalive) ===\n");
        std::mt19937 ka_rng(seed);
        auto ka_gw = std::make_unique<GroupedWeights>(ctx, E, K, N, ka_rng);
        auto ka_gi = std::make_unique<GroupedInput>(ctx, M, K, E, ka_rng);
        (void)nvfp4_dequant_lut(ctx);
        // Just run the kernel directly (no graph) to confirm correctness.
        {
            GpuBuffer<bf16> C((size_t)M * N, q);
            matmul_nvfp4_grouped_rows_xe2(ctx, ka_gi->A_packed.data(),
                ka_gi->A_scale.data(), K, ka_gi->row_expert.data(), M,
                ka_gw->w_coal_dev.data(), ka_gw->w_scale_dev.data(),
                ka_gw->dst_scale_dev.data(), N, C.data());
            q.wait();
            std::vector<bf16> out((size_t)M * N);
            C.download(out.data(), out.size());
            double mn = 1e30, mx = -1e30;
            for (auto v : out) { float f = bf16_to_float(v); mn = std::min(mn,(double)f); mx = std::max(mx,(double)f); }
            std::printf("  run1 direct ref range=[%.4g, %.4g]\n", mn, mx);
        }
        // Keep ka_gw/ka_gi alive, run second xe2 with fresh weights.
        std::printf("\n=== XE2 DPAS run 2 (first weights STILL ALIVE) ===\n");
        all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
    } else if (pollute == "xe2-verify-weights") {
        // After a weight alloc/free cycle, verify whether the coalesced weight
        // DATA itself is correct (not just whether the kernel produces correct
        // output). Create+destroy weights, then create new weights and download
        // the coalesced data to check against the raw packed data.
        std::printf("\n=== POLLUTE: create+destroy weights ===\n");
        { std::mt19937 prng(seed); GroupedWeights pw(ctx, E, K, N, prng); q.wait(); }
        q.wait();
        std::printf("\n=== VERIFY: create new weights, download coalesced data ===\n");
        std::mt19937 prng2(seed);
        GroupedWeights gw2(ctx, E, K, N, prng2);
        q.wait();
        // Download expert 0's coalesced weight and raw packed weight.
        int halfK = K / 2;
        int ktiles = K / 16;
        std::vector<uint8_t> coal_host((size_t)N * halfK), raw_host((size_t)K * N / 2);
        q.memcpy(coal_host.data(), gw2.linears[0].weight_coal.data(), (size_t)N * halfK).wait();
        q.memcpy(raw_host.data(), gw2.linears[0].weight_packed.data(), (size_t)K * N / 2).wait();
        // Verify the reorder: dst[(n/16)*ktiles*128 + kt*128 + (n%16)*8 + j] == src[n*halfK + b]
        int mismatches = 0, checked = 0;
        for (int n = 0; n < N; ++n) {
            for (int b = 0; b < halfK; ++b) {
                int kt = b / 8, j = b % 8;
                size_t coal_idx = (size_t)(n / 16) * ktiles * 128 + (size_t)kt * 128 + (n % 16) * 8 + j;
                size_t raw_idx = (size_t)n * halfK + b;
                ++checked;
                if (coal_host[coal_idx] != raw_host[raw_idx]) ++mismatches;
            }
        }
        std::printf("  coalesced weight verification: %d/%d mismatched (%.2f%%)\n",
                    mismatches, checked, 100.0 * mismatches / checked);
        if (mismatches > 0)
            std::printf("  => COALESCED WEIGHT DATA IS WRONG (reorder kernel failed on recycled memory)\n");
        // Now run the xe2 kernel with these weights.
        std::printf("\n=== XE2 DPAS (after verified weights) ===\n");
        all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
    } else if (pollute == "xe2-direct-after-recycle") {
        // After a weight alloc/free cycle, run the xe2 kernel DIRECTLY (no
        // graph capture). If this fails, the issue is at the kernel level
        // (DPAS + recycled memory), not graph capture.
        std::printf("\n=== POLLUTE: create+destroy weights ===\n");
        { std::mt19937 prng(seed); GroupedWeights pw(ctx, E, K, N, prng); q.wait(); }
        q.wait();
        std::printf("\n=== XE2 direct (no graph, after recycling) ===\n");
        std::mt19937 prng2(seed);
        GroupedWeights gw(ctx, E, K, N, prng2);
        GroupedInput gi(ctx, M, K, E, prng2);
        (void)nvfp4_dequant_lut(ctx);
        // Warm
        { GpuBuffer<bf16> Cw((size_t)M * N, q); matmul_nvfp4_grouped_rows_xe2(ctx,
            gi.A_packed.data(), gi.A_scale.data(), K, gi.row_expert.data(), M,
            gw.w_coal_dev.data(), gw.w_scale_dev.data(), gw.dst_scale_dev.data(),
            N, Cw.data()); q.wait(); }
        // Direct ref
        std::vector<bf16> ref((size_t)M * N);
        { GpuBuffer<bf16> C((size_t)M * N, q); matmul_nvfp4_grouped_rows_xe2(ctx,
            gi.A_packed.data(), gi.A_scale.data(), K, gi.row_expert.data(), M,
            gw.w_coal_dev.data(), gw.w_scale_dev.data(), gw.dst_scale_dev.data(),
            N, C.data()); q.wait(); C.download(ref.data(), ref.size()); }
        ref_is_nontrivial(ref);
        // Also run scalar with same weights for comparison
        std::vector<bf16> sref((size_t)M * N);
        { GpuBuffer<bf16> C((size_t)M * N, q); matmul_nvfp4_grouped_rows_custom(q,
            gi.A_packed.data(), gi.A_scale.data(), K, gi.row_expert.data(), M,
            gw.w_raw_dev.data(), gw.w_scale_dev.data(), gw.dst_scale_dev.data(),
            N, C.data()); q.wait(); C.download(sref.data(), sref.size()); }
        DiffResult x = diff_buffers(sref, ref);
        print_diff("xe2-vs-scalar (after recycle)", x);
        all_ok = (x.mismatched == 0);
        std::printf("[xe2-direct] %s\n", all_ok ? "PASS" : "FAIL");
    } else if (pollute == "weights-twice") {
        // Create+destroy weights TWICE, then run xe2. If this PASSES and
        // "weights" FAILS, the extra construction cycle is the fix (not the
        // memcpy in verify-weights).
        std::printf("\n=== POLLUTE: create+destroy weights (round 1) ===\n");
        { std::mt19937 prng(seed); GroupedWeights pw(ctx, E, K, N, prng); q.wait(); }
        q.wait();
        std::printf("\n=== POLLUTE: create+destroy weights (round 2) ===\n");
        { std::mt19937 prng2(seed); GroupedWeights pw2(ctx, E, K, N, prng2); q.wait(); }
        q.wait();
        std::printf("\n=== XE2 DPAS (after double weight cycling) ===\n");
        all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
    } else if (pollute == "weights" || pollute == "coal-only") {
        // Create+destroy scalar GroupedWeights (no kernel execution) to test if
        // weight alloc/free alone pollutes the xe2 kernel.
        std::printf("\n=== POLLUTE: creating+destroying %s weights (no kernel) ===\n",
                    pollute == "coal-only" ? "coalesced" : "scalar");
        {
            std::mt19937 prng(seed);
            GroupedWeights pw(ctx, E, K, N, prng);
            if (pollute == "coal-only") {
                // Only keep the coalesced weights; skip raw packed by not touching
                // w_raw_dev (still allocated, just not used by any kernel).
            }
            q.wait();
            std::printf("[pollute] weights built, now destroying...\n");
        }
        q.wait();
        std::printf("\n=== XE2 DPAS (after weight-only pollution) ===\n");
        all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
    } else if (pollute == "kernel") {
        // Run the scalar kernel with its own weights+input, then xe2 with fresh
        // weights. Tests if scalar kernel *execution* (not weight alloc) pollutes.
        std::printf("\n=== POLLUTE: scalar kernel warmup only ===\n");
        {
            std::mt19937 prng(seed);
            GroupedWeights pw(ctx, E, K, N, prng);
            GroupedInput pi(ctx, M, K, E, prng);
            (void)nvfp4_dequant_lut(ctx);
            GpuBuffer<bf16> Cw((size_t)M * N, q);
            matmul_nvfp4_grouped_rows_custom(q, pi.A_packed.data(),
                pi.A_scale.data(), K, pi.row_expert.data(), M,
                pw.w_raw_dev.data(), pw.w_scale_dev.data(), pw.dst_scale_dev.data(),
                N, Cw.data());
            q.wait();
            std::printf("[pollute] scalar kernel ran, now destroying weights...\n");
        }
        q.wait();
        std::printf("\n=== XE2 DPAS (after scalar kernel pollution) ===\n");
        all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
    } else {
        // Default: scalar first, then xe2 (the original failing order).
        if (!env_skip("XE2_SKIP_SCALAR")) {
            std::printf("\n=== SCALAR control: matmul_nvfp4_grouped_rows_custom ===\n");
            all_ok = probe_grouped_kernel(ctx, "scalar", 1, M, K, N, E, seed) && all_ok;
        }
        if (!env_skip("XE2_SKIP_XE2")) {
            std::printf("\n=== XE2 DPAS: matmul_nvfp4_grouped_rows_xe2 ===\n");
            all_ok = probe_grouped_kernel(ctx, "xe2", 0, M, K, N, E, seed) && all_ok;
        }
    }

    std::printf("\n==== OVERALL: %s ====\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

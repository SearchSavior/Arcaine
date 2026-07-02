// src/nvfp4_batched_probe.cpp
//
// Phase 0, Probe 1 — does oneDNN accept a BATCHED f4_e2m1 matmul with per-batch
// f8 weight scales + a shared f32 dst scale? A single f4_e2m1 matmul works
// (matmul_nvfp4_packed, nvfp4.hpp:268); a batched sub-byte matmul is unverified
// on Arc (cf. the q8 *grouped* failure logged in memory). This probe asks the
// narrower question: does plain batched (not grouped/varlen) sub-byte work?
//
// Two weight-scale layouts are tried; for each we print whether the
// primitive_desc builds and whether execute() runs:
//   A) batched weight scales [E,G,N]  (mask=7, groups{1,16,1})
//   B) shared  weight scales [G,N]    (mask=6, groups{16,1})  -- broadcast over E
// Both use batched src scales [E,M,G] (mask=7, groups{1,1,16}) and one shared
// f32 dst scale (mask=0, groups{}). Tiny case only; correctness is irrelevant —
// we only need to know whether oneDNN accepts the config.
//
// Build:
//   cmake --build build --target nvfp4_batched_probe
// Run:
//   ZE_AFFINITY_MASK=0 ./build/nvfp4_batched_probe

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include "common/gpu/engine.hpp"

using dt = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;

static const char* verdict(bool ok) { return ok ? "ACCEPT" : "REJECT"; }

// Build the primitive_desc for a given weight-scale config; return true + run a
// best-effort execute on a tiny zeroed buffer if it does.
static bool try_variant(const char* name, GpuEngine& ctx,
                        int E, int M, int K, int N, int G,
                        int w_mask, const dnnl::memory::dims& w_groups,
                        const dnnl::memory::dims& w_scale_dims) {
    std::printf("[probe] %s: E=%d M=%d K=%d N=%d G=%d w_mask=%d w_scale_dims=[",
                name, E, M, K, N, G, w_mask);
    for (size_t i = 0; i < w_scale_dims.size(); ++i)
        std::printf("%s%lld", i ? "," : "", (long long)w_scale_dims[i]);
    std::printf("]\n");

    try {
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_SRC,     7, {1, 1, 16}, dt::f8_e4m3);  // [E,M,G]
        attr.set_scales(DNNL_ARG_WEIGHTS, w_mask, w_groups, dt::f8_e4m3);
        attr.set_scales(DNNL_ARG_DST,     0, {},          dt::f32);      // [1]

        auto src_md     = dnnl::memory::desc({E, M, K}, dt::f4_e2m1, tag::abc);
        auto weights_md = dnnl::memory::desc({E, K, N}, dt::f4_e2m1, tag::any);
        auto dst_md     = dnnl::memory::desc({E, M, N}, dt::bf16,     tag::abc);

        dnnl::matmul::primitive_desc pd(ctx.engine, src_md, weights_md, dst_md, attr);
        std::printf("[probe]   pd OK: impl=%s weights_bytes=%zu weights_dims=[",
                    pd.impl_info_str(), (size_t)pd.weights_desc().get_size());
        for (auto d : pd.weights_desc().get_dims()) std::printf("%lld,", (long long)d);
        std::printf("]\n");

        // Best-effort execute on a tiny zeroed buffer. Correctness is irrelevant;
        // we only confirm execute() does not throw. Buffers are zeroed via a
        // sycl memset on the underlying USM pointer.
        auto mk = [&](const dnnl::memory::desc& md) {
            return dnnl::sycl_interop::make_memory(
                md, ctx.engine, dnnl::sycl_interop::memory_kind::usm);
        };
        auto concrete_w = pd.weights_desc();
        auto src_scales_md = dnnl::memory::desc({E, M, G}, dt::f8_e4m3, tag::abc);
        auto w_scales_md   = dnnl::memory::desc(w_scale_dims, dt::f8_e4m3, tag::abc);
        auto dst_scale_md  = dnnl::memory::desc({1}, dt::f32, tag::a);

        auto src_m  = mk(src_md);
        auto w_m    = mk(concrete_w);
        auto dst_m  = mk(dst_md);
        auto ssc_m  = mk(src_scales_md);
        auto wsc_m  = mk(w_scales_md);
        auto dsc_m  = mk(dst_scale_md);

        // Zero the buffers via their USM pointers.
        auto zero = [&](const dnnl::memory& m) {
            void* p = m.get_data_handle();
            ctx.queue.memset(p, 0, m.get_desc().get_size()).wait();
        };
        zero(src_m); zero(w_m); zero(dst_m); zero(ssc_m); zero(wsc_m);
        // dst_scale = 1.0f so a zeroed matmul is well-defined (avoids any NaN).
        float one = 1.0f;
        ctx.queue.memset(dsc_m.get_data_handle(), 0, 4).wait();
        ctx.queue.memcpy(dsc_m.get_data_handle(), &one, 4).wait();

        dnnl::matmul prim(pd);
        prim.execute(ctx.stream, {
            {DNNL_ARG_SRC, src_m},
            {DNNL_ARG_WEIGHTS, w_m},
            {DNNL_ARG_DST, dst_m},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, ssc_m},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wsc_m},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dsc_m},
        });
        ctx.stream.wait();
        std::printf("[probe]   execute OK\n");
        return true;
    } catch (const dnnl::error& e) {
        std::printf("[probe]   REJECT dnnl::error status=%d msg: %s\n",
                    e.status, e.message);
        return false;
    } catch (const std::exception& e) {
        std::printf("[probe]   REJECT std::exception: %s\n", e.what());
        return false;
    }
}

int main() {
    GpuEngine& ctx = GpuEngine::get(0);
    std::string dev = ctx.queue.get_device().get_info<sycl::info::device::name>();
    while (!dev.empty() && (unsigned char)dev.back() <= ' ') dev.pop_back();
    std::printf("[probe] device: %s | oneDNN batched f4_e2m1 feasibility\n", dev.c_str());

    constexpr int E = 2, M = 16, K = 32, N = 16;
    constexpr int G = K / 16;

    // Variant A: per-batch weight scales [E,G,N].
    bool a = try_variant("A:batched-wscales", ctx, E, M, K, N, G,
                         /*w_mask*/ 7, /*groups*/ {1, 16, 1},
                         /*w_scale_dims*/ {E, G, N});

    // Variant B: shared weight scales [G,N] (broadcast over the batch dim).
    bool b = try_variant("B:shared-wscales", ctx, E, M, K, N, G,
                         /*w_mask*/ 6, /*groups*/ {16, 1},
                         /*w_scale_dims*/ {G, N});

    std::printf("[probe] SUMMARY: A(batched)=%s  B(shared)=%s\n",
                verdict(a), verdict(b));
    std::printf("[probe] %s\n",
                (a || b) ? "Direction B (oneDNN batched) FEASIBLE — proceed to Phase 1."
                         : "Direction B DEAD — oneDNN rejects batched f4_e2m1. "
                           "Reallocate effort to Direction A (xe2 block-load).");
    return 0;
}

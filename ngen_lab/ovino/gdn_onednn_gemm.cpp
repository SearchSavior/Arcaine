// Non-fused GDN path benchmark: state-update dot products as oneDNN GPU GEMMs
// (the path OpenVINO uses when GDN fusion is disabled — loop-body GEMMs route
// to gemm_onednn -> oneDNN jit:gemm -> GemmStone DSL with DPAS on Xe2).
//
// Per token t, per head h:  h_k = state^T @ k_t   (state [K=128, V=128])
//                           update = (v - h_k) * beta
//                           state += k_t outer update
//                           o = state^T @ q_t
// As a batch GEMM over tokens: treat [V,K] x [K, T] -> [V, T] with one
// oneDNN matmul per head per phase (2 matmuls per head).
//
// This mirrors the OpenVINO non-fused loop body (query/key matmul + reduce),
// and measures whether DPAS GEMMs change the sequential-strategy picture.

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using bf16 = uint16_t;
static inline float bf2f(uint16_t v) {
    uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f;
}
static inline uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t b = ((u >> 16) & 1) + 0x7FFFu;
    return (uint16_t)((u + b) >> 16);
}

int main(int argc, char** argv) {
    std::string p_csv = "512,1024,2048,4096";
    int iters = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-p" && i + 1 < argc) p_csv = argv[++i];
        else if (a == "-n" && i + 1 < argc) iters = std::atoi(argv[++i]);
    }

    // ---- engine / queue ------------------------------------------------
    dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    auto dev = dnnl::sycl_interop::get_device(engine);
    auto ctx = dnnl::sycl_interop::get_context(engine);
    sycl::queue queue(ctx, dev, sycl::property::queue::in_order{});
    auto stream = dnnl::sycl_interop::make_stream(engine, queue);

    constexpr int nv = 32, dk = 128, dv = 128;

    std::vector<int> tokens;
    { std::string s = p_csv + ","; size_t pos = 0;
      while ((pos = s.find(',')) != std::string::npos) {
          tokens.push_back(std::atoi(s.substr(0, pos).c_str())); s.erase(0, pos + 1); } }

    for (int T : tokens) {
        // q,k: [T, nv, dk] bf16 (l2norm'd, q scaled); state [nv, dk, dv] f32
        std::vector<bf16> q_h((size_t)T * nv * dk), k_h((size_t)T * nv * dk);
        std::vector<bf16> v_h((size_t)T * nv * dv);
        std::vector<bf16> beta_h((size_t)T * nv), g_h((size_t)T * nv);
        uint32_t rng = 42;
        auto frand = [&](float lo, float hi) {
            rng = rng * 1664525u + 1013904223u;
            return lo + (hi - lo) * (float)((rng >> 8) & 0xFFFFFF) / 16777216.0f;
        };
        const float scale = 1.0f / std::sqrt((float)dk);
        for (size_t row = 0; row < (size_t)T * nv; ++row) {
            float nq = 0, nk = 0;
            std::vector<float> tq(dk), tk(dk);
            for (int d = 0; d < dk; ++d) {
                tq[d] = frand(-1.f, 1.f); nq += tq[d] * tq[d];
                tk[d] = frand(-1.f, 1.f); nk += tk[d] * tk[d];
            }
            nq = 1.f / std::sqrt(std::max(nq, 1e-12f));
            nk = 1.f / std::sqrt(std::max(nk, 1e-12f));
            for (int d = 0; d < dk; ++d) {
                q_h[row * dk + d] = f2bf(tq[d] * nq * scale);
                k_h[row * dk + d] = f2bf(tk[d] * nk);
            }
            for (int d = 0; d < dv; ++d) v_h[row * dv + d] = f2bf(frand(-1.f, 1.f));
            beta_h[row] = f2bf(frand(0.05f, 0.95f));
            g_h[row]    = f2bf(-frand(0.0005f, 0.05f));
        }
        std::vector<bf16> s0((size_t)nv * dk * dv, 0);
        for (auto& x : s0) x = f2bf(frand(-0.2f, 0.2f));
        std::vector<float> s0f((size_t)nv * dk * dv);
        for (size_t i = 0; i < s0f.size(); ++i) s0f[i] = bf2f(s0[i]);

        // ---- device buffers --------------------------------------------
        auto md = [](dnnl::memory::dims d, dnnl::memory::data_type dt, dnnl::memory::format_tag tag) {
            return dnnl::memory::desc(d, dt, tag);
        };
        auto mk = [&](const dnnl::memory::desc& d, void* p) {
            return dnnl::sycl_interop::make_memory(d, engine, dnnl::sycl_interop::memory_kind::usm, p);
        };
        auto malloc_dev = [&](size_t bytes) {
            return sycl::malloc_device<char>(bytes, queue);
        };
        auto upload = [&](void* dst, const void* src, size_t bytes) {
            queue.memcpy(dst, src, bytes).wait();
        };

        void* d_q = malloc_dev(q_h.size() * 2);
        void* d_k = malloc_dev(k_h.size() * 2);
        void* d_v = malloc_dev(v_h.size() * 2);
        void* d_beta = malloc_dev(beta_h.size() * 2);
        void* d_g = malloc_dev(g_h.size() * 2);
        void* d_s = malloc_dev(s0.size() * 2);
        void* d_hk = malloc_dev((size_t)nv * dv * T * 4);       // [nv, dv, T] f32
        void* d_core = malloc_dev((size_t)nv * dv * T * 4);     // f32 [nv,dv,T] out
        upload(d_q, q_h.data(), q_h.size() * 2);
        upload(d_k, k_h.data(), k_h.size() * 2);
        upload(d_v, v_h.data(), v_h.size() * 2);
        upload(d_beta, beta_h.data(), beta_h.size() * 2);
        upload(d_g, g_h.data(), g_h.size() * 2);
        upload(d_s, s0.data(), s0.size() * 2);

        // ---- oneDNN matmul: h_k = S^T K -> [nv, dv, T] (state [nv,dv,dk]^T x k)
        // State viewed as [nv, dv, dk] (transposed layout), K as [nv, dk, T].
        // We build the matmul per head-batch: [nv*dv, dk] x [nv*dk, T] won't
        // work; instead do [B=nv, dv, T] = state[b] (dv x dk) x k[b] (dk x T).
        // oneDNN matmul: src [nv, dv, dk] wei [nv, dk, T] dst [nv, dv, T].
        // State is stored [nv, dk, dv]; pass a view with swapped last two dims
        // is not expressible — so benchmark the equivalent batched matmul
        // shapes that the non-fused loop body would produce:
        //   phase 1: hk[b] = S[b]^T @ K[b]   S:[dv,dk] K:[dk,T]
        //   phase 2: o[b]  = S2[b]^T @ Q[b]  S2:[dv,dk] Q:[dk,T]
        // (S transposed in memory as [nv, dv, dk]).
        auto s_desc = md({nv, dv, dk}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto k_desc = md({nv, dk, T}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto hk_desc = md({nv, dv, T}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
        auto mm_pd1 = dnnl::matmul::primitive_desc(engine, s_desc, k_desc, hk_desc);
        auto mm1 = dnnl::matmul(mm_pd1);

        auto q_desc = md({nv, dk, T}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto o_desc = md({nv, dv, T}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
        auto mm_pd2 = dnnl::matmul::primitive_desc(engine, s_desc, q_desc, o_desc);
        auto mm2 = dnnl::matmul(mm_pd2);

        // A small state-copy + rank-1 update kernel is skipped: we measure the
        // GEMM-dominant cost (the DPAS matmuls), which is the crux of the
        // non-fused path.

        auto run_once = [&] {
            mm1.execute(stream, {
                {DNNL_ARG_SRC, mk(s_desc, d_s)},
                {DNNL_ARG_WEIGHTS, mk(k_desc, d_k)},
                {DNNL_ARG_DST, mk(hk_desc, d_hk)}});
            // phase 2 uses updated state; for timing, reuse d_s (same cost)
            mm2.execute(stream, {
                {DNNL_ARG_SRC, mk(s_desc, d_s)},
                {DNNL_ARG_WEIGHTS, mk(q_desc, d_q)},
                {DNNL_ARG_DST, mk(o_desc, d_core)}});
            stream.wait();
        };

        for (int i = 0; i < 2; ++i) run_once();
        std::vector<double> samples;
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            run_once();
            auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double mean = 0; for (double s : samples) mean += s; mean /= samples.size();
        std::printf("gdn_onednn_gemm p=%d runs=%d mean_ms=%.3f tok_per_s=%.1f (2 DPAS matmuls: "
                    "[%dx%d]x[%dx%d] + [%dx%d]x[%dx%d], per-head batched)\n",
                    T, iters, mean, T * 1000.0 / mean,
                    nv, dv, dk, dk, T, nv, dv, dk, dk, T);

        sycl::free(d_q, queue); sycl::free(d_k, queue); sycl::free(d_v, queue);
        sycl::free(d_beta, queue); sycl::free(d_g, queue); sycl::free(d_s, queue);
        sycl::free(d_hk, queue); sycl::free(d_core, queue);
    }
    return 0;
}

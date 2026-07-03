// src/nvfp4_dpas_dist_probe.cpp
// Distinguish ESIMD xmx::dpas operand-distribution semantics across the 16-lane
// subgroup. The earlier probe (nvfp4_dpas_probe.cpp) used IDENTICAL A/B on every
// lane, so it could not tell whether dpas is (a) per-lane independent or
// (b) subgroup-distributed (A by K, B by N, C by N) — both give the same
// replicated result for identical inputs. This probe gives lanes DIFFERENT A/B
// and checks which interpretation holds:
//   mode 1 (control):  full A, full B  on all lanes            -> expect full C = Cref
//   mode 2 (dist-A):   lane l has A with ONLY k=l col populated; full B  -> tests A-by-K
//   mode 3 (dist-AB):  lane l has A[k=l] col + B[n=l] col      -> tests full v2-style dist
// If mode 3 gives lane l the correct C[*][n=l], the v2 lane=ncolumn model works
// with ESIMD dpas (and lsc_load_2d can supply B). M=8,K=16,N=16 (bf16).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include "common/gpu/engine.hpp"
#include "common/gpu/buffer.hpp"

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using bf16_t = sycl::ext::oneapi::bfloat16;

static constexpr int M = 8, K = 16, N = 16;
static inline float Aval(int m, int k) { return ((m * 7 + k * 3) % 11) / 10.0f - 0.5f; }
static inline float Bval(int k, int n) { return ((k * 5 + n * 2) % 7) / 10.0f - 0.3f; }

// VNNI layout L2 (validated): B[(kp*N + n)*2 + kparity], kparity 0=even, 1=odd.
static inline void fillB_full(esimd::simd<bf16_t, K * N>& B) {
    for (int n = 0; n < N; ++n)
        for (int kp = 0; kp < K / 2; ++kp) {
            B[(kp * N + n) * 2 + 0] = bf16_t(Bval(2 * kp + 0, n));
            B[(kp * N + n) * 2 + 1] = bf16_t(Bval(2 * kp + 1, n));
        }
}
// lane l's B with ONLY its n=l column populated (VNNI L2), rest zero.
static inline void fillB_lane(esimd::simd<bf16_t, K * N>& B, int l) {
    for (int i = 0; i < K * N; ++i) B[i] = bf16_t(0.0f);
    for (int kp = 0; kp < K / 2; ++kp) {
        B[(kp * N + l) * 2 + 0] = bf16_t(Bval(2 * kp + 0, l));
        B[(kp * N + l) * 2 + 1] = bf16_t(Bval(2 * kp + 1, l));
    }
}

int main() {
    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;

    float Cref[M * N] = {0};
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += Aval(m, k) * Bval(k, n);
            Cref[m * N + n] = s;
        }

    constexpr int NM = 3;
    // per mode: 16 lanes x 128 floats
    std::vector<float> out((size_t)NM * 16 * 128, -7777.0f);
    GpuBuffer<float> out_dev((size_t)NM * 16 * 128, q);
    float* outp = out_dev.data();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>{16}, sycl::range<1>{16}),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                int lane = (int)it.get_local_id(0);

                // ---- full A (row-major) ----
                esimd::simd<bf16_t, M * K> A_full;
                for (int m = 0; m < M; ++m)
                    for (int k = 0; k < K; ++k) A_full[m * K + k] = bf16_t(Aval(m, k));
                // ---- lane A: only k=lane col populated ----
                esimd::simd<bf16_t, M * K> A_lane;
                for (int i = 0; i < M * K; ++i) A_lane[i] = bf16_t(0.0f);
                for (int m = 0; m < M; ++m) A_lane[m * K + lane] = bf16_t(Aval(m, lane));

                esimd::simd<bf16_t, K * N> B_full;
                fillB_full(B_full);
                esimd::simd<bf16_t, K * N> B_lane;
                fillB_lane(B_lane, lane);

                auto store = [&](int mode, esimd::simd<float, M * N>& C) {
                    size_t base = (size_t)mode * 16 * 128 + (size_t)lane * 128;
                    for (int i = 0; i < M * N; ++i) outp[base + i] = C[i];
                };

                {
                    esimd::simd<float, M * N> C(0.0f);
                    C = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(C, B_full, A_full);
                    store(0, C);
                }
                {
                    esimd::simd<float, M * N> C(0.0f);
                    C = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(C, B_full, A_lane);
                    store(1, C);
                }
                {
                    esimd::simd<float, M * N> C(0.0f);
                    C = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(C, B_lane, A_lane);
                    store(2, C);
                }
            });
    });
    q.wait();
    out_dev.download(out.data(), (size_t)NM * 16 * 128);

    auto eps = [](float a, float b) { return std::fabs(a - b); };
    const char* names[NM] = {"full-A full-B", "lane-A(k=l) full-B", "lane-A(k=l) lane-B(n=l)"};

    for (int mode = 0; mode < NM; ++mode) {
        const float* lane0 = &out[(size_t)mode * 16 * 128];
        // (a) lane0 full C vs Cref (row-major)
        float err_full = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                err_full = std::max(err_full, eps(lane0[m * N + n], Cref[m * N + n]));
        // (b) distributed-by-N: lane l holds C[m][l]? check lane l's C[m*N+l].
        float err_distN = 0;
        for (int l = 0; l < N; ++l) {
            const float* L = &out[(size_t)mode * 16 * 128 + (size_t)l * 128];
            for (int m = 0; m < M; ++m)
                err_distN = std::max(err_distN, eps(L[m * N + l], Cref[m * N + l]));
        }
        // (c) is the result identical across lanes (replicated)?
        float err_repl = 0;
        for (int l = 1; l < 16; ++l)
            for (int i = 0; i < M * N; ++i)
                err_repl = std::max(err_repl, eps(out[(size_t)mode*16*128 + (size_t)l*128 + i],
                                                  lane0[i]));
        std::printf("[dist-probe] %-22s : err_full=%.4g err_distN=%.4g err_repl=%.4g\n",
                    names[mode], err_full, err_distN, err_repl);
    }

    std::printf("[dist-probe] verdict:\n");
    const float* m2lane0 = &out[2 * 16 * 128];
    float e2_full = 0, e2_distN = 0;
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n)
        e2_full = std::max(e2_full, eps(m2lane0[m*N+n], Cref[m*N+n]));
    for (int l = 0; l < N; ++l) {
        const float* L = &out[2*16*128 + (size_t)l*128];
        for (int m = 0; m < M; ++m) e2_distN = std::max(e2_distN, eps(L[m*N+l], Cref[m*N+l]));
    }
    if (e2_full < 0.05f)
        std::printf("  mode3 C is REPLICATED (full A+B cooperative, one m-tile/subgroup)\n");
    else if (e2_distN < 0.05f)
        std::printf("  mode3 C is DISTRIBUTED by N (lane l -> n=l): v2-style works w/ ESIMD dpas\n");
    else
        std::printf("  mode3 NEITHER replicated nor N-distributed matches -> need other layout\n");
    return 0;
}

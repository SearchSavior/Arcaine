// src/nvfp4_dpas_probe.cpp
// Phase-2 probe: determine esimd::dpas B VNNI packing for a bf16 GEMM tile
// (M=8,K=16,N=16). Confirmed: result is PER-LANE REPLICATED (each lane gets
// the full 8x16=128 floats, identical across lanes for identical inputs).
// A is row-major [m*K+k] (VNNI implicit via consecutive K). This probe sweeps 4
// B packings to find the one matching the host reference.

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

// B layout variants. Each fills B (256 bf16) so its int-view is a VNNI pack
// (bf16[2i]=X, bf16[2i+1]=Y, int = X|Y<<16) under a different (kp,n) ordering
// and/or K0/K1 swap.
//   L1: i = n*(K/2)+kp ; X=Bval(2kp,n)   Y=Bval(2kp+1,n)   (n-outer, K0 low)
//   L2: i = kp*N+n     ; X=Bval(2kp,n)   Y=Bval(2kp+1,n)   (kp-outer, K0 low)
//   L4: i = n*(K/2)+kp ; X=Bval(2kp+1,n) Y=Bval(2kp,n)     (n-outer, K1 low)
//   L5: i = kp*N+n     ; X=Bval(2kp+1,n) Y=Bval(2kp,n)     (kp-outer, K1 low)
static inline void fillB(esimd::simd<bf16_t, K * N>& B, int layout) {
    for (int n = 0; n < N; ++n)
        for (int kp = 0; kp < K / 2; ++kp) {
            float k0 = Bval(2 * kp + 0, n);
            float k1 = Bval(2 * kp + 1, n);
            int i;
            if (layout == 1 || layout == 4) i = n * (K / 2) + kp;
            else                              i = kp * N + n;
            float X = (layout == 4 || layout == 5) ? k1 : k0;
            float Y = (layout == 4 || layout == 5) ? k0 : k1;
            B[2 * i + 0] = bf16_t(X);
            B[2 * i + 1] = bf16_t(Y);
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

    constexpr int NL = 4;  // layouts tested
    int layouts[NL] = {1, 2, 4, 5};
    std::vector<float> out((size_t)NL * 16 * 128, -7777.0f);
    GpuBuffer<float> out_dev((size_t)NL * 16 * 128, q);
    float* outp = out_dev.data();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>{16}, sycl::range<1>{16}),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                int lane = (int)it.get_local_id(0);
                esimd::simd<bf16_t, M * K> A;
                for (int m = 0; m < M; ++m)
                    for (int k = 0; k < K; ++k) A[m * K + k] = bf16_t(Aval(m, k));
                esimd::simd<bf16_t, K * N> B;
                esimd::simd<float, M * N> C;
                for (int li = 0; li < NL; ++li) {
                    int layout = layouts[li];
                    fillB(B, layout);
                    C = esimd::simd<float, M * N>(0.0f);
                    C = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(C, B, A);
                    size_t base = (size_t)li * 16 * 128 + (size_t)lane * 128;
                    for (int i = 0; i < M * N; ++i) outp[base + i] = C[i];
                }
            });
    });
    q.wait();
    out_dev.download(out.data(), (size_t)NL * 16 * 128);

    auto eps = [](float a, float b) { return std::fabs(a - b); };
    for (int li = 0; li < NL; ++li) {
        const float* lane0 = &out[(size_t)li * 16 * 128];
        float err_row = 0, err_col = 0;  // C row-major [m*N+n] or col-major [n*M+m]
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                err_row = std::max(err_row, eps(lane0[m * N + n], Cref[m * N + n]));
                err_col = std::max(err_col, eps(lane0[n * M + m], Cref[m * N + n]));
            }
        std::printf("[dpas-probe] B-layout L%d : err(row-major C)=%.4g  err(col-major C)=%.4g\n",
                    layouts[li], err_row, err_col);
    }
    // best
    int best = -1; double besterr = 1e9; const char* bestc = "";
    for (int li = 0; li < NL; ++li) {
        const float* lane0 = &out[(size_t)li * 16 * 128];
        for (int cm = 0; cm < 2; ++cm) {
            float e = 0;
            for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n)
                e = std::max(e, eps(lane0[cm ? n * M + m : m * N + n], Cref[m * N + n]));
            if (e < besterr) { besterr = e; best = layouts[li]; bestc = cm ? "col-major" : "row-major"; }
        }
    }
    std::printf("[dpas-probe] BEST: B-L%d C-%s err=%.4g => %s\n",
                best, bestc, besterr, besterr < 0.05f ? "PASS" : "FAIL");
    return besterr < 0.05f ? 0 : 1;
}

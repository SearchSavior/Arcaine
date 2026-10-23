// Call the real prefill_xmx::gemm_q4_xmx and prefill::gemm_q4<true> on a
// tiny exact-fit shape and dump both.
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <sycl/sycl.hpp>
#include "modeling/flint/weights.hpp"
#include "modeling/flint/prefill.hpp"
#include "modeling/flint/prefill_xmx.hpp"
using namespace flint;

int main(int argc, char** argv) {
    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                  sycl::property::queue::in_order{});
    const int M = argc > 1 ? atoi(argv[1]) : 16;
    const int N = argc > 2 ? atoi(argv[2]) : 32;
    const int K = argc > 3 ? atoi(argv[3]) : 64;
    std::vector<int8_t> a((size_t)M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) a[m * K + k] = (int8_t)((m * 7 + k * 3) % 61 - 30);
    std::vector<uint8_t> w((size_t)N * K / 2);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; k += 2) {
            int c0 = (n * 5 + k * 11) % 16, c1 = (n * 5 + (k + 1) * 11) % 16;
            w[(size_t)n * (K / 2) + k / 2] = (uint8_t)(c0 | (c1 << 4));
        }
    const int G = K / 64;
    std::vector<bf16> s((size_t)N * G), b((size_t)N * G);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < G; ++g) {
            s[(size_t)n * G + g] = bf16(0.01f + 0.001f * (float)((n + g) % 7));
            b[(size_t)n * G + g] = bf16(-0.002f + 0.0005f * (float)((n + g) % 5));
        }
    std::vector<float> xs((size_t)M * G, 1.f);
    std::vector<int> xi((size_t)M * G, 0);
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < G; ++g) {
            xs[(size_t)m * G + g] = 0.02f + 0.001f * (float)((m + g) % 9);
            int t = 0;
            for (int k = g * 64; k < g * 64 + 64; ++k) t += a[(size_t)m * K + k];
            xi[(size_t)m * G + g] = t;
        }
    QTensor wm;
    wm.w = sycl::malloc_device<uint8_t>(w.size(), q);
    wm.s = sycl::malloc_device<bf16>(s.size(), q);
    wm.b = sycl::malloc_device<bf16>(b.size(), q);
    wm.n = N; wm.k = K; wm.experts = 1;
    wm.stride_w = w.size(); wm.stride_s = s.size(); wm.stride_b = b.size();
    q.memcpy(const_cast<uint8_t*>(wm.w), w.data(), w.size());
    q.memcpy(const_cast<bf16*>(wm.s), s.data(), s.size() * 2);
    q.memcpy(const_cast<bf16*>(wm.b), b.data(), b.size() * 2);
    int8_t* d_a = sycl::malloc_device<int8_t>(a.size(), q);
    float* d_xs = sycl::malloc_device<float>(xs.size(), q);
    int* d_xi = sycl::malloc_device<int>(xi.size(), q);
    bf16* d_cv = sycl::malloc_device<bf16>((size_t)M * N, q);
    bf16* d_cx = sycl::malloc_device<bf16>((size_t)M * N, q);
    q.memcpy(d_a, a.data(), a.size());
    q.memcpy(d_xs, xs.data(), xs.size() * 4);
    q.memcpy(d_xi, xi.data(), xi.size() * 4).wait();
    const bool ok = prefill_xmx::gemm_q4_xmx(q, wm, d_a, d_xs, d_xi, nullptr, d_cx, M, N);
    prefill::gemm_q4<true>(q, wm, nullptr, d_a, d_xs, nullptr, d_cv, M, N);
    q.wait();
    std::printf("accepted=%d\n", ok);
    std::vector<bf16> cv((size_t)M * N), cx((size_t)M * N);
    q.copy(d_cv, cv.data(), cv.size()).wait();
    q.copy(d_cx, cx.data(), cx.size()).wait();
    auto s4 = [](int cc) { return cc; };
    double d_max = 0, d_min = 0;
    int bad_v = 0, bad_x = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float gold = 0;
            for (int g = 0; g < G; ++g) {
                int ai = 0;
                for (int k = g * 64; k < g * 64 + 64; ++k) {
                    int cc = (k & 1) ? (w[(size_t)n * (K / 2) + k / 2] >> 4) & 15
                                     : w[(size_t)n * (K / 2) + k / 2] & 15;
                    ai += (int)a[(size_t)m * K + k] * s4(cc);
                }
                gold += (float(s[(size_t)n * G + g]) * ai +
                         float(b[(size_t)n * G + g]) * xi[(size_t)m * G + g]) *
                        xs[(size_t)m * G + g];
            }
            const double vx = std::fabs(float(cx[(size_t)m * N + n]) - gold);
            const double vv = std::fabs(float(cv[(size_t)m * N + n]) - gold);
            d_max = std::fmax(d_max, vx);
            if (vx > 1e-2 * (1 + std::fabs(gold))) ++bad_x;
            if (vv > 1e-2 * (1 + std::fabs(gold))) ++bad_v;
        }
    std::printf("M=%d N=%d K=%d  xmx_bad=%d vec_bad=%d  max_abs=%.5g\n", M, N, K,
                bad_x, bad_v, d_max);
}

// Weight-only NVFP4 DPAS GEMM: C[M,N] = (A_bf16 @ dequant(W_fp4)) / weight_global
// Validates the tiled DPAS kernel mechanics against a host reference on the real
// expert shapes, before integrating into expert_parallel.
//
// Build:
//   icpx -fsycl -O2 -fsycl-targets=intel_gpu_bmg_g31 \
//     -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate \
//     tools/dpas_gemm_test.cpp -o dpas_gemm_test
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

using v8s = short __attribute__((ext_vector_type(8)));
using v8i = int   __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));

SYCL_EXTERNAL v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, v8s A, v8i B, v8f C, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return v8f{}; }
#endif
constexpr int kBF16 = 0x3000;  // MatrixAPackedBFloat16INTEL | MatrixBPackedBFloat16INTEL

// ---- tiny shared codecs (host + device) ----
static inline uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t bias = ((u >> 16) & 1) + 0x7FFFu;
    return (uint16_t)((u + bias) >> 16);
}
static inline float bf2f(uint16_t v) {
    uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f;
}
static inline float e2m1(uint8_t bits) {
    float mag;
    switch (bits & 7) {
        case 0: mag = 0; break; case 1: mag = 0.5f; break; case 2: mag = 1; break;
        case 3: mag = 1.5f; break; case 4: mag = 2; break; case 5: mag = 3; break;
        case 6: mag = 4; break; default: mag = 6; break;
    }
    return (bits & 8) ? -mag : mag;
}
static inline float e4m3(uint8_t bits) {
    if (bits == 0) return 0.0f;
    float sign = (bits & 0x80) ? -1.0f : 1.0f;
    int exp = (bits >> 3) & 0x0f, mant = bits & 0x07;
    float v = exp == 0 ? (mant / 8.0f) * sycl::exp2(-6.0f)
                       : (1.0f + mant / 8.0f) * sycl::exp2((float)exp - 7.0f);
    return sign * v;
}

// One sub-group (SG16) computes an 8x16 output tile. lane = output column offset
// AND the K offset for the A operand load (DPAS distributes data internally).
static void dpas_weightonly(sycl::queue& q, const uint16_t* A, const uint8_t* w_packed,
                            const uint8_t* w_scale, float inv_wglobal,
                            uint16_t* C, int M, int K, int N) {
    int Mt = M / 8, Nt = N / 16, G = K / 16, halfK = K / 2;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(Mt, N), sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int m0 = (int)it.get_group(0) * 8;
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                v8f c = {0,0,0,0,0,0,0,0};
                for (int k0 = 0; k0 < K; k0 += 16) {
                    float scale = e4m3(w_scale[(size_t)(k0 / 16) * N + n]);
                    const uint8_t* wrow = w_packed + (size_t)n * halfK + k0 / 2;
                    v8i b;
                    for (int j = 0; j < 8; ++j) {
                        uint8_t byte = wrow[j];
                        b[j] = (uint32_t)f2bf(e2m1(byte & 0x0f) * scale)
                             | ((uint32_t)f2bf(e2m1(byte >> 4) * scale) << 16);
                    }
                    v8s a;
                    for (int m = 0; m < 8; ++m)
                        a[m] = (short)A[(size_t)(m0 + m) * K + k0 + lane];
                    c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c, kBF16);
                }
                (void)G;
                for (int m = 0; m < 8; ++m)
                    C[(size_t)(m0 + m) * N + n] = f2bf(c[m] * inv_wglobal);
            });
    }).wait();
}

// Full-NVFP4: A is packed FP4 [M,K/2] + per-(row,16) f8 scale; same weight path.
// C = (dequant(A) @ dequant(W)) / (input_global * weight_global)
static void dpas_full(sycl::queue& q, const uint8_t* a_packed, const uint8_t* a_scale,
                      const uint8_t* w_packed, const uint8_t* w_scale, float inv_global,
                      uint16_t* C, int M, int K, int N) {
    int Mt = M / 8, halfK = K / 2, G = K / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(Mt, N), sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int m0 = (int)it.get_group(0) * 8;
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                v8f c = {0,0,0,0,0,0,0,0};
                for (int k0 = 0; k0 < K; k0 += 16) {
                    int g = k0 / 16;
                    float wscale = e4m3(w_scale[(size_t)g * N + n]);
                    const uint8_t* wrow = w_packed + (size_t)n * halfK + k0 / 2;
                    v8i b;
                    for (int j = 0; j < 8; ++j) {
                        uint8_t byte = wrow[j];
                        b[j] = (uint32_t)f2bf(e2m1(byte & 0x0f) * wscale)
                             | ((uint32_t)f2bf(e2m1(byte >> 4) * wscale) << 16);
                    }
                    v8s a;
                    int kk = k0 + lane;
                    for (int m = 0; m < 8; ++m) {
                        uint8_t byte = a_packed[(size_t)(m0 + m) * halfK + kk / 2];
                        uint8_t nibv = (kk & 1) ? (byte >> 4) : (byte & 0x0f);
                        float av = e2m1(nibv) * e4m3(a_scale[(size_t)(m0 + m) * G + g]);
                        a[m] = (short)f2bf(av);
                    }
                    c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c, kBF16);
                }
                for (int m = 0; m < 8; ++m)
                    C[(size_t)(m0 + m) * N + n] = f2bf(c[m] * inv_global);
            });
    }).wait();
}

static void test_full(sycl::queue& q, int M, int K, int N, const char* name) {
    std::mt19937 rng(99 + M + K + N);
    std::uniform_int_distribution<int> nib(0, 15), sc(40, 110);
    int halfK = K / 2, G = K / 16;
    float iglobal = 0.8f, wglobal = 1.5f, inv = 1.0f / (iglobal * wglobal);
    std::vector<uint8_t> ap((size_t)M * halfK), as((size_t)M * G), wp((size_t)N * halfK), ws((size_t)G * N);
    for (auto& v : ap) v = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& v : as) v = (uint8_t)sc(rng);
    for (auto& v : wp) v = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& v : ws) v = (uint8_t)sc(rng);
    std::vector<float> ref((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                uint8_t ab = ap[(size_t)m * halfK + k / 2];
                float a = bf2f(f2bf(e2m1((k & 1) ? (ab >> 4) : (ab & 0x0f)) * e4m3(as[(size_t)m * G + k / 16])));
                uint8_t wb = wp[(size_t)n * halfK + k / 2];
                float w = bf2f(f2bf(e2m1((k & 1) ? (wb >> 4) : (wb & 0x0f)) * e4m3(ws[(size_t)(k / 16) * N + n])));
                acc += a * w;
            }
            ref[(size_t)m * N + n] = acc * inv;
        }
    uint8_t *apd = sycl::malloc_device<uint8_t>(ap.size(), q), *asd = sycl::malloc_device<uint8_t>(as.size(), q);
    uint8_t *wpd = sycl::malloc_device<uint8_t>(wp.size(), q), *wsd = sycl::malloc_device<uint8_t>(ws.size(), q);
    uint16_t* Cd = sycl::malloc_device<uint16_t>((size_t)M * N, q);
    q.memcpy(apd, ap.data(), ap.size()).wait(); q.memcpy(asd, as.data(), as.size()).wait();
    q.memcpy(wpd, wp.data(), wp.size()).wait(); q.memcpy(wsd, ws.data(), ws.size()).wait();
    dpas_full(q, apd, asd, wpd, wsd, inv, Cd, M, K, N);
    std::vector<uint16_t> C((size_t)M * N); q.memcpy(C.data(), Cd, C.size() * 2).wait();
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i) { float g = bf2f(C[i]); dot += (double)g * ref[i]; na += (double)g * g; nb += (double)ref[i] * ref[i]; }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    std::printf("%-14s M=%4d K=%4d N=%4d  cos=%.6f  %s\n", name, M, K, N, cos, cos > 0.999 ? "PASS" : "FAIL");
    sycl::free(apd, q); sycl::free(asd, q); sycl::free(wpd, q); sycl::free(wsd, q); sycl::free(Cd, q);
}

static void test(sycl::queue& q, int M, int K, int N, const char* name) {
    std::mt19937 rng(1234 + M + K + N);
    std::uniform_int_distribution<int> nib(0, 15), sc(40, 110);  // e4m3 ~ O(1) scales
    std::normal_distribution<float> an(0.f, 1.f);
    int halfK = K / 2, G = K / 16;
    float wglobal = 1.5f, inv_wglobal = 1.0f / wglobal;

    std::vector<uint16_t> A(M * K);
    std::vector<uint8_t> wp((size_t)N * halfK), ws((size_t)G * N);
    for (auto& a : A) a = f2bf(an(rng));
    for (auto& v : wp) v = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& v : ws) v = (uint8_t)sc(rng);

    // host reference
    std::vector<float> ref((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                uint8_t byte = wp[(size_t)n * halfK + k / 2];
                uint8_t nibv = (k & 1) ? (byte >> 4) : (byte & 0x0f);
                float w = bf2f(f2bf(e2m1(nibv) * e4m3(ws[(size_t)(k / 16) * N + n])));
                acc += bf2f(A[(size_t)m * K + k]) * w;
            }
            ref[(size_t)m * N + n] = acc * inv_wglobal;
        }

    uint16_t* Ad = sycl::malloc_device<uint16_t>(A.size(), q);
    uint8_t* wpd = sycl::malloc_device<uint8_t>(wp.size(), q);
    uint8_t* wsd = sycl::malloc_device<uint8_t>(ws.size(), q);
    uint16_t* Cd = sycl::malloc_device<uint16_t>((size_t)M * N, q);
    q.memcpy(Ad, A.data(), A.size() * 2).wait();
    q.memcpy(wpd, wp.data(), wp.size()).wait();
    q.memcpy(wsd, ws.data(), ws.size()).wait();
    dpas_weightonly(q, Ad, wpd, wsd, inv_wglobal, Cd, M, K, N);
    std::vector<uint16_t> C((size_t)M * N);
    q.memcpy(C.data(), Cd, C.size() * 2).wait();

    double dot = 0, na = 0, nb = 0, mae = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        float got = bf2f(C[i]), exp = ref[i];
        dot += (double)got * exp; na += (double)got * got; nb += (double)exp * exp;
        mae += std::fabs(got - exp);
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    std::printf("%-14s M=%4d K=%4d N=%4d  cos=%.6f  mae=%.5f  %s\n",
                name, M, K, N, cos, mae / ((size_t)M * N),
                cos > 0.999 ? "PASS" : "FAIL");
    sycl::free(Ad, q); sycl::free(wpd, q); sycl::free(wsd, q); sycl::free(Cd, q);
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("-- weight-only --\n");
    test(q, 8, 2816, 4224, "gate/up M8");
    test(q, 64, 2816, 4224, "gate/up M64");
    test(q, 64, 2112, 2816, "down M64");
    test(q, 256, 704, 2816, "down-sm M256");
    std::printf("-- full nvfp4 --\n");
    test_full(q, 8, 2816, 4224, "gate/up M8");
    test_full(q, 64, 2112, 2816, "down M64");
    test_full(q, 256, 704, 2816, "down-sm M256");
    return 0;
}

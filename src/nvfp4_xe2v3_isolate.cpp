// src/nvfp4_xe2v3_isolate.cpp
//
// Isolate xe2v3's data path vs a host reference. Now multi-tile: K=32 (2
// k-tiles), N=32 (2 n-tiles), 1 expert, 128 rows (16 m-tiles). Replicates v3's
// EXACT loop structure (m-tile stride of LANES, kt accumulation, n-tile
// work-groups, replicated lsc_load_2d B, per-lane A, per-lane dpas). Locates the
// v3 correctness bug by bisection vs the single-tile probe (which passed with
// a=b=c=0).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include "common/gpu/engine.hpp"
#include "common/gpu/buffer.hpp"

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace esimd_x = sycl::ext::intel::experimental::esimd;
using bf16_t = sycl::ext::oneapi::bfloat16;

static inline float nvfp4_e4m3_fast(uint8_t b) {
    uint32_t exp = (b >> 3) & 0x0f, mant = b & 0x07;
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    if (exp == 0) return sign * (float)mant * (1.0f / 512.0f);
    uint32_t bits = ((uint32_t)(b & 0x80) << 24) | ((exp - 7 + 127) << 23) | (mant << 20);
    float out; __builtin_memcpy(&out, &bits, 4); return out;
}
static inline float nvfp4_e2m1_fast(uint8_t bits) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (bits & 8) ? -mag[bits & 7] : mag[bits & 7];
}

static constexpr int M_TOT = 128, K = 32, N = 32;
static constexpr int TM = 8, TN = 16, TK = 16, BW = 8, LANES = 16;
static constexpr int LBW = 16, LBH = 8;
static constexpr int KTILES = K / 16, NTILES = N / 16;

int main() {
    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    int halfK = K / 2;

    // ---- known packed data ----
    std::vector<uint8_t> A_pk((size_t)M_TOT * halfK), A_sc((size_t)M_TOT * KTILES);
    std::vector<uint8_t> W_pk((size_t)N * halfK), W_sc((size_t)KTILES * N);
    for (int r = 0; r < M_TOT; ++r)
        for (int b = 0; b < halfK; ++b)
            A_pk[(size_t)r * halfK + b] = (uint8_t)((r * 13 + b * 7 + 1) & 0xff);
    for (int r = 0; r < M_TOT; ++r)
        A_sc[(size_t)r * KTILES + 0] = (uint8_t)(0x60 | ((r * 3) & 0x0f));
    for (int r = 0; r < M_TOT; ++r)
        A_sc[(size_t)r * KTILES + 1] = (uint8_t)(0x50 | ((r * 5) & 0x0f));
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < halfK; ++b)
            W_pk[(size_t)n * halfK + b] = (uint8_t)((n * 5 + b * 11 + 2) & 0xff);
    for (int kt = 0; kt < KTILES; ++kt)
        for (int n = 0; n < N; ++n)
            W_sc[(size_t)kt * N + n] = (uint8_t)(0x68 | ((n * 2 + kt) & 0x07));
    float dst_scale = 1.0f, inv_dst = 1.0f / dst_scale;

    // ---- host coalesced weight: [nt][kt][nloc][byte], 128 bytes/slab ----
    std::vector<uint8_t> wcoal((size_t)NTILES * KTILES * 128);
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < halfK; ++b) {
            int kt = b / 8, j = b % 8;
            int nt = n / 16, nloc = n % 16;
            wcoal[(size_t)nt * KTILES * 128 + (size_t)kt * 128 + nloc * 8 + j] =
                W_pk[(size_t)n * halfK + b];
        }

    // ---- host ground-truth dequant ----
    auto A_dq = [&](int row, int k) -> float {
        uint8_t byte = A_pk[(size_t)row * halfK + k / 2];
        uint8_t nib = (k & 1) ? ((byte >> 4) & 0x0f) : (byte & 0x0f);
        return nvfp4_e2m1_fast(nib) * nvfp4_e4m3_fast(A_sc[(size_t)row * KTILES + k / 16]);
    };
    auto W_dq = [&](int n, int k) -> float {
        uint8_t byte = W_pk[(size_t)n * halfK + k / 2];
        uint8_t nib = (k & 1) ? ((byte >> 4) & 0x0f) : (byte & 0x0f);
        return nvfp4_e2m1_fast(nib) * nvfp4_e4m3_fast(W_sc[(size_t)(k / 16) * N + n]);
    };
    std::vector<float> C_ref((size_t)M_TOT * N, 0.0f);
    for (int row = 0; row < M_TOT; ++row)
        for (int n = 0; n < N; ++n) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += A_dq(row, k) * W_dq(n, k);
            C_ref[(size_t)row * N + n] = s * inv_dst;
        }

    // bf16-rounded reference: round each dequant product to bf16 BEFORE accumulating
    // (this is exactly what the dpas path sees: A,B are bf16 inputs).
    auto f2b = [](float f) -> uint16_t {
        uint32_t u; __builtin_memcpy(&u, &f, 4);
        uint16_t b = (uint16_t)(u >> 16);
        if (u & 0x8000) b += (b & 1);  // round-to-nearest-even
        return b;
    };
    auto b2f = [](uint16_t b) -> float {
        uint32_t u = ((uint32_t)b) << 16; float f; __builtin_memcpy(&f, &u, 4); return f;
    };
    auto A_dqb = [&](int row, int k) -> float {
        return b2f(f2b(A_dq(row, k)));
    };
    auto W_dqb = [&](int n, int k) -> float {
        return b2f(f2b(W_dq(n, k)));
    };
    std::vector<float> C_refb((size_t)M_TOT * N, 0.0f);
    for (int row = 0; row < M_TOT; ++row)
        for (int n = 0; n < N; ++n) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += A_dqb(row, k) * W_dqb(n, k);
            C_refb[(size_t)row * N + n] = s * inv_dst;
        }

    // ---- device buffers ----
    GpuBuffer<uint8_t> A_pk_dev((size_t)M_TOT * halfK, q); A_pk_dev.upload(A_pk.data(), (size_t)M_TOT * halfK);
    GpuBuffer<uint8_t> A_sc_dev((size_t)M_TOT * KTILES, q); A_sc_dev.upload(A_sc.data(), (size_t)M_TOT * KTILES);
    GpuBuffer<uint8_t> wcoal_dev((size_t)NTILES * KTILES * 128, q); wcoal_dev.upload(wcoal.data(), (size_t)NTILES * KTILES * 128);
    GpuBuffer<uint8_t> ws_dev((size_t)KTILES * N, q); ws_dev.upload(W_sc.data(), (size_t)KTILES * N);
    GpuBuffer<bf16_t> C_dev((size_t)M_TOT * N, q);
    const uint8_t* Ap = A_pk_dev.data();
    const uint8_t* As = A_sc_dev.data();
    const uint8_t* wc = wcoal_dev.data();
    const uint8_t* ws = ws_dev.data();
    bf16_t* Cp = C_dev.data();
    int offset = 0, count = M_TOT, mt = (count + TM - 1) / TM;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(1, (size_t)NTILES * LANES),
                               sycl::range<2>(1, (size_t)LANES)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int e = (int)it.get_group(0);
                int lane = (int)it.get_local_id(1);
                int nt = (int)it.get_group(1);
                const uint8_t* wcoal_nt = wc + (size_t)nt * KTILES * 128;
                for (int tile0 = 0; tile0 < mt; tile0 += LANES) {
                    int m_tile = tile0 + lane;
                    bool active = (m_tile < mt);
                    esimd::simd<float, TM * TN> c(0.0f);
                    for (int kt = 0; kt < KTILES; ++kt) {
                        auto v = esimd_x::lsc_load_2d<uint8_t, LBW, LBH, 1, false, false>(
                            wcoal_nt + (size_t)kt * 128, LBW - 1, LBH - 1, LBW - 1, 0, 0);
                        esimd::simd<bf16_t, TK * TN> b;
                        #pragma unroll
                        for (int r = 0; r < LBH; ++r) {
                            #pragma unroll
                            for (int ccol = 0; ccol < LBW; ++ccol) {
                                int n = (ccol < BW) ? 2 * r : 2 * r + 1;
                                int kp = ccol & (BW - 1);
                                uint8_t byte = v[r * LBW + ccol];
                                float wscale = nvfp4_e4m3_fast(ws[(size_t)kt * N + nt * TN + n]);
                                float lo = nvfp4_e2m1_fast(byte & 0x0f) * wscale;
                                float hi = nvfp4_e2m1_fast(byte >> 4) * wscale;
                                b[(kp * TN + n) * 2 + 0] = bf16_t(lo);
                                b[(kp * TN + n) * 2 + 1] = bf16_t(hi);
                            }
                        }
                        esimd::simd<bf16_t, TM * TK> a;
                        #pragma unroll
                        for (int m = 0; m < TM; ++m) {
                            int row = offset + m_tile * TM + m;
                            bool ok = active && ((row - offset) < count);
                            if (ok) {
                                float ascale = nvfp4_e4m3_fast(As[(size_t)row * KTILES + kt]);
                                #pragma unroll
                                for (int k = 0; k < TK; ++k) {
                                    uint8_t byte = Ap[(size_t)row * halfK + (kt * TK + k) / 2];
                                    uint8_t nib = (k & 1) ? ((byte >> 4) & 0x0f) : (byte & 0x0f);
                                    a[m * TK + k] = bf16_t(nvfp4_e2m1_fast(nib) * ascale);
                                }
                            } else {
                                #pragma unroll
                                for (int k = 0; k < TK; ++k) a[m * TK + k] = bf16_t(0.0f);
                            }
                        }
                        c = xmx::dpas<8, 8, float, float, bf16_t, bf16_t>(c, b, a);
                    }
                    if (active) {
                        #pragma unroll
                        for (int m = 0; m < TM; ++m) {
                            int row = offset + m_tile * TM + m;
                            if ((row - offset) >= count) continue;
                            #pragma unroll
                            for (int n = 0; n < TN; ++n)
                                Cp[(size_t)row * N + nt * TN + n] =
                                    bf16_t(c[m * TN + n] * inv_dst);
                        }
                    }
                }
            });
    });
    q.wait();

    std::vector<bf16_t> hCdev((size_t)M_TOT * N);
    C_dev.download(hCdev.data(), (size_t)M_TOT * N);
    double full = 0; int argmax = -1, nbad = 0;
    for (size_t i = 0; i < hCdev.size(); ++i) {
        double d = std::fabs((float)hCdev[i] - C_ref[i]);
        if (d > full) { full = d; argmax = (int)i; }
        if (d > 1.0) ++nbad;
    }
    std::printf("[iso2] K=%d N=%d ktiles=%d ntiles=%d | full C max_abs=%.4g at row=%d n=%d (dev=%.4g ref=%.4g) nbad(>1)=%d\n",
                K, N, KTILES, NTILES, full, argmax / N, argmax % N,
                (float)hCdev[argmax], C_ref[argmax], nbad);
    // device vs bf16-rounded ref
    double fullb = 0; int argmaxb = -1, nbadb = 0;
    for (size_t i = 0; i < hCdev.size(); ++i) {
        double d = std::fabs((float)hCdev[i] - C_refb[i]);
        if (d > fullb) { fullb = d; argmaxb = (int)i; }
        if (d > 1.0) ++nbadb;
    }
    std::printf("[iso2] dev-vs-bf16ref max_abs=%.4g at row=%d n=%d (dev=%.4g bf16ref=%.4g) nbad(>1)=%d\n",
                fullb, argmaxb / N, argmaxb % N,
                (argmaxb >= 0 ? (float)hCdev[argmaxb] : 0.f),
                (argmaxb >= 0 ? C_refb[argmaxb] : 0.f), nbadb);
    // sample first 8 of n-tile 0 vs ref
    std::printf("[iso2] C[0][0..7] dev: ");
    for (int n = 0; n < 8; ++n) std::printf("%.1f ", (float)hCdev[0 * N + n]);
    std::printf("\n[iso2] C[0][0..7] ref: ");
    for (int n = 0; n < 8; ++n) std::printf("%.1f ", C_ref[0 * N + n]);
    std::printf("\n[iso2] C[0][16..23] dev: ");
    for (int n = 16; n < 24; ++n) std::printf("%.1f ", (float)hCdev[0 * N + n]);
    std::printf("\n[iso2] C[0][16..23] ref: ");
    for (int n = 16; n < 24; ++n) std::printf("%.1f ", C_ref[0 * N + n]);
    std::printf("\n");
    return 0;
}

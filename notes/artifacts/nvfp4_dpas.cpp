#include "nvfp4_dpas.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <vector>

// Native Clang vector types (NOT sycl::vec — a struct return forces an sret
// pointer that breaks the matrix-mad operand count). Per lane these hold:
//   A (v8s): 8 bf16 bits = A[m][k=k0+lane] for m=0..7   (A column k0+lane)
//   B (v8i): 8 ints VNNI-packed = B[k0+2j..][n=n0+lane] (B column n0+lane)
//   C (v8f): 8 f32 = C[m][n0+lane] for m=0..7
using v8s = short __attribute__((ext_vector_type(8)));
using v8i = int   __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));

SYCL_EXTERNAL v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, v8s MatrixA, v8i MatrixB, v8f MatrixC, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return v8f{}; }   // host stub: never executed, satisfies host link
#endif

// MatrixAPackedBFloat16INTEL | MatrixBPackedBFloat16INTEL; the only legal bf16
// operand for K=16 / f32 result / i16 A / i32 B on this HW.
static constexpr int kBF16Ops = 0x3000;

// Fast f8 e4m3 -> float by constructing the f32 bits directly (no sycl::exp2).
static inline float e4m3_fast(uint8_t b) {
    uint32_t exp = (b >> 3) & 0x0f, mant = b & 0x07;
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    if (exp == 0) return sign * (float)mant * (1.0f / 512.0f);  // subnormal: mant * 2^-9
    uint32_t bits = ((uint32_t)(b & 0x80) << 24) | ((exp - 7 + 127) << 23) | (mant << 20);
    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}
static inline float e2m1_fast(uint8_t bits) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (bits & 8) ? -mag[bits & 7] : mag[bits & 7];
}

static void check_dims(const char* who, int M, int K, int N) {
    if (M % 8 || K % 16 || N % 16)
        throw std::runtime_error(std::string(who) + ": M%8/K%16/N%16 constraint violated");
}

// K-split factor: run KS sub-groups per 8x16 output tile over strided K-slices,
// reducing partial f32 accumulators in SLM, to fill the 2048 HW threads when
// MoE buckets are tiny.
static int ksplit_factor(int M, int K, int N) {
    static int target = [] {
        const char* env = std::getenv("DIFF_NVFP4_DPAS_OCC");
        int v = env ? std::atoi(env) : 2048;
        return v > 0 ? v : 2048;
    }();
    int base_groups = (M / 8) * (N / 16);
    int total_tiles = K / 16;
    int ks = (target + base_groups - 1) / std::max(base_groups, 1);
    return std::clamp(ks, 1, std::min(total_tiles, 32));
}

// Global (per-GPU) dequant LUT: lut[f8_scale_byte*16 + e2m1_nibble] = bf16 bits
// of e2m1(nibble) * e4m3(scale). 256*16 = 8 KB, L1/constant-resident. This
// replaces all per-element e2m1/e4m3/f2bf math in the weight dequant with one
// table load — the microbench showed it ~doubles matmul throughput (the kernel
// is dequant-ALU bound, not bandwidth bound). The weight global scale is applied
// separately at the store, so one LUT serves every weight.
static const uint16_t* dequant_lut(GpuEngine& ctx) {
    static std::vector<GpuBuffer<uint16_t>>* luts =
        new std::vector<GpuBuffer<uint16_t>>(GpuEngine::count());
    auto& buf = (*luts)[ctx.index];
    if (buf.empty()) {
        std::vector<uint16_t> h(256 * 16);
        for (int sb = 0; sb < 256; ++sb)
            for (int nb = 0; nb < 16; ++nb)
                h[sb * 16 + nb] = float_to_bf16(e2m1_fast((uint8_t)nb) * e4m3_fast((uint8_t)sb));
        buf = GpuBuffer<uint16_t>(256 * 16, ctx.queue);
        buf.upload(h.data(), 256 * 16);
    }
    return buf.data();
}

// Lazily build the coalesced/blocked copy of weight_packed for this GPU.
// src [N, K/2] row-major -> dst [n/16][k/16][16 cols][8 bytes] so the 16 lanes
// of a sub-group read 128 contiguous bytes per K-tile (coalesced).
static const uint8_t* ensure_weight_coal(const Nvfp4Linear& W, int K, int N, GpuEngine& ctx) {
    if (!W.weight_coal.empty() && W.weight_coal_gpu == ctx.index)
        return W.weight_coal.data();
    int Khalf = K / 2, Ktiles = K / 16;
    W.weight_coal = GpuBuffer<uint8_t>((size_t)N * Khalf, ctx.queue);
    W.weight_coal_gpu = ctx.index;
    const uint8_t* src = W.weight_packed.data();
    uint8_t* dst = W.weight_coal.data();
    ctx.queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(N, Khalf), [=](sycl::id<2> id) {
            int n = (int)id[0], b = (int)id[1];
            int kt = b / 8, j = b % 8;
            dst[(size_t)(n / 16) * Ktiles * 128 + (size_t)kt * 128 + (n % 16) * 8 + j] =
                src[(size_t)n * Khalf + b];
        });
    });
    ctx.queue.wait();
    return dst;
}

// Dequant FP4 weight column n for the K-tile kt into a VNNI bf16 B operand,
// reading the coalesced layout and using the LUT (one load per nibble).
static inline v8i dequant_b_coal(const uint8_t* wcoal, const uint16_t* lut,
                                 const uint8_t* wscale, int n, int lane, int kt,
                                 int K, int N) {
    const uint16_t* lrow = lut + (size_t)wscale[(size_t)kt * N + n] * 16;
    const uint8_t* row = wcoal + (size_t)(n / 16) * (K / 16) * 128 + (size_t)kt * 128 + lane * 8;
    v8i b;
    for (int j = 0; j < 8; ++j) {
        uint8_t by = row[j];
        b[j] = (int)((uint32_t)lrow[by & 0x0f] | ((uint32_t)lrow[by >> 4] << 16));
    }
    return b;
}

void matmul_nvfp4_dpas_weightonly(GpuEngine& ctx, const bf16* A, int M, int K,
                                  const Nvfp4Linear& W, bf16* C) {
    auto& q = ctx.queue;
    int N = W.out_features;
    if (W.in_features != K) throw std::runtime_error("dpas weightonly: K mismatch");
    check_dims("dpas weightonly", M, K, N);
    const uint8_t* wcoal = ensure_weight_coal(W, K, N, ctx);
    const uint16_t* lut = dequant_lut(ctx);
    const uint8_t* wscale = W.weight_scale.data();
    float inv = 1.0f / W.weight_global_scale;
    int total_tiles = K / 16, KS = ksplit_factor(M, K, N);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm((size_t)KS * 8 * 16, h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)(M / 8) * KS, N),
                                         sycl::range<2>(KS, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int m0 = (int)it.get_group(0) * 8;
                int s = (int)it.get_local_id(0);
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int kt = s; kt < total_tiles; kt += KS) {
                    v8i b = dequant_b_coal(wcoal, lut, wscale, n, lane, kt, K, N);
                    v8s a;
                    int k0 = kt * 16;
                    for (int m = 0; m < 8; ++m)
                        a[m] = (short)A[(size_t)(m0 + m) * K + k0 + lane];
                    c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c, kBF16Ops);
                }
                for (int m = 0; m < 8; ++m) slm[(size_t)(s * 8 + m) * 16 + lane] = c[m];
                it.barrier(sycl::access::fence_space::local_space);
                if (s == 0)
                    for (int m = 0; m < 8; ++m) {
                        float sum = 0;
                        for (int ss = 0; ss < KS; ++ss) sum += slm[(size_t)(ss * 8 + m) * 16 + lane];
                        C[(size_t)(m0 + m) * N + n] = float_to_bf16(sum * inv);
                    }
            });
    });
}

void matmul_nvfp4_dpas_full(GpuEngine& ctx, const uint8_t* A_packed,
                            const uint8_t* A_scale, int M, int K,
                            const Nvfp4Linear& W, bf16* C) {
    auto& q = ctx.queue;
    int N = W.out_features;
    if (W.in_features != K) throw std::runtime_error("dpas full: K mismatch");
    check_dims("dpas full", M, K, N);
    const uint8_t* wcoal = ensure_weight_coal(W, K, N, ctx);
    const uint16_t* lut = dequant_lut(ctx);
    const uint8_t* wscale = W.weight_scale.data();
    float inv = 1.0f / (W.input_global_scale * W.weight_global_scale);
    int halfK = K / 2, total_tiles = K / 16, KS = ksplit_factor(M, K, N);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm((size_t)KS * 8 * 16, h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)(M / 8) * KS, N),
                                         sycl::range<2>(KS, 16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                int m0 = (int)it.get_group(0) * 8;
                int s = (int)it.get_local_id(0);
                int lane = (int)it.get_local_id(1);
                int n = (int)it.get_group(1) * 16 + lane;
                v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int kt = s; kt < total_tiles; kt += KS) {
                    v8i b = dequant_b_coal(wcoal, lut, wscale, n, lane, kt, K, N);
                    int k0 = kt * 16, kk = k0 + lane;
                    v8s a;
                    for (int m = 0; m < 8; ++m) {
                        uint8_t byte = A_packed[(size_t)(m0 + m) * halfK + kk / 2];
                        uint8_t nibv = (kk & 1) ? (byte >> 4) : (byte & 0x0f);
                        a[m] = (short)float_to_bf16(e2m1_fast(nibv) *
                                   e4m3_fast(A_scale[(size_t)(m0 + m) * total_tiles + kt]));
                    }
                    c = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c, kBF16Ops);
                }
                for (int m = 0; m < 8; ++m) slm[(size_t)(s * 8 + m) * 16 + lane] = c[m];
                it.barrier(sycl::access::fence_space::local_space);
                if (s == 0)
                    for (int m = 0; m < 8; ++m) {
                        float sum = 0;
                        for (int ss = 0; ss < KS; ++ss) sum += slm[(size_t)(ss * 8 + m) * 16 + lane];
                        C[(size_t)(m0 + m) * N + n] = float_to_bf16(sum * inv);
                    }
            });
    });
}

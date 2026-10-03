// Prototype + operand-sweep for the Intel sub-group matrix-mad builtin
// (OpSubgroupMatrixMultiplyAccumulateINTEL) on Battlemage G31, where SYCL
// joint_matrix is runtime-blocked (aspect::ext_intel_matrix == 0).
//
// Shape: bf16 A[M=8,K=16] * bf16 B[K=16,N=16] -> f32 C[8,16], subgroup size 16.
// Per lane l (= output column n): A column k=l across 8 rows -> short8;
// B column n=l across 16 k, VNNI 2-per-int -> int8; result C col n=l -> float8.
//
// We don't know the exact bf16 "Operands" literal, so sweep candidates and
// report which reproduces the host reference. Build:
//   icpx -fsycl -O2 -fsycl-targets=intel_gpu_bmg_g31 tools/dpas_mad_probe.cpp -o dpas_mad_probe
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <type_traits>

// Native Clang vector types (NOT sycl::vec, which is a struct and would force
// an sret pointer, breaking the intrinsic's operand count).
using v8s = short __attribute__((ext_vector_type(8)));
using v8i = int   __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));

// SPIR-V INTEL extension op. The translator matches the demangled name.
// Operands must be a compile-time constant 32-bit int (bf16 type bits).
SYCL_EXTERNAL v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, v8s MatrixA, v8i MatrixB, v8f MatrixC, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return v8f{}; }  // host stub: never executed, satisfies host link
#endif

static inline uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t bias = ((u >> 16) & 1) + 0x7FFFu;
    return (uint16_t)((u + bias) >> 16);
}

constexpr int M = 8, N = 16, K = 16, SG = 16;

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<float> Aref(M * K), Bref(K * N);
    for (int i = 0; i < M * K; ++i) Aref[i] = ((i * 5) % 11) * 0.25f - 1.0f;
    for (int i = 0; i < K * N; ++i) Bref[i] = ((i * 7) % 9) * 0.5f - 2.0f;

    // Per-lane operands: a_col[l*8 + m] = bf16(A[m][l]); b_col[l*8 + j] packs
    // B[2j][l] (low) and B[2j+1][l] (high); c_out[l*8 + m] = C[m][l].
    short* a_dev = sycl::malloc_shared<short>(SG * 8, q);
    int*   b_dev = sycl::malloc_shared<int>(SG * 8, q);
    float* c_dev = sycl::malloc_shared<float>(SG * 8, q);
    for (int l = 0; l < SG; ++l) {
        for (int m = 0; m < M; ++m)
            a_dev[l * 8 + m] = (short)f2bf(Aref[m * K + l]);   // A[m][k=l]
        for (int j = 0; j < 8; ++j) {
            uint16_t lo = f2bf(Bref[(2 * j) * N + l]);          // B[2j][n=l]
            uint16_t hi = f2bf(Bref[(2 * j + 1) * N + l]);      // B[2j+1][n=l]
            b_dev[l * 8 + j] = (int)((uint32_t)lo | ((uint32_t)hi << 16));
        }
    }

    std::vector<float> Cref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0;
            for (int k = 0; k < K; ++k) acc += Aref[m * K + k] * Bref[k * N + n];
            Cref[m * N + n] = acc;
        }

    auto run = [&](auto OPc) {
        constexpr int OP = decltype(OPc)::value;
        std::memset(c_dev, 0, SG * 8 * sizeof(float));
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(SG, SG),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    int l = it.get_local_id(0);
                    v8s a; v8i b; v8f c = {0,0,0,0,0,0,0,0};
                    for (int i = 0; i < 8; ++i) { a[i] = a_dev[l * 8 + i]; b[i] = b_dev[l * 8 + i]; }
                    v8f r = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(K, a, b, c, OP);
                    for (int i = 0; i < 8; ++i) c_dev[l * 8 + i] = r[i];
                });
        }).wait();
        double max_err = 0;
        for (int l = 0; l < N; ++l)
            for (int m = 0; m < M; ++m)
                max_err = std::max(max_err, (double)std::fabs(Cref[m * N + l] - c_dev[l * 8 + m]));
        std::printf("operand=0x%-5x max_err=%g  %s\n", OP, max_err,
                    max_err < 0.5 ? "<-- MATCH" : "");
    };
    // Compiler confirmed: for K=16, f32 result, i16 A, i32 B, the only valid
    // bf16 operand is 0x3000 = MatrixAPackedBFloat16INTEL|MatrixBPackedBFloat16INTEL.
    run(std::integral_constant<int, 0x3000>{});
    sycl::free(a_dev, q); sycl::free(b_dev, q); sycl::free(c_dev, q);
    return 0;
}

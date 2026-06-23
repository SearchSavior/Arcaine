// Empirically determine a working joint_matrix bf16 tile shape on this GPU.
// Computes C[TM,TN] = A[TM,TK] * B[TK,TN] for a known input and checks the
// result on host. B is staged in VNNI/ext_intel_packed layout as the bf16
// use::b matrix requires.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

namespace jm = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int TM = 8, TN = 16, TK = 16;
constexpr int SG = 16;  // Intel sub-group size

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // A: TM x TK row-major.  B: TK x TN, staged VNNI-packed as [TK/2][TN][2].
    bf16* A = sycl::malloc_shared<bf16>(TM * TK, q);
    bf16* Bv = sycl::malloc_shared<bf16>(TK * TN, q);   // packed buffer, same count
    float* C = sycl::malloc_shared<float>(TM * TN, q);

    std::vector<float> Aref(TM * TK), Bref(TK * TN);
    for (int i = 0; i < TM * TK; ++i) { Aref[i] = (i % 7) * 0.5f - 1.0f; A[i] = bf16(Aref[i]); }
    for (int i = 0; i < TK * TN; ++i) Bref[i] = ((i * 3) % 5) * 0.25f - 0.5f;
    // VNNI pack: Bv[(k/2)*TN*2 + n*2 + (k%2)] = B[k][n]
    for (int k = 0; k < TK; ++k)
        for (int n = 0; n < TN; ++n)
            Bv[(k / 2) * TN * 2 + n * 2 + (k % 2)] = bf16(Bref[k * TN + n]);
    std::memset(C, 0, TM * TN * sizeof(float));

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(SG), sycl::range<1>(SG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, TM, TK,
                                 jm::layout::row_major> ma;
                jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, TK, TN,
                                 jm::layout::ext_intel_packed> mb;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, TN> mc;
                jm::joint_matrix_fill(sg, mc, 0.0f);
                jm::joint_matrix_load(sg, ma,
                    sycl::multi_ptr<bf16, sycl::access::address_space::global_space>(A), TK);
                jm::joint_matrix_load(sg, mb,
                    sycl::multi_ptr<bf16, sycl::access::address_space::global_space>(Bv), TN * 2);
                jm::joint_matrix_mad(sg, mc, ma, mb, mc);
                jm::joint_matrix_store(sg, mc,
                    sycl::multi_ptr<float, sycl::access::address_space::global_space>(C),
                    TN, jm::layout::row_major);
            });
    }).wait();

    double max_err = 0;
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n) {
            float ref = 0;
            for (int k = 0; k < TK; ++k) ref += Aref[m * TK + k] * Bref[k * TN + n];
            max_err = std::max(max_err, (double)std::fabs(ref - C[m * TN + n]));
        }
    std::printf("TM=%d TN=%d TK=%d  max_err=%g  %s\n", TM, TN, TK, max_err,
                max_err < 0.05 ? "PASS" : "FAIL");
    sycl::free(A, q); sycl::free(Bv, q); sycl::free(C, q);
    return 0;
}

#pragma once

#include "../weights.hpp"

namespace flint::awq_joint_matrix {

// Native joint_matrix W4A16 prefill. The selector is independent of the CuTe
// path: ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas or dpas-sg32.
enum class Mode { disabled, dpas, dpas_sg32 };
Mode mode();
// ARCAINE_FLINT_AWQ_JOINT_MATRIX_TILE=16x32, 32x64 or 64x64.
// Defaults: 32x64 at SG16, 64x64 at SG32; 16x32 retains the original kernel.
enum class Tile { m16n32, m32n64, m64n64 };
Tile tile();
const char* tile_name();
// Packed staging is the default. STAGE=scalar retains the previous tiled kernel.
enum class Staging { scalar, packed };
Staging staging();
const char* staging_name();
// GLOBAL_A=0 disables the default global-load specialization. It requires
// full tiles, no input gather, and a 64-byte-aligned A; other calls stage A.
bool global_a_enabled();
bool supported(const sycl::device& device, Mode selected);

// False means no work was submitted. A/C aliasing is rejected;
// residual==C is supported. Row maps must contain valid source/destination
// indices, with unique destinations, as required by the vector prefill path.
// On success, completion optionally receives the actual kernel submission event.
bool gemm(sycl::queue& q, const QTensor& t, const bf16* A,
          const int* row_index, bf16* C, int M, int N,
          const bf16* residual = nullptr, const int* out_index = nullptr,
          int out_stride = 0, int expert = 0, sycl::event* completion = nullptr);

} // namespace flint::awq_joint_matrix

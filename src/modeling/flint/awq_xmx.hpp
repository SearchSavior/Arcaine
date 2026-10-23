#pragma once

#include "weights.hpp"

namespace flint::awq_xmx {

bool gemm(sycl::queue& q, const QTensor& t, const bf16* A,
          const int* row_index, bf16* C, int M, int N,
          const bf16* residual = nullptr, const int* out_index = nullptr,
          int out_stride = 0, int expert = 0);

}

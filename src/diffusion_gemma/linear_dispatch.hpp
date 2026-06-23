#pragma once
// Dispatch a DiffLinearWeight matmul to the BF16, NVFP4, or int4 backend.
// Shared by attention and the dense MLP so each projection can be quantized
// independently of the others.
#include <stdexcept>
#include "weights.hpp"
#include "../common/gpu/ops.hpp"
#include "../common/gpu/nvfp4.hpp"
#include "../common/gpu/int4.hpp"
#include "../common/gpu/q8_0.hpp"

// C (M,N) = A (M,K) @ W^T, where W is (N=out_features, K=in_features).
inline void matmul_linear_weight(const bf16* A, int M, int K,
                                 const DiffLinearWeight& W, int N,
                                 bf16* C, GpuEngine& ctx) {
    if (W.is_int4()) {
        if (W.int4.out_features != N || W.int4.in_features != K)
            throw std::runtime_error("int4 linear shape mismatch");
        matmul_int4(A, M, K, W.int4, C, ctx);
    } else if (W.is_q8()) {
        if (W.q8.out_features != N || W.q8.in_features != K)
            throw std::runtime_error("Q8_0 linear shape mismatch");
        matmul_q8_0(A, M, K, W.q8, C, ctx);
    } else if (W.nvfp4) {
        if (W.fp4.out_features != N || W.fp4.in_features != K)
            throw std::runtime_error("NVFP4 linear shape mismatch");
        matmul_nvfp4(A, M, K, W.fp4, C, ctx);
    } else {
        matmul_bf16(A, M, K, W.bf16.data(), N, C, ctx);
    }
}

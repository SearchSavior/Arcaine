#pragma once
#include <sycl/sycl.hpp>
#include "buffer.hpp"
#include "nvfp4.hpp"

// Tiled DPAS NVFP4 expert matmul for Battlemage (Xe2/XMX). Reaches the systolic
// array via the SPIR-V Intel sub-group matrix-mad builtin, because portable
// SYCL joint_matrix is runtime-blocked on this stack (aspect::ext_intel_matrix
// is 0). See docs/diffusion_gemma/NVFP4_DPAS_KERNEL.md for the full derivation.
//
// Both entries compute C[M,N] = A @ dequant(W) with f32 accumulation, weight
// FP4 dequantized to bf16 on chip. Constraints: M % 8 == 0, K % 16 == 0,
// N % 16 == 0 (callers pad the bucket M up to a multiple of 8; model K/N are
// already multiples of 16). The implementation TU must be built AOT for
// intel_gpu_bmg_g31 with SPV_INTEL_subgroup_matrix_multiply_accumulate enabled.

// Weight-only: activations stay bf16 (no activation quant). Divides by the
// weight global scale only. Uses a coalesced weight layout + a (scale,nibble)
// dequant LUT (built lazily per GPU); ctx provides the queue and device index.
void matmul_nvfp4_dpas_weightonly(GpuEngine& ctx, const bf16* A, int M, int K,
                                  const Nvfp4Linear& W, bf16* C);

// Full NVFP4: activations are packed FP4 + per-(row,16) f8 scales, matching the
// oneDNN path. Divides by input_global * weight_global.
void matmul_nvfp4_dpas_full(GpuEngine& ctx, const uint8_t* A_packed,
                            const uint8_t* A_scale, int M, int K,
                            const Nvfp4Linear& W, bf16* C);

// Note: a fused gate/up+GeGLU kernel was tried and measured ~15% slower than the
// separate path (dual-accumulator register/SLM pressure outweighs the ~1%
// intermediate-traffic saving). Not kept. See NVFP4_DPAS_KERNEL.md.

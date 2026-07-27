#pragma once

#include <cstdint>

namespace arcaine::diffusion_gemma {

// Kernel-role manifest for explicit ownership + navigation. Maps block-
// diffusion model phases to the implementation/benchmark locations under
// kernels/. For navigation only; it does not force kernels through a dynamic
// generic interface.
enum class KernelRole : std::uint16_t {
    TokenEmbedding,
    EncoderAttention,
    DenoiserAttention,
    MoeRouter,
    MoeExpertGemm,
    DiffusionDenoise,
    Logits,
    Entropy,
    Acceptance,
    LogitSampling,
};

const char* kernel_role_name(KernelRole role);

}  // namespace arcaine::diffusion_gemma

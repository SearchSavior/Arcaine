#include "manifest.hpp"

namespace arcaine::diffusion_gemma {

const char* kernel_role_name(KernelRole role) {
    switch (role) {
        case KernelRole::TokenEmbedding:    return "token_embedding";
        case KernelRole::EncoderAttention:  return "encoder_attention";
        case KernelRole::DenoiserAttention: return "denoiser_attention";
        case KernelRole::MoeRouter:         return "moe_router";
        case KernelRole::MoeExpertGemm:     return "moe_expert_gemm";
        case KernelRole::DiffusionDenoise:  return "diffusion_denoise";
        case KernelRole::Logits:            return "logits";
        case KernelRole::Entropy:           return "entropy";
        case KernelRole::Acceptance:        return "acceptance";
        case KernelRole::LogitSampling:     return "logit_sampling";
    }
    return "unknown";
}

}  // namespace arcaine::diffusion_gemma

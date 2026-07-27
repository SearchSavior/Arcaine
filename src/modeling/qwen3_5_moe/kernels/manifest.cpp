#include "manifest.hpp"

namespace arcaine::qwen3_5_moe {

const char* kernel_role_name(KernelRole role) {
    switch (role) {
        case KernelRole::TokenEmbedding:   return "token_embedding";
        case KernelRole::AttentionPrefill: return "attention_prefill";
        case KernelRole::AttentionDecode:  return "attention_decode";
        case KernelRole::DeltaNetPrefill:   return "deltanet_prefill";
        case KernelRole::DeltaNetDecode:    return "deltanet_decode";
        case KernelRole::MoeRouter:         return "moe_router";
        case KernelRole::MoeExpertGemm:     return "moe_expert_gemm";
        case KernelRole::DenseMlp:          return "dense_mlp";
        case KernelRole::OutputProjection:  return "output_projection";
        case KernelRole::LogitSampling:      return "logit_sampling";
    }
    return "unknown";
}

}  // namespace arcaine::qwen3_5_moe

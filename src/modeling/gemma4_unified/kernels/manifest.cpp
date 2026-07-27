#include "manifest.hpp"

namespace arcaine::gemma4_unified {

const char* kernel_role_name(KernelRole role) {
    switch (role) {
        case KernelRole::TokenEmbedding:    return "token_embedding";
        case KernelRole::AttentionPrefill:  return "attention_prefill";
        case KernelRole::AttentionDecode:   return "attention_decode";
        case KernelRole::DenseMlp:           return "dense_mlp";
        case KernelRole::OutputProjection:  return "output_projection";
        case KernelRole::LogitSampling:     return "logit_sampling";
    }
    return "unknown";
}

}  // namespace arcaine::gemma4_unified

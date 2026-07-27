#pragma once

#include <cstdint>

namespace arcaine::qwen3_5 {

// Kernel-role manifest for explicit ownership + navigation. Maps model phases
// to the implementation/benchmark locations under kernels/. For navigation
// only; it does not force kernels through a dynamic generic interface.
enum class KernelRole : std::uint16_t {
    TokenEmbedding,
    AttentionPrefill,
    AttentionDecode,
    DeltaNetPrefill,
    DeltaNetDecode,
    DenseMlp,
    OutputProjection,
    LogitSampling,
};

const char* kernel_role_name(KernelRole role);

}  // namespace arcaine::qwen3_5

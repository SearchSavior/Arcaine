#pragma once

#include <cstdint>
#include <vector>

namespace arcaine::gemma4_unified {

// Kernel-role manifest for explicit ownership + navigation. Maps model phases
// to the implementation/benchmark locations under kernels/. This is for
// navigation only; it does not force kernels through a dynamic generic
// interface.
enum class KernelRole : std::uint16_t {
    TokenEmbedding,
    AttentionPrefill,
    AttentionDecode,
    DenseMlp,
    OutputProjection,
    LogitSampling,
};

const char* kernel_role_name(KernelRole role);

}  // namespace arcaine::gemma4_unified

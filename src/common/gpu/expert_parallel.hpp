#pragma once
#include "../../modeling/diffusion_gemma/weights.hpp"
#include "engine.hpp"
#include "buffer.hpp"
#include <vector>

// NVFP4 expert-kernel selection. Default comes from DIFF_NVFP4_EXPERT_KERNEL,
// but the benchmark tool overrides it at runtime via set_nvfp4_kernel() so it
// can sweep kernels in one process (no model reload). (A DPAS/XMX kernel was
// also explored and removed; see docs/artifacts/.)
enum class Nvfp4Kernel { Default, Hybrid, Custom };
void set_nvfp4_kernel(Nvfp4Kernel kernel);
const char* nvfp4_kernel_name(Nvfp4Kernel kernel);

struct ExpertProfileLabels {
    const char* setup_gather;
    const char* pack;
    const char* gateup_mm;
    const char* geglu_pack;
    const char* down_mm;
    const char* combine;
};

inline const ExpertProfileLabels& expert_profile_labels(bool is_encoder, bool is_full) {
    static constexpr ExpertProfileLabels labels[4] = {
        {
            "exp.enc.sliding.setup+gather",
            "exp.enc.sliding.pack",
            "exp.enc.sliding.gateup_mm",
            "exp.enc.sliding.geglu+pack",
            "exp.enc.sliding.down_mm",
            "exp.enc.sliding.combine",
        },
        {
            "exp.enc.full.setup+gather",
            "exp.enc.full.pack",
            "exp.enc.full.gateup_mm",
            "exp.enc.full.geglu+pack",
            "exp.enc.full.down_mm",
            "exp.enc.full.combine",
        },
        {
            "exp.dec.sliding.setup+gather",
            "exp.dec.sliding.pack",
            "exp.dec.sliding.gateup_mm",
            "exp.dec.sliding.geglu+pack",
            "exp.dec.sliding.down_mm",
            "exp.dec.sliding.combine",
        },
        {
            "exp.dec.full.setup+gather",
            "exp.dec.full.pack",
            "exp.dec.full.gateup_mm",
            "exp.dec.full.geglu+pack",
            "exp.dec.full.down_mm",
            "exp.dec.full.combine",
        },
    };
    return labels[(is_encoder ? 0 : 2) + (is_full ? 1 : 0)];
}

void expert_parallel_forward(
    GpuEngine& owner,
    const DiffMoE& moe,
    const bf16* expert_in,
    bf16* out,
    int seq, int H, int E, int top_k, int moe_intermediate,
    const std::vector<int>& idx,
    const std::vector<float>& weight,
    const ExpertProfileLabels& prof);

void expert_parallel_forward(
    GpuEngine& owner,
    const DiffMoE& moe,
    const bf16* expert_in,
    bf16* out,
    int seq, int H, int E, int top_k, int moe_intermediate,
    const int* idx_dev,
    const float* weight_dev,
    const ExpertProfileLabels& prof);

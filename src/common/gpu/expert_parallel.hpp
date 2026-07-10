#pragma once
#include "../../modeling/diffusion_gemma/weights.hpp"
#include "engine.hpp"
#include "buffer.hpp"
#include <vector>

// NVFP4 expert-kernel selection. Default comes from DIFF_NVFP4_EXPERT_KERNEL,
// but the benchmark tool overrides it at runtime via set_nvfp4_kernel() so it
// can sweep kernels in one process (no model reload). (A DPAS/XMX kernel was
// also explored and removed; see docs/artifacts/.)
enum class Nvfp4Kernel { Default, Hybrid, Custom, GroupedDpas };
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

// Build (idempotent) the persistent raw-weight pointer tables for an NVFP4
// expert shard -- the per-expert device pointers + input_global_scales the
// grouped-GEMM kernels read. Call once at load (outside any session); the
// tables are then reused every denoising step without a host->device upload.
// No-op for non-NVFP4 shards. See DiffExpertShard::pt_*.
void ensure_expert_pointer_tables_raw(DiffExpertShard& shard, GpuEngine& ctx);

// Build the persistent COALESCED (xe2 DPAS) weight pointer tables for an NVFP4
// shard, once at load. Pre-warms nvfp4_dequant_lut + nvfp4_coalesced_weight per
// expert (which cache on first call and return stable USM ptrs thereafter), then
// uploads the coalesced weight ptr table to pt_gate_w_coal/pt_down_w_coal. The
// scale/dst/input tables (pt_gate_s etc.) are shared with the raw tables and are
// NOT rebuilt. This eliminates the xe2 path's per-step pointer-table upload +
// the lazy nvfp4_coalesced_weight ctx.queue.wait(), making the xe2 DPAS path
// SYCL-graph-capturable. Requires the raw tables to be built first. No-op for
// non-NVFP4 shards. See DiffExpertShard::pt_*_coal.
void ensure_expert_pointer_tables_coalesced(DiffExpertShard& shard, GpuEngine& ctx,
                                             int moe_intermediate, int hidden_size);

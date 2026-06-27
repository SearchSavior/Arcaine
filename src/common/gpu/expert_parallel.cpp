#include "expert_parallel.hpp"
#include "../../modeling/diffusion_gemma/arena.hpp"
#include "ops.hpp"
#include "../kernels/elementwise.hpp"
#include "../../utils/profile.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace {

static constexpr int kTailCap = 64;
static int round_up(int v, int m) { return (v + m - 1) / m * m; }

// Tail capacity for the batched cold-expert GEMM.  Cold experts each occupy T
// padded rows; experts hotter than T spill to exact-size GEMMs.  Larger T moves
// more experts into the (padded) batched path; smaller T tightens padding waste
// but pushes more experts onto the per-expert exact path.  Hot-path tuning knob
// for the MoE load distribution (inspect with DIFF_MOE_STATS).
static int moe_tail_cap() {
    static int value = [] {
        const char* env = std::getenv("DIFF_MOE_TAIL_CAP");
        int v = env ? std::atoi(env) : kTailCap;
        return v > 0 ? v : kTailCap;
    }();
    return value;
}

// Expert activations are carved from the per-GPU liveness arena (see arena.hpp).
// run_shard allocates ~16 temporaries per layer per pass; the arena hands them
// out from a pre-sized device chunk (no per-call sycl::malloc churn — the same
// motivation as the old pool) but also frees each the moment its last consumer
// runs, so the Xe→gu→act→Ye pipeline reuses storage instead of holding all four
// live at once.
static int nvfp4_expert_round_multiple() {
    static int value = [] {
        const char* env = std::getenv("DIFF_NVFP4_EXPERT_ROUND");
        if (!env) return 8;
        int v = std::atoi(env);
        if (v == 1 || v == 8 || v == 16 || v == 32 || v == 64)
            return v;
        return 8;
    }();
    return value;
}

// Runtime-settable expert-kernel selection. Initialized lazily from the
// documented DIFF_NVFP4_EXPERT_KERNEL env var, overridable by set_nvfp4_kernel()
// for in-process benchmarking.
static Nvfp4Kernel& g_kernel() {
    static Nvfp4Kernel k = [] {
        const char* env = std::getenv("DIFF_NVFP4_EXPERT_KERNEL");
        if (!env) return Nvfp4Kernel::Default;
        if (!std::strcmp(env, "hybrid") || !std::strcmp(env, "grouped-pack") ||
            !std::strcmp(env, "pack")) return Nvfp4Kernel::Hybrid;
        if (!std::strcmp(env, "custom") || !std::strcmp(env, "grouped") ||
            !std::strcmp(env, "1")) return Nvfp4Kernel::Custom;
        return Nvfp4Kernel::Default;
    }();
    return k;
}

static bool nvfp4_use_custom_expert_kernel() { return g_kernel() == Nvfp4Kernel::Custom; }
static bool nvfp4_use_hybrid_expert_kernel() { return g_kernel() == Nvfp4Kernel::Hybrid; }

static bool nvfp4_gpu_layout_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_NVFP4_GPU_LAYOUT");
        if (!env) return false;
        if (!std::strcmp(env, "0") || !std::strcmp(env, "off") ||
            !std::strcmp(env, "false") || !std::strcmp(env, "no"))
            return false;
        return !std::strcmp(env, "1") || !std::strcmp(env, "on") ||
               !std::strcmp(env, "true") || !std::strcmp(env, "gpu");
    }();
    return enabled;
}

static int nvfp4_gpu_layout_max_seq() {
    static int value = [] {
        const char* env = std::getenv("DIFF_NVFP4_GPU_LAYOUT_MAX_SEQ");
        if (!env) return 64;
        int v = std::atoi(env);
        return v > 0 ? v : 64;
    }();
    return value;
}

static bool nvfp4_grouped_gemm_xe2_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_NVFP4_GROUPED_GEMM");
        if (!env) return false;
        return !std::strcmp(env, "xe2") || !std::strcmp(env, "dpas") ||
               !std::strcmp(env, "xmx") || !std::strcmp(env, "1");
    }();
    return enabled;
}

static bool q8_use_hybrid_expert_kernel() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_Q8_EXPERT_KERNEL");
        if (!env) return false;
        if (!std::strcmp(env, "hybrid") || !std::strcmp(env, "batched") ||
            !std::strcmp(env, "1") || !std::strcmp(env, "on"))
            return true;
        if (!std::strcmp(env, "simple") || !std::strcmp(env, "exact") ||
            !std::strcmp(env, "default") || !std::strcmp(env, "0") ||
            !std::strcmp(env, "off"))
            return false;
        return false;
    }();
    return enabled;
}

static bool q8_use_fused_expert_geglu() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_Q8_FUSED_EXPERT_GEGLU");
        return env && (!std::strcmp(env, "1") || !std::strcmp(env, "on") ||
                       !std::strcmp(env, "true") || !std::strcmp(env, "fused"));
    }();
    return enabled;
}

static bool q8_use_grouped_expert_kernel() {
    static bool enabled = [] {
        const char* env = std::getenv("DIFF_Q8_GROUPED_EXPERT");
        if (!env) return true;
        if (!std::strcmp(env, "0") || !std::strcmp(env, "off") ||
            !std::strcmp(env, "false") || !std::strcmp(env, "simple"))
            return false;
        return !std::strcmp(env, "1") || !std::strcmp(env, "on") ||
               !std::strcmp(env, "true") || !std::strcmp(env, "grouped");
    }();
    return enabled;
}

} // namespace

void set_nvfp4_kernel(Nvfp4Kernel kernel) { g_kernel() = kernel; }
const char* nvfp4_kernel_name(Nvfp4Kernel kernel) {
    switch (kernel) {
        case Nvfp4Kernel::Hybrid: return "hybrid";
        case Nvfp4Kernel::Custom: return "custom";
        default:                  return "default";
    }
}

namespace {

static void geglu_strided_grouped(sycl::queue& q,
                                  const bf16* gate_up,
                                  bf16* gate_act,
                                  const int32_t* row_slot,
                                  int rows,
                                  int inter) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEF = 0.044715f;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)rows, (size_t)inter), [=](sycl::id<2> id) {
            int r = (int)id[0];
            int dim = (int)id[1];
            int slot = row_slot[r];
            const bf16* row = gate_up + (size_t)slot * 2 * inter;
            float g = bf16_to_float(row[dim]);
            float u = bf16_to_float(row[inter + dim]);
            float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
            gate_act[(size_t)slot * inter + dim] =
                float_to_bf16(0.5f * g * (1.0f + sycl::tanh(inner)) * u);
        });
    });
}

static void geglu_strided_counts(sycl::queue& q,
                                 const bf16* gate_up,
                                 bf16* gate_act,
                                 const int32_t* row_expert,
                                 int max_rows,
                                 int inter) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEF = 0.044715f;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)max_rows, (size_t)inter), [=](sycl::id<2> id) {
            int row = (int)id[0];
            int dim = (int)id[1];
            if (row_expert[row] < 0) return;

            const bf16* src = gate_up + (size_t)row * 2 * inter;
            float g = bf16_to_float(src[dim]);
            float u = bf16_to_float(src[inter + dim]);
            float inner = SQRT_2_OVER_PI * (g + COEF * g * g * g);
            gate_act[(size_t)row * inter + dim] =
                float_to_bf16(0.5f * g * (1.0f + sycl::tanh(inner)) * u);
        });
    });
}

static void transfer(sycl::queue& src_q, const bf16* src,
                     sycl::queue& dst_q, bf16* dst, size_t n) {
    std::vector<bf16> stage(n);
    src_q.memcpy(stage.data(), src, n * sizeof(bf16)).wait();
    dst_q.memcpy(dst, stage.data(), n * sizeof(bf16)).wait();
}

template <class T>
static void transfer_t(sycl::queue& src_q, const T* src,
                       sycl::queue& dst_q, T* dst, size_t n) {
    std::vector<T> stage(n);
    src_q.memcpy(stage.data(), src, n * sizeof(T)).wait();
    dst_q.memcpy(dst, stage.data(), n * sizeof(T)).wait();
}

template <class T>
static void upload_alloc(sycl::queue& q, diffarena::Alloc<T>& dst,
                         const std::vector<T>& src) {
    if (!src.empty())
        q.memcpy(dst.data(), src.data(), src.size() * sizeof(T)).wait();
}

static bool run_shard_nvfp4_gpu_layout(
    GpuEngine& ctx,
    const DiffExpertShard& shard,
    const bf16* expert_in,
    bf16* out,
    int seq, int H, int top_k, int moe_inter,
    const ExpertProfileLabels& prof,
    const int* idx_dev,
    const float* weight_dev)
{
    auto& q = ctx.queue;
    auto& ar = diffarena::arena(ctx.index);
    int localE = shard.num_experts;
    int A_all = seq * top_k;
    if (!idx_dev || !weight_dev || !shard.nvfp4 || A_all <= 0)
        return false;
    if ((int)shard.gate_up_proj_fp4.size() != localE ||
        (int)shard.down_proj_fp4.size() != localE)
        throw std::runtime_error("NVFP4 expert shard is missing fused projection weights");

    q.memset(out, 0, (size_t)seq * H * sizeof(bf16));

    auto t_setup = diffprof::tic(q);
    auto rows_per_expert_dev = ar.alloc<int32_t>(localE);
    auto expert_offsets_dev = ar.alloc<int32_t>(localE);
    auto unpermuted_to_permuted_dev = ar.alloc<int32_t>(A_all);
    auto row_expert_dev = ar.alloc<int32_t>(A_all);
    auto slot_dev = ar.alloc<int32_t>(A_all);
    auto w_dev = ar.alloc<float>(A_all);
    auto Xe = ar.alloc<bf16>((size_t)A_all * H);

    q.memset(rows_per_expert_dev.data(), 0, (size_t)localE * sizeof(int32_t));
    q.memset(unpermuted_to_permuted_dev.data(), 0xFF, (size_t)A_all * sizeof(int32_t));
    q.memset(row_expert_dev.data(), 0xFF, (size_t)A_all * sizeof(int32_t));
    q.memset(slot_dev.data(), 0xFF, (size_t)A_all * sizeof(int32_t));
    q.memset(w_dev.data(), 0, (size_t)A_all * sizeof(float));

    int first = shard.first_expert;
    int32_t* rows_per_expert = rows_per_expert_dev.data();
    int32_t* unpermuted_to_permuted = unpermuted_to_permuted_dev.data();
    constexpr int kRemapWG = 256;
    size_t groups = ((size_t)A_all + kRemapWG - 1) / kRemapWG;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> local_counts(sycl::range<1>(localE), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * kRemapWG),
                              sycl::range<1>(kRemapWG)),
            [=](sycl::nd_item<1> it) {
                int lid = (int)it.get_local_id(0);
                int lsize = (int)it.get_local_range(0);
                int a = (int)it.get_global_id(0);
                for (int e = lid; e < localE; e += lsize)
                    local_counts[e] = 0;
                it.barrier(sycl::access::fence_space::local_space);

                int local_e = -1;
                if (a < A_all) {
                    local_e = idx_dev[a] - first;
                    if (local_e >= 0 && local_e < localE) {
                        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            local_count(local_counts[local_e]);
                        unpermuted_to_permuted[a] = local_count.fetch_add(1);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int e = lid; e < localE; e += lsize) {
                    int count = local_counts[e];
                    if (count > 0) {
                        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            global_count(rows_per_expert[e]);
                        local_counts[e] = global_count.fetch_add(count);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                if (a < A_all && local_e >= 0 && local_e < localE)
                    unpermuted_to_permuted[a] += local_counts[local_e];
            });
    });

    int32_t* expert_offsets = expert_offsets_dev.data();
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            int running = 0;
            for (int e = 0; e < localE; ++e) {
                expert_offsets[e] = running;
                running += rows_per_expert[e];
            }
        });
    });

    const int32_t* offsets = expert_offsets_dev.data();
    const int32_t* permuted = unpermuted_to_permuted_dev.data();
    int32_t* row_expert = row_expert_dev.data();
    int32_t* slot_ptr = slot_dev.data();
    float* w_ptr = w_dev.data();
    bf16* xe = Xe.data();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)A_all, (size_t)H), [=](sycl::id<2> id) {
            int a = (int)id[0];
            int d = (int)id[1];
            int e = idx_dev[a] - first;
            bool local = e >= 0 && e < localE;
            int slot = local ? offsets[e] + permuted[a] : -1;
            if (d == 0) {
                slot_ptr[a] = slot;
                w_ptr[a] = local ? weight_dev[a] : 0.0f;
                if (slot >= 0) row_expert[slot] = e;
            }
            if (slot >= 0)
                xe[(size_t)slot * H + d] = expert_in[(size_t)(a / top_k) * H + d];
        });
    });
    diffprof::toc(q, prof.setup_gather, t_setup);

    int HG = H / 16;
    int IG = moe_inter / 16;
    bool use_xe2_gemm = nvfp4_grouped_gemm_xe2_enabled();
    std::vector<const uint8_t*> gate_w(localE), gate_s(localE), down_w(localE), down_s(localE);
    std::vector<const float*> gate_dst(localE), down_dst(localE);
    std::vector<float> gate_input(localE), down_input(localE);
    for (int e = 0; e < localE; ++e) {
        gate_w[e] = use_xe2_gemm
            ? nvfp4_coalesced_weight(shard.gate_up_proj_fp4[e], H, 2 * moe_inter, ctx)
            : shard.gate_up_proj_fp4[e].weight_packed.data();
        gate_s[e] = shard.gate_up_proj_fp4[e].weight_scale.data();
        gate_dst[e] = shard.gate_up_proj_fp4[e].dst_scale.data();
        gate_input[e] = shard.gate_up_proj_fp4[e].input_global_scale;
        down_w[e] = use_xe2_gemm
            ? nvfp4_coalesced_weight(shard.down_proj_fp4[e], moe_inter, H, ctx)
            : shard.down_proj_fp4[e].weight_packed.data();
        down_s[e] = shard.down_proj_fp4[e].weight_scale.data();
        down_dst[e] = shard.down_proj_fp4[e].dst_scale.data();
        down_input[e] = shard.down_proj_fp4[e].input_global_scale;
    }

    auto gate_w_dev = ar.alloc<const uint8_t*>(localE);
    auto gate_s_dev = ar.alloc<const uint8_t*>(localE);
    auto down_w_dev = ar.alloc<const uint8_t*>(localE);
    auto down_s_dev = ar.alloc<const uint8_t*>(localE);
    auto gate_dst_dev = ar.alloc<const float*>(localE);
    auto down_dst_dev = ar.alloc<const float*>(localE);
    auto gate_input_dev = ar.alloc<float>(localE);
    auto down_input_dev = ar.alloc<float>(localE);
    upload_alloc(q, gate_w_dev, gate_w);
    upload_alloc(q, gate_s_dev, gate_s);
    upload_alloc(q, gate_dst_dev, gate_dst);
    upload_alloc(q, gate_input_dev, gate_input);
    upload_alloc(q, down_w_dev, down_w);
    upload_alloc(q, down_s_dev, down_s);
    upload_alloc(q, down_dst_dev, down_dst);
    upload_alloc(q, down_input_dev, down_input);

    auto xe_packed = ar.alloc<uint8_t>((size_t)A_all * H / 2);
    auto xe_scale = ar.alloc<uint8_t>((size_t)A_all * HG);
    { DIFF_PROF(q, prof.pack);
      pack_bf16_to_nvfp4_grouped_rows(q, Xe.data(), H,
                                      row_expert_dev.data(), A_all,
                                      gate_input_dev.data(), xe_packed.data(), xe_scale.data()); }
    Xe.reset();
    gate_input_dev.reset();

    auto gu = ar.alloc<bf16>((size_t)A_all * 2 * moe_inter);
    { DIFF_PROF(q, prof.gateup_mm);
      if (use_xe2_gemm) {
          matmul_nvfp4_grouped_rows_xe2(ctx, xe_packed.data(), xe_scale.data(), H,
                                        row_expert_dev.data(), A_all,
                                        gate_w_dev.data(), gate_s_dev.data(), gate_dst_dev.data(),
                                        2 * moe_inter, gu.data());
      } else {
          matmul_nvfp4_grouped_rows_custom(q, xe_packed.data(), xe_scale.data(), H,
                                           row_expert_dev.data(), A_all,
                                           gate_w_dev.data(), gate_s_dev.data(), gate_dst_dev.data(),
                                           2 * moe_inter, gu.data());
      } }
    gate_w_dev.reset(); gate_s_dev.reset(); gate_dst_dev.reset();
    xe_packed.reset(); xe_scale.reset();

    auto act = ar.alloc<bf16>((size_t)A_all * moe_inter);
    auto act_packed = ar.alloc<uint8_t>((size_t)A_all * moe_inter / 2);
    auto act_scale = ar.alloc<uint8_t>((size_t)A_all * IG);
    { DIFF_PROF(q, prof.geglu_pack);
      geglu_strided_counts(q, gu.data(), act.data(),
                           row_expert_dev.data(), A_all, moe_inter);
      pack_bf16_to_nvfp4_grouped_rows(q, act.data(), moe_inter,
                                      row_expert_dev.data(), A_all,
                                      down_input_dev.data(), act_packed.data(), act_scale.data()); }
    gu.reset();
    act.reset();
    down_input_dev.reset();

    auto Ye = ar.alloc<bf16>((size_t)A_all * H);
    { DIFF_PROF(q, prof.down_mm);
      if (use_xe2_gemm) {
          matmul_nvfp4_grouped_rows_xe2(ctx, act_packed.data(), act_scale.data(), moe_inter,
                                        row_expert_dev.data(), A_all,
                                        down_w_dev.data(), down_s_dev.data(), down_dst_dev.data(),
                                        H, Ye.data());
      } else {
          matmul_nvfp4_grouped_rows_custom(q, act_packed.data(), act_scale.data(), moe_inter,
                                           row_expert_dev.data(), A_all,
                                           down_w_dev.data(), down_s_dev.data(), down_dst_dev.data(),
                                           H, Ye.data());
      } }
    down_w_dev.reset(); down_s_dev.reset(); down_dst_dev.reset();
    act_packed.reset(); act_scale.reset();

    auto t_combine = diffprof::tic(q);
    {
        const int32_t* slot = slot_dev.data();
        const float* wt = w_dev.data();
        const bf16* ye = Ye.data();
        int K = top_k;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>((size_t)seq, (size_t)H), [=](sycl::id<2> id) {
                int t = (int)id[0], d = (int)id[1];
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int a = t * K + k;
                    int s = slot[a];
                    if (s >= 0) acc += wt[a] * bf16_to_float(ye[(size_t)s * H + d]);
                }
                out[(size_t)t * H + d] = float_to_bf16(acc);
            });
        });
    }
    q.wait();
    diffprof::toc(q, prof.combine, t_combine);
    return true;
}

static void run_shard(
    GpuEngine& ctx,
    const DiffExpertShard& shard,
	    const bf16* expert_in,
	    bf16* out,
	    int seq, int H, int top_k, int moe_inter,
	    const std::vector<int>& idx,
	    const std::vector<float>& weight,
	    const ExpertProfileLabels& prof,
	    const int* idx_dev = nullptr,
	    const float* weight_dev = nullptr)
	{
	    auto& q = ctx.queue;
	    auto& ar = diffarena::arena(ctx.index);
	    auto t_setup = diffprof::tic(q);
	    int localE = shard.num_experts;
	    int A_all = seq * top_k;
	    bool device_routes = idx_dev != nullptr && weight_dev != nullptr;

        if (device_routes && shard.nvfp4 && nvfp4_gpu_layout_enabled() &&
            seq <= nvfp4_gpu_layout_max_seq()) {
            if (run_shard_nvfp4_gpu_layout(ctx, shard, expert_in, out, seq, H,
                                           top_k, moe_inter, prof,
                                           idx_dev, weight_dev))
                return;
        }
	
	    std::vector<int> count(localE, 0);
	    int A = 0;
	    if (device_routes) {
	        auto count_dev = ar.alloc<int32_t>(localE);
	        q.memset(count_dev.data(), 0, (size_t)localE * sizeof(int32_t));
	        int first = shard.first_expert;
	        int32_t* count_ptr = count_dev.data();
	        q.submit([&](sycl::handler& h) {
	            h.parallel_for(sycl::range<1>(A_all), [=](sycl::id<1> id) {
	                int a = (int)id[0];
	                int e = idx_dev[a] - first;
	                if (e >= 0 && e < localE) {
	                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
	                                     sycl::memory_scope::device,
	                                     sycl::access::address_space::global_space>
	                        cnt(count_ptr[e]);
	                    cnt.fetch_add(1);
	                }
	            });
	        });
	        std::vector<int32_t> count_i(localE, 0);
	        q.memcpy(count_i.data(), count_dev.data(), (size_t)localE * sizeof(int32_t)).wait();
	        for (int e = 0; e < localE; ++e) { count[e] = count_i[e]; A += count[e]; }
	    } else {
	        for (int a = 0; a < A_all; ++a) {
	            int e = idx[a] - shard.first_expert;
	            if (e >= 0 && e < localE) { count[e]++; A++; }
	        }
	    }

    q.memset(out, 0, (size_t)seq * H * sizeof(bf16));
    if (A == 0) return;

    int T = std::min(moe_tail_cap(), round_up(seq, 8));
    bool q8_compact_exact = shard.q8 && !q8_use_hybrid_expert_kernel();
    struct Hot { int expert, rows_off, m; };
    std::vector<Hot> hot;
    std::vector<int> base(localE), bucket_cap(localE);
    int total_rows = q8_compact_exact ? 0 : localE * T;
    if (q8_compact_exact) {
        for (int e = 0; e < localE; ++e) {
            if (count[e] == 0) { base[e] = 0; bucket_cap[e] = 0; continue; }
            int m = round_up(count[e], 8);
            base[e] = total_rows;
            bucket_cap[e] = m;
            total_rows += m;
        }
    } else {
        for (int e = 0; e < localE; ++e) {
            if (count[e] > T) {
                int m = round_up(count[e], 32);
                base[e] = total_rows; bucket_cap[e] = m;
                hot.push_back({e, total_rows, m});
                total_rows += m;
            } else {
                base[e] = e * T; bucket_cap[e] = T;
            }
        }
    }

	    auto slot_dev = ar.alloc<int32_t>(A_all);
	    auto w_dev = ar.alloc<float>(A_all);
	    auto Xe = ar.alloc<bf16>((size_t)total_rows * H);
	    q.memset(Xe.data(), 0, (size_t)total_rows * H * sizeof(bf16));
	    if (device_routes) {
	        auto base_dev = ar.alloc<int32_t>(localE);
	        auto cursor_dev = ar.alloc<int32_t>(localE);
	        std::vector<int32_t> base_i(localE);
	        for (int e = 0; e < localE; ++e) base_i[e] = base[e];
	        upload_alloc(q, base_dev, base_i);
	        q.memset(cursor_dev.data(), 0, (size_t)localE * sizeof(int32_t));
	        q.memset(slot_dev.data(), 0xFF, (size_t)A_all * sizeof(int32_t));
	        q.memset(w_dev.data(), 0, (size_t)A_all * sizeof(float));
	        int first = shard.first_expert;
	        const int32_t* base_ptr = base_dev.data();
	        int32_t* cursor_ptr = cursor_dev.data();
	        int32_t* slot_ptr = slot_dev.data();
	        float* w_ptr = w_dev.data();
	        q.submit([&](sycl::handler& h) {
	            h.parallel_for(sycl::range<1>(A_all), [=](sycl::id<1> id) {
	                int a = (int)id[0];
	                int e = idx_dev[a] - first;
	                if (e < 0 || e >= localE) return;
	                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
	                                 sycl::memory_scope::device,
	                                 sycl::access::address_space::global_space>
	                    cur(cursor_ptr[e]);
	                int pos = cur.fetch_add(1);
	                int slot = base_ptr[e] + pos;
	                slot_ptr[a] = slot;
	                w_ptr[a] = weight_dev[a];
	            });
	        });
	        const int32_t* slot = slot_dev.data();
	        bf16* xe = Xe.data();
	        q.submit([&](sycl::handler& h) {
	            h.parallel_for(sycl::range<2>(A_all, H), [=](sycl::id<2> id) {
	                int a = (int)id[0], d = (int)id[1];
	                int s = slot[a];
	                if (s >= 0)
	                    xe[(size_t)s * H + d] = expert_in[(size_t)(a / top_k) * H + d];
	            });
	        });
	    } else {
	        std::vector<int32_t> assign_token;
	        std::vector<int32_t> assign_slot;
	        std::vector<int32_t> slot_for_tk((size_t)A_all, -1);
	        std::vector<float> weight_for_tk((size_t)A_all, 0.0f);
	        assign_token.reserve(A);
	        assign_slot.reserve(A);
	        std::vector<int> cursor(localE, 0);
	        for (int a = 0; a < A_all; ++a) {
	            int e = idx[a] - shard.first_expert;
	            if (e < 0 || e >= localE) continue;
	            int slot = base[e] + cursor[e]++;
	            assign_token.push_back(a / top_k);
	            assign_slot.push_back(slot);
	            slot_for_tk[a] = slot;
	            weight_for_tk[a] = weight[a];
	        }
	        auto atok_dev = ar.alloc<int32_t>(A);
	        auto aslot_dev = ar.alloc<int32_t>(A);
	        upload_alloc(q, atok_dev, assign_token);
	        upload_alloc(q, aslot_dev, assign_slot);
	        upload_alloc(q, slot_dev, slot_for_tk);
	        upload_alloc(q, w_dev, weight_for_tk);
	        {
	            const int32_t* at = atok_dev.data();
	            const int32_t* as = aslot_dev.data();
	            bf16* xe = Xe.data();
	            q.submit([&](sycl::handler& h) {
	                h.parallel_for(sycl::range<2>(A, H), [=](sycl::id<2> id) {
	                    int a = (int)id[0], d = (int)id[1];
	                    xe[(size_t)as[a] * H + d] = expert_in[(size_t)at[a] * H + d];
	                });
	            });
	        }
	        atok_dev.reset();
	        aslot_dev.reset();
	    }
	
	    diffprof::toc(q, prof.setup_gather, t_setup);
	
	    diffarena::Alloc<bf16> Ye;
    if (shard.nvfp4) {
        if ((int)shard.gate_up_proj_fp4.size() != localE ||
            (int)shard.down_proj_fp4.size() != localE)
            throw std::runtime_error("NVFP4 expert shard is missing fused projection weights");

        int HG = H / 16;
        int IG = moe_inter / 16;
        int round_m = nvfp4_expert_round_multiple();

        if (nvfp4_use_custom_expert_kernel()) {
            std::vector<int32_t> compute_slot;
            std::vector<int32_t> compute_expert;
            compute_slot.reserve((size_t)A + localE * round_m);
            compute_expert.reserve((size_t)A + localE * round_m);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                for (int r = 0; r < m; ++r) {
                    compute_slot.push_back(off + r);
                    compute_expert.push_back(e);
                }
            }
            int compute_rows = (int)compute_slot.size();
            auto compute_slot_dev = ar.alloc<int32_t>(compute_rows);
            auto compute_expert_dev = ar.alloc<int32_t>(compute_rows);
            upload_alloc(q, compute_slot_dev, compute_slot);
            upload_alloc(q, compute_expert_dev, compute_expert);

            std::vector<const uint8_t*> gate_w(localE), gate_s(localE), down_w(localE), down_s(localE);
            std::vector<const float*> gate_dst(localE), down_dst(localE);
            std::vector<float> gate_input(localE), down_input(localE);
            for (int e = 0; e < localE; ++e) {
                gate_w[e] = shard.gate_up_proj_fp4[e].weight_packed.data();
                gate_s[e] = shard.gate_up_proj_fp4[e].weight_scale.data();
                gate_dst[e] = shard.gate_up_proj_fp4[e].dst_scale.data();
                gate_input[e] = shard.gate_up_proj_fp4[e].input_global_scale;
                down_w[e] = shard.down_proj_fp4[e].weight_packed.data();
                down_s[e] = shard.down_proj_fp4[e].weight_scale.data();
                down_dst[e] = shard.down_proj_fp4[e].dst_scale.data();
                down_input[e] = shard.down_proj_fp4[e].input_global_scale;
            }

            auto gate_w_dev = ar.alloc<const uint8_t*>(localE);
            auto gate_s_dev = ar.alloc<const uint8_t*>(localE);
            auto down_w_dev = ar.alloc<const uint8_t*>(localE);
            auto down_s_dev = ar.alloc<const uint8_t*>(localE);
            auto gate_dst_dev = ar.alloc<const float*>(localE);
            auto down_dst_dev = ar.alloc<const float*>(localE);
            auto gate_input_dev = ar.alloc<float>(localE);
            auto down_input_dev = ar.alloc<float>(localE);
            upload_alloc(q, gate_w_dev, gate_w);
            upload_alloc(q, gate_s_dev, gate_s);
            upload_alloc(q, gate_dst_dev, gate_dst);
            upload_alloc(q, gate_input_dev, gate_input);
            upload_alloc(q, down_w_dev, down_w);
            upload_alloc(q, down_s_dev, down_s);
            upload_alloc(q, down_dst_dev, down_dst);
            upload_alloc(q, down_input_dev, down_input);

            auto xe_packed = ar.alloc<uint8_t>((size_t)total_rows * H / 2);
            auto xe_scale = ar.alloc<uint8_t>((size_t)total_rows * HG);
            pack_bf16_to_nvfp4_grouped(q, Xe.data(), H,
                                       compute_slot_dev.data(), compute_expert_dev.data(), compute_rows,
                                       gate_input_dev.data(), xe_packed.data(), xe_scale.data());
            Xe.reset();

            auto gu = ar.alloc<bf16>((size_t)total_rows * 2 * moe_inter);
            matmul_nvfp4_grouped_custom(q, xe_packed.data(), xe_scale.data(), H,
                                        compute_slot_dev.data(), compute_expert_dev.data(), compute_rows,
                                        gate_w_dev.data(), gate_s_dev.data(), gate_dst_dev.data(),
                                        2 * moe_inter, gu.data());
            gate_w_dev.reset(); gate_s_dev.reset(); gate_dst_dev.reset();
            gate_input_dev.reset(); xe_packed.reset(); xe_scale.reset();

            auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
            geglu_strided_grouped(q, gu.data(), act.data(), compute_slot_dev.data(),
                                  compute_rows, moe_inter);
            gu.reset();

            auto act_packed = ar.alloc<uint8_t>((size_t)total_rows * moe_inter / 2);
            auto act_scale = ar.alloc<uint8_t>((size_t)total_rows * IG);
            pack_bf16_to_nvfp4_grouped(q, act.data(), moe_inter,
                                       compute_slot_dev.data(), compute_expert_dev.data(), compute_rows,
                                       down_input_dev.data(), act_packed.data(), act_scale.data());
            act.reset();

            Ye = ar.alloc<bf16>((size_t)total_rows * H);
            matmul_nvfp4_grouped_custom(q, act_packed.data(), act_scale.data(), moe_inter,
                                        compute_slot_dev.data(), compute_expert_dev.data(), compute_rows,
                                        down_w_dev.data(), down_s_dev.data(), down_dst_dev.data(),
                                        H, Ye.data());
            compute_slot_dev.reset(); compute_expert_dev.reset();
            down_w_dev.reset(); down_s_dev.reset(); down_dst_dev.reset();
            down_input_dev.reset(); act_packed.reset(); act_scale.reset();
        } else if (nvfp4_use_hybrid_expert_kernel()) {
            std::vector<int32_t> compute_slot;
            std::vector<int32_t> compute_expert;
            compute_slot.reserve((size_t)A + localE * round_m);
            compute_expert.reserve((size_t)A + localE * round_m);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                for (int r = 0; r < m; ++r) {
                    compute_slot.push_back(off + r);
                    compute_expert.push_back(e);
                }
            }
            int compute_rows = (int)compute_slot.size();
            auto compute_slot_dev = ar.alloc<int32_t>(compute_rows);
            auto compute_expert_dev = ar.alloc<int32_t>(compute_rows);
            upload_alloc(q, compute_slot_dev, compute_slot);
            upload_alloc(q, compute_expert_dev, compute_expert);

            std::vector<float> gate_input(localE), down_input(localE);
            for (int e = 0; e < localE; ++e) {
                gate_input[e] = shard.gate_up_proj_fp4[e].input_global_scale;
                down_input[e] = shard.down_proj_fp4[e].input_global_scale;
            }
            auto gate_input_dev = ar.alloc<float>(localE);
            auto down_input_dev = ar.alloc<float>(localE);
            upload_alloc(q, gate_input_dev, gate_input);
            upload_alloc(q, down_input_dev, down_input);

            auto xe_packed = ar.alloc<uint8_t>((size_t)total_rows * H / 2);
            auto xe_scale = ar.alloc<uint8_t>((size_t)total_rows * HG);
            { DIFF_PROF(q, prof.pack);
              pack_bf16_to_nvfp4_grouped(q, Xe.data(), H,
                                       compute_slot_dev.data(), compute_expert_dev.data(), compute_rows,
                                       gate_input_dev.data(), xe_packed.data(), xe_scale.data()); }
            Xe.reset();
            gate_input_dev.reset();

            auto gu = ar.alloc<bf16>((size_t)total_rows * 2 * moe_inter);
            { DIFF_PROF(q, prof.gateup_mm);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                matmul_nvfp4_packed(xe_packed.data() + (size_t)off * H / 2,
                                    xe_scale.data() + (size_t)off * HG,
                                    m, H, shard.gate_up_proj_fp4[e],
                                    gu.data() + (size_t)off * 2 * moe_inter, ctx);
            } }
            xe_packed.reset(); xe_scale.reset();

            auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
            auto act_packed = ar.alloc<uint8_t>((size_t)total_rows * moe_inter / 2);
            auto act_scale = ar.alloc<uint8_t>((size_t)total_rows * IG);
            { DIFF_PROF(q, prof.geglu_pack);
              geglu_strided_grouped(q, gu.data(), act.data(), compute_slot_dev.data(),
                                  compute_rows, moe_inter);
              pack_bf16_to_nvfp4_grouped(q, act.data(), moe_inter,
                                       compute_slot_dev.data(), compute_expert_dev.data(), compute_rows,
                                       down_input_dev.data(), act_packed.data(), act_scale.data()); }
            gu.reset();
            act.reset();
            compute_slot_dev.reset(); compute_expert_dev.reset();
            down_input_dev.reset();

            Ye = ar.alloc<bf16>((size_t)total_rows * H);
            { DIFF_PROF(q, prof.down_mm);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                matmul_nvfp4_packed(act_packed.data() + (size_t)off * moe_inter / 2,
                                    act_scale.data() + (size_t)off * IG,
                                    m, moe_inter, shard.down_proj_fp4[e],
                                    Ye.data() + (size_t)off * H, ctx);
            } }
            act_packed.reset(); act_scale.reset();
        } else {
            auto xe_packed = ar.alloc<uint8_t>((size_t)total_rows * H / 2);
            auto xe_scale = ar.alloc<uint8_t>((size_t)total_rows * HG);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                uint8_t* packed = xe_packed.data() + (size_t)off * H / 2;
                uint8_t* scales = xe_scale.data() + (size_t)off * HG;
                pack_bf16_to_nvfp4(q, Xe.data() + (size_t)off * H, packed, scales,
                                   m, H, shard.gate_up_proj_fp4[e].input_global_scale);
            }
            Xe.reset();

            auto gu = ar.alloc<bf16>((size_t)total_rows * 2 * moe_inter);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                uint8_t* packed = xe_packed.data() + (size_t)off * H / 2;
                uint8_t* scales = xe_scale.data() + (size_t)off * HG;
                matmul_nvfp4_packed(packed, scales, m, H,
                                    shard.gate_up_proj_fp4[e],
                                    gu.data() + (size_t)off * 2 * moe_inter, ctx);
            }
            xe_packed.reset(); xe_scale.reset();

            auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                geglu_strided(q, gu.data() + (size_t)off * 2 * moe_inter,
                              act.data() + (size_t)off * moe_inter, m, moe_inter);
            }
            gu.reset();

            auto act_packed = ar.alloc<uint8_t>((size_t)total_rows * moe_inter / 2);
            auto act_scale = ar.alloc<uint8_t>((size_t)total_rows * IG);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                uint8_t* packed = act_packed.data() + (size_t)off * moe_inter / 2;
                uint8_t* scales = act_scale.data() + (size_t)off * IG;
                pack_bf16_to_nvfp4(q, act.data() + (size_t)off * moe_inter,
                                   packed, scales, m, moe_inter,
                                   shard.down_proj_fp4[e].input_global_scale);
            }
            act.reset();

            Ye = ar.alloc<bf16>((size_t)total_rows * H);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], round_m));
                int off = base[e];
                uint8_t* packed = act_packed.data() + (size_t)off * moe_inter / 2;
                uint8_t* scales = act_scale.data() + (size_t)off * IG;
                matmul_nvfp4_packed(packed, scales, m, moe_inter,
                                    shard.down_proj_fp4[e], Ye.data() + (size_t)off * H, ctx);
            }
            act_packed.reset(); act_scale.reset();
        }
    } else if (shard.int4) {
        if ((int)shard.gate_up_proj_int4.size() != localE ||
            (int)shard.down_proj_int4.size() != localE)
            throw std::runtime_error("int4 expert shard is missing fused projection weights");

        // W4A16: activations stay BF16, so no packing — one s4 decompress GEMM
        // per active expert at its bucket offset (m rounded to bound the oneDNN
        // primitive-cache shapes), then GeGLU, then the down projection.
        auto gu = ar.alloc<bf16>((size_t)total_rows * 2 * moe_inter);
        { DIFF_PROF(q, prof.gateup_mm);
          for (int e = 0; e < localE; ++e) {
              if (count[e] == 0) continue;
              int m = std::min(bucket_cap[e], round_up(count[e], 32));
              int off = base[e];
              matmul_int4(Xe.data() + (size_t)off * H, m, H,
                          shard.gate_up_proj_int4[e],
                          gu.data() + (size_t)off * 2 * moe_inter, ctx);
          } }
        Xe.reset();

        // One GeGLU launch over the whole row space instead of one per expert.
        // The tiny per-expert kernels are enqueue-bound; gaps (padding / empty
        // experts) compute garbage into act rows that combine never reads.
        auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
        { DIFF_PROF(q, prof.geglu_pack);
          geglu_strided(q, gu.data(), act.data(), total_rows, moe_inter); }
        gu.reset();

        Ye = ar.alloc<bf16>((size_t)total_rows * H);
        { DIFF_PROF(q, prof.down_mm);
          for (int e = 0; e < localE; ++e) {
              if (count[e] == 0) continue;
              int m = std::min(bucket_cap[e], round_up(count[e], 32));
              int off = base[e];
              matmul_int4(act.data() + (size_t)off * moe_inter, m, moe_inter,
                          shard.down_proj_int4[e],
                          Ye.data() + (size_t)off * H, ctx);
          } }
        act.reset();
    } else if (shard.q8) {
        if (shard.gate_up_proj_q8_batch.empty() || shard.down_proj_q8_batch.empty())
            throw std::runtime_error("Q8_0 expert shard is missing batched projection weights");

        if (q8_use_hybrid_expert_kernel()) {
            auto gu = ar.alloc<bf16>((size_t)total_rows * 2 * moe_inter);
            { DIFF_PROF(q, prof.gateup_mm);
              matmul_q8_0_batched(Xe.data(), localE, T, H,
                                  shard.gate_up_proj_q8_batch, gu.data(), ctx);
              for (auto& hx : hot) {
                  matmul_q8_0_expert(Xe.data() + (size_t)hx.rows_off * H,
                                     hx.m, H, shard.gate_up_proj_q8_batch,
                                     hx.expert,
                                     gu.data() + (size_t)hx.rows_off * 2 * moe_inter,
                                     ctx);
              } }
            Xe.reset();

            auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
            { DIFF_PROF(q, prof.geglu_pack);
              geglu_strided(q, gu.data(), act.data(), total_rows, moe_inter); }
            gu.reset();

            Ye = ar.alloc<bf16>((size_t)total_rows * H);
            { DIFF_PROF(q, prof.down_mm);
              matmul_q8_0_batched(act.data(), localE, T, moe_inter,
                                  shard.down_proj_q8_batch, Ye.data(), ctx);
              for (auto& hx : hot) {
                  matmul_q8_0_expert(act.data() + (size_t)hx.rows_off * moe_inter,
                                     hx.m, moe_inter, shard.down_proj_q8_batch,
                                     hx.expert,
                                     Ye.data() + (size_t)hx.rows_off * H, ctx);
              } }
            act.reset();
        } else if (q8_use_grouped_expert_kernel()) {
            std::vector<int32_t> block_slot;
            std::vector<int32_t> block_expert;
            block_slot.reserve((size_t)total_rows / 8);
            block_expert.reserve((size_t)total_rows / 8);
            for (int e = 0; e < localE; ++e) {
                if (count[e] == 0) continue;
                int m = std::min(bucket_cap[e], round_up(count[e], 8));
                int off = base[e];
                for (int r = 0; r < m; r += 8) {
                    block_slot.push_back(off + r);
                    block_expert.push_back(e);
                }
            }
            int blocks = (int)block_slot.size();
            auto block_slot_dev = ar.alloc<int32_t>(blocks);
            auto block_expert_dev = ar.alloc<int32_t>(blocks);
            upload_alloc(q, block_slot_dev, block_slot);
            upload_alloc(q, block_expert_dev, block_expert);

            auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
            if (q8_use_fused_expert_geglu()) {
                { DIFF_PROF(q, prof.gateup_mm);
                  matmul_q8_0_grouped_expert_gateup_geglu(
                      Xe.data(), H, shard.gate_up_proj_q8_batch,
                      block_slot_dev.data(), block_expert_dev.data(), blocks,
                      moe_inter, act.data(), ctx); }
                Xe.reset();
            } else {
                auto gu = ar.alloc<bf16>((size_t)total_rows * 2 * moe_inter);
                { DIFF_PROF(q, prof.gateup_mm);
                  matmul_q8_0_grouped_expert(
                      Xe.data(), H, shard.gate_up_proj_q8_batch,
                      block_slot_dev.data(), block_expert_dev.data(), blocks,
                      gu.data(), ctx); }
                Xe.reset();

                { DIFF_PROF(q, prof.geglu_pack);
                  geglu_strided(q, gu.data(), act.data(), total_rows, moe_inter); }
                gu.reset();
            }

            Ye = ar.alloc<bf16>((size_t)total_rows * H);
            { DIFF_PROF(q, prof.down_mm);
              matmul_q8_0_grouped_expert(
                  act.data(), moe_inter, shard.down_proj_q8_batch,
                  block_slot_dev.data(), block_expert_dev.data(), blocks,
                  Ye.data(), ctx); }
            act.reset();
        } else if (q8_use_fused_expert_geglu()) {
            auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
            { DIFF_PROF(q, prof.gateup_mm);
              for (int e = 0; e < localE; ++e) {
                  if (count[e] == 0) continue;
                  int m = std::min(bucket_cap[e], round_up(count[e], 8));
                  int off = base[e];
                  matmul_q8_0_expert_gateup_geglu(
                      Xe.data() + (size_t)off * H, m, H,
                      shard.gate_up_proj_q8_batch, e, moe_inter,
                      act.data() + (size_t)off * moe_inter, ctx);
              } }
            Xe.reset();

            Ye = ar.alloc<bf16>((size_t)total_rows * H);
            { DIFF_PROF(q, prof.down_mm);
              for (int e = 0; e < localE; ++e) {
                  if (count[e] == 0) continue;
                  int m = std::min(bucket_cap[e], round_up(count[e], 8));
                  int off = base[e];
                  matmul_q8_0_expert(act.data() + (size_t)off * moe_inter,
                                     m, moe_inter, shard.down_proj_q8_batch, e,
                                     Ye.data() + (size_t)off * H, ctx);
              } }
            act.reset();
        } else {
            auto gu = ar.alloc<bf16>((size_t)total_rows * 2 * moe_inter);
            { DIFF_PROF(q, prof.gateup_mm);
              for (int e = 0; e < localE; ++e) {
                  if (count[e] == 0) continue;
                  int m = std::min(bucket_cap[e], round_up(count[e], 8));
                  int off = base[e];
                  matmul_q8_0_expert(Xe.data() + (size_t)off * H,
                                     m, H, shard.gate_up_proj_q8_batch, e,
                                     gu.data() + (size_t)off * 2 * moe_inter,
                                     ctx);
              } }
            Xe.reset();

            auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
            { DIFF_PROF(q, prof.geglu_pack);
              geglu_strided(q, gu.data(), act.data(), total_rows, moe_inter); }
            gu.reset();

            Ye = ar.alloc<bf16>((size_t)total_rows * H);
            { DIFF_PROF(q, prof.down_mm);
              for (int e = 0; e < localE; ++e) {
                  if (count[e] == 0) continue;
                  int m = std::min(bucket_cap[e], round_up(count[e], 8));
                  int off = base[e];
                  matmul_q8_0_expert(act.data() + (size_t)off * moe_inter,
                                     m, moe_inter, shard.down_proj_q8_batch, e,
                                     Ye.data() + (size_t)off * H, ctx);
              } }
            act.reset();
        }
    } else {
        size_t gu_stride = (size_t)2 * moe_inter * H;
        size_t down_stride = (size_t)H * moe_inter;

        auto gu = ar.alloc<bf16>((size_t)total_rows * 2 * moe_inter);
        matmul_bf16_batched(Xe.data(), localE, T, H,
                            shard.gate_up_proj.data(), 2 * moe_inter, true,
                            gu.data(), ctx);
        for (auto& hx : hot) {
            matmul_bf16(Xe.data() + (size_t)hx.rows_off * H, hx.m, H,
                        shard.gate_up_proj.data() + (size_t)hx.expert * gu_stride,
                        2 * moe_inter,
                        gu.data() + (size_t)hx.rows_off * 2 * moe_inter, ctx);
        }
        Xe.reset();

        auto act = ar.alloc<bf16>((size_t)total_rows * moe_inter);
        geglu_strided(q, gu.data(), act.data(), total_rows, moe_inter);
        gu.reset();

        Ye = ar.alloc<bf16>((size_t)total_rows * H);
        matmul_bf16_batched(act.data(), localE, T, moe_inter,
                            shard.down_proj.data(), H, true,
                            Ye.data(), ctx);
        for (auto& hx : hot) {
            matmul_bf16(act.data() + (size_t)hx.rows_off * moe_inter, hx.m, moe_inter,
                        shard.down_proj.data() + (size_t)hx.expert * down_stride, H,
                        Ye.data() + (size_t)hx.rows_off * H, ctx);
        }
        act.reset();
    }

    auto t_combine = diffprof::tic(q);
    {
        const int32_t* slot = slot_dev.data();
        const float* wt = w_dev.data();
        const bf16* ye = Ye.data();
        int K = top_k;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(seq, H), [=](sycl::id<2> id) {
                int t = (int)id[0], d = (int)id[1];
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int a = t * K + k;
                    int s = slot[a];
                    if (s >= 0) acc += wt[a] * bf16_to_float(ye[(size_t)s * H + d]);
                }
                out[(size_t)t * H + d] = float_to_bf16(acc);
            });
        });
    }
    q.wait();
    diffprof::toc(q, prof.combine, t_combine);
}

} // namespace

void expert_parallel_forward(
    GpuEngine& owner,
    const DiffMoE& moe,
    const bf16* expert_in,
    bf16* out,
    int seq, int H, int E, int top_k, int moe_intermediate,
    const std::vector<int>& idx,
    const std::vector<float>& weight,
    const ExpertProfileLabels& prof)
{
    auto& owner_q = owner.queue;
    size_t N = (size_t)seq * H;
    owner_q.memset(out, 0, N * sizeof(bf16));

    for (const auto& shard : moe.expert_shards) {
        GpuEngine& ctx = GpuEngine::get(shard.gpu);
        auto& q = ctx.queue;
        bool local = shard.gpu == owner.index;

        auto& shard_ar = diffarena::arena(ctx.index);
        diffarena::Alloc<bf16> remote_in;
        const bf16* shard_in = expert_in;
        if (!local) {
            remote_in = shard_ar.alloc<bf16>(N);
            transfer(owner_q, expert_in, q, remote_in.data(), N);
            shard_in = remote_in.data();
        }

        auto shard_out_a = shard_ar.alloc<bf16>(N);
        bf16* shard_out = shard_out_a.data();
        run_shard(ctx, shard, shard_in, shard_out, seq, H, top_k,
                  moe_intermediate, idx, weight, prof);
        remote_in.reset();

        if (local) {
            add_inplace(owner_q, out, shard_out, (int)N);
            owner_q.wait();
        } else {
            auto tmp = diffarena::arena(owner.index).alloc<bf16>(N);
            transfer(q, shard_out, owner_q, tmp.data(), N);
            add_inplace(owner_q, out, tmp.data(), (int)N);
            owner_q.wait();
        }
    }
}

void expert_parallel_forward(
    GpuEngine& owner,
    const DiffMoE& moe,
    const bf16* expert_in,
    bf16* out,
    int seq, int H, int E, int top_k, int moe_intermediate,
    const int* idx_dev,
    const float* weight_dev,
    const ExpertProfileLabels& prof)
{
    (void)E;
    auto& owner_q = owner.queue;
    size_t N = (size_t)seq * H;
    int A_all = seq * top_k;
    owner_q.memset(out, 0, N * sizeof(bf16));

    for (const auto& shard : moe.expert_shards) {
        GpuEngine& ctx = GpuEngine::get(shard.gpu);
        auto& q = ctx.queue;
        bool local = shard.gpu == owner.index;

        auto& shard_ar = diffarena::arena(ctx.index);
        diffarena::Alloc<bf16> remote_in;
        const bf16* shard_in = expert_in;
        if (!local) {
            remote_in = shard_ar.alloc<bf16>(N);
            transfer(owner_q, expert_in, q, remote_in.data(), N);
            shard_in = remote_in.data();
        }

        diffarena::Alloc<int> remote_idx;
        diffarena::Alloc<float> remote_weight;
        const int* shard_idx = idx_dev;
        const float* shard_weight = weight_dev;
        if (!local) {
            remote_idx = shard_ar.alloc<int>(A_all);
            remote_weight = shard_ar.alloc<float>(A_all);
            transfer_t(owner_q, idx_dev, q, remote_idx.data(), A_all);
            transfer_t(owner_q, weight_dev, q, remote_weight.data(), A_all);
            shard_idx = remote_idx.data();
            shard_weight = remote_weight.data();
        }

        auto shard_out_a = shard_ar.alloc<bf16>(N);
        bf16* shard_out = shard_out_a.data();
        static const std::vector<int> empty_idx;
        static const std::vector<float> empty_weight;
        run_shard(ctx, shard, shard_in, shard_out, seq, H, top_k,
                  moe_intermediate, empty_idx, empty_weight, prof,
                  shard_idx, shard_weight);
        remote_in.reset();
        remote_idx.reset();
        remote_weight.reset();

        if (local) {
            add_inplace(owner_q, out, shard_out, (int)N);
            owner_q.wait();
        } else {
            auto tmp = diffarena::arena(owner.index).alloc<bf16>(N);
            transfer(q, shard_out, owner_q, tmp.data(), N);
            add_inplace(owner_q, out, tmp.data(), (int)N);
            owner_q.wait();
        }
    }
}

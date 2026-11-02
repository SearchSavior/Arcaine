#pragma once

#include "grouped_moe.hpp"

namespace flint::flash_decode_variants {

// Ten global decode experiments share the existing FP32 score workspace and
// partition merge. Each numbered variant instantiates a distinct pair of GPU
// kernels. Score strategies (pairs 1/2 through 9/10): defer max reduction,
// butterfly max per tile, stream Q, scalar packed-K loads, four subgroups per
// work-group. Even variants defer the PV sum reduction; variant 10 also assigns
// all 128 output channels to each subgroup rather than splitting into two.
template<int Variant>
inline sycl::event partials(sycl::queue& q, const bf16* query, const bf16* kc,
                            const bf16* vc, const bf16* sink, float* partial,
                            float* stats, float* scores, int kv_heads, int cap,
                            int minimum_parts, int max_parts, const int* position) {
    static_assert(Variant >= 1 && Variant <= 10);
    constexpr int strategy = (Variant - 1) / 2;
    constexpr bool deferred_max = strategy != 1;
    constexpr bool hoist_query = strategy != 2;
    constexpr bool vector_key = strategy != 3;
    constexpr int score_subgroups = strategy == 4 ? 4 : 1;
    constexpr bool deferred_sum = Variant % 2 == 0;
    constexpr int value_tiles = Variant == 10 ? 1 : 2;
    constexpr int channel_blocks = 8 / value_tiles;
    const size_t score_groups = size_t(kv_heads) * max_parts;
    const size_t score_local = score_subgroups * 16;
    const size_t score_global = ((score_groups + score_subgroups - 1) / score_subgroups) * score_local;

    auto first_event = q.parallel_for(
        sycl::nd_range<1>(score_global, score_local),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int work = int(it.get_global_linear_id() / 16);
            if (size_t(work) >= score_groups) return;
            const int part = work % max_parts, kv = work / max_parts;
            const int lane = int(it.get_local_linear_id() % 16);
            const int count = *position + 1;
            const int parts = sycl::max(minimum_parts,
                count <= 1024 ? 1 : count <= 4096 ? 16 : count <= 16384 ? 32 : 64);
            if (part >= parts) return;
            const int partition = (count + parts - 1) / parts;
            const int begin = part * partition;
            const int nkeys = sycl::max(0, sycl::min(partition, count - begin));
            const auto sg = it.get_sub_group();
            flint_moe_v8s queries[8];
            float hi[8];
            if constexpr (hoist_query) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
#pragma unroll
                    for (int m = 0; m < 8; ++m)
                        queries[k][m] = sycl::bit_cast<short>(
                            query[size_t(kv * 8 + m) * 128 + k * 16 + lane]);
                }
            }
#pragma unroll
            for (int m = 0; m < 8; ++m)
                hi[m] = part ? -3.402823466e+38f : float(sink[kv * 8 + m]);
            for (int first = 0; first < nkeys; first += 16) {
                const int key = first + lane;
                const int absolute = begin + key;
                const bool valid = key < nkeys;
                flint_moe_v8f score{};
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    flint_moe_v8s a;
                    if constexpr (hoist_query) a = queries[k];
                    else {
#pragma unroll
                        for (int m = 0; m < 8; ++m)
                            a[m] = sycl::bit_cast<short>(
                                query[size_t(kv * 8 + m) * 128 + k * 16 + lane]);
                    }
                    flint_moe_v8i b{};
                    const int safe_absolute = valid ? absolute : begin + first;
                    const bf16* keys = kc + (size_t(kv) * cap + safe_absolute) * 128 + k * 16;
                    if constexpr (vector_key) {
                        sycl::vec<uint32_t, 8> packed{};
                        if (valid) packed.load(0, reinterpret_cast<const uint32_t*>(keys));
#pragma unroll
                        for (int j = 0; j < 8; ++j) b[j] = int(packed[j]);
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const uint16_t lo = valid ? sycl::bit_cast<uint16_t>(keys[2*j]) : 0;
                            const uint16_t high = valid ? sycl::bit_cast<uint16_t>(keys[2*j+1]) : 0;
                            b[j] = int(uint32_t(lo) | (uint32_t(high) << 16));
                        }
                    }
                    score = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, score, 0x3000);
                }
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const float value = valid ? score[m] * 0.08838834764831845f : -3.402823466e+38f;
                    if (valid) scores[size_t(kv * 8 + m) * cap + absolute] = value;
                    if constexpr (deferred_max) hi[m] = sycl::fmax(hi[m], value);
                    else {
                        float tile_max = value;
#pragma unroll
                        for (int mask = 8; mask > 0; mask >>= 1)
                            tile_max = sycl::fmax(tile_max,
                                sycl::permute_group_by_xor(sg, tile_max, mask));
                        hi[m] = sycl::fmax(hi[m], tile_max);
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                if constexpr (deferred_max)
                    hi[m] = sycl::reduce_over_group(sg, hi[m], sycl::maximum<>());
                if (!lane) stats[(size_t(kv * 8 + m) * max_parts + part) * 2] = hi[m];
            }
        });

    q.parallel_for(sycl::nd_range<1>(size_t(kv_heads) * max_parts * value_tiles * 16, 16),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int work = int(it.get_group(0)), tile = work % value_tiles;
            const int part = (work / value_tiles) % max_parts;
            const int kv = work / (value_tiles * max_parts);
            const int lane = int(it.get_local_id(0)), count = *position + 1;
            const int parts = sycl::max(minimum_parts,
                count <= 1024 ? 1 : count <= 4096 ? 16 : count <= 16384 ? 32 : 64);
            if (part >= parts) return;
            const int partition = (count + parts - 1) / parts;
            const int begin = part * partition;
            const int nkeys = sycl::max(0, sycl::min(partition, count - begin));
            const auto sg = it.get_sub_group();
            flint_moe_v8f accum[channel_blocks];
            float hi[8], z[8];
#pragma unroll
            for (int n = 0; n < channel_blocks; ++n) accum[n] = flint_moe_v8f{};
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                hi[m] = stats[(size_t(kv * 8 + m) * max_parts + part) * 2];
                const float sink_weight = part ? 0.f :
                    sycl::native::exp2((float(sink[kv * 8 + m]) - hi[m]) * 1.4426950408889634f);
                z[m] = deferred_sum && lane ? 0.f : sink_weight;
            }
            for (int first = 0; first < nkeys; first += 16) {
                const int key = first + lane, absolute = begin + key;
                const bool valid = key < nkeys;
                flint_moe_v8s p_hi, p_lo;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const float value = valid ? scores[size_t(kv * 8 + m) * cap + absolute] : 0.f;
                    const float probability = valid ?
                        sycl::native::exp2((value - hi[m]) * 1.4426950408889634f) : 0.f;
                    if constexpr (deferred_sum) z[m] += probability;
                    else z[m] += sycl::reduce_over_group(sg, probability, sycl::plus<>());
                    const bf16 high = bf16(probability);
                    p_hi[m] = sycl::bit_cast<short>(high);
                    p_lo[m] = sycl::bit_cast<short>(bf16(probability - float(high)));
                }
#pragma unroll
                for (int n = 0; n < channel_blocks; ++n) {
                    flint_moe_v8i b;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int key0 = first + 2*j, key1 = key0 + 1;
                        const int d = (tile * channel_blocks + n) * 16 + lane;
                        const uint16_t lo = key0 < nkeys ? sycl::bit_cast<uint16_t>(
                            vc[(size_t(kv) * cap + begin + key0) * 128 + d]) : 0;
                        const uint16_t high = key1 < nkeys ? sycl::bit_cast<uint16_t>(
                            vc[(size_t(kv) * cap + begin + key1) * 128 + d]) : 0;
                        b[j] = int(uint32_t(lo) | (uint32_t(high) << 16));
                    }
                    accum[n] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, p_hi, b, accum[n], 0x3000);
                    accum[n] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, p_lo, b, accum[n], 0x3000);
                }
            }
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                if constexpr (deferred_sum)
                    z[m] = sycl::reduce_over_group(sg, z[m], sycl::plus<>());
                const size_t row = size_t(kv * 8 + m) * max_parts + part;
#pragma unroll
                for (int n = 0; n < channel_blocks; ++n)
                    partial[row * 128 + (tile * channel_blocks + n) * 16 + lane] = accum[n][m];
                if (!lane && !tile) stats[row * 2 + 1] = z[m];
            }
        });
    return first_event;
}

inline sycl::event launch(int variant, sycl::queue& q, const bf16* query, const bf16* kc,
                          const bf16* vc, const bf16* sink, float* partial,
                          float* stats, float* scores, int kv_heads, int cap,
                          int minimum_parts, int max_parts, const int* position) {
#define ARCAINE_GLOBAL_VARIANT(N) case N: return partials<N>(q, query, kc, vc, sink, partial, stats, scores, kv_heads, cap, minimum_parts, max_parts, position)
    switch (variant) {
        ARCAINE_GLOBAL_VARIANT(1); ARCAINE_GLOBAL_VARIANT(2);
        ARCAINE_GLOBAL_VARIANT(3); ARCAINE_GLOBAL_VARIANT(4);
        ARCAINE_GLOBAL_VARIANT(5); ARCAINE_GLOBAL_VARIANT(6);
        ARCAINE_GLOBAL_VARIANT(7); ARCAINE_GLOBAL_VARIANT(8);
        ARCAINE_GLOBAL_VARIANT(9); ARCAINE_GLOBAL_VARIANT(10);
    }
#undef ARCAINE_GLOBAL_VARIANT
    throw std::invalid_argument("invalid Flint global attention variant");
}
}  // namespace flint::flash_decode_variants

#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "weights.hpp"

// bf16 DPAS: eight activation rows x sixteen output channels x K=16.
// Keep W4 weights in the loader's row-major layout; no duplicate weight bank.
using flint_moe_v8s = short __attribute__((ext_vector_type(8)));
using flint_moe_v8i = int __attribute__((ext_vector_type(8)));
using flint_moe_v8f = float __attribute__((ext_vector_type(8)));
#ifndef __SYCL_DEVICE_ONLY__
inline
#endif
SYCL_EXTERNAL flint_moe_v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int, flint_moe_v8s, flint_moe_v8i, flint_moe_v8f, int)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return {}; }
#endif

namespace flint::grouped_moe {

inline bool enabled() {
    const char* value = std::getenv("ARCAINE_FLINT_AWQ_GROUPED_MOE");
    return value && value[0] == '1' && value[1] == '\0';
}

// counts[E], offsets[E+1], cursors[E], tile_offsets[E+1].
// pairs[] holds the compact expert-major permutation of token/top-k slots.
inline size_t metadata_size(int experts) { return size_t(experts) * 4 + 2; }

inline bool supported(sycl::queue& q, const QTensor& gate,
                      const QTensor& up, const QTensor& down) {
    if (!q.has_property<sycl::property::queue::in_order>()) return false;
    auto tensor_ok = [](const QTensor& t) {
        return t.format == WeightFormat::AWQ && t.w && t.awq_s && t.awq_z &&
               t.experts > 0 && t.k > 0 && t.k % 64 == 0 && t.n > 0 &&
               t.n % 16 == 0 &&
               t.stride_w == size_t(t.n) * (t.k / 2) &&
               t.stride_s == size_t(t.n) * (t.k / 64) &&
               t.stride_b == t.stride_s;
    };
    if (!tensor_ok(gate) || !tensor_ok(up) || !tensor_ok(down) ||
        gate.n != up.n || gate.k != up.k || gate.n != down.k ||
        gate.k != down.n || gate.experts != up.experts ||
        gate.experts != down.experts) return false;
    const auto device = q.get_device();
    const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    return device.is_gpu() &&
           device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
           device.has(sycl::aspect::fp16) &&
           device.has(sycl::aspect::ext_intel_matrix) &&
           device.has(sycl::aspect::usm_device_allocations) &&
           std::find(sizes.begin(), sizes.end(), size_t(16)) != sizes.end();
}

inline void build_routes(sycl::queue& q, const int* ids, int slots,
                         int experts, int* metadata, int* pairs) {
    int* counts = metadata;
    int* offsets = counts + experts;
    int* cursors = offsets + experts + 1;
    int* tiles = cursors + experts;
    q.memset(counts, 0, size_t(experts) * sizeof(int));
    q.parallel_for(sycl::range<1>(size_t(slots)), [=](sycl::id<1> i) {
        const int e = ids[i];
        // Router IDs are in [0,E). Invalid IDs are a caller error.
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space> count(counts[e]);
        count.fetch_add(1);
    });
    q.single_task([=] {
        int row = 0, tile = 0;
        for (int e = 0; e < experts; ++e) {
            offsets[e] = cursors[e] = row;
            tiles[e] = tile;
            row += counts[e];
            tile += (counts[e] + 7) / 8;
        }
        offsets[experts] = row;
        tiles[experts] = tile;
    });
    q.parallel_for(sycl::range<1>(size_t(slots)), [=](sycl::id<1> i) {
        const int e = ids[i];
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space> cursor(cursors[e]);
        pairs[cursor.fetch_add(1)] = int(i[0]);
    });
}

inline flint_moe_v8i weight_fragment(const QTensor& t, int expert,
                                    int n, int k0) {
    const size_t param = size_t(expert) * t.stride_s + size_t(k0 / 64) * t.n + n;
    const float scale = float(t.awq_s[param]);
    const int zero = int(t.awq_z[size_t(expert) * t.stride_b +
                                 size_t(k0 / 64) * t.n + n]);
    const uint8_t* row = t.w + size_t(expert) * t.stride_w +
                         size_t(n) * (t.k / 2) + k0 / 2;
    flint_moe_v8i fragment;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned byte = row[j];
        const auto lo = sycl::bit_cast<uint16_t>(bf16(float(int(byte & 15) - zero) * scale));
        const auto hi = sycl::bit_cast<uint16_t>(bf16(float(int(byte >> 4) - zero) * scale));
        fragment[j] = int(uint32_t(lo) | (uint32_t(hi) << 16));
    }
    return fragment;
}

// The upper bound ceil(slots/8)+E-1 needs no device counter readback.
// Each work-group owns one 8x16 tile and looks up its expert on device.
template <bool GateUp>
inline void project(sycl::queue& q, const QTensor& weight, const QTensor& up,
                    const bf16* input, bf16* output, int slots, int top_k,
                    const int* metadata, const int* pairs) {
    const int experts = weight.experts, N = weight.n, K = weight.k;
    const int* offsets = metadata + experts;
    const int* tiles = metadata + 3 * experts + 1;
    const size_t max_tiles = (size_t(slots) + 7) / 8 + experts - 1;
    q.parallel_for(
        sycl::nd_range<2>({max_tiles, size_t(N)}, {1, 16}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int tile = int(it.get_group(0));
            if (tile >= tiles[experts]) return;
            int lo = 0, hi = experts;
            while (lo + 1 < hi) {
                const int mid = (lo + hi) / 2;
                if (tiles[mid] <= tile) lo = mid;
                else hi = mid;
            }
            const int e = lo;
            const int start = offsets[e] + (tile - tiles[e]) * 8;
            const int end = offsets[e + 1];
            const int lane = int(it.get_local_id(1));
            const int n = int(it.get_group(1)) * 16 + lane;
            int slot[8];
#pragma unroll
            for (int m = 0; m < 8; ++m)
                slot[m] = start + m < end ? pairs[start + m] : -1;
            flint_moe_v8f g{}, u{};
            for (int k0 = 0; k0 < K; k0 += 16) {
                flint_moe_v8s a;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const int source = GateUp ? slot[m] / top_k : slot[m];
                    a[m] = slot[m] < 0 ? short(0) :
                        sycl::bit_cast<short>(input[size_t(source) * K + k0 + lane]);
                }
                g = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                    16, a, weight_fragment(weight, e, n, k0), g, 0x3000);
                if constexpr (GateUp)
                    u = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, a, weight_fragment(up, e, n, k0), u, 0x3000);
            }
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                if (slot[m] < 0) continue;
                bf16 value(g[m]);
                if constexpr (GateUp) {
                    const float gf = float(value);
                    const float tail = 1.f / (1.f + sycl::exp(sycl::fabs(gf)));
                    const float sigmoid = gf < 0.f ? tail : 1.f - tail;
                    value = bf16(float(bf16(gf * sigmoid)) * float(bf16(u[m])));
                }
                output[size_t(slot[m]) * N + n] = value;
            }
        });
}

// Workspace belongs to the caller and must outlive queued work. No allocations,
// waits, host routing, weight repacks, or activation gather/scatter copies.
inline void run(sycl::queue& q, const QTensor& gate, const QTensor& up,
                const QTensor& down, const bf16* input, const int* ids,
                bf16* activation, bf16* output, int tokens, int top_k,
                int* metadata, int* pairs) {
    if (tokens <= 0 || top_k <= 0 || gate.experts <= 0)
        throw std::invalid_argument("flint grouped MoE: invalid token/top-k extent");
    const size_t extent = size_t(tokens) * size_t(top_k);
    const size_t max_tiles = (extent + 7) / 8 + size_t(gate.experts) - 1;
    if (extent > size_t(std::numeric_limits<int>::max()) ||
        max_tiles > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("flint grouped MoE: route extent exceeds int indexing");
    const int slots = int(extent);
    build_routes(q, ids, slots, gate.experts, metadata, pairs);
    project<true>(q, gate, up, input, activation, slots, top_k, metadata, pairs);
    project<false>(q, down, down, activation, output, slots, top_k, metadata, pairs);
}

}  // namespace flint::grouped_moe

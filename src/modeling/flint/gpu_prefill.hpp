#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "kernels.hpp"
#include "flash_decode.hpp"

// All query tokens are scheduled by the GPU. Current-chunk K/V remain outside
// the ring until attention completes, preserving old slots needed by early queries.
namespace flint::gpu_prefill {

inline bool enabled() {
    const char* value = std::getenv("ARCAINE_FLINT_GPU_PREFILL");
    return value && value[0] == '1' && value[1] == '\0';
}

inline int decode_min_parts(bool global=false) {
    const char* p = global?std::getenv("ARCAINE_FLINT_DECODE_GLOBAL_ATTN_SPLIT"):nullptr;
    if(!p)p=std::getenv("ARCAINE_FLINT_DECODE_ATTN_SPLIT");
    if (!p) return 1;
    if (!std::strcmp(p, "4")) return 4;
    if (!std::strcmp(p, "8")) return 8;
    if (!std::strcmp(p, "16")) return 16;
    if (!std::strcmp(p, "32")) return 32;
    if (!std::strcmp(p, "64")) return 64;
    if (!std::strcmp(p, "128")) return 128;
    if (!std::strcmp(p, "256")) return 256;
    if (!std::strcmp(p, "512")) return 512;
    if (!std::strcmp(p, "1024")) return 1024;
    return 1;
}

inline int parts_for(int count) {
    return count <= 1024 ? 1 : count <= 4096 ? 16 : count <= 16384 ? 32 : 64;
}

// K/V input: [M,kv_heads,128], already normalized and rotated.
// Cache: [kv_heads,cap,128]. Commit only the last cap rows to avoid
// overlapping ring writes when the chunk is longer than the sliding window.
inline void commit(sycl::queue& q, const bf16* k, const bf16* v,
                   bf16* kcache, bf16* vcache, int M, int kv_heads,
                   bool global, int base, int cap) {
    const int first = global ? 0 : std::max(0, M - cap);
    q.parallel_for(sycl::range<1>(size_t(M - first) * kv_heads * 128),
        [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int token = first + int(i / (kv_heads * 128));
            const int head = int(i / 128) % kv_heads;
            const int d = int(i % 128);
            const int slot = global ? base + token : (base + token) % cap;
            const size_t src = (size_t(token) * kv_heads + head) * 128 + d;
            const size_t dst = (size_t(head) * cap + slot) * 128 + d;
            kcache[dst] = k[src];
            vcache[dst] = v[src];
        });
}

// Reads historical cache without modifying it. The partition and reduction
// order match kernels::attention[_split/_merge], including attention sinks.
// partial/stats capacity: M*heads*parts_for(global ? base+M : cap) * {128,2}.
inline void attention(sycl::queue& q, const bf16* queries,
                      const bf16* k, const bf16* v,
                      const bf16* kcache, const bf16* vcache,
                      const bf16* sink, bf16* output,
                      float* partial, float* stats, int M, int heads,
                      int kv_heads, bool global, int base, int cap,
                      const int* device_position = nullptr,float* decode_scores=nullptr,const uint32_t* decode_packed_v=nullptr,
                      sycl::event* partial_event=nullptr,sycl::event* merge_event=nullptr) {
    const bool shallow_two_pass=device_position&&global&&!decode_scores&&flash_decode_twopass::adaptive();
    const int minimum_parts = device_position ? decode_min_parts(global&&!shallow_two_pass) : 1;
    const int max_parts = std::max(minimum_parts,
        parts_for(device_position ? cap : (global ? base + M : std::min(base + M, cap))));
    const int repeat = heads / kv_heads;
    sycl::event partial_submission;
    if(device_position && repeat==8 && max_parts>1 && flash_decode::tile()) {
        partial_submission=flash_decode::launch(q,queries,kcache,vcache,sink,partial,stats,kv_heads,
                             global,cap,minimum_parts,max_parts,device_position,decode_scores,decode_packed_v);
    } else partial_submission=q.parallel_for(
        sycl::nd_range<1>(size_t(M) * heads * max_parts * 32, 32),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const size_t group = it.get_group(0);
            const int part = int(group % max_parts);
            const int head = int(group / max_parts) % heads;
            const int token = int(group / (size_t(max_parts) * heads));
            const int pos = device_position ? *device_position : base + token;
            const int count = global ? pos + 1 : sycl::min(pos + 1, cap);
            const int parts = sycl::max(minimum_parts,
                count <= 1024 ? 1 : count <= 4096 ? 16 : count <= 16384 ? 32 : 64);
            if (part >= parts) return;
            const int start = pos + 1 - count;
            const int kvh = head / repeat;
            const auto sg = it.get_sub_group();
            const int lane = int(it.get_local_id(0));
            float qv[4], acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int d = 0; d < 4; ++d)
                qv[d] = float(queries[(size_t(token) * heads + head) * 128 +
                                      lane * 4 + d]) * 0.08838834764831845f;
            float hi = part == 0 ? float(sink[head]) : -3.402823466e+38f;
            float z = part == 0 ? 1.f : 0.f;
            for (int t = start + part; t <= pos; t += parts) {
                const bool current = !device_position && t >= base;
                const size_t index = current ?
                    (size_t(t - base) * kv_heads + kvh) * 128 + lane * 4 :
                    (size_t(kvh) * cap + (global ? t : t % cap)) * 128 + lane * 4;
                const bf16* kp = (current ? k : kcache) + index;
                const bf16* vp = (current ? v : vcache) + index;
                float score = 0.f;
#pragma unroll
                for (int d = 0; d < 4; ++d) score += qv[d] * float(kp[d]);
                score = sycl::reduce_over_group(sg, score, sycl::plus<>());
                const float next = sycl::fmax(hi, score);
                const float old_w = sycl::exp(hi - next);
                const float new_w = sycl::exp(score - next);
                z = z * old_w + new_w;
#pragma unroll
                for (int d = 0; d < 4; ++d)
                    acc[d] = acc[d] * old_w + new_w * float(vp[d]);
                hi = next;
            }
            const size_t row = (size_t(token) * heads + head) * max_parts + part;
            if (max_parts == 1) {
                const float inv = z == 0.f ? 0.f : 1.f / z;
#pragma unroll
                for (int d = 0; d < 4; ++d)
                    output[(size_t(token) * heads + head) * 128 + lane * 4 + d] =
                        bf16(acc[d] * inv);
            } else {
#pragma unroll
                for (int d = 0; d < 4; ++d) partial[row * 128 + lane * 4 + d] = acc[d];
                if (lane == 0) {
                    stats[row * 2] = hi;
                    stats[row * 2 + 1] = z;
                }
            }
        });
    if(partial_event)*partial_event=partial_submission;
    if (max_parts == 1) return;
    if(device_position && flash_decode::tile() &&
       flash_decode::merge(q,partial,stats,output,heads,global,cap,minimum_parts,max_parts,device_position,merge_event)){
        return;
    }
    const auto merge_submission=q.parallel_for(
        sycl::nd_range<1>(size_t(M) * heads * 32, 32),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const size_t query = it.get_group(0);
            const int token = int(query / heads);
            const int pos = device_position ? *device_position : base + token;
            const int count = global ? pos + 1 : sycl::min(pos + 1, cap);
            const int parts = sycl::max(minimum_parts,
                count <= 1024 ? 1 : count <= 4096 ? 16 : count <= 16384 ? 32 : 64);
            const int lane = int(it.get_local_id(0));
            float hi = -3.402823466e+38f;
            for (int p = 0; p < parts; ++p)
                hi = sycl::fmax(hi, stats[(query * max_parts + p) * 2]);
            float z = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int p = 0; p < parts; ++p) {
                const size_t row = query * max_parts + p;
                const float factor = sycl::exp(stats[row * 2] - hi);
                z += factor * stats[row * 2 + 1];
#pragma unroll
                for (int d = 0; d < 4; ++d)
                    acc[d] += factor * partial[row * 128 + lane * 4 + d];
            }
            const float inv = z == 0.f ? 0.f : 1.f / z;
#pragma unroll
            for (int d = 0; d < 4; ++d)
                output[query * 128 + lane * 4 + d] = bf16(acc[d] * inv);
        });
    if(merge_event)*merge_event=merge_submission;
}

}  // namespace flint::gpu_prefill

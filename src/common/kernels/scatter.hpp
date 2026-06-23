#pragma once
#include <sycl/sycl.hpp>
#include "../gpu/buffer.hpp"

// Replace positions where mask[i] == true in seq_embeds with rows from modal_embeds.
// modal_embeds rows are consumed in order of the true positions in mask.
// seq_embeds: (total_seq, H)
// modal_embeds: (num_modal_tokens, H)
// mask: (total_seq,) bool — must have exactly num_modal_tokens true entries.
// mask: (total_seq,) uint8 (1=replace, 0=keep)
inline void masked_scatter_bf16(
    sycl::queue& q,
    bf16* seq_embeds,
    const bf16* modal_embeds,
    const int32_t* offsets,  // precomputed prefix-sum → modal row index per seq pos
    const uint8_t* mask,
    int total_seq, int H
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(total_seq, H), [=](sycl::id<2> id) {
            int i = id[0], d = id[1];
            if (mask[i]) {
                int modal_row = offsets[i];
                seq_embeds[i * H + d] = modal_embeds[modal_row * H + d];
            }
        });
    });
}

// Compute prefix-sum offset array from a host uint8 mask and upload to GPU.
inline GpuBuffer<int32_t> compute_scatter_offsets(
    sycl::queue& q,
    const uint8_t* host_mask,
    int total_seq
) {
    std::vector<int32_t> offsets(total_seq);
    int count = 0;
    for (int i = 0; i < total_seq; ++i) {
        offsets[i] = count;
        if (host_mask[i]) ++count;
    }
    GpuBuffer<int32_t> buf(total_seq);
    buf.upload(offsets.data(), total_seq);
    return buf;
}

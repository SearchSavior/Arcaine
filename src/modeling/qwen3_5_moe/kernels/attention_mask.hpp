#pragma once
#include <sycl/sycl.hpp>
#include <limits>
#include <climits>

// Fill a float mask buffer (q_len × kv_len) with 0.0 or -inf.
// Masking rules (applied together):
//   causal:  kv_j > q_global_pos  → -inf   (future token)
//   sliding: kv_j < q_global_pos - sliding_window + 1 → -inf  (too far back)
// past_offset: absolute position of kv[0] in the full sequence.
// sliding_window: pass INT_MAX for full (global) attention layers.
inline void fill_causal_mask(
    sycl::queue& q,
    float* mask,           // (q_len, kv_len) device ptr
    int q_len, int kv_len,
    int past_offset,       // absolute position of the first KV in cache
    int sliding_window     // INT_MAX = no sliding constraint
) {
    static const float NEG_INF = -std::numeric_limits<float>::infinity();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(q_len, kv_len), [=](sycl::id<2> id) {
            int qi  = id[0];   // query index within this call
            int kvj = id[1];   // KV index within the cache

            int q_global  = past_offset + qi; // absolute query position
            int kv_global = past_offset - (kv_len - q_len) + kvj;
            // kv_len = past_kv_filled + q_len;
            // kv_global[0] = past_offset - past_kv_filled = first cached KV position

            bool causal  = kv_global > q_global;
            bool too_far = (sliding_window != INT_MAX) &&
                           (kv_global < q_global - sliding_window + 1);

            mask[qi * kv_len + kvj] = (causal || too_far) ? NEG_INF : 0.0f;
        });
    });
}

// Same as fill_causal_mask, plus Gemma4 Unified's bidirectional vision blocks.
// block_ids contains -1 for non-vision tokens and a non-negative group id for
// each contiguous image/video block. Tokens in the same non-negative block id
// are allowed to attend bidirectionally within that block.
inline void fill_causal_mask_with_block_ids(
    sycl::queue& q,
    float* mask,
    const int32_t* block_ids,
    int q_len, int kv_len,
    int past_offset,
    int sliding_window
) {
    static const float NEG_INF = -std::numeric_limits<float>::infinity();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(q_len, kv_len), [=](sycl::id<2> id) {
            int qi  = id[0];
            int kvj = id[1];

            int q_global  = past_offset + qi;
            int kv_global = past_offset - (kv_len - q_len) + kvj;

            int q_block  = block_ids ? block_ids[q_global] : -1;
            int kv_block = block_ids ? block_ids[kv_global] : -1;
            bool same_vision_block = q_block >= 0 && q_block == kv_block;

            bool causal  = !same_vision_block && kv_global > q_global;
            bool too_far = (sliding_window != INT_MAX) &&
                           (kv_global < q_global - sliding_window + 1);

            mask[qi * kv_len + kvj] = (causal || too_far) ? NEG_INF : 0.0f;
        });
    });
}

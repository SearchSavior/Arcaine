#pragma once
#include <sycl/sycl.hpp>
#include "../gpu/buffer.hpp"
#include <cmath>

// Embedding lookup with scaling.
// out[i, :] = table[ids[i], :] * scale
// scale = sqrt(hidden_size) per Gemma convention.
inline void embedding_lookup(
    sycl::queue& q,
    const bf16* table,     // (vocab_size, H)
    const int32_t* ids,    // (seq_len,) device ptr
    bf16* out,             // (seq_len, H)
    int seq_len, int H,
    float scale
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(seq_len, H), [=](sycl::id<2> id) {
            int tok = id[0];
            int d   = id[1];
            int token_id = ids[tok];
            float v = bf16_to_float(table[token_id * H + d]) * scale;
            out[tok * H + d] = float_to_bf16(v);
        });
    });
}

#pragma once
#include "../../common/gpu/buffer.hpp"
#include "weights.hpp"

// FFN forward: out = (GELU(x @ gate.T) * (x @ up.T)) @ down.T
// x: (seq, hidden_size), out: (seq, hidden_size)
// ctx: GPU context; all buffers must live on ctx's device.
void ffn_forward(
    GpuEngine& ctx,
    const FfnWeights& w,
    const bf16* x,
    bf16* out,
    int seq_len,
    int hidden_size,
    int intermediate_size
);

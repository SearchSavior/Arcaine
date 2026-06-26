#include "ffn.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/kernels/elementwise.hpp"

void ffn_forward(
    GpuEngine& ctx,
    const FfnWeights& w,
    const bf16* x,
    bf16* out,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    auto& q = ctx.queue;

    GpuBuffer<bf16> gate((size_t)seq_len * intermediate_size, q);
    GpuBuffer<bf16> up  ((size_t)seq_len * intermediate_size, q);

    // Both matmuls read x — submit to queue back-to-back (both async).
    matmul_bf16(x, seq_len, hidden_size, w.gate_proj.data(), intermediate_size, gate.data(), ctx);
    matmul_bf16(x, seq_len, hidden_size, w.up_proj.data(),   intermediate_size, up.data(),   ctx);

    // Fused GeGLU: gate[i] = gelu(gate[i]) * up[i]
    geglu_inplace(q, gate.data(), up.data(), seq_len * intermediate_size);

    matmul_bf16(gate.data(), seq_len, intermediate_size, w.down_proj.data(), hidden_size, out, ctx);
}

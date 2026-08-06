#include "ffn.hpp"
#include "../../runtime/gpu/ops.hpp"
#include "../../runtime/gpu/engine.hpp"
#include "runtime/kernels/elementwise.hpp"

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
    proj_matmul(w.gate_proj, x, seq_len, hidden_size, intermediate_size, gate.data(), ctx);
    proj_matmul(w.up_proj, x, seq_len, hidden_size, intermediate_size, up.data(), ctx);

    // Fused GeGLU: gate[i] = gelu(gate[i]) * up[i]
    geglu_inplace(q, gate.data(), up.data(), seq_len * intermediate_size);

    proj_matmul(w.down_proj, gate.data(), seq_len, intermediate_size, hidden_size, out, ctx);
}

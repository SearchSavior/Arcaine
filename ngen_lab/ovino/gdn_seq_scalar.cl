// Scalar sequential GDN prefill kernel (OpenVINO ref strategy, scalar path).
// One work-item per (h, v-block): serializes over t and k. Uses ONLY
// get_global_id(0/1) indexing proven bit-exact on this driver path.
// grid {nv, vblocks} local {1,1}.  q,k,v,beta,g = bf16(ushort), state fp32.
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))

inline float bf16_to_f32(ushort v) {
    uint u = ((uint)v) << 16;
    return as_float(u);
}
inline ushort f32_to_bf16(float f) {
    uint u = as_uint(f);
    uint rounding_bias = ((u >> 16) & 1) + 0x7FFFu;
    return (ushort)((u + rounding_bias) >> 16);
}

__kernel void gdn_seq_scalar(
    __global ushort* q,       // [S, nv, dk]
    __global ushort* k,       // [S, nv, dk]
    __global ushort* v,       // [S, nv, dv]
    __global float* state,    // [nv, dk, dv] in/out
    __global ushort* beta,    // [S, nv]
    __global ushort* g,       // [S, nv]
    __global ushort* output,  // [S, nv, dv]
    int seq_len, int q_t_stride, int k_t_stride, int v_t_stride) {
    const int h = get_global_id(0);          // value head
    const int vb = get_global_id(1);         // v-block index
    const int K_len = 128, V_len = 128, V_BLOCK = 4;
    const int start_iv = vb * V_BLOCK;       // [0, 128)

    float S[V_BLOCK][K_len];                 // 4 x 128 state columns
    const int STATE_BASE = h * (K_len * V_len);
    for (int v_idx = 0; v_idx < V_BLOCK; v_idx++) {
        const int cv = start_iv + v_idx;
        for (int dk = 0; dk < K_len; dk++)
            S[v_idx][dk] = state[STATE_BASE + dk * V_len + cv];
    }

    for (int t = 0; t < seq_len; t++) {
        const size_t row = (size_t)t * 32 + h;
        const float eg = exp(bf16_to_f32(g[row]));
        const float bb = bf16_to_f32(beta[row]);
        const ushort* qrow = q + (size_t)row * K_len;
        const ushort* krow = k + (size_t)row * K_len;
        const ushort* vrow = v + (size_t)row * V_len;
        for (int v_idx = 0; v_idx < V_BLOCK; v_idx++) {
            const int cv = start_iv + v_idx;
            float kv_mem = 0.0f;
            for (int dk = 0; dk < K_len; dk++)
                kv_mem += S[v_idx][dk] * eg * bf16_to_f32(krow[dk]);
            const float delta = (bf16_to_f32(vrow[cv]) - kv_mem) * bb;
            float o = 0.0f;
            for (int dk = 0; dk < K_len; dk++) {
                const float s = S[v_idx][dk] * eg + bf16_to_f32(krow[dk]) * delta;
                S[v_idx][dk] = s;
                o += s * bf16_to_f32(qrow[dk]);
            }
            output[(size_t)row * V_len + cv] = f32_to_bf16(o);
        }
    }

    for (int v_idx = 0; v_idx < V_BLOCK; v_idx++) {
        const int cv = start_iv + v_idx;
        for (int dk = 0; dk < K_len; dk++)
            state[STATE_BASE + dk * V_len + cv] = S[v_idx][dk];
    }
}

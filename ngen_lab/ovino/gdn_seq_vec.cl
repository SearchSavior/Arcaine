// Vectorized sequential GDN prefill (OpenVINO strategy, SLM-reduced).
// One WG per (h, v-block): 16 lanes x 8 K-rows = 128 K; 4 V-cols per WG.
// K-dot reduction via SLM+barrier (subgroup builtins are unreliable on the
// L0/SPIR-V path). Layout x-first (proven bit-exact): grid {vblocks*16, nv, 1},
// local (16,1,1); v_lane=get_global_id(0), h=get_global_id(1), lid=glid0%16.
// q,k,v,beta,g = bf16 (ushort raw bits); state = fp32; core out = bf16.

inline float bf16_to_f32(ushort v) {
    uint u = ((uint)v) << 16;
    return as_float(u);
}
inline ushort f32_to_bf16(float f) {
    uint u = as_uint(f);
    uint rounding_bias = ((u >> 16) & 1) + 0x7FFFu;
    return (ushort)((u + rounding_bias) >> 16);
}

__kernel void gdn_seq_vec(
    __global ushort* q,       // [S, nv, dk]
    __global ushort* k,       // [S, nv, dk]
    __global ushort* v,       // [S, nv, dv]
    __global float* state,    // [nv, dk, dv] in/out
    __global ushort* beta,    // [S, nv]
    __global ushort* g,       // [S, nv]
    __global ushort* output,  // [S, nv, dv]
    int seq_len, int q_t_stride, int k_t_stride, int v_t_stride) {
    const int v_lane = get_global_id(0);
    const int start_iv = (v_lane / 16) * 4;
    const int h = get_global_id(1);
    const int lid = get_local_id(0);  // workgroup-local lane
    const int K_len = 128, V_len = 128;
    const int K_CHUNKS = K_len / 128;  // 1: lane holds 8 k-rows

    __local float slm_k[4][16];  // per v-col, per-lane partial dot_k
    __local float slm_q[4][16];  // per v-col, per-lane partial dot_q
    __local float slm_out[4];    // broadcast output

    float h_state[4][8];         // 4 v-cols x 8 k-rows per lane
    const int STATE_BASE = h * (K_len * V_len);
    for (int v_idx = 0; v_idx < 4; v_idx++) {
        const int cv = start_iv + v_idx;
        for (int j = 0; j < 8; j++) {
            const int dk = lid * 8 + j;
            h_state[v_idx][j] = state[STATE_BASE + dk * V_len + cv];
        }
    }

    for (int t = 0; t < seq_len; t++) {
        const size_t row = (size_t)t * 32 + h;
        const float eg = exp(bf16_to_f32(g[row]));
        const float bb = bf16_to_f32(beta[row]);
        const ushort* qrow = q + (size_t)row * K_len;
        const ushort* krow = k + (size_t)row * K_len;
        const ushort* vrow = v + (size_t)row * V_len;
        const int kb = lid * 8;  // this lane's 8 k-rows

        // partial dot_k per v-col (lane-local over 8 k)
        for (int v_idx = 0; v_idx < 4; v_idx++) {
            float acc = 0.0f;
            for (int j = 0; j < 8; j++) {
                const float s = h_state[v_idx][j] * eg;
                h_state[v_idx][j] = s;
                acc += s * bf16_to_f32(krow[kb + j]);
            }
            slm_k[v_idx][lid] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // reduce over 16 lanes: h_k per v-col
        float h_k[4];
        for (int v_idx = 0; v_idx < 4; v_idx++) {
            float s = 0.0f;
            for (int l = 0; l < 16; l++) s += slm_k[v_idx][l];
            h_k[v_idx] = s;
        }

        // update + partial dot_q
        for (int v_idx = 0; v_idx < 4; v_idx++) {
            const float delta = (bf16_to_f32(vrow[start_iv + v_idx]) - h_k[v_idx]) * bb;
            float acc = 0.0f;
            for (int j = 0; j < 8; j++) {
                const float s = h_state[v_idx][j] + bf16_to_f32(krow[kb + j]) * delta;
                h_state[v_idx][j] = s;
                acc += s * bf16_to_f32(qrow[kb + j]);
            }
            slm_q[v_idx][lid] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int v_idx = 0; v_idx < 4; v_idx++) {
            float s = 0.0f;
            for (int l = 0; l < 16; l++) s += slm_q[v_idx][l];
            slm_out[v_idx] = s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0)
            for (int v_idx = 0; v_idx < 4; v_idx++)
                output[(size_t)row * V_len + start_iv + v_idx] = f32_to_bf16(slm_out[v_idx]);
    }

    for (int v_idx = 0; v_idx < 4; v_idx++) {
        const int cv = start_iv + v_idx;
        for (int j = 0; j < 8; j++) {
            const int dk = lid * 8 + j;
            state[STATE_BASE + dk * V_len + cv] = h_state[v_idx][j];
        }
    }
}

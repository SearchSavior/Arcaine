// Decay-only test: S'[k][v] = S[k][v] * exp(g[t=0][h]) for all k,v.
// grid {1, nv, vblocks*16} local (1,1,16) -- exact GDN dispatch.
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
inline float bf16_to_f32(ushort v) { uint u = ((uint)v) << 16; return as_float(u); }
REQD_SUB_GROUP_SIZE(16)
__kernel void decay_test(__global float* state, __global ushort* g) {
    const int v_lane = get_global_id(2);
    const int start_iv = (v_lane / 16) * 4;
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int lid = get_sub_group_local_id();
    const int K_len = 128, V_len = 128;
    const int STATE_BASE = (b * 32 + h) * (K_len * V_len);

    float eg = exp(bf16_to_f32(g[h]));  // g[t=0][h], bf16 bits
    for (int v_idx = 0; v_idx < 4; v_idx++) {
        int curr_iv = start_iv + v_idx;
        for (int j = 0; j < 8; j++) {
            int row_idx = lid * 8 + j;
            state[STATE_BASE + row_idx * V_len + curr_iv] *= eg;
        }
    }
}

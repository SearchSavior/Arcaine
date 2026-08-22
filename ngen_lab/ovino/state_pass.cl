// State passthrough: loads initial_state[h][k][v] and writes it back verbatim
// using the GDN kernel's EXACT indexing (x-first). If the mapping is right,
// out state == in state bit-exact. Also writes a marker to core[0].
//   v_lane = get_global_id(0); start_iv = (v_lane/16)*4; h = get_global_id(1)
//   lane lid holds rows [lid*8, lid*8+8), columns [start_iv, start_iv+4)
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void state_pass(__global float* initial_state, __global ushort* output) {
    const int v_lane = get_global_id(0);
    const int start_iv = (v_lane / 16) * 4;
    const int h = get_global_id(1);
    const int lid = get_sub_group_local_id();
    const int K_len = 128, V_len = 128;
    const int STATE_BASE = h * (K_len * V_len);

    for (int v_idx = 0; v_idx < 4; v_idx++) {
        int curr_iv = start_iv + v_idx;
        for (int j = 0; j < 8; j++) {
            int row_idx = lid * 8 + j;
            float s = initial_state[STATE_BASE + row_idx * V_len + curr_iv];
            initial_state[STATE_BASE + row_idx * V_len + curr_iv] = s;  // round-trip
        }
    }
    if (lid == 0 && h == 0 && start_iv == 0)
        output[0] = 0x3F00;  // 0.5 bf16 marker
}

// QKV passthrough: lane lid loads q/k (8 elems) and v (4 elems) for (t=0, h)
// using the GDN kernel's EXACT x-first indexing, writes them back to output
// buffers. Verifies the load addressing is bit-exact.
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void qkv_pass(__global ushort* q, __global ushort* k,
                       __global ushort* v, __global ushort* qout,
                       __global ushort* kout, __global ushort* vout) {
    const int v_lane = get_global_id(0);
    const int start_iv = (v_lane / 16) * 4;
    const int h = get_global_id(1);
    const int lid = get_sub_group_local_id();
    const int lane_k_base = lid * 8;
    const int t = 0;
    const int Q_T_STRIDE = 4096, K_T_STRIDE = 4096, V_T_STRIDE = 4096;

    int q_offset = t * Q_T_STRIDE + h * 128;
    int k_offset = t * K_T_STRIDE + h * 128;
    int v_offset = t * V_T_STRIDE + h * 128;

    for (int j = 0; j < 8; j++) {
        qout[h * 128 + lane_k_base + j] = q[q_offset + lane_k_base + j];
        kout[h * 128 + lane_k_base + j] = k[k_offset + lane_k_base + j];
    }
    for (int j = 0; j < 4; j++)
        vout[h * 128 + start_iv + j] = v[v_offset + start_iv + j];
}

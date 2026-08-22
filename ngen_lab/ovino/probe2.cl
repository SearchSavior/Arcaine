// Extended probe: exercises the exact features the GDN kernel uses.
//  - 3D grid {1, nv, vblocks*16}, local {1,1,16}
//  - get_global_id(0/1), get_group_id(2), get_sub_group_local_id()
//  - scalar int args (seq_len, strides)
//  - vload8/vload4 vector loads
//  - sub_group_reduce_add over float4 components
//  - lid==0 conditional store
// Kernel: for WG (g0,g1,g2) with lanes 0..15:
//   out[g1 * vblocks*16 + g2 + lid] = (float)(g0*1000 + g1*100 + g2 + lid)
//   scalar_out[g1*vblocks + g2/16 + lid] = (float)(seq_len + q_t_stride)
//   red[g1*vblocks + g2/16] = reduce_add of lane values from a vload8 (sum of
//   v[g1*8 + lid*8 + 0..8) cast to float)  -> 8*v_base + 8*(lid*8) + 28
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void probe2(__global ushort* v, __global float* out,
                     __global float* scalar_out, __global float* red,
                     int seq_len, int q_t_stride) {
    const int g0 = get_global_id(0);
    const int g1 = get_global_id(1);
    const int g2 = get_group_id(2);
    const int lid = get_sub_group_local_id();

    // per-lane scattered store (all lanes)
    out[g1 * 512 + g2 + lid] = (float)(g0 * 1000 + g1 * 100 + g2 + lid);

    // scalar args
    if (lid == 0)
        scalar_out[g1 * 32 + g2 / 16] = (float)(seq_len + q_t_stride);

    // vload8 + reduce_add
    float8 vv = convert_float8(vload8(0, v + g1 * 128 + lid * 8));
    float red_sum = vv.s0 + vv.s1 + vv.s2 + vv.s3 + vv.s4 + vv.s5 + vv.s6 + vv.s7;
    float r = sub_group_reduce_add(red_sum);
    if (lid == 0)
        red[g1 * 32 + g2 / 16] = r;
}

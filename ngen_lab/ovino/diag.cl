// Diagnostic: dump local/global/group ids to reveal the driver's actual
// workgroup <-> subgroup mapping for grid {1,1,4} local {1,1,16}.
// Each lane writes: out[glid_global_flat] = (get_sub_group_local_id(), get_group_id(2))
// packed as lid*1000 + group_id2.
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void diag(__global uint* out, __global uint* sizes) {
    const int lid = get_sub_group_local_id();
    const int g2 = get_group_id(2);
    const int gl2 = get_global_id(2);
    if (lid == 0) {
        out[g2 * 3 + 0] = (uint)g2;
        out[g2 * 3 + 1] = (uint)gl2;
        out[g2 * 3 + 2] = (uint)lid;
    }
}

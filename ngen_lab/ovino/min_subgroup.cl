// Minimal subgroup SPIR-V probe for the L0 loader path.
// out[lid] = lid ; out[16+lid] = sub_group_reduce_add((float)(lid+1))
// Expected: out[0..16)=0..15, out[16..32)=136 (sum of 1..16).
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void min_subgroup(__global float* out) {
    int lid = get_sub_group_local_id();
    out[lid] = (float)lid;
    float lane_val = (float)(lid + 1);
    float red = sub_group_reduce_add(lane_val);
    if (lid == 0)
        out[16] = red;
}

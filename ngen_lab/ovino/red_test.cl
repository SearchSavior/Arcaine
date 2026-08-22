// Subgroup reduce with multi-group x-first: grid {512,32,1} local (16,1,1).
// Each lane l (0..15) of group g contributes (g*16+l+1); the reduce for group g
// should equal sum_{l=0..15} (g*16+l+1) = 16*g*16 + 136.
// red[h*32 + g] must == 256*g + 136 (only lane 0 writes).
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void red_test(__global float* red, __global float* size_out) {
    const int g = get_global_id(0) / 16;   // group index in x
    const int h = get_global_id(1);
    const int lid = get_sub_group_local_id();
    float lane_val = (float)(g * 16 + lid + 1);
    float r = sub_group_reduce_add(lane_val);
    if (lid == 0) {
        red[h * 32 + g] = r;
        if (h == 0 && g == 0) size_out[0] = (float)get_sub_group_size();
    }
}

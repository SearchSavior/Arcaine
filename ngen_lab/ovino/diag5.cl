#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void diag5(__global uint* out) {
    const int glid0 = get_global_id(0);
    out[glid0] = (uint)glid0;
}

// Diagnostic 3: x-first subgroup layout. Grid {512,1,1}, local {16,1,1}.
// Records (get_global_id(0), get_group_id(0), get_sub_group_local_id(), sg size).
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void diag3(__global uint* out) {
    const int glid0 = get_global_id(0);
    const int gid0 = get_group_id(0);
    const int lid = get_sub_group_local_id();

    out[glid0 * 4 + 0] = (uint)glid0;
    out[glid0 * 4 + 1] = (uint)gid0;
    out[glid0 * 4 + 2] = (uint)lid;
    out[glid0 * 4 + 3] = (uint)glid0;
}

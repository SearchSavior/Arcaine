// Diagnostic: record (get_global_id(2), get_group_id(2), get_sub_group_local_id(),
// get_sub_group_size()) for every z-slot into a generously sized buffer.
// Layout: out[(glid2*4 + 0..3)] where glid2 in [0,512).
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void diag2(__global uint* out) {
    const int glid2 = get_global_id(2);
    const int gid2 = get_group_id(2);
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_size();
    out[glid2 * 4 + 0] = (uint)glid2;
    out[glid2 * 4 + 1] = (uint)gid2;
    out[glid2 * 4 + 2] = (uint)lid;
    out[glid2 * 4 + 3] = (uint)sg;
}

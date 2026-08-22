__kernel void diag4(__global uint* out) {
    const int glid0 = get_global_id(0);
    out[glid0] = (uint)glid0;
}

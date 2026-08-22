// Multi-group barrier probe: grid {512,1,1}, local (16,1,1).
// Lane l writes scratch[l] = glid0; barrier; each lane checks that all 16
// scratch slots are set and sums them; lane 0 writes the sum to out[group].
// If barrier works, sum == sum over the 16 glid0 values in this group.
__kernel void barrier_probe(__global float* out) {
    __local float scratch[16];
    const int glid0 = get_global_id(0);
    const int lid = get_local_id(0);
    const int g = get_group_id(0);
    scratch[lid] = (float)glid0;
    barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.0f;
    for (int i = 0; i < 16; ++i) s += scratch[i];
    if (lid == 0) out[g] = s;
}

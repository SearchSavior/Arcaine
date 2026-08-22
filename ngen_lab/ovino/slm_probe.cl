// SLM probe: 16 lanes each write 1 value to SLM, barrier, lane 0 sums them.
__kernel void slm_probe(__global float* out) {
    __local float scratch[16];
    const int lid = get_local_id(0);
    scratch[lid] = (float)(lid + 1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        float s = 0;
        for (int i = 0; i < 16; ++i) s += scratch[i];
        out[0] = s;
    }
}

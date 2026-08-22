// Canonical oneDNN-style DPAS-recognizable kernel: 8x8x16 tile GEMM microkernel.
// Each lane loads A (8x8 bf16, lane-major) and B (8x16 bf16), does 8-deep
// fma chains -> the pattern IGC's DPAS auto-detection is designed to catch.
// grid {1,1,1}, local (16,1,1). Simple enough to verify by disassembly.
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void dpas_probe(__global const ushort* A, __global const ushort* B,
                         __global float* C) {
    const int lid = get_sub_group_local_id();
    // lane l holds row l of an 8x8 tile: A[l][0..7]
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int k = 0; k < 8; k++) {
        float8 a = vload8(0, A + (lid * 8 + k) * 8);
        float8 b = vload8(0, B + (k * 8 + lid * 0) * 8);
        // A row lid, B row k: outer-product accumulate -> 8x8 tile
        for (int j = 0; j < 8; j++) {
            acc[0] = fma(a.s0, b.s0, acc[0]);
            acc[1] = fma(a.s1, b.s1, acc[1]);
            acc[2] = fma(a.s2, b.s2, acc[2]);
            acc[3] = fma(a.s3, b.s3, acc[3]);
            acc[4] = fma(a.s4, b.s4, acc[4]);
            acc[5] = fma(a.s5, b.s5, acc[5]);
            acc[6] = fma(a.s6, b.s6, acc[6]);
            acc[7] = fma(a.s7, b.s7, acc[7]);
        }
    }
    for (int j = 0; j < 8; j++)
        C[lid * 8 + j] = acc[j];
}

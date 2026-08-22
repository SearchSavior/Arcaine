// Test whether IGC auto-lowers FP32 FMA chains to DPAS when the dummy DPAS
// builtin is referenced (oneDNN xe_hp_systolic_copy.cl mechanism).
void dummy_dpas() {
    if (get_sub_group_local_id() >= 16) {
        int __builtin_IB_sub_group_idpas_s8_s8_8_1(int, int, int8)
                __attribute__((const));
        global volatile int *_;
        int z = __builtin_IB_sub_group_idpas_s8_s8_8_1(0, _[0], 1);
        for (int i = 0; i < z; i++) (void)_[0];
    }
}
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void dpas_trick(__global const ushort* A, __global const ushort* B,
                         __global float* C) {
    dummy_dpas();
    const int lid = get_sub_group_local_id();
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int k = 0; k < 8; k++) {
        float8 a = convert_float8(vload8(0, A + (lid * 8 + k) * 8));
        float8 b = convert_float8(vload8(0, B + k * 8));
        acc[0] = fma(a.s0, b.s0, acc[0]);
        acc[1] = fma(a.s1, b.s1, acc[1]);
        acc[2] = fma(a.s2, b.s2, acc[2]);
        acc[3] = fma(a.s3, b.s3, acc[3]);
        acc[4] = fma(a.s4, b.s4, acc[4]);
        acc[5] = fma(a.s5, b.s5, acc[5]);
        acc[6] = fma(a.s6, b.s6, acc[6]);
        acc[7] = fma(a.s7, b.s7, acc[7]);
    }
    for (int j = 0; j < 8; j++) C[lid * 8 + j] = acc[j];
}

// Actively use the DPAS builtin -- should produce a real dpas instruction.
#define REQD_SUB_GROUP_SIZE(sg) __attribute__((intel_reqd_sub_group_size(sg)))
REQD_SUB_GROUP_SIZE(16)
__kernel void dpas_live(__global const int* A, __global const int* B,
                        __global int* C) {
    if (get_sub_group_local_id() < 16) {
        int __builtin_IB_sub_group_idpas_s8_s8_8_1(int, int, int8)
                __attribute__((const));
        global volatile int *_;
        int z = __builtin_IB_sub_group_idpas_s8_s8_8_1(0, _[0], 1);
        C[0] = z;
    }
}

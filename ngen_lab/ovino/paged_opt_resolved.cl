// JIT-baked for Arcaine Qwen3.5-MoE GDN: bf16 in, fp32 state, bf16 out.
#define KERNEL(name) __kernel void name
#define FUNC(name) name
#define FUNC_CALL(name) name
#define INPUT0_TYPE ushort
#define INPUT1_TYPE ushort
#define INPUT2_TYPE ushort
#define INPUT3_TYPE float
#define INPUT4_TYPE ushort
#define INPUT5_TYPE ushort
#define INPUT6_TYPE int
#define INPUT7_TYPE int
#define INPUT8_TYPE int
#define INPUT9_TYPE int
#define INPUT10_TYPE int
#define OUTPUT_TYPE ushort
#define TO_OUTPUT_TYPE(v) f32_to_bf16(v)
#define K_HEAD_NUM 32
#define V_HEAD_NUM 32
#define K_HEAD_DIM 128
#define V_HEAD_DIM 128
#define V_BLOCK_SIZE 4
#define SUBGROUP_SIZE 16
#define K_VEC_SIZE 8
#define FUSE_QK_L2NORM 0
#define SCALE_FACTOR 1.0f
#define Q_L2_NORM_EPS 1e-6f
#define K_L2_NORM_EPS 1e-6f
inline float bf16_to_f32(ushort v) { uint u = ((uint)v) << 16; return as_float(u); }
inline ushort f32_to_bf16(float f) { uint u = as_uint(f); uint b = ((u >> 16) & 1) + 0x7FFFu; return (ushort)((u + b) >> 16); }
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ==== inlined include/batch_headers/common.cl ====
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))

#if OPT_HINTS_SUPPORTED
#   define ASSUME_HINT(x) __builtin_assume(x)
#else
#   define ASSUME_HINT(x) do { } while (0)
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b)      ((a) < (b) ? (a) : (b))
#define MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type)  elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type)  CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type)  CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type)  CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type)  CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size)   CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE_PREFIX_uchar as_
#define AS_TYPE_PREFIX_char as_
#define AS_TYPE_PREFIX_fp4e2m1_t _as_
#define AS_TYPE_PREFIX_fp8e5m2_t _as_
#define AS_TYPE_PREFIX_fp8e4m3_t _as_
#define AS_TYPE_PREFIX_fp8e8m0_t _as_
#define AS_TYPE_PREFIX_ushort as_
#define AS_TYPE_PREFIX_short as_
#define AS_TYPE_PREFIX_half as_
#define AS_TYPE_PREFIX_int as_
#define AS_TYPE_PREFIX_uint as_
#define AS_TYPE_PREFIX_float as_
#define AS_TYPE_PREFIX_ulong as_
#define AS_TYPE_PREFIX_long as_

#define AS_TYPE_EXT(type, val, src_type) CAT(CAT(AS_TYPE_PREFIX_, src_type), type)(val)
#define AS_TYPE(type, val) CAT(as_, type)(val)

// ====================================================================================================================
// TYPE_SIZE(type) - evaluates to size of "type" in bytes
// type [PP] - Must evaluate to non-vectorized type.
// ====================================================================================================================
#define TYPE_SIZE_uchar  1
#define TYPE_SIZE_char   1
#define TYPE_SIZE_fp8e5m2_t 1
#define TYPE_SIZE_fp8e4m3_t 1
#define TYPE_SIZE_fp8e8m0_t 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short  2
#define TYPE_SIZE_half   2
#define TYPE_SIZE_int    4
#define TYPE_SIZE_uint   4
#define TYPE_SIZE_float  4
#define TYPE_SIZE_ulong  8
#define TYPE_SIZE_long   8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif

// ==== inlined include/batch_headers/sub_group_block_read.cl ====
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ==== inlined common.cl ====
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))

#if OPT_HINTS_SUPPORTED
#   define ASSUME_HINT(x) __builtin_assume(x)
#else
#   define ASSUME_HINT(x) do { } while (0)
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b)      ((a) < (b) ? (a) : (b))
#define MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type)  elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type)  CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type)  CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type)  CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type)  CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size)   CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE_PREFIX_uchar as_
#define AS_TYPE_PREFIX_char as_
#define AS_TYPE_PREFIX_fp4e2m1_t _as_
#define AS_TYPE_PREFIX_fp8e5m2_t _as_
#define AS_TYPE_PREFIX_fp8e4m3_t _as_
#define AS_TYPE_PREFIX_fp8e8m0_t _as_
#define AS_TYPE_PREFIX_ushort as_
#define AS_TYPE_PREFIX_short as_
#define AS_TYPE_PREFIX_half as_
#define AS_TYPE_PREFIX_int as_
#define AS_TYPE_PREFIX_uint as_
#define AS_TYPE_PREFIX_float as_
#define AS_TYPE_PREFIX_ulong as_
#define AS_TYPE_PREFIX_long as_

#define AS_TYPE_EXT(type, val, src_type) CAT(CAT(AS_TYPE_PREFIX_, src_type), type)(val)
#define AS_TYPE(type, val) CAT(as_, type)(val)

// ====================================================================================================================
// TYPE_SIZE(type) - evaluates to size of "type" in bytes
// type [PP] - Must evaluate to non-vectorized type.
// ====================================================================================================================
#define TYPE_SIZE_uchar  1
#define TYPE_SIZE_char   1
#define TYPE_SIZE_fp8e5m2_t 1
#define TYPE_SIZE_fp8e4m3_t 1
#define TYPE_SIZE_fp8e8m0_t 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short  2
#define TYPE_SIZE_half   2
#define TYPE_SIZE_int    4
#define TYPE_SIZE_uint   4
#define TYPE_SIZE_float  4
#define TYPE_SIZE_ulong  8
#define TYPE_SIZE_long   8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif


// ====================================================================================================================
// BLOCK_READN(type, vector_size, ptr, offset)
//    - evaluates to intel_sub_group_block_read operation for specified "type" and "vector size", reading
//      "vector_size" elements from memory starting at "ptr" + "offset"
//  For more details and description of intel_sub_group_block_read functions please,
//  refer to cl_intel_subgroups extension documentation.
//
// BLOCK_READN_SLM(type, vector_size, ptr, offset)
//    - performs same operation as BLOCK_READN, but with "ptr" being in __local address space.
//
// type        [PP] - Must evaluate to non-vectorized type, ex. float, half, char, etc..
// vector_size [PP] - Number of elements to read/write, ex 2 for intel_sub_group_block_read2.
// ptr              - Pointer to global memory where to read from/write to.
// offset           - Additional offset added to ptr in "type" elements, equivalent to passing ((ptr) + (offset)) as "ptr".
// val              - For write function vector of "vector_size" of "type" elements (or scalar) to write.
//
// ====================================================================================================================
// Pre-defined commonly used definitions:
//   DT_<tensor>_BLOCK_READ<n>(ptr, offset)
// Where:
//    <tensor> is one of: INPUT - referencing type jitted as INPUT0,
//                        BIAS,
//                        FILTER
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will be a scalar
//
// ====================================================================================================================

#define BLOCK_READ_TYPE_size1 uchar
#define BLOCK_READ_TYPE_size2 ushort
#define BLOCK_READ_TYPE_size4 uint
#define BLOCK_READ_TYPE_size8 ulong
#define BLOCK_READ_TYPE(type_size) CAT(BLOCK_READ_TYPE_size, type_size)

#define BLOCK_READ_FUNC_size1       _sub_group_block_read_uc
#define BLOCK_READ_FUNC_size2       _sub_group_block_read_us
#define BLOCK_READ_FUNC_size4       _sub_group_block_read
#define BLOCK_READ_FUNC_size8       _sub_group_block_read_ul
#define BLOCK_READ_FUNC(type_size)  CAT(BLOCK_READ_FUNC_size, type_size)

// __local-pointer variants. cl_intel_subgroups* only defines __global overloads of
// intel_sub_group_block_read*; the __local overloads come from the separate
// cl_intel_subgroup_local_block_io extension, which NEO 23.x+ no longer declares
// implicitly. The _slm-suffixed family below routes to the intrinsic when the
// extension is advertised and falls back to an inline-emulation per-lane gather
// otherwise — see DECLARE_BLOCK_READ_SLM_EMULATION and the macro blocks at the
// bottom of this file.
#define BLOCK_READ_FUNC_SLM_size1       _sub_group_block_read_slm_uc
#define BLOCK_READ_FUNC_SLM_size2       _sub_group_block_read_slm_us
#define BLOCK_READ_FUNC_SLM_size4       _sub_group_block_read_slm
#define BLOCK_READ_FUNC_SLM_size8       _sub_group_block_read_slm_ul
#define BLOCK_READ_FUNC_SLM(type_size)  CAT(BLOCK_READ_FUNC_SLM_size, type_size)

#define BLOCK_READN_FUNC_SIZE_DEF(type_size, vector_size)   MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vector_size)
#define BLOCK_READN_FUNC_size1(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_READN_FUNC_size2(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_size4(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC_size8(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_READN_FUNC(type_size, vector_size)            CAT(BLOCK_READN_FUNC_size, type_size)(vector_size)

#define BLOCK_READN_FUNC_SLM_SIZE_DEF(type_size, vector_size)   MAKE_VECTOR_TYPE(BLOCK_READ_FUNC_SLM(type_size), vector_size)
#define BLOCK_READN_FUNC_SLM_size1(vector_size)                 BLOCK_READN_FUNC_SLM_SIZE_DEF(1, vector_size)
#define BLOCK_READN_FUNC_SLM_size2(vector_size)                 BLOCK_READN_FUNC_SLM_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_SLM_size4(vector_size)                 BLOCK_READN_FUNC_SLM_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC_SLM_size8(vector_size)                 BLOCK_READN_FUNC_SLM_SIZE_DEF(8, vector_size)
#define BLOCK_READN_FUNC_SLM(type_size, vector_size)            CAT(BLOCK_READN_FUNC_SLM_size, type_size)(vector_size)

#define BLOCK_READN_RAW(type_size, vector_size, addr_space, ptr, offset)                                        \
    BLOCK_READN_FUNC(type_size, vector_size)((const addr_space BLOCK_READ_TYPE(type_size)*)(ptr) + (offset))

#define BLOCK_READN_RAW_SLM(type_size, vector_size, ptr, offset)                                                \
    BLOCK_READN_FUNC_SLM(type_size, vector_size)((const __local BLOCK_READ_TYPE(type_size)*)(ptr) + (offset))

#define BLOCK_READN(type, vector_size, ptr, offset)                                                             \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset))

#define BLOCK_READN_SLM(type, vector_size, ptr, offset)                                                         \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW_SLM(TYPE_SIZE(type), vector_size, ptr, offset))

#define DT_INPUT_BLOCK_READ(ptr, offset)            BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset)          BLOCK_READN(INPUT0_TYPE, 16, ptr, offset)

#define DT_BIAS_BLOCK_READ(ptr, offset)             BLOCK_READN(BIAS_TYPE, 1, ptr, offset)
#define DT_BIAS_BLOCK_READ2(ptr, offset)            BLOCK_READN(BIAS_TYPE, 2, ptr, offset)
#define DT_BIAS_BLOCK_READ4(ptr, offset)            BLOCK_READN(BIAS_TYPE, 4, ptr, offset)
#define DT_BIAS_BLOCK_READ8(ptr, offset)            BLOCK_READN(BIAS_TYPE, 8, ptr, offset)
#define DT_BIAS_BLOCK_READ16(ptr, offset)           BLOCK_READN(BIAS_TYPE, 16, ptr, offset)

#define DT_FILTER_BLOCK_READ(ptr, offset)           BLOCK_READN(FILTER_TYPE, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset)          BLOCK_READN(FILTER_TYPE, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset)          BLOCK_READN(FILTER_TYPE, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset)          BLOCK_READN(FILTER_TYPE, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset)         BLOCK_READN(FILTER_TYPE, 16, ptr, offset)


#define BLOCK_READ_IMPL_1 ret = ptr[idx];

#define BLOCK_READ_IMPL_2                                   \
        ret.s0 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s1 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_4                                   \
        BLOCK_READ_IMPL_2                                   \
        ret.s2 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s3 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_8                                   \
        BLOCK_READ_IMPL_4                                   \
        ret.s4 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s5 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s6 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s7 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_16                                  \
        BLOCK_READ_IMPL_8                                   \
        ret.s8 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s9 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sa = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sb = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sc = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sd = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.se = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sf = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL(vec_size) CAT(BLOCK_READ_IMPL_, vec_size)
#define BLOCK_READ_FUNC_NAME(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_READ_EMULATION(type_size, vec_size) \
    inline MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) BLOCK_READ_FUNC_NAME(type_size, vec_size)(const __global BLOCK_READ_TYPE(type_size)* ptr) { \
    uint idx = get_sub_group_local_id(); \
    MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) ret; \
    BLOCK_READ_IMPL(vec_size) \
    return ret; \
}

#define BLOCK_READ_FUNC_NAME_SLM(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_READ_FUNC_SLM(type_size), vec_size)
#define DECLARE_BLOCK_READ_SLM_EMULATION(type_size, vec_size) \
    inline MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) BLOCK_READ_FUNC_NAME_SLM(type_size, vec_size)(const __local BLOCK_READ_TYPE(type_size)* ptr) { \
    uint idx = get_sub_group_local_id(); \
    MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) ret; \
    BLOCK_READ_IMPL(vec_size) \
    return ret; \
}

#if defined(cl_intel_subgroups)
    #define _sub_group_block_read(ptr) intel_sub_group_block_read(ptr)
    #define _sub_group_block_read2(ptr) intel_sub_group_block_read2(ptr)
    #define _sub_group_block_read4(ptr) intel_sub_group_block_read4(ptr)
    #define _sub_group_block_read8(ptr) intel_sub_group_block_read8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(4, 1)
    DECLARE_BLOCK_READ_EMULATION(4, 2)
    DECLARE_BLOCK_READ_EMULATION(4, 4)
    DECLARE_BLOCK_READ_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroup_local_block_io)
    #define _sub_group_block_read_slm(ptr)  intel_sub_group_block_read(ptr)
    #define _sub_group_block_read_slm2(ptr) intel_sub_group_block_read2(ptr)
    #define _sub_group_block_read_slm4(ptr) intel_sub_group_block_read4(ptr)
    #define _sub_group_block_read_slm8(ptr) intel_sub_group_block_read8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_SLM_EMULATION(4, 1)
    DECLARE_BLOCK_READ_SLM_EMULATION(4, 2)
    DECLARE_BLOCK_READ_SLM_EMULATION(4, 4)
    DECLARE_BLOCK_READ_SLM_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroups_short)
    #define _sub_group_block_read_us(ptr) intel_sub_group_block_read_us(ptr)
    #define _sub_group_block_read_us2(ptr) intel_sub_group_block_read_us2(ptr)
    #define _sub_group_block_read_us4(ptr) intel_sub_group_block_read_us4(ptr)
    #define _sub_group_block_read_us8(ptr) intel_sub_group_block_read_us8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(2, 1)
    DECLARE_BLOCK_READ_EMULATION(2, 2)
    DECLARE_BLOCK_READ_EMULATION(2, 4)
    DECLARE_BLOCK_READ_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroup_local_block_io)
    #define _sub_group_block_read_slm_us(ptr)  intel_sub_group_block_read_us(ptr)
    #define _sub_group_block_read_slm_us2(ptr) intel_sub_group_block_read_us2(ptr)
    #define _sub_group_block_read_slm_us4(ptr) intel_sub_group_block_read_us4(ptr)
    #define _sub_group_block_read_slm_us8(ptr) intel_sub_group_block_read_us8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_SLM_EMULATION(2, 1)
    DECLARE_BLOCK_READ_SLM_EMULATION(2, 2)
    DECLARE_BLOCK_READ_SLM_EMULATION(2, 4)
    DECLARE_BLOCK_READ_SLM_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroups_char)
    #define _sub_group_block_read_uc(ptr) intel_sub_group_block_read_uc(ptr)
    #define _sub_group_block_read_uc2(ptr) intel_sub_group_block_read_uc2(ptr)
    #define _sub_group_block_read_uc4(ptr) intel_sub_group_block_read_uc4(ptr)
    #define _sub_group_block_read_uc8(ptr) intel_sub_group_block_read_uc8(ptr)
    #define _sub_group_block_read_uc16(ptr) intel_sub_group_block_read_uc16(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(1, 1)
    DECLARE_BLOCK_READ_EMULATION(1, 2)
    DECLARE_BLOCK_READ_EMULATION(1, 4)
    DECLARE_BLOCK_READ_EMULATION(1, 8)
    DECLARE_BLOCK_READ_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroup_local_block_io)
    #define _sub_group_block_read_slm_uc(ptr)   intel_sub_group_block_read_uc(ptr)
    #define _sub_group_block_read_slm_uc2(ptr)  intel_sub_group_block_read_uc2(ptr)
    #define _sub_group_block_read_slm_uc4(ptr)  intel_sub_group_block_read_uc4(ptr)
    #define _sub_group_block_read_slm_uc8(ptr)  intel_sub_group_block_read_uc8(ptr)
    #define _sub_group_block_read_slm_uc16(ptr) intel_sub_group_block_read_uc16(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 1)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 2)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 4)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 8)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroups_long)
    #define _sub_group_block_read_ul(ptr)  intel_sub_group_block_read_ul(ptr)
    #define _sub_group_block_read_ul2(ptr) intel_sub_group_block_read_ul2(ptr)
    #define _sub_group_block_read_ul4(ptr) intel_sub_group_block_read_ul4(ptr)
    #define _sub_group_block_read_ul8(ptr) intel_sub_group_block_read_ul8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(8, 1)
    DECLARE_BLOCK_READ_EMULATION(8, 2)
    DECLARE_BLOCK_READ_EMULATION(8, 4)
    DECLARE_BLOCK_READ_EMULATION(8, 8)
#endif

#if defined(cl_intel_subgroup_local_block_io)
    #define _sub_group_block_read_slm_ul(ptr)  intel_sub_group_block_read_ul(ptr)
    #define _sub_group_block_read_slm_ul2(ptr) intel_sub_group_block_read_ul2(ptr)
    #define _sub_group_block_read_slm_ul4(ptr) intel_sub_group_block_read_ul4(ptr)
    #define _sub_group_block_read_slm_ul8(ptr) intel_sub_group_block_read_ul8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_SLM_EMULATION(8, 1)
    DECLARE_BLOCK_READ_SLM_EMULATION(8, 2)
    DECLARE_BLOCK_READ_SLM_EMULATION(8, 4)
    DECLARE_BLOCK_READ_SLM_EMULATION(8, 8)
#endif

// ==== inlined include/batch_headers/sub_group_block_write.cl ====
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ==== inlined common.cl ====
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))

#if OPT_HINTS_SUPPORTED
#   define ASSUME_HINT(x) __builtin_assume(x)
#else
#   define ASSUME_HINT(x) do { } while (0)
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b)      ((a) < (b) ? (a) : (b))
#define MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type)  elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type)  CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type)  CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type)  CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type)  CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size)   CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE_PREFIX_uchar as_
#define AS_TYPE_PREFIX_char as_
#define AS_TYPE_PREFIX_fp4e2m1_t _as_
#define AS_TYPE_PREFIX_fp8e5m2_t _as_
#define AS_TYPE_PREFIX_fp8e4m3_t _as_
#define AS_TYPE_PREFIX_fp8e8m0_t _as_
#define AS_TYPE_PREFIX_ushort as_
#define AS_TYPE_PREFIX_short as_
#define AS_TYPE_PREFIX_half as_
#define AS_TYPE_PREFIX_int as_
#define AS_TYPE_PREFIX_uint as_
#define AS_TYPE_PREFIX_float as_
#define AS_TYPE_PREFIX_ulong as_
#define AS_TYPE_PREFIX_long as_

#define AS_TYPE_EXT(type, val, src_type) CAT(CAT(AS_TYPE_PREFIX_, src_type), type)(val)
#define AS_TYPE(type, val) CAT(as_, type)(val)

// ====================================================================================================================
// TYPE_SIZE(type) - evaluates to size of "type" in bytes
// type [PP] - Must evaluate to non-vectorized type.
// ====================================================================================================================
#define TYPE_SIZE_uchar  1
#define TYPE_SIZE_char   1
#define TYPE_SIZE_fp8e5m2_t 1
#define TYPE_SIZE_fp8e4m3_t 1
#define TYPE_SIZE_fp8e8m0_t 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short  2
#define TYPE_SIZE_half   2
#define TYPE_SIZE_int    4
#define TYPE_SIZE_uint   4
#define TYPE_SIZE_float  4
#define TYPE_SIZE_ulong  8
#define TYPE_SIZE_long   8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif


// ====================================================================================================================
// BLOCK_WRITEN(type, vector_size, ptr, offset, val)
//    - evaluates to intel_sub_group_block_write operation for specified "type" and "vector size", writing
//      "vector_size"-element vector "val" to memory starting at "ptr" + "offset"
//  For more details and description of intel_sub_group_block_read/write functions please,
//  refer to cl_intel_subgroups extension documentation.
//
// BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)
//    - performs same operation as BLOCK_READN, but with "ptr" being in __local address space.
//
// type        [PP] - Must evaluate to non-vectorized type, ex. float, half, char, etc..
// vector_size [PP] - Number of elements to read/write, ex 2 for intel_sub_group_block_read2.
// ptr              - Pointer to global memory where to read from/write to.
// offset           - Additional offset added to ptr in "type" elements, equivalent to passing ((ptr) + (offset)) as "ptr".
// val              - For write function vector of "vector_size" of "type" elements (or scalar) to write.
//
// ====================================================================================================================
// Pre-defined commonly used definitions:
//   DT_<tensor>_BLOCK_WRITE<n>(ptr, offset, offset)
// Where:
//    <tensor> is usually OUTPUT,
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will be a scalar
//
// ====================================================================================================================

#define BLOCK_WRITE_TYPE_size1 uchar
#define BLOCK_WRITE_TYPE_size2 ushort
#define BLOCK_WRITE_TYPE_size4 uint
#define BLOCK_WRITE_TYPE_size8 ulong
#define BLOCK_WRITE_TYPE(type_size) CAT(BLOCK_WRITE_TYPE_size, type_size)

#define BLOCK_WRITE_FUNC_size1       _sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_size2       _sub_group_block_write_us
#define BLOCK_WRITE_FUNC_size4       _sub_group_block_write
#define BLOCK_WRITE_FUNC_size8       _sub_group_block_write_ul
#define BLOCK_WRITE_FUNC(type_size)  CAT(BLOCK_WRITE_FUNC_size, type_size)

#define BLOCK_WRITEN_FUNC_SIZE_DEF(type_size, vector_size)  MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vector_size)
#define BLOCK_WRITEN_FUNC_size1(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_WRITEN_FUNC_size2(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_WRITEN_FUNC_size4(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_WRITEN_FUNC_size8(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_WRITEN_FUNC(type_size, vector_size)           CAT(BLOCK_WRITEN_FUNC_size, type_size)(vector_size)

#define BLOCK_WRITEN_RAW(type_size, vector_size, addr_space, ptr, offset, val, src_type)                        \
    BLOCK_WRITEN_FUNC(type_size, vector_size)(                                                                  \
        (addr_space BLOCK_WRITE_TYPE(type_size)*)(ptr) + (offset),                                              \
        AS_TYPE_EXT(MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vector_size), val, src_type))

#define BLOCK_WRITEN(type, vector_size, ptr, offset, val)                                                       \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset, val, type)

#define BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)                                                   \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset, val, type)

#define DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)     BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE2(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 2, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE4(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 4, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE8(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 8, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE16(ptr, offset, val)   BLOCK_WRITEN(OUTPUT_TYPE, 16, ptr, offset, val)

#define BLOCK_WRITE_IMPL_1 out_ptr[idx] = v;
#define BLOCK_WRITE_IMPL_2                                    \
        out_ptr[idx] = v.s0; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s1; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_4                                    \
        BLOCK_WRITE_IMPL_2                                    \
        out_ptr[idx] = v.s2; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s3; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_8                                    \
        BLOCK_WRITE_IMPL_4                                    \
        out_ptr[idx] = v.s4; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s5; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s6; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s7; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_16                                   \
        BLOCK_WRITE_IMPL_8                                    \
        out_ptr[idx] = v.s8; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s9; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sa; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sb; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sc; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sd; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.se; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sf; idx += get_max_sub_group_size();

#define BLOCK_WRITE_IMPL(vec_size) CAT(BLOCK_WRITE_IMPL_, vec_size)
#define BLOCK_WRITE_FUNC_NAME(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_WRITE_EMULATION(type_size, vec_size) \
    inline void BLOCK_WRITE_FUNC_NAME(type_size, vec_size)(__global BLOCK_WRITE_TYPE(type_size)* out_ptr, \
                                                           MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vec_size) v) { \
    uint idx = get_sub_group_local_id(); \
    BLOCK_WRITE_IMPL(vec_size) \
}

#if defined(cl_intel_subgroups)
    #define _sub_group_block_write(ptr, v) intel_sub_group_block_write(ptr, v)
    #define _sub_group_block_write2(ptr, v) intel_sub_group_block_write2(ptr, v)
    #define _sub_group_block_write4(ptr, v) intel_sub_group_block_write4(ptr, v)
    #define _sub_group_block_write8(ptr, v) intel_sub_group_block_write8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(4, 1)
    DECLARE_BLOCK_WRITE_EMULATION(4, 2)
    DECLARE_BLOCK_WRITE_EMULATION(4, 4)
    DECLARE_BLOCK_WRITE_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroups_short)
    #define _sub_group_block_write_us(ptr, v) intel_sub_group_block_write_us(ptr, v)
    #define _sub_group_block_write_us2(ptr, v) intel_sub_group_block_write_us2(ptr, v)
    #define _sub_group_block_write_us4(ptr, v) intel_sub_group_block_write_us4(ptr, v)
    #define _sub_group_block_write_us8(ptr, v) intel_sub_group_block_write_us8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(2, 1)
    DECLARE_BLOCK_WRITE_EMULATION(2, 2)
    DECLARE_BLOCK_WRITE_EMULATION(2, 4)
    DECLARE_BLOCK_WRITE_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroups_char)
    #define _sub_group_block_write_uc(ptr, v) intel_sub_group_block_write_uc(ptr, v)
    #define _sub_group_block_write_uc2(ptr, v) intel_sub_group_block_write_uc2(ptr, v)
    #define _sub_group_block_write_uc4(ptr, v) intel_sub_group_block_write_uc4(ptr, v)
    #define _sub_group_block_write_uc8(ptr, v) intel_sub_group_block_write_uc8(ptr, v)
    #define _sub_group_block_write_uc16(ptr, v) intel_sub_group_block_write_uc16(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(1, 1)
    DECLARE_BLOCK_WRITE_EMULATION(1, 2)
    DECLARE_BLOCK_WRITE_EMULATION(1, 4)
    DECLARE_BLOCK_WRITE_EMULATION(1, 8)
    DECLARE_BLOCK_WRITE_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroups_long)
    #define _sub_group_block_write_ul(ptr, v)  intel_sub_group_block_write_ul(ptr, v)
    #define _sub_group_block_write_ul2(ptr, v) intel_sub_group_block_write_ul2(ptr, v)
    #define _sub_group_block_write_ul4(ptr, v) intel_sub_group_block_write_ul4(ptr, v)
    #define _sub_group_block_write_ul8(ptr, v) intel_sub_group_block_write_ul8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(8, 1)
    DECLARE_BLOCK_WRITE_EMULATION(8, 2)
    DECLARE_BLOCK_WRITE_EMULATION(8, 4)
    DECLARE_BLOCK_WRITE_EMULATION(8, 8)
#endif

// ==== inlined include/batch_headers/sub_group_shuffle.cl ====
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ==== inlined common.cl ====
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))

#if OPT_HINTS_SUPPORTED
#   define ASSUME_HINT(x) __builtin_assume(x)
#else
#   define ASSUME_HINT(x) do { } while (0)
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b)      ((a) < (b) ? (a) : (b))
#define MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type)  elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type)  CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type)  CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type)  CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type)  CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size)   CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE_PREFIX_uchar as_
#define AS_TYPE_PREFIX_char as_
#define AS_TYPE_PREFIX_fp4e2m1_t _as_
#define AS_TYPE_PREFIX_fp8e5m2_t _as_
#define AS_TYPE_PREFIX_fp8e4m3_t _as_
#define AS_TYPE_PREFIX_fp8e8m0_t _as_
#define AS_TYPE_PREFIX_ushort as_
#define AS_TYPE_PREFIX_short as_
#define AS_TYPE_PREFIX_half as_
#define AS_TYPE_PREFIX_int as_
#define AS_TYPE_PREFIX_uint as_
#define AS_TYPE_PREFIX_float as_
#define AS_TYPE_PREFIX_ulong as_
#define AS_TYPE_PREFIX_long as_

#define AS_TYPE_EXT(type, val, src_type) CAT(CAT(AS_TYPE_PREFIX_, src_type), type)(val)
#define AS_TYPE(type, val) CAT(as_, type)(val)

// ====================================================================================================================
// TYPE_SIZE(type) - evaluates to size of "type" in bytes
// type [PP] - Must evaluate to non-vectorized type.
// ====================================================================================================================
#define TYPE_SIZE_uchar  1
#define TYPE_SIZE_char   1
#define TYPE_SIZE_fp8e5m2_t 1
#define TYPE_SIZE_fp8e4m3_t 1
#define TYPE_SIZE_fp8e8m0_t 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short  2
#define TYPE_SIZE_half   2
#define TYPE_SIZE_int    4
#define TYPE_SIZE_uint   4
#define TYPE_SIZE_float  4
#define TYPE_SIZE_ulong  8
#define TYPE_SIZE_long   8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif


#ifdef cl_intel_subgroups
#define _sub_group_shuffle(v, c) intel_sub_group_shuffle(v, c)
#define _sub_group_shuffle_up(c, n, d) intel_sub_group_shuffle_up(c, n, d)
#define _sub_group_shuffle_down(c, n, d) intel_sub_group_shuffle_down(c, n, d)
#elif (__OPENCL_C_VERSION__ >= 200)

// The spec for intel_subgroup_shuffle says that index (c) need not be the same value for all work-items in
// a subgroup while sub_group_broadcast requires that.
// However, most of our kernels uses shuffle in a way that produces same index for all work-items,
// so for now we use this solution.
// In case of accuracy issues we may switch to something like this:
// #define MAX_SG_SIZE 32
// #define DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)
// inline type _sub_group_shuffle(type v, uint c) __attribute__((overloadable)) {
//     type vals[MAX_SG_SIZE];
//     for (size_t i = 0; i < get_max_sub_group_size(); i++) {
//         vals[i] = AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v), i));
//     }
//     return vals[c];
// }

#define DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)                                               \
inline type _sub_group_shuffle(type v, uint c) __attribute__((overloadable)) {                    \
    return AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v), c));                          \
}

#define DECLARE_SUB_GROUP_SHUFFLE2(type, cast_type)                                               \
inline CAT(type, 2) _sub_group_shuffle(CAT(type, 2) v, uint c) __attribute__((overloadable)) {    \
    return (CAT(type, 2))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)));      \
}

#define DECLARE_SUB_GROUP_SHUFFLE4(type, cast_type)                                               \
inline CAT(type, 4) _sub_group_shuffle(CAT(type, 4) v, uint c) __attribute__((overloadable)) {    \
   return (CAT(type, 4))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)));       \
}

#define DECLARE_SUB_GROUP_SHUFFLE8(type, cast_type)                                               \
inline CAT(type, 8) _sub_group_shuffle(CAT(type, 8) v, uint c) __attribute__((overloadable)) {    \
   return (CAT(type, 8))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s4), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s5), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s6), c)),        \
                          AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s7), c)));       \
}

#define DECLARE_SUB_GROUP_SHUFFLE16(type, cast_type)                                              \
inline CAT(type, 16) _sub_group_shuffle(CAT(type, 16) v, uint c) __attribute__((overloadable)) {  \
   return (CAT(type, 16))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s4), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s5), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s6), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s7), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s8), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s9), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sa), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sb), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sc), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sd), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.se), c)),       \
                           AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sf), c)));      \
}


#define DECLARE_SUB_GROUP_SHUFFLE(type)    \
    DECLARE_SUB_GROUP_SHUFFLE1(type, type) \
    DECLARE_SUB_GROUP_SHUFFLE2(type, type) \
    DECLARE_SUB_GROUP_SHUFFLE4(type, type) \
    DECLARE_SUB_GROUP_SHUFFLE8(type, type) \
    DECLARE_SUB_GROUP_SHUFFLE16(type, type)

#define DECLARE_SUB_GROUP_SHUFFLE_CASTED(type, cast_type) \
    DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)           \
    DECLARE_SUB_GROUP_SHUFFLE2(type, cast_type)           \
    DECLARE_SUB_GROUP_SHUFFLE4(type, cast_type)           \
    DECLARE_SUB_GROUP_SHUFFLE8(type, cast_type)           \
    DECLARE_SUB_GROUP_SHUFFLE16(type, cast_type)


DECLARE_SUB_GROUP_SHUFFLE(int)
DECLARE_SUB_GROUP_SHUFFLE(uint)
DECLARE_SUB_GROUP_SHUFFLE(float)

#if defined(cl_khr_fp16)
    DECLARE_SUB_GROUP_SHUFFLE(half)
    DECLARE_SUB_GROUP_SHUFFLE_CASTED(short, half)
    DECLARE_SUB_GROUP_SHUFFLE_CASTED(ushort, half)
#endif

#endif


#ifndef FUSE_QK_L2NORM
#    define FUSE_QK_L2NORM 0
#endif

#ifndef Q_L2_NORM_EPS
#    define Q_L2_NORM_EPS 1e-6f
#endif

#ifndef K_L2_NORM_EPS
#    define K_L2_NORM_EPS 1e-6f
#endif

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    return rsqrt(sum + eps) * extra_scale;
}

inline float FUNC(sum8)(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}

inline void FUNC(normalize_kq_128)(float8* b_k, float8* b_q) {
#if FUSE_QK_L2NORM
    float k_sum = FUNC(sum8)((*b_k) * (*b_k));
    k_sum = sub_group_reduce_add(k_sum);
    const float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
    *b_k *= k_scale;

    float q_sum = FUNC(sum8)((*b_q) * (*b_q));
    q_sum = sub_group_reduce_add(q_sum);
    const float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);
    *b_q *= q_scale;
#else
    *b_q *= SCALE_FACTOR;
#endif
}

#ifndef K_VEC_SIZE
#    define K_VEC_SIZE 1
#endif

#if ((K_HEAD_DIM % 16) != 0) || ((V_HEAD_DIM % 16) != 0)
#    error "paged_gated_delta_net_opt requires K_HEAD_DIM and V_HEAD_DIM divisible by 16"
#endif

#define K_LANE_ELEMS (K_HEAD_DIM / SUBGROUP_SIZE)

typedef MAKE_VECTOR_TYPE(float, K_VEC_SIZE) K_VEC_TYPE;
#if (K_VEC_SIZE == 1)
#    define K_VEC_LOAD_Q(ptr, idx)     convert_float(BLOCK_READN(INPUT0_TYPE, 1, (ptr), (idx)))
#    define K_VEC_LOAD_K(ptr, idx)     convert_float(BLOCK_READN(INPUT1_TYPE, 1, (ptr), (idx)))
#    define K_VEC_LOAD_STATE(ptr, idx) convert_float(BLOCK_READN(INPUT3_TYPE, 1, (ptr), (idx)))
#    define K_VEC_TO_STATE(vec)        ((INPUT3_TYPE)(vec))
#    define K_VEC_DOT(a, b)            ((a) * (b))
#    define K_VEC_SUM_SQ(a)            ((a) * (a))
#elif (K_VEC_SIZE == 8)
#    define K_VEC_LOAD_Q(ptr, idx)     convert_float8(BLOCK_READN(INPUT0_TYPE, 8, (ptr), (idx)))
#    define K_VEC_LOAD_K(ptr, idx)     convert_float8(BLOCK_READN(INPUT1_TYPE, 8, (ptr), (idx)))
#    define K_VEC_LOAD_STATE(ptr, idx) convert_float8(BLOCK_READN(INPUT3_TYPE, 8, (ptr), (idx)))
#    define K_VEC_TO_STATE(vec)        CAT(convert_, CAT(INPUT3_TYPE, 8))(vec)
#    define K_VEC_DOT(a, b)            FUNC(sum8)((a) * (b))
#    define K_VEC_SUM_SQ(a)            FUNC(sum8)((a) * (a))
#else
#    define K_VEC_LOAD_Q(ptr, idx)     CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT0_TYPE, K_VEC_SIZE, (ptr), (idx)))
#    define K_VEC_LOAD_K(ptr, idx)     CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT1_TYPE, K_VEC_SIZE, (ptr), (idx)))
#    define K_VEC_LOAD_STATE(ptr, idx) CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT3_TYPE, K_VEC_SIZE, (ptr), (idx)))
#    define K_VEC_TO_STATE(vec)        CAT(convert_, CAT(INPUT3_TYPE, K_VEC_SIZE))(vec)
#    define K_VEC_DOT(a, b)            dot((a), (b))
#    define K_VEC_SUM_SQ(a)            dot((a), (a))
#endif

#define K_VEC_COUNT (K_LANE_ELEMS / K_VEC_SIZE)

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(paged_gated_delta_net_opt)
(__global INPUT0_TYPE* query,
 __global INPUT1_TYPE* key,
 __global INPUT2_TYPE* value,
 __global INPUT3_TYPE* recurrent_state_table,
 __global INPUT4_TYPE* gate,
 __global INPUT5_TYPE* beta,
 __global INPUT6_TYPE* subsequence_begins,
 __global INPUT7_TYPE* block_indices,
 __global INPUT8_TYPE* block_indices_begins,
 __global INPUT9_TYPE* past_lens,
 __global INPUT10_TYPE* cache_interval,
 __global OUTPUT_TYPE* output,
 int num_sequences,
 int query_head_offset,
 int key_head_offset,
 int value_head_offset,
 int q_token_stride,
 int q_head_stride,
 int k_token_stride,
 int k_head_stride,
 int v_token_stride,
 int v_head_stride) {
    const int seq = get_global_id(0);
    const int h = get_global_id(1);
    const int v_block = get_group_id(2);
    const int lid = get_sub_group_local_id();
    const int start_iv = v_block * V_BLOCK_SIZE;

    const int token_begin = subsequence_begins[seq];
    const int token_end = subsequence_begins[seq + 1];
    const int block_begin = block_indices_begins[seq];
    const int past_len = past_lens[seq];
    const int interval = cache_interval[seq];
    const int prev_nums = interval > 0 ? past_len % interval : 0;

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;
    const int q_head_base = (hk + query_head_offset) * q_head_stride;
    const int k_head_base = (hk + key_head_offset) * k_head_stride;
    const int v_head_base = (h + value_head_offset) * v_head_stride;
    const int state_stride = K_HEAD_DIM * V_HEAD_DIM;

    K_VEC_TYPE state[V_BLOCK_SIZE][K_VEC_COUNT];
    K_VEC_TYPE q_norm[K_VEC_COUNT];
    K_VEC_TYPE k_norm[K_VEC_COUNT];

    const int initial_block_id = block_indices[block_begin];
    const int initial_block_base = (initial_block_id * V_HEAD_NUM + h) * state_stride;

#pragma unroll
    for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
        int curr_iv = start_iv + v_idx;
        int base = initial_block_base + curr_iv * K_HEAD_DIM;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
        state[v_idx][0] = K_VEC_LOAD_STATE(recurrent_state_table, base);
#else
#    pragma unroll
        for (int kc = 0; kc < K_VEC_COUNT; kc++) {
            const int k_base = kc * K_VEC_SIZE * SUBGROUP_SIZE;
            state[v_idx][kc] = K_VEC_LOAD_STATE(recurrent_state_table, base + k_base);
        }
#endif
    }

    int token = token_begin;
    int slot = 1;
    int tokens_to_next_boundary = interval > 0 ? (prev_nums > 0 ? (interval - prev_nums) : interval) : (token_end - token_begin);
    while (token < token_end) {
        const int chunk_end = min(token + tokens_to_next_boundary, token_end);

        int q_base = token * q_token_stride + q_head_base;
        int k_base = token * k_token_stride + k_head_base;
        int v_base = token * v_token_stride + v_head_base;
        int g_idx = token * V_HEAD_NUM + h;

        for (; token < chunk_end; token++, q_base += q_token_stride, k_base += k_token_stride, v_base += v_token_stride, g_idx += V_HEAD_NUM) {
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
            q_norm[0] = K_VEC_LOAD_Q(query, q_base);
            k_norm[0] = K_VEC_LOAD_K(key, k_base);
            FUNC(normalize_kq_128)(&k_norm[0], &q_norm[0]);
#else
            float q_sum_local = 0.0f;
            float k_sum_local = 0.0f;
#    pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                const int offset = kc * K_VEC_SIZE * SUBGROUP_SIZE;
                q_norm[kc] = K_VEC_LOAD_Q(query, q_base + offset);
                k_norm[kc] = K_VEC_LOAD_K(key, k_base + offset);
                q_sum_local += K_VEC_SUM_SQ(q_norm[kc]);
                k_sum_local += K_VEC_SUM_SQ(k_norm[kc]);
            }

            float q_scale = SCALE_FACTOR;
            float k_scale = 1.0f;
#    if FUSE_QK_L2NORM
            const float q_sum = sub_group_reduce_add(q_sum_local);
            const float k_sum = sub_group_reduce_add(k_sum_local);

            q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);
            k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
#    endif
#    pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                q_norm[kc] *= q_scale;
                k_norm[kc] *= k_scale;
            }
#endif

            const float b_g = exp(convert_float(gate[g_idx]));
            const float b_beta = convert_float(beta[g_idx]);

            float b_v_block[V_BLOCK_SIZE];
            float h_k_block[V_BLOCK_SIZE];
            float update_block[V_BLOCK_SIZE];
            float out_block[V_BLOCK_SIZE];
#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                const int curr_iv = start_iv + v_idx;
                const int v_base_aligned = v_base + (curr_iv & ~(SUBGROUP_SIZE - 1));
                const int v_lane = curr_iv & (SUBGROUP_SIZE - 1);
                const float v_val = convert_float(BLOCK_READN(INPUT2_TYPE, 1, value, v_base_aligned));
                b_v_block[v_idx] = sub_group_broadcast(v_val, v_lane);
            }

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                float h_k_local = 0.0f;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
                state[v_idx][0] *= b_g;
                h_k_local = FUNC(sum8)(state[v_idx][0] * k_norm[0]);
#else
#    pragma unroll
                for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                    state[v_idx][kc] *= b_g;
                    h_k_local += K_VEC_DOT(state[v_idx][kc], k_norm[kc]);
                }
#endif
                h_k_block[v_idx] = sub_group_reduce_add(h_k_local);
            }

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                update_block[v_idx] = (b_v_block[v_idx] - h_k_block[v_idx]) * b_beta;
            }

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                float out_val_local = 0.0f;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
                state[v_idx][0] = fma(k_norm[0], update_block[v_idx], state[v_idx][0]);
                out_val_local = FUNC(sum8)(state[v_idx][0] * q_norm[0]);
#else
#    pragma unroll
                for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                    state[v_idx][kc] = fma(k_norm[kc], update_block[v_idx], state[v_idx][kc]);
                    out_val_local += K_VEC_DOT(state[v_idx][kc], q_norm[kc]);
                }
#endif
                out_block[v_idx] = sub_group_reduce_add(out_val_local);
            }

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                const float out_val = out_block[v_idx];

                if (lid == 0) {
                    const int out_offset = (token * V_HEAD_NUM + h) * V_HEAD_DIM + curr_iv;
                    output[out_offset] = TO_OUTPUT_TYPE(out_val);
                }
            }
        }

        const int block_id = block_indices[block_begin + slot];
        slot++;
        const int block_base = (block_id * V_HEAD_NUM + h) * state_stride;
#pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            int base = block_base + curr_iv * K_HEAD_DIM;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
            BLOCK_WRITEN(INPUT3_TYPE, K_VEC_SIZE, recurrent_state_table, base, K_VEC_TO_STATE(state[v_idx][0]));
#else
#    pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                const int k_base = kc * K_VEC_SIZE * SUBGROUP_SIZE;
                BLOCK_WRITEN(INPUT3_TYPE, K_VEC_SIZE, recurrent_state_table, base + k_base, K_VEC_TO_STATE(state[v_idx][kc]));
            }
#endif
        }

        if (interval > 0)
            tokens_to_next_boundary = interval;
    }
}

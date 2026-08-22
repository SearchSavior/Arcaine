// Resolved lowering of OpenVINO gated_delta_net_ref.cl for the Arcaine
// Qwen3.5-MoE GDN prefill op (sequential O(T) strategy).
//
// JIT macros baked for the Arcaine harness:
//   K_HEAD_NUM=32  V_HEAD_NUM=32  K_HEAD_DIM=128  V_HEAD_DIM=128
//   V_BLOCK_SIZE=4 SUBGROUP_SIZE=16 FUSE_QK_L2NORM=0 SCALE_FACTOR=1.0
//   OUTPUT_STATE=0  (final state written back into initial_state in place)
//   q,k,v,beta,g = bf16 (ushort raw bits); state = fp32; core out = bf16.
//
// Layouts match the Arcaine qwen35-gdn bench 1:1:
//   q,k,v: [S, n_v, d]  token stride = 32*128 = 4096
//   beta,g: [S, n_v]
//   ssm_state: fp32 [n_v, d_k, d_v] in/out (kernel writes back in place)
//   core: bf16 [S, n_v, d_v]
//
// NOTE: the original kernel's `#if INPUT0_TYPE_SIZE == 2` fast path uses
// vload8((half*)p), which is only valid for f16. For bf16 the byte layout
// differs, so the loads below use explicit bf16->f32 conversion
// (bit-reinterpret ushort as float with mantissa shifted left 16).

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))

// --- bf16 <-> f32 software conversions (round-to-nearest-even f32->bf16) ---
inline float bf16_to_f32(ushort v) {
    uint u = ((uint)v) << 16;
    return as_float(u);
}
inline ushort f32_to_bf16(float f) {
    uint u = as_uint(f);
    uint rounding_bias = ((u >> 16) & 1) + 0x7FFFu;
    return (ushort)((u + rounding_bias) >> 16);
}
inline float8 bf16x8_to_f32x8(ushort8 v) {
    return (float8)(bf16_to_f32(v.s0), bf16_to_f32(v.s1), bf16_to_f32(v.s2),
                    bf16_to_f32(v.s3), bf16_to_f32(v.s4), bf16_to_f32(v.s5),
                    bf16_to_f32(v.s6), bf16_to_f32(v.s7));
}
inline float4 bf16x4_to_f32x4(ushort4 v) {
    return (float4)(bf16_to_f32(v.s0), bf16_to_f32(v.s1), bf16_to_f32(v.s2),
                    bf16_to_f32(v.s3));
}

#define K_HEAD_DIM 128
#define V_HEAD_DIM 128
#define V_HEAD_NUM 32
#define K_HEAD_NUM 32
#define V_BLOCK_SIZE 4
#define SUBGROUP_SIZE 16
#define FUSE_QK_L2NORM 0
#define SCALE_FACTOR 1.0f
#define Q_L2_NORM_EPS 1e-6f
#define K_L2_NORM_EPS 1e-6f
#define OUTPUT_STATE 0

inline float l2norm_scale(float sum, float extra_scale, float eps) {
    sum = sub_group_reduce_add(sum);
    sum = sub_group_broadcast(sum, 0);
    return rsqrt(sum + eps) * extra_scale;
}

inline float dot8_fma(float8 a, float8 b) {
    float acc = a.s0 * b.s0;
    acc = fma(a.s1, b.s1, acc);
    acc = fma(a.s2, b.s2, acc);
    acc = fma(a.s3, b.s3, acc);
    acc = fma(a.s4, b.s4, acc);
    acc = fma(a.s5, b.s5, acc);
    acc = fma(a.s6, b.s6, acc);
    acc = fma(a.s7, b.s7, acc);
    return acc;
}

inline float8 load_q8_as_float8(const __global ushort* p) {
    return bf16x8_to_f32x8(vload8(0, p));
}
inline float8 load_k8_as_float8(const __global ushort* p) {
    return bf16x8_to_f32x8(vload8(0, p));
}
inline float4 load_v4_as_float4(const __global ushort* p) {
    return bf16x4_to_f32x4(vload4(0, p));
}

inline void prepare_qk(const __global ushort* q, const __global ushort* k,
                       int q_offset, int k_offset, int lane_k_base,
                       float8* b_q, float8* b_k) {
    const int K_CHUNKS = K_HEAD_DIM / (SUBGROUP_SIZE * 8);  // 1 for K=128, SS=16
#pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        int idx = lane_k_base + c * 8;
        float8 lane_k = load_k8_as_float8(k + k_offset + idx);
        float8 lane_q = load_q8_as_float8(q + q_offset + idx);
        b_k[c] = lane_k;
        b_q[c] = lane_q * (float8)(SCALE_FACTOR);
    }
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__kernel void gated_delta_net_ref(
    __global ushort* q,
    __global ushort* k,
    __global ushort* v,
    __global float* initial_state,
    __global ushort* g,
    __global ushort* beta,
    __global ushort* output,
    int seq_len,
    int query_offset,
    int key_offset,
    int value_offset,
    int q_t_stride,
    int k_t_stride,
    int v_t_stride) {
    const int T_len = seq_len;
    const int H_len = V_HEAD_NUM;
    const int HK_len = K_HEAD_NUM;
    const int K_len = K_HEAD_DIM;
    const int V_len = V_HEAD_DIM;

    const int v_lane = get_global_id(0);       // [0, vblocks*16), x-first subgroup
    const int start_iv = (v_lane / SUBGROUP_SIZE) * V_BLOCK_SIZE;  // v-block
    const int b = get_global_id(2);            // batch
    const int h = get_global_id(1);            // value head
    const int lid = get_sub_group_local_id();  // [0,16) along x

    const int Q_T_STRIDE = q_t_stride;
    const int K_T_STRIDE = k_t_stride;
    const int V_T_STRIDE = v_t_stride;
    const int Q_B_STRIDE = T_len * Q_T_STRIDE;
    const int K_B_STRIDE = T_len * K_T_STRIDE;
    const int V_B_STRIDE = T_len * V_T_STRIDE;
    const int STATE_BASE = (b * H_len + h) * (K_len * V_len);
    const int group_size = H_len / HK_len;  // 1 for this harness
    const int hk = h / group_size;
    const int out_bh_base = b * T_len * H_len * V_len + h * V_len;

    const int K_CHUNKS = K_HEAD_DIM / (SUBGROUP_SIZE * 8);
    float8 h_state[V_BLOCK_SIZE][K_CHUNKS];

    // 1. LOAD STATE BLOCK: 4 V-columns, 8 K-rows per lane.
#pragma unroll
    for (int row_chunk = 0; row_chunk < K_CHUNKS; row_chunk++) {
#pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            float8 lane_state = (float8)(0.0f);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_idx = lid * (K_HEAD_DIM / SUBGROUP_SIZE) + row_chunk * 8 + j;
                lane_state[j] = convert_float(initial_state[STATE_BASE + row_idx * V_len + curr_iv]);
            }
            h_state[v_idx][row_chunk] = lane_state;
        }
    }

    for (int t = 0; t < T_len; t++) {
        // 2. LOAD COMMON TIMESTEP DATA
        int g_idx = (b * T_len + t) * H_len + h;
        float b_g = exp(bf16_to_f32(g[g_idx]));
        float b_beta = bf16_to_f32(beta[g_idx]);
        const int lane_k_base = lid * (K_HEAD_DIM / SUBGROUP_SIZE);

        int q_offset = b * Q_B_STRIDE + t * Q_T_STRIDE + (hk + query_offset) * K_len;
        int k_offset = b * K_B_STRIDE + t * K_T_STRIDE + (hk + key_offset) * K_len;
        int v_offset = b * V_B_STRIDE + t * V_T_STRIDE + (h + value_offset) * V_len;
        int out_offset = out_bh_base + t * H_len * V_len;
        float8 b_k[K_CHUNKS];
        float8 b_q[K_CHUNKS];

        prepare_qk(q, k, q_offset, k_offset, lane_k_base, b_q, b_k);

        // V load
        float4 b_v_vec = load_v4_as_float4(v + v_offset + start_iv);

        // 3. RECURRENT UPDATE
        float4 dot_part_k_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 dot_part_q_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            float dot_part_k = 0.0f;
#pragma unroll
            for (int c = 0; c < K_CHUNKS; c++) {
                float8 s = h_state[v_idx][c] * (float8)(b_g);
                h_state[v_idx][c] = s;
                dot_part_k += dot8_fma(s, b_k[c]);
            }
            dot_part_k_vec[v_idx] = dot_part_k;
        }

        float4 h_k_vec = (float4)(sub_group_reduce_add(dot_part_k_vec.s0),
                                  sub_group_reduce_add(dot_part_k_vec.s1),
                                  sub_group_reduce_add(dot_part_k_vec.s2),
                                  sub_group_reduce_add(dot_part_k_vec.s3));

        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            float update_val = (b_v_vec[v_idx] - h_k_vec[v_idx]) * b_beta;
            float dot_part_q = 0.0f;
#pragma unroll
            for (int c = 0; c < K_CHUNKS; c++) {
                float8 s = fma(b_k[c], (float8)(update_val), h_state[v_idx][c]);
                h_state[v_idx][c] = s;
                dot_part_q += dot8_fma(s, b_q[c]);
            }
            dot_part_q_vec[v_idx] = dot_part_q;
        }

        float4 b_output_vec = (float4)(sub_group_reduce_add(dot_part_q_vec.s0),
                                       sub_group_reduce_add(dot_part_q_vec.s1),
                                       sub_group_reduce_add(dot_part_q_vec.s2),
                                       sub_group_reduce_add(dot_part_q_vec.s3));

        if (lid == 0) {
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                output[out_offset + curr_iv] = f32_to_bf16(b_output_vec[v_idx]);
            }
        }
    }

    // 4. WRITE BACK STATE BLOCK (in place into initial_state)
#pragma unroll
    for (int row_chunk = 0; row_chunk < K_CHUNKS; row_chunk++) {
#pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_idx = lid * (K_HEAD_DIM / SUBGROUP_SIZE) + row_chunk * 8 + j;
                initial_state[STATE_BASE + row_idx * V_len + curr_iv] = h_state[v_idx][row_chunk][j];
            }
        }
    }
}

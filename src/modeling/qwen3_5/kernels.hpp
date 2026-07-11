#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sycl/ext/intel/esimd.hpp>

#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/nvfp4.hpp"
#include "../qwen3_5_moe/kernels.hpp"

inline void qwen35_add_bias(sycl::queue& queue, bf16* x, const bf16* bias,
                            int rows, int cols) {
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> id) {
            size_t at = (size_t)id[0] * cols + id[1];
            x[at] = float_to_bf16(bf16_to_float(x[at]) + bf16_to_float(bias[id[1]]));
        });
    });
}

inline void qwen35_gelu_tanh_inplace(sycl::queue& queue, bf16* x, size_t count) {
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            float value = bf16_to_float(x[id]);
            constexpr float k = 0.7978845608028654f;
            float inner = k * (value + 0.044715f * value * value * value);
            x[id] = float_to_bf16(0.5f * value * (1.0f + sycl::tanh(inner)));
        });
    });
}

inline void qwen35_split_q_gate(sycl::queue& queue, const bf16* projected,
                                bf16* query, bf16* gate,
                                int seq, int heads, int head_dim) {
    size_t count = (size_t)seq * heads * head_dim;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            size_t linear = id[0];
            int dim = linear % head_dim;
            int head = (linear / head_dim) % heads;
            int token = linear / ((size_t)heads * head_dim);
            size_t src = ((size_t)token * heads + head) * (2 * head_dim) + dim;
            query[linear] = projected[src];
            gate[linear] = projected[src + head_dim];
        });
    });
}

inline void qwen35_split_q_gate_kv(
    sycl::queue& queue, const bf16* projected, bf16* query, bf16* gate,
    bf16* key, bf16* value, int seq, int query_heads, int key_heads,
    int head_dim) {
    int query_dim = query_heads * head_dim;
    int q_proj_dim = 2 * query_dim;
    int kv_dim = key_heads * head_dim;
    int row_stride = q_proj_dim + 2 * kv_dim;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::range<2>((size_t)seq, (size_t)query_dim),
            [=](sycl::id<2> id) {
                int token = static_cast<int>(id[0]);
                int q_col = static_cast<int>(id[1]);
                int dim = q_col % head_dim;
                int head = q_col / head_dim;
                size_t row = (size_t)token * row_stride;
                query[(size_t)token * query_dim + q_col] =
                    projected[row + (size_t)head * 2 * head_dim + dim];
                gate[(size_t)token * query_dim + q_col] =
                    projected[row + (size_t)head * 2 * head_dim + head_dim + dim];
                if (q_col < kv_dim) {
                    key[(size_t)token * kv_dim + q_col] =
                        projected[row + q_proj_dim + q_col];
                    value[(size_t)token * kv_dim + q_col] =
                        projected[row + q_proj_dim + kv_dim + q_col];
                }
            });
    });
}

inline void qwen35_copy_strided(sycl::queue& queue, const bf16* source,
                                int source_stride, int source_offset,
                                bf16* destination, int rows, int columns) {
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::range<2>((size_t)rows, (size_t)columns),
            [=](sycl::id<2> id) {
                destination[id[0] * columns + id[1]] =
                    source[id[0] * source_stride + source_offset + id[1]];
            });
    });
}

inline void qwen35_extract_qkv(sycl::queue& queue, const bf16* mixed,
                               bf16* query, bf16* key, bf16* value,
                               int seq, int key_heads, int value_heads,
                               int key_head_dim, int value_head_dim) {
    if (value_heads % key_heads != 0)
        throw std::runtime_error("DeltaNet value heads must be divisible by key heads");
    int repeat = value_heads / key_heads;
    int key_dim = key_heads * key_head_dim;
    int value_dim = value_heads * value_head_dim;
    int conv_dim = 2 * key_dim + value_dim;
    size_t count = (size_t)seq * value_heads * key_head_dim;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            size_t linear = id[0];
            int dim = linear % key_head_dim;
            int value_head = (linear / key_head_dim) % value_heads;
            int token = linear / ((size_t)value_heads * key_head_dim);
            int key_head = value_head / repeat;
            size_t base = (size_t)token * conv_dim;
            query[linear] = mixed[base + (size_t)key_head * key_head_dim + dim];
            key[linear] = mixed[base + key_dim + (size_t)key_head * key_head_dim + dim];
            if (dim < value_head_dim) {
                value[(size_t)token * value_dim + (size_t)value_head * value_head_dim + dim] =
                    mixed[base + 2 * key_dim + (size_t)value_head * value_head_dim + dim];
            }
        });
    });
}

inline void qwen35_conv_causal(sycl::queue& queue, const bf16* input,
                               const bf16* weight, const bf16* old_state,
                               bf16* output, int seq, int channels, int kernel,
                               bool has_state, int input_stride = 0) {
    if (input_stride == 0) input_stride = channels;
    size_t count = (size_t)seq * channels;
    int history = kernel - 1;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            int channel = id[0] % channels;
            int token = id[0] / channels;
            float sum = 0.0f;
            for (int tap = 0; tap < kernel; ++tap) {
                int source_token = token - history + tap;
                float sample = 0.0f;
                if (source_token >= 0) {
                    sample = bf16_to_float(input[(size_t)source_token * input_stride + channel]);
                } else if (has_state) {
                    int state_index = history + source_token;
                    sample = bf16_to_float(old_state[(size_t)channel * history + state_index]);
                }
                sum += bf16_to_float(weight[(size_t)channel * kernel + tap]) * sample;
            }
            output[id] = float_to_bf16(sum / (1.0f + sycl::exp(-sum)));
        });
    });
}

inline void qwen35_update_conv_state(sycl::queue& queue, const bf16* input,
                                     bf16* state, int seq, int channels,
                                     int kernel, bool had_state,
                                     int input_stride = 0) {
    if (input_stride == 0) input_stride = channels;
    int history = kernel - 1;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(channels), [=](sycl::id<1> id) {
            int channel = id[0];
            bf16 next[8];
            for (int slot = 0; slot < history; ++slot) {
                int source_token = seq - history + slot;
                if (source_token >= 0) {
                    next[slot] = input[(size_t)source_token * input_stride + channel];
                } else if (had_state) {
                    next[slot] = state[(size_t)channel * history + history + source_token];
                } else {
                    next[slot] = bf16{0};
                }
            }
            for (int slot = 0; slot < history; ++slot)
                state[(size_t)channel * history + slot] = next[slot];
        });
    });
}

inline void qwen35_compute_g(sycl::queue& queue, const bf16* a,
                             const bf16* A_log, const bf16* dt_bias,
                             bf16* g, int seq, int heads) {
    size_t count = (size_t)seq * heads;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            int head = id[0] % heads;
            float x = bf16_to_float(a[id]) + bf16_to_float(dt_bias[head]);
            float softplus = sycl::fmax(x, 0.0f) + sycl::log(1.0f + sycl::exp(-sycl::fabs(x)));
            g[id] = float_to_bf16(-sycl::exp(bf16_to_float(A_log[head])) * softplus);
        });
    });
}

// Work-group per value head. Each of 128 work-items owns one value column of
// the 128x128 FP32 recurrent state, which fits in 64 KiB SLM per work-group.
inline void qwen35_recurrent_delta(sycl::queue& queue,
                                   const bf16* query, const bf16* key,
                                   const bf16* value, const bf16* beta,
                                   const bf16* g, float* state, bf16* output,
                                   int seq, int heads, int key_dim, int value_dim) {
    if (value_dim > 256 || key_dim <= 0)
        throw std::runtime_error("Unsupported DeltaNet recurrent tile");
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> local_state((size_t)key_dim * value_dim, handler);
        handler.parallel_for(
            sycl::nd_range<1>((size_t)heads * value_dim, (size_t)value_dim),
            [=](sycl::nd_item<1> item) {
                int head = item.get_group(0);
                int value_index = item.get_local_id(0);
                size_t state_base = (size_t)head * key_dim * value_dim;
                for (int k = 0; k < key_dim; ++k)
                    local_state[(size_t)k * value_dim + value_index] =
                        state[state_base + (size_t)k * value_dim + value_index];

                for (int token = 0; token < seq; ++token) {
                    size_t qk_base = ((size_t)token * heads + head) * key_dim;
                    size_t v_base = ((size_t)token * heads + head) * value_dim;
                    float decay = sycl::exp(bf16_to_float(g[(size_t)token * heads + head]));
                    float memory = 0.0f;
                    for (int k = 0; k < key_dim; ++k) {
                        float current = local_state[(size_t)k * value_dim + value_index] * decay;
                        local_state[(size_t)k * value_dim + value_index] = current;
                        memory += current * bf16_to_float(key[qk_base + k]);
                    }
                    float delta = (bf16_to_float(value[v_base + value_index]) - memory) *
                        bf16_to_float(beta[(size_t)token * heads + head]);
                    float result = 0.0f;
                    for (int k = 0; k < key_dim; ++k) {
                        float current = local_state[(size_t)k * value_dim + value_index] +
                            bf16_to_float(key[qk_base + k]) * delta;
                        local_state[(size_t)k * value_dim + value_index] = current;
                        result += current * bf16_to_float(query[qk_base + k]);
                    }
                    output[v_base + value_index] = float_to_bf16(result);
                }

                for (int k = 0; k < key_dim; ++k)
                    state[state_base + (size_t)k * value_dim + value_index] =
                        local_state[(size_t)k * value_dim + value_index];
            });
    });
}

namespace qwen35_esimd {

namespace esimd = sycl::ext::intel::esimd;
using native_bf16 = sycl::ext::oneapi::bfloat16;

ESIMD_INLINE float dot128(esimd::simd<float, 64> a0,
                          esimd::simd<float, 64> a1,
                          esimd::simd<float, 64> b0,
                          esimd::simd<float, 64> b1) {
    esimd::simd<float, 64> product = a0 * b0 + a1 * b1;
    product.select<32, 1>(0) += product.select<32, 1>(32);
    product.select<16, 1>(0) += product.select<16, 1>(16);
    product.select<8, 1>(0) += product.select<8, 1>(8);
    product.select<4, 1>(0) += product.select<4, 1>(4);
    product.select<2, 1>(0) += product.select<2, 1>(2);
    return product[0] + product[1];
}

ESIMD_INLINE float load_bf16_scalar(const bf16* base, size_t index) {
    size_t aligned = index & ~size_t{15};
    int lane = static_cast<int>(index & size_t{15});
    auto* source = reinterpret_cast<const native_bf16*>(base + aligned);
    esimd::simd<native_bf16, 16> values = esimd::block_load<native_bf16, 16>(source);
    esimd::simd<float, 16> converted = values;
    return converted[lane];
}

}  // namespace qwen35_esimd

// Sequential Gated DeltaNet recurrence specialized for Qwen3.5's
// H=48, K=V=128 shape. Each 64-thread ESIMD work-group owns one value head;
// each work-item keeps two value rows of the recurrent state in registers for
// the entire sequence. Q/K are staged once per token in 2 KiB of SLM. The
// state layout for this path is [head, value, key], unlike the scalar/SIMT
// baseline's [head, key, value]; a cache must not switch layouts mid-stream.
inline void qwen35_recurrent_delta_esimd(
    sycl::queue& queue, const bf16* query, const bf16* key,
    const bf16* value, const bf16* beta, const bf16* g,
    float* state, bf16* output, int seq, int heads,
    int key_dim, int value_dim) {
    if (key_dim != 128 || value_dim != 128)
        throw std::runtime_error("Qwen3.5 ESIMD DeltaNet requires K=V=128");
    constexpr int work_group = 64;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>((size_t)heads * work_group, work_group),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                namespace esimd = sycl::ext::intel::esimd;
                using native_bf16 = sycl::ext::oneapi::bfloat16;
                esimd::slm_init<2048>();
                int head = static_cast<int>(item.get_group(0));
                int lane = static_cast<int>(item.get_local_id(0));
                int value0 = lane * 2;
                int value1 = value0 + 1;
                size_t state_base = (size_t)head * value_dim * key_dim;

                esimd::simd<float, 64> state00 = esimd::block_load<float, 64>(
                    state + state_base + (size_t)value0 * key_dim);
                esimd::simd<float, 64> state01 = esimd::block_load<float, 64>(
                    state + state_base + (size_t)value0 * key_dim + 64);
                esimd::simd<float, 64> state10 = esimd::block_load<float, 64>(
                    state + state_base + (size_t)value1 * key_dim);
                esimd::simd<float, 64> state11 = esimd::block_load<float, 64>(
                    state + state_base + (size_t)value1 * key_dim + 64);

                for (int token = 0; token < seq; ++token) {
                    size_t qk_base = ((size_t)token * heads + head) * key_dim;
                    if (lane < 4) {
                        const bf16* source = lane < 2 ? query : key;
                        int half = lane & 1;
                        auto* native_source = reinterpret_cast<const native_bf16*>(
                            source + qk_base + half * 64);
                        esimd::simd<native_bf16, 64> raw =
                            esimd::block_load<native_bf16, 64>(native_source);
                        esimd::simd<float, 64> converted = raw;
                        esimd::slm_block_store<float, 64>(lane * 256, converted);
                    }
                    esimd::barrier();
                    esimd::simd<float, 64> q0 =
                        esimd::slm_block_load<float, 64>(0);
                    esimd::simd<float, 64> q1 =
                        esimd::slm_block_load<float, 64>(256);
                    esimd::simd<float, 64> k0 =
                        esimd::slm_block_load<float, 64>(512);
                    esimd::simd<float, 64> k1 =
                        esimd::slm_block_load<float, 64>(768);

                    size_t gate_index = (size_t)token * heads + head;
                    float decay = esimd::exp(
                        esimd::simd<float, 8>(
                            qwen35_esimd::load_bf16_scalar(g, gate_index)))[0];
                    float beta_value =
                        qwen35_esimd::load_bf16_scalar(beta, gate_index);
                    state00 *= decay;
                    state01 *= decay;
                    state10 *= decay;
                    state11 *= decay;

                    float memory0 = qwen35_esimd::dot128(state00, state01, k0, k1);
                    float memory1 = qwen35_esimd::dot128(state10, state11, k0, k1);
                    size_t value_base = ((size_t)token * heads + head) * value_dim;
                    float input0 = qwen35_esimd::load_bf16_scalar(value, value_base + value0);
                    float input1 = qwen35_esimd::load_bf16_scalar(value, value_base + value1);
                    float delta0 = (input0 - memory0) * beta_value;
                    float delta1 = (input1 - memory1) * beta_value;
                    state00 += delta0 * k0;
                    state01 += delta0 * k1;
                    state10 += delta1 * k0;
                    state11 += delta1 * k1;

                    esimd::simd<float, 2> result;
                    result[0] = qwen35_esimd::dot128(state00, state01, q0, q1);
                    result[1] = qwen35_esimd::dot128(state10, state11, q0, q1);
                    auto* destination = reinterpret_cast<native_bf16*>(
                        output + value_base + value0);
                    esimd::block_store<native_bf16, 2>(
                        destination, esimd::simd<native_bf16, 2>(result));
                    esimd::barrier();
                }

                esimd::block_store<float, 64>(
                    state + state_base + (size_t)value0 * key_dim, state00);
                esimd::block_store<float, 64>(
                    state + state_base + (size_t)value0 * key_dim + 64, state01);
                esimd::block_store<float, 64>(
                    state + state_base + (size_t)value1 * key_dim, state10);
                esimd::block_store<float, 64>(
                    state + state_base + (size_t)value1 * key_dim + 64, state11);
            });
    });
}

inline int qwen35_mrope_axis(int frequency, const int* sections) {
    for (int axis = 1; axis <= 2; ++axis) {
        int limit = sections[axis] * 3;
        if (frequency >= axis && frequency < limit && frequency % 3 == axis)
            return axis;
    }
    return 0;
}

inline void qwen35_apply_mrope(sycl::queue& queue, bf16* query, bf16* key,
                               const int32_t* positions, int seq,
                               int query_heads, int key_heads, int head_dim,
                               int rotary_dim, float theta,
                               const std::vector<int>& section_vector) {
    if (section_vector.size() != 3)
        throw std::runtime_error("Qwen3.5 MRoPE needs three sections");
    int sections[3] = {section_vector[0], section_vector[1], section_vector[2]};
    int half = rotary_dim / 2;
    auto submit = [&](bf16* tensor, int heads) {
        size_t count = (size_t)seq * heads * half;
        queue.submit([&](sycl::handler& handler) {
            handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                int frequency = id[0] % half;
                int head = (id[0] / half) % heads;
                int token = id[0] / ((size_t)heads * half);
                int axis = qwen35_mrope_axis(frequency, sections);
                int position = positions[(size_t)axis * seq + token];
                float inv = 1.0f / sycl::pow(theta, 2.0f * frequency / rotary_dim);
                float angle = position * inv;
                float cosine = sycl::cos(angle);
                float sine = sycl::sin(angle);
                bf16* row = tensor + ((size_t)token * heads + head) * head_dim;
                float x0 = bf16_to_float(row[frequency]);
                float x1 = bf16_to_float(row[frequency + half]);
                row[frequency] = float_to_bf16(x0 * cosine - x1 * sine);
                row[frequency + half] = float_to_bf16(x0 * sine + x1 * cosine);
            });
        });
    };
    submit(query, query_heads);
    submit(key, key_heads);
}

// Online-softmax causal GQA attention. One work-group owns one (query token,
// query head); 256 work-items own output dimensions and stream the KV cache.
inline void qwen35_online_attention(sycl::queue& queue,
                                    const bf16* query, const bf16* key,
                                    const bf16* value, bf16* output,
                                    int seq, int past, int query_heads,
                                    int key_heads, int head_dim, float scale) {
    if (head_dim > 256)
        throw std::runtime_error("Qwen3.5 online attention head_dim exceeds tile");
    size_t local = 256;
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> reduce(local + 2, handler);
        handler.parallel_for(
            sycl::nd_range<1>((size_t)seq * query_heads * local, local),
            [=](sycl::nd_item<1> item) {
                int group = item.get_group(0);
                int token = group / query_heads;
                int query_head = group % query_heads;
                int key_head = query_head / (query_heads / key_heads);
                int dim = item.get_local_id(0);
                float out_acc = 0.0f;
                float running_max = -INFINITY;
                float running_sum = 0.0f;
                const bf16* qrow = query + ((size_t)token * query_heads + query_head) * head_dim;
                int visible = past + token + 1;
                for (int position = 0; position < visible; ++position) {
                    const bf16* krow = key + ((size_t)position * key_heads + key_head) * head_dim;
                    float partial = dim < head_dim
                        ? bf16_to_float(qrow[dim]) * bf16_to_float(krow[dim]) : 0.0f;
                    reduce[dim] = partial;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = 128; stride > 0; stride >>= 1) {
                        if (dim < stride) reduce[dim] += reduce[dim + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (dim == 0) {
                        float score = reduce[0] * scale;
                        float next_max = sycl::fmax(running_max, score);
                        reduce[256] = sycl::exp(running_max - next_max);
                        reduce[257] = sycl::exp(score - next_max);
                        running_sum = running_sum * reduce[256] + reduce[257];
                        running_max = next_max;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    if (dim < head_dim) {
                        const bf16* vrow = value +
                            ((size_t)position * key_heads + key_head) * head_dim;
                        out_acc = out_acc * reduce[256] +
                            bf16_to_float(vrow[dim]) * reduce[257];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (dim == 0) reduce[256] = running_sum;
                item.barrier(sycl::access::fence_space::local_space);
                if (dim < head_dim) {
                    output[((size_t)token * query_heads + query_head) * head_dim + dim] =
                        float_to_bf16(out_acc / reduce[256]);
                }
            });
    });
}

// Xe2-optimized counterpart of qwen35_online_attention. The baseline performs
// an eight-stage work-group reduction through SLM for every visible KV row.
// BMG has native 16-wide subgroup reductions, so reduce each subgroup in
// registers and combine the sixteen subgroup totals with one more subgroup.
// The online-softmax state uses separate SLM slots, which also removes the
// baseline's trailing barrier from every KV iteration.
inline void qwen35_online_attention_subgroup(
    sycl::queue& queue, const bf16* query, const bf16* key, const bf16* value,
    bf16* output, int seq, int past, int query_heads, int key_heads,
    int head_dim, float scale) {
    if (head_dim != 256)
        throw std::runtime_error(
            "Qwen3.5 subgroup attention currently requires head_dim=256");
    constexpr size_t local = 256;
    constexpr int subgroup_size = 16;
    constexpr int subgroups = static_cast<int>(local) / subgroup_size;
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> shared(subgroups + 2, handler);
        handler.parallel_for(
            sycl::nd_range<1>((size_t)seq * query_heads * local, local),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                int group = static_cast<int>(item.get_group(0));
                int token = group / query_heads;
                int query_head = group % query_heads;
                int key_head = query_head / (query_heads / key_heads);
                int dim = static_cast<int>(item.get_local_id(0));
                auto subgroup = item.get_sub_group();
                int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
                int lane = static_cast<int>(subgroup.get_local_linear_id());
                float out_acc = 0.0f;
                float running_max = -INFINITY;
                float running_sum = 0.0f;
                const bf16* qrow = query +
                    ((size_t)token * query_heads + query_head) * head_dim;
                int visible = past + token + 1;
                for (int position = 0; position < visible; ++position) {
                    const bf16* krow = key +
                        ((size_t)position * key_heads + key_head) * head_dim;
                    float partial = bf16_to_float(qrow[dim]) *
                                    bf16_to_float(krow[dim]);
                    float subgroup_sum = sycl::reduce_over_group(
                        subgroup, partial, sycl::plus<float>());
                    if (lane == 0) shared[subgroup_id] = subgroup_sum;
                    item.barrier(sycl::access::fence_space::local_space);

                    if (subgroup_id == 0) {
                        float block_partial = lane < subgroups ? shared[lane] : 0.0f;
                        float score = sycl::reduce_over_group(
                            subgroup, block_partial, sycl::plus<float>()) * scale;
                        if (lane == 0) {
                            float next_max = sycl::fmax(running_max, score);
                            shared[subgroups] = sycl::exp(running_max - next_max);
                            shared[subgroups + 1] = sycl::exp(score - next_max);
                            running_sum = running_sum * shared[subgroups] +
                                          shared[subgroups + 1];
                            running_max = next_max;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    const bf16* vrow = value +
                        ((size_t)position * key_heads + key_head) * head_dim;
                    out_acc = out_acc * shared[subgroups] +
                              bf16_to_float(vrow[dim]) * shared[subgroups + 1];
                }
                if (dim == 0) shared[subgroups] = running_sum;
                item.barrier(sycl::access::fence_space::local_space);
                output[((size_t)token * query_heads + query_head) * head_dim + dim] =
                    float_to_bf16(out_acc / shared[subgroups]);
            });
    });
}

// Flash-style causal GQA attention mapped to Battlemage XMX through the
// SPV_INTEL_subgroup_matrix_multiply_accumulate DPAS intrinsic. One work-group
// owns an 8-query x 256-output tile for one query head:
//   subgroup 0: Q[8,256] x K[16,256]^T -> score[8,16]
//   all 16 SGs: P[8,16] x V[16,256] -> O[8,256]
// Scores are consumed a 16-key block at a time with online softmax, so no
// score matrix is materialized. This is the architecture path; subgroup
// collectives only implement the softmax reductions around the two XMX GEMMs.
inline void qwen35_xmx_attention(
    sycl::queue& queue, const bf16* query, const bf16* key, const bf16* value,
    bf16* output, int seq, int past, int query_heads, int key_heads,
    int head_dim, float scale) {
    if (head_dim != 256)
        throw std::runtime_error("Qwen3.5 XMX attention requires head_dim=256");
    if (query_heads % key_heads != 0)
        throw std::runtime_error("Qwen3.5 XMX attention requires integral GQA ratio");
    constexpr int sg_size = 16;
    constexpr int query_tile = 8;
    constexpr int output_tiles = 16;
    constexpr int wg_size = sg_size * output_tiles;
    int query_tiles = (seq + query_tile - 1) / query_tile;
    queue.submit([&](sycl::handler& handler) {
        // First 8*16 floats are the current softmax block. The following 8
        // hold output rescale factors, then final denominators.
        sycl::local_accessor<float, 1> shared(
            query_tile * sg_size + 2 * query_tile, handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                (size_t)query_tiles * query_heads * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                int workgroup = static_cast<int>(item.get_group(0));
                int tile = workgroup / query_heads;
                int query_head = workgroup % query_heads;
                int key_head = query_head / (query_heads / key_heads);
                int query0 = tile * query_tile;
                auto subgroup = item.get_sub_group();
                int output_tile = static_cast<int>(subgroup.get_group_linear_id());
                int lane = static_cast<int>(subgroup.get_local_linear_id());

                diff_dpas_v8f output_acc = {0, 0, 0, 0, 0, 0, 0, 0};
                float running_max[query_tile];
                float running_sum[query_tile];
                for (int m = 0; m < query_tile; ++m) {
                    running_max[m] = -INFINITY;
                    running_sum[m] = 0.0f;
                }

                int maximum_visible = past + sycl::min(query0 + query_tile, seq);
                int key_blocks = (maximum_visible + sg_size - 1) / sg_size;
                for (int block = 0; block < key_blocks; ++block) {
                    int key0 = block * sg_size;
                    if (output_tile == 0) {
                        diff_dpas_v8f scores = {0, 0, 0, 0, 0, 0, 0, 0};
                        for (int kt = 0; kt < head_dim / sg_size; ++kt) {
                            int k = kt * sg_size + lane;
                            diff_dpas_v8s q_fragment;
                            for (int m = 0; m < query_tile; ++m) {
                                int query_token = query0 + m;
                                q_fragment[m] = query_token < seq
                                    ? static_cast<short>(query[
                                        ((size_t)query_token * query_heads +
                                         query_head) * head_dim + k])
                                    : 0;
                            }
                            diff_dpas_v8i k_fragment;
                            int key_position = key0 + lane;
                            for (int pair = 0; pair < 8; ++pair) {
                                uint32_t packed = 0;
                                if (key_position < maximum_visible) {
                                    const bf16* krow = key +
                                        ((size_t)key_position * key_heads + key_head) *
                                            head_dim;
                                    packed = static_cast<uint32_t>(krow[kt * 16 + 2 * pair]) |
                                        (static_cast<uint32_t>(
                                             krow[kt * 16 + 2 * pair + 1]) << 16);
                                }
                                k_fragment[pair] = static_cast<int>(packed);
                            }
                            scores = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                                16, q_fragment, k_fragment, scores,
                                kNvfp4DpasBF16);
                        }

                        for (int m = 0; m < query_tile; ++m) {
                            int query_token = query0 + m;
                            int key_position = key0 + lane;
                            bool visible = query_token < seq &&
                                           key_position <= past + query_token;
                            float score = visible ? scores[m] * scale : -INFINITY;
                            float block_max = sycl::reduce_over_group(
                                subgroup, score, sycl::maximum<float>());
                            float next_max = sycl::fmax(running_max[m], block_max);
                            float alpha = sycl::exp(running_max[m] - next_max);
                            float probability = visible
                                ? sycl::exp(score - next_max) : 0.0f;
                            float block_sum = sycl::reduce_over_group(
                                subgroup, probability, sycl::plus<float>());
                            running_sum[m] = running_sum[m] * alpha + block_sum;
                            running_max[m] = next_max;
                            shared[m * sg_size + lane] = probability;
                            if (lane == 0)
                                shared[query_tile * sg_size + m] = alpha;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int m = 0; m < query_tile; ++m)
                        output_acc[m] *= shared[query_tile * sg_size + m];
                    diff_dpas_v8s probability_fragment;
                    for (int m = 0; m < query_tile; ++m)
                        probability_fragment[m] = static_cast<short>(float_to_bf16(
                            shared[m * sg_size + lane]));
                    diff_dpas_v8i value_fragment;
                    int output_dim = output_tile * sg_size + lane;
                    for (int pair = 0; pair < 8; ++pair) {
                        int key_position0 = key0 + pair * 2;
                        uint32_t packed = 0;
                        if (key_position0 < maximum_visible) {
                            bf16 lo = value[
                                ((size_t)key_position0 * key_heads + key_head) *
                                    head_dim + output_dim];
                            bf16 hi = 0;
                            if (key_position0 + 1 < maximum_visible)
                                hi = value[
                                    ((size_t)(key_position0 + 1) * key_heads +
                                     key_head) * head_dim + output_dim];
                            packed = static_cast<uint32_t>(lo) |
                                     (static_cast<uint32_t>(hi) << 16);
                        }
                        value_fragment[pair] = static_cast<int>(packed);
                    }
                    output_acc = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
                        16, probability_fragment, value_fragment, output_acc,
                        kNvfp4DpasBF16);
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (output_tile == 0 && lane == 0)
                    for (int m = 0; m < query_tile; ++m)
                        shared[query_tile * (sg_size + 1) + m] = running_sum[m];
                item.barrier(sycl::access::fence_space::local_space);
                for (int m = 0; m < query_tile; ++m) {
                    int query_token = query0 + m;
                    if (query_token < seq) {
                        float denominator = shared[query_tile * (sg_size + 1) + m];
                        output[((size_t)query_token * query_heads + query_head) *
                                   head_dim + output_tile * sg_size + lane] =
                            float_to_bf16(output_acc[m] / denominator);
                    }
                }
            });
    });
}

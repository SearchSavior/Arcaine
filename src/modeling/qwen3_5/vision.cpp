#include "vision.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "kernels.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/kernels/elementwise.hpp"

namespace {

void layer_norm_async(sycl::queue& queue, const bf16* input, const bf16* weight,
                      const bf16* bias, bf16* output, int rows, int cols,
                      float epsilon = 1e-6f) {
    size_t local = 256;
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> sums(local * 2, handler);
        handler.parallel_for(
            sycl::nd_range<1>((size_t)rows * local, local),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lane = item.get_local_id(0);
                float sum = 0.0f;
                float squares = 0.0f;
                for (int col = lane; col < cols; col += 256) {
                    float value = bf16_to_float(input[(size_t)row * cols + col]);
                    sum += value;
                    squares += value * value;
                }
                sums[lane] = sum;
                sums[256 + lane] = squares;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = 128; stride > 0; stride >>= 1) {
                    if (lane < stride) {
                        sums[lane] += sums[lane + stride];
                        sums[256 + lane] += sums[256 + lane + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float mean = sums[0] / cols;
                float variance = sums[256] / cols - mean * mean;
                float inverse = sycl::rsqrt(variance + epsilon);
                for (int col = lane; col < cols; col += 256) {
                    size_t at = (size_t)row * cols + col;
                    float value = (bf16_to_float(input[at]) - mean) * inverse;
                    output[at] = float_to_bf16(
                        value * bf16_to_float(weight[col]) + bf16_to_float(bias[col]));
                }
            });
    });
}

void add_position_embedding(sycl::queue& queue, bf16* hidden,
                            const bf16* table, int tokens, int hidden_size,
                            int grid_t, int grid_h, int grid_w,
                            int merge, int side) {
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<2>(tokens, hidden_size), [=](sycl::id<2> id) {
            int token = id[0];
            int dim = id[1];
            int per_frame = grid_h * grid_w;
            int spatial = token % per_frame;
            int blocks_w = grid_w / merge;
            int block = spatial / (merge * merge);
            int within = spatial % (merge * merge);
            int y = (block / blocks_w) * merge + within / merge;
            int x = (block % blocks_w) * merge + within % merge;
            float yf = grid_h == 1 ? 0.0f : y * (side - 1.0f) / (grid_h - 1.0f);
            float xf = grid_w == 1 ? 0.0f : x * (side - 1.0f) / (grid_w - 1.0f);
            int y0 = static_cast<int>(sycl::floor(yf));
            int x0 = static_cast<int>(sycl::floor(xf));
            int y1 = sycl::min(y0 + 1, side - 1);
            int x1 = sycl::min(x0 + 1, side - 1);
            float wy = yf - y0;
            float wx = xf - x0;
            float position =
                bf16_to_float(table[((size_t)y0 * side + x0) * hidden_size + dim]) * (1-wy) * (1-wx) +
                bf16_to_float(table[((size_t)y0 * side + x1) * hidden_size + dim]) * (1-wy) * wx +
                bf16_to_float(table[((size_t)y1 * side + x0) * hidden_size + dim]) * wy * (1-wx) +
                bf16_to_float(table[((size_t)y1 * side + x1) * hidden_size + dim]) * wy * wx;
            size_t at = (size_t)token * hidden_size + dim;
            hidden[at] = float_to_bf16(bf16_to_float(hidden[at]) + position);
        });
    });
}

void split_vision_qkv(sycl::queue& queue, const bf16* projected,
                      bf16* query, bf16* key, bf16* value,
                      int tokens, int hidden) {
    size_t count = (size_t)tokens * hidden;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            int token = id[0] / hidden;
            int dim = id[0] % hidden;
            const bf16* row = projected + (size_t)token * 3 * hidden;
            query[id] = row[dim];
            key[id] = row[hidden + dim];
            value[id] = row[2 * hidden + dim];
        });
    });
}

void apply_vision_rope(sycl::queue& queue, bf16* query, bf16* key,
                       int tokens, int heads, int head_dim,
                       int grid_h, int grid_w, int merge) {
    int half = head_dim / 2;
    int axis_frequencies = half / 2;
    auto submit = [&](bf16* tensor) {
        size_t count = (size_t)tokens * heads * half;
        queue.submit([&](sycl::handler& handler) {
            handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                int frequency = id[0] % half;
                int head = (id[0] / half) % heads;
                int token = id[0] / ((size_t)heads * half);
                int spatial = token % (grid_h * grid_w);
                int blocks_w = grid_w / merge;
                int block = spatial / (merge * merge);
                int within = spatial % (merge * merge);
                int y = (block / blocks_w) * merge + within / merge;
                int x = (block % blocks_w) * merge + within % merge;
                bool width_axis = frequency >= axis_frequencies;
                int axis_frequency = width_axis ? frequency - axis_frequencies : frequency;
                int position = width_axis ? x : y;
                float inv = 1.0f / sycl::pow(10000.0f,
                    2.0f * axis_frequency / (2.0f * axis_frequencies));
                float angle = position * inv;
                float cosine = sycl::cos(angle);
                float sine = sycl::sin(angle);
                bf16* row = tensor + ((size_t)token * heads + head) * head_dim;
                float first = bf16_to_float(row[frequency]);
                float second = bf16_to_float(row[frequency + half]);
                row[frequency] = float_to_bf16(first * cosine - second * sine);
                row[frequency + half] = float_to_bf16(first * sine + second * cosine);
            });
        });
    };
    submit(query);
    submit(key);
}

void vision_online_attention(sycl::queue& queue, const bf16* query,
                             const bf16* key, const bf16* value, bf16* output,
                             int tokens, int heads, int head_dim,
                             int tokens_per_frame) {
    size_t local = 128;
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> reduce(local + 2, handler);
        handler.parallel_for(
            sycl::nd_range<1>((size_t)tokens * heads * local, local),
            [=](sycl::nd_item<1> item) {
                int group = item.get_group(0);
                int token = group / heads;
                int head = group % heads;
                int lane = item.get_local_id(0);
                int frame_start = (token / tokens_per_frame) * tokens_per_frame;
                int frame_end = frame_start + tokens_per_frame;
                const bf16* qrow = query + ((size_t)token * heads + head) * head_dim;
                float out_acc = 0.0f;
                float running_max = -INFINITY;
                float running_sum = 0.0f;
                for (int position = frame_start; position < frame_end; ++position) {
                    const bf16* krow = key + ((size_t)position * heads + head) * head_dim;
                    reduce[lane] = lane < head_dim
                        ? bf16_to_float(qrow[lane]) * bf16_to_float(krow[lane]) : 0.0f;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = 64; stride > 0; stride >>= 1) {
                        if (lane < stride) reduce[lane] += reduce[lane + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lane == 0) {
                        float score = reduce[0] / sycl::sqrt((float)head_dim);
                        float next_max = sycl::fmax(running_max, score);
                        reduce[128] = sycl::exp(running_max - next_max);
                        reduce[129] = sycl::exp(score - next_max);
                        running_sum = running_sum * reduce[128] + reduce[129];
                        running_max = next_max;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    if (lane < head_dim) {
                        const bf16* vrow = value + ((size_t)position * heads + head) * head_dim;
                        out_acc = out_acc * reduce[128] + bf16_to_float(vrow[lane]) * reduce[129];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (lane == 0) reduce[128] = running_sum;
                item.barrier(sycl::access::fence_space::local_space);
                if (lane < head_dim)
                    output[((size_t)token * heads + head) * head_dim + lane] =
                        float_to_bf16(out_acc / reduce[128]);
            });
    });
}

void gather_merge(sycl::queue& queue, const bf16* input, bf16* output,
                  int merged_tokens, int hidden, int merge_count) {
    int merged_hidden = hidden * merge_count;
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<2>(merged_tokens, merged_hidden), [=](sycl::id<2> id) {
            int token = id[0];
            int dim = id[1];
            int source_patch = dim / hidden;
            int source_dim = dim % hidden;
            output[(size_t)token * merged_hidden + dim] =
                input[((size_t)token * merge_count + source_patch) * hidden + source_dim];
        });
    });
}

}  // namespace

GpuBuffer<bf16> qwen35_vision_forward(const Qwen35VisionWeights& weights,
                                      const Qwen35VisionConfig& config,
                                      const ImageInput& image) {
    auto& context = GpuEngine::get(0);
    auto& queue = context.queue;
    int tokens = image.raw_patches;
    int patch_dim = config.in_channels * config.temporal_patch_size *
                    config.patch_size * config.patch_size;
    if (tokens <= 0 || image.patch_dim != patch_dim ||
        (int)image.pixel_values.size() != tokens * patch_dim)
        throw std::runtime_error("Invalid Qwen3.5 preprocessed image tensor");
    int grid_t = image.grid_thw[0];
    int grid_h = image.grid_thw[1];
    int grid_w = image.grid_thw[2];
    if (tokens != grid_t * grid_h * grid_w)
        throw std::runtime_error("Qwen3.5 image grid does not match patch count");

    std::vector<bf16> host_pixels(image.pixel_values.size());
    for (size_t i = 0; i < host_pixels.size(); ++i)
        host_pixels[i] = float_to_bf16(image.pixel_values[i]);
    GpuBuffer<bf16> pixels(host_pixels.size(), queue);
    pixels.upload(host_pixels.data(), host_pixels.size());
    int H = config.hidden_size;
    GpuBuffer<bf16> hidden((size_t)tokens * H, queue);
    matmul_bf16(pixels.data(), tokens, patch_dim, weights.patch_weight.data(), H,
                hidden.data(), context);
    qwen35_add_bias(queue, hidden.data(), weights.patch_bias.data(), tokens, H);
    int position_side = static_cast<int>(std::sqrt(config.num_position_embeddings));
    add_position_embedding(queue, hidden.data(), weights.position_embedding.data(),
                           tokens, H, grid_t, grid_h, grid_w,
                           config.spatial_merge_size, position_side);

    size_t wide = (size_t)tokens * std::max(3 * H, config.intermediate_size);
    GpuBuffer<bf16> norm((size_t)tokens * H, queue);
    GpuBuffer<bf16> wide_buffer(wide, queue);
    GpuBuffer<bf16> query((size_t)tokens * H, queue);
    GpuBuffer<bf16> key((size_t)tokens * H, queue);
    GpuBuffer<bf16> value((size_t)tokens * H, queue);
    int head_dim = H / config.num_heads;

    for (const auto& block : weights.blocks) {
        layer_norm_async(queue, hidden.data(), block.norm1_weight.data(),
                         block.norm1_bias.data(), norm.data(), tokens, H);
        matmul_bf16(norm.data(), tokens, H, block.qkv_weight.data(), 3 * H,
                    wide_buffer.data(), context);
        qwen35_add_bias(queue, wide_buffer.data(), block.qkv_bias.data(), tokens, 3 * H);
        split_vision_qkv(queue, wide_buffer.data(), query.data(), key.data(), value.data(), tokens, H);
        apply_vision_rope(queue, query.data(), key.data(), tokens, config.num_heads,
                          head_dim, grid_h, grid_w, config.spatial_merge_size);
        vision_online_attention(queue, query.data(), key.data(), value.data(), query.data(),
                                tokens, config.num_heads, head_dim, grid_h * grid_w);
        matmul_bf16(query.data(), tokens, H, block.proj_weight.data(), H,
                    norm.data(), context);
        qwen35_add_bias(queue, norm.data(), block.proj_bias.data(), tokens, H);
        add_inplace(queue, hidden.data(), norm.data(), (size_t)tokens * H);

        layer_norm_async(queue, hidden.data(), block.norm2_weight.data(),
                         block.norm2_bias.data(), norm.data(), tokens, H);
        matmul_bf16(norm.data(), tokens, H, block.fc1_weight.data(),
                    config.intermediate_size, wide_buffer.data(), context);
        qwen35_add_bias(queue, wide_buffer.data(), block.fc1_bias.data(),
                        tokens, config.intermediate_size);
        qwen35_gelu_tanh_inplace(queue, wide_buffer.data(),
                                 (size_t)tokens * config.intermediate_size);
        matmul_bf16(wide_buffer.data(), tokens, config.intermediate_size,
                    block.fc2_weight.data(), H, norm.data(), context);
        qwen35_add_bias(queue, norm.data(), block.fc2_bias.data(), tokens, H);
        add_inplace(queue, hidden.data(), norm.data(), (size_t)tokens * H);
    }

    layer_norm_async(queue, hidden.data(), weights.merger_norm_weight.data(),
                     weights.merger_norm_bias.data(), norm.data(), tokens, H);
    int merge_count = config.spatial_merge_size * config.spatial_merge_size;
    int merged_tokens = tokens / merge_count;
    int merged_hidden = H * merge_count;
    GpuBuffer<bf16> merged((size_t)merged_tokens * merged_hidden, queue);
    gather_merge(queue, norm.data(), merged.data(), merged_tokens, H, merge_count);
    GpuBuffer<bf16> merger_hidden((size_t)merged_tokens * merged_hidden, queue);
    matmul_bf16(merged.data(), merged_tokens, merged_hidden,
                weights.merger_fc1_weight.data(), merged_hidden,
                merger_hidden.data(), context);
    qwen35_add_bias(queue, merger_hidden.data(), weights.merger_fc1_bias.data(),
                    merged_tokens, merged_hidden);
    qwen35_gelu_tanh_inplace(queue, merger_hidden.data(),
                             (size_t)merged_tokens * merged_hidden);
    GpuBuffer<bf16> output((size_t)merged_tokens * config.out_hidden_size, queue);
    matmul_bf16(merger_hidden.data(), merged_tokens, merged_hidden,
                weights.merger_fc2_weight.data(), config.out_hidden_size,
                output.data(), context);
    qwen35_add_bias(queue, output.data(), weights.merger_fc2_bias.data(),
                    merged_tokens, config.out_hidden_size);
    queue.wait();
    return output;
}

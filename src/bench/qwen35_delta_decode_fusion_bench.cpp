#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/gpu/buffer.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"
#include "common/kernels/elementwise.hpp"
#include "modeling/qwen3_5/kernels.hpp"

namespace {
struct Stat { double mean = 0.0, sd = 0.0; };
Stat aggregate(const std::vector<double>& values) {
    Stat result;
    for (double value : values) result.mean += value;
    result.mean /= values.size();
    if (values.size() > 1) {
        for (double value : values)
            result.sd += (value - result.mean) * (value - result.mean);
        result.sd = std::sqrt(result.sd / (values.size() - 1));
    }
    return result;
}
void usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s [--p 1] [--n N] [--w N] [--r N] [--device N]\n",
        program);
}
}  // namespace

int main(int argc, char** argv) {
    int prompt = 1, tokens = 32, warmup = 1, runs = 5;
    std::string device;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (arg == "-p" || arg == "--p") prompt = std::stoi(next());
        else if (arg == "-n" || arg == "--n") tokens = std::stoi(next());
        else if (arg == "-w" || arg == "--w") warmup = std::stoi(next());
        else if (arg == "-r" || arg == "--r") runs = std::stoi(next());
        else if (arg == "--device") device = next();
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (prompt != 1 || tokens <= 0 || warmup < 0 || runs <= 0)
        throw std::runtime_error("decode fusion benchmark requires --p 1");
    if (!device.empty()) gpu_device_control::apply_device_index(device);

    constexpr int key_heads = 16, value_heads = 48;
    constexpr int key_dim = 128, value_dim = 128;
    constexpr int conv_dim = 10240, projected_stride = 16384;
    constexpr int kernel = 4;
    constexpr float epsilon = 1e-6f;
    auto& queue = GpuEngine::get(0).queue;

    uint32_t random = 0x35dec0deu;
    auto sample = [&] {
        random = random * 1664525u + 1013904223u;
        return (static_cast<int>((random >> 16) & 2047u) - 1024) / 8192.0f;
    };
    std::vector<bf16> host_projected((size_t)tokens * projected_stride);
    std::vector<bf16> host_ba((size_t)tokens * 2 * value_heads);
    std::vector<bf16> host_weight((size_t)conv_dim * kernel);
    std::vector<bf16> host_weight_t(host_weight.size());
    std::vector<bf16> host_A(value_heads), host_dt(value_heads);
    for (bf16& value : host_projected) value = float_to_bf16(sample());
    for (bf16& value : host_ba) value = float_to_bf16(sample());
    for (int channel = 0; channel < conv_dim; ++channel)
        for (int tap = 0; tap < kernel; ++tap) {
            bf16 value = float_to_bf16(sample());
            host_weight[(size_t)channel * kernel + tap] = value;
            host_weight_t[(size_t)tap * conv_dim + channel] = value;
        }
    for (int head = 0; head < value_heads; ++head) {
        host_A[head] = float_to_bf16(-1.0f + 0.01f * sample());
        host_dt[head] = float_to_bf16(sample());
    }

    GpuBuffer<bf16> projected(host_projected.size(), queue), ba(host_ba.size(), queue),
        weight(host_weight.size(), queue), weight_t(host_weight_t.size(), queue),
        A(host_A.size(), queue), dt(host_dt.size(), queue);
    projected.upload(host_projected.data(), host_projected.size());
    ba.upload(host_ba.data(), host_ba.size());
    weight.upload(host_weight.data(), host_weight.size());
    weight_t.upload(host_weight_t.data(), host_weight_t.size());
    A.upload(host_A.data(), host_A.size());
    dt.upload(host_dt.data(), host_dt.size());

    size_t state_count = (size_t)value_heads * key_dim * value_dim;
    GpuBuffer<bf16> conv_baseline((size_t)3 * conv_dim, queue),
        conv_fused((size_t)3 * conv_dim, queue), conv_output(conv_dim, queue),
        query((size_t)value_heads * key_dim, queue),
        key((size_t)value_heads * key_dim, queue),
        value((size_t)value_heads * value_dim, queue), gate(2 * value_heads, queue),
        core_fused((size_t)value_heads * value_dim, queue),
        z_baseline((size_t)value_heads * value_dim, queue),
        z_fused((size_t)value_heads * value_dim, queue);
    GpuBuffer<float> state_baseline(state_count, queue), state_fused(state_count, queue);

    auto reset = [&] {
        conv_baseline.zero(); conv_fused.zero();
        state_baseline.zero(); state_fused.zero();
    };
    auto baseline = [&] {
        for (int token = 0; token < tokens; ++token) {
            const bf16* row = projected.data() + (size_t)token * projected_stride;
            qwen35_conv_causal(queue, row, weight.data(), conv_baseline.data(),
                               conv_output.data(), 1, conv_dim, kernel, token > 0,
                               projected_stride);
            qwen35_update_conv_state(queue, row, conv_baseline.data(), 1, conv_dim,
                                     kernel, token > 0, projected_stride);
            qwen35_extract_qkv(queue, conv_output.data(), query.data(), key.data(),
                               value.data(), 1, key_heads, value_heads,
                               key_dim, value_dim);
            l2norm(queue, query.data(), query.data(), value_heads, key_dim, epsilon);
            l2norm(queue, key.data(), key.data(), value_heads, key_dim, epsilon);
            scale_inplace(queue, query.data(), (size_t)value_heads * key_dim,
                          0.08838834764831845f);
            queue.memcpy(gate.data(), ba.data() + (size_t)token * 2 * value_heads,
                         (size_t)2 * value_heads * sizeof(bf16));
            sigmoid_inplace(queue, gate.data(), value_heads);
            qwen35_compute_g(queue, gate.data() + value_heads, A.data(), dt.data(),
                             gate.data() + value_heads, 1, value_heads);
            qwen35_recurrent_delta_esimd(
                queue, query.data(), key.data(), value.data(), gate.data(),
                gate.data() + value_heads, state_baseline.data(), value.data(),
                1, value_heads, key_dim, value_dim);
            qwen35_copy_strided(queue, row, projected_stride, conv_dim,
                                z_baseline.data(), 1, value_heads * value_dim);
        }
    };
    auto fused = [&] {
        for (int token = 0; token < tokens; ++token) {
            const bf16* row = projected.data() + (size_t)token * projected_stride;
            qwen35_delta_decode_fused_esimd(
                queue, row, projected_stride, weight_t.data(), conv_fused.data(),
                A.data(), dt.data(), ba.data() + (size_t)token * 2 * value_heads,
                state_fused.data(), core_fused.data(), z_fused.data(), key_heads,
                value_heads, key_dim, value_dim, conv_dim, kernel, epsilon);
            qwen35_update_conv_state_time_major(
                queue, row, projected_stride, conv_fused.data(), conv_dim);
        }
    };

    reset(); baseline(); fused(); queue.wait();
    std::vector<bf16> expected((size_t)value_heads * value_dim),
                      actual(expected.size()), expected_z(expected.size()),
                      actual_z(expected.size());
    value.download(expected.data(), expected.size());
    core_fused.download(actual.data(), actual.size());
    z_baseline.download(expected_z.data(), expected_z.size());
    z_fused.download(actual_z.data(), actual_z.size());
    float max_abs = 0.0f, z_max_abs = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(
            bf16_to_float(actual[i]) - bf16_to_float(expected[i])));
        z_max_abs = std::max(z_max_abs, std::fabs(
            bf16_to_float(actual_z[i]) - bf16_to_float(expected_z[i])));
    }

    auto measure = [&](const char* name, auto&& operation) {
        for (int i = 0; i < warmup; ++i) { reset(); operation(); queue.wait(); }
        std::vector<double> samples;
        for (int i = 0; i < runs; ++i) {
            reset();
            auto start = std::chrono::steady_clock::now();
            operation(); queue.wait();
            samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count() / tokens);
        }
        Stat stat = aggregate(samples);
        std::printf("qwen35_delta_decode kernel=%s p=1 n=%d mean_ms_per_token=%.6f "
                    "sd_ms=%.6f core_max_abs=%.6f z_max_abs=%.6f\n",
                    name, tokens, stat.mean, stat.sd, max_abs, z_max_abs);
    };
    measure("baseline", baseline);
    measure("fused-esimd", fused);
    return 0;
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/gpu/buffer.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"
#include "modeling/qwen3_5/kernels.hpp"

namespace {

std::vector<std::string> split_csv(const std::string& value) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= value.size()) {
        size_t end = value.find(',', begin);
        if (end == std::string::npos) end = value.size();
        if (end > begin) result.push_back(value.substr(begin, end - begin));
        begin = end + 1;
    }
    return result;
}

std::vector<int> parse_int_csv(const std::string& value) {
    std::vector<int> result;
    for (const auto& token : split_csv(value)) result.push_back(std::stoi(token));
    return result;
}

struct Stat { double mean = 0.0, sd = 0.0; };

Stat aggregate(const std::vector<double>& values) {
    Stat stat;
    for (double value : values) stat.mean += value;
    stat.mean /= values.size();
    if (values.size() > 1) {
        for (double value : values)
            stat.sd += (value - stat.mean) * (value - stat.mean);
        stat.sd = std::sqrt(stat.sd / (values.size() - 1));
    }
    return stat;
}

void usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --p <csv>          prefill query lengths       (default: 128,512)\n"
        "  --d <csv>          decode KV depths           (default: 0,512,1024)\n"
        "  --w <N>            warmup runs per cell       (default: 1)\n"
        "  --r <N>            timed runs per cell        (default: 5)\n"
        "  --kernels <csv>    baseline,subgroup,xmx      (default: baseline,xmx)\n"
        "  --device <N>       visible Level Zero GPU\n",
        program);
}

}  // namespace

int main(int argc, char** argv) {
    std::string p_csv = "128,512";
    std::string d_csv = "0,512,1024";
    std::string kernels_csv = "baseline,xmx";
    std::string device;
    int warmup = 1;
    int runs = 5;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (arg == "-p" || arg == "--p") p_csv = next();
        else if (arg == "-d" || arg == "--d") d_csv = next();
        else if (arg == "-w" || arg == "--w") warmup = std::stoi(next());
        else if (arg == "-r" || arg == "--r") runs = std::stoi(next());
        else if (arg == "--kernels") kernels_csv = next();
        else if (arg == "--device") device = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", arg.c_str()); return 1; }
    }
    if (!device.empty()) gpu_device_control::apply_device_index(device);
    std::vector<int> prefills = parse_int_csv(p_csv);
    std::vector<int> depths = parse_int_csv(d_csv);
    std::vector<std::string> kernels = split_csv(kernels_csv);
    if (prefills.empty() || depths.empty() || kernels.empty() || warmup < 0 || runs <= 0)
        throw std::runtime_error("invalid benchmark arguments");

    constexpr int query_heads = 24;
    constexpr int key_heads = 4;
    constexpr int head_dim = 256;
    constexpr float scale = 0.0625f;
    int max_query = *std::max_element(prefills.begin(), prefills.end());
    int max_depth = *std::max_element(depths.begin(), depths.end());
    int max_kv = std::max(max_query, max_depth + 1);
    auto& queue = GpuEngine::get(0).queue;

    std::vector<bf16> host_q((size_t)max_query * query_heads * head_dim);
    std::vector<bf16> host_k((size_t)max_kv * key_heads * head_dim);
    std::vector<bf16> host_v(host_k.size());
    uint32_t state = 42;
    auto fill = [&](std::vector<bf16>& values) {
        for (bf16& value : values) {
            state = state * 1664525u + 1013904223u;
            float x = (static_cast<int>((state >> 16) & 1023u) - 512) / 1024.0f;
            value = float_to_bf16(x);
        }
    };
    fill(host_q); fill(host_k); fill(host_v);
    GpuBuffer<bf16> q(host_q.size(), queue), k(host_k.size(), queue),
                      v(host_v.size(), queue);
    GpuBuffer<bf16> output((size_t)max_query * query_heads * head_dim, queue);
    GpuBuffer<bf16> reference(output.count(), queue);
    q.upload(host_q.data(), host_q.size());
    k.upload(host_k.data(), host_k.size());
    v.upload(host_v.data(), host_v.size());

    std::printf("[bench] Qwen3.5 full attention: q_heads=24 kv_heads=4 head_dim=256\n");
    auto run_kernel = [&](const std::string& kernel, bf16* destination,
                          int seq, int past) {
        if (kernel == "baseline")
            qwen35_online_attention(queue, q.data(), k.data(), v.data(), destination,
                                    seq, past, query_heads, key_heads, head_dim, scale);
        else if (kernel == "subgroup")
            qwen35_online_attention_subgroup(
                queue, q.data(), k.data(), v.data(), destination, seq, past,
                query_heads, key_heads, head_dim, scale);
        else if (kernel == "xmx")
            qwen35_xmx_attention(queue, q.data(), k.data(), v.data(), destination,
                                 seq, past, query_heads, key_heads, head_dim, scale);
        else
            throw std::runtime_error("unknown kernel: " + kernel);
    };

    auto benchmark_cell = [&](const char* kind, int seq, int past) {
        run_kernel("baseline", reference.data(), seq, past);
        queue.wait();
        std::vector<bf16> host_reference((size_t)seq * query_heads * head_dim);
        reference.download(host_reference.data(), host_reference.size());
        for (const std::string& kernel : kernels) {
            run_kernel(kernel, output.data(), seq, past);
            queue.wait();
            std::vector<bf16> host_output(host_reference.size());
            output.download(host_output.data(), host_output.size());
            float max_abs = 0.0f;
            float max_rel = 0.0f;
            for (size_t i = 0; i < host_output.size(); ++i) {
                float expected = bf16_to_float(host_reference[i]);
                float actual = bf16_to_float(host_output[i]);
                float error = std::fabs(expected - actual);
                max_abs = std::max(max_abs, error);
                max_rel = std::max(max_rel, error / std::max(1e-3f, std::fabs(expected)));
            }
            for (int i = 0; i < warmup; ++i) run_kernel(kernel, output.data(), seq, past);
            queue.wait();
            std::vector<double> samples;
            for (int i = 0; i < runs; ++i) {
                auto start = std::chrono::steady_clock::now();
                run_kernel(kernel, output.data(), seq, past);
                queue.wait();
                samples.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            }
            Stat stat = aggregate(samples);
            std::printf("qwen35_attention kind=%s kernel=%s q=%d kv=%d mean_ms=%.3f "
                        "sd_ms=%.3f max_abs=%.6f max_rel=%.6f\n",
                        kind, kernel.c_str(), seq, past + seq, stat.mean, stat.sd,
                        max_abs, max_rel);
        }
    };

    for (int seq : prefills) benchmark_cell("prefill", seq, 0);
    for (int depth : depths) benchmark_cell("decode", 1, depth);
    return 0;
}

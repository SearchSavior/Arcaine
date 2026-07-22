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
        "  --p <csv>          prefill lengths       (default: 512,1024)\n"
        "  --n <N>            sequential decode tokens (default: 32)\n"
        "  --w <N>            warmup runs per cell (default: 1)\n"
        "  --r <N>            timed runs per cell  (default: 5)\n"
        "  --kernels <csv>    baseline,esimd       (default: baseline,esimd)\n"
        "  --device <N>       visible Level Zero GPU\n",
        program);
}

}  // namespace

int main(int argc, char** argv) {
    std::string p_csv = "512,1024";
    std::string kernels_csv = "baseline,esimd";
    std::string device;
    int decode_tokens = 32;
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
        else if (arg == "-n" || arg == "--n") decode_tokens = std::stoi(next());
        else if (arg == "-w" || arg == "--w") warmup = std::stoi(next());
        else if (arg == "-r" || arg == "--r") runs = std::stoi(next());
        else if (arg == "--kernels") kernels_csv = next();
        else if (arg == "--device") device = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", arg.c_str()); return 1; }
    }
    if (!device.empty()) gpu_device_control::apply_device_index(device);
    std::vector<int> prefills = parse_int_csv(p_csv);
    std::vector<std::string> kernels = split_csv(kernels_csv);
    if (prefills.empty() || kernels.empty() || decode_tokens <= 0 ||
        warmup < 0 || runs <= 0)
        throw std::runtime_error("invalid benchmark arguments");

    constexpr int heads = 48;
    constexpr int key_dim = 128;
    constexpr int value_dim = 128;
    int max_tokens = std::max(decode_tokens,
        *std::max_element(prefills.begin(), prefills.end()));
    size_t vectors = (size_t)max_tokens * heads * key_dim;
    size_t state_values = (size_t)heads * key_dim * value_dim;
    auto& queue = GpuEngine::get(0).queue;

    std::vector<bf16> host_q(vectors), host_k(vectors), host_v(vectors);
    std::vector<bf16> host_beta((size_t)max_tokens * heads);
    std::vector<bf16> host_g(host_beta.size());
    uint32_t random = 0x35d31a5u;
    auto sample = [&]() {
        random = random * 1664525u + 1013904223u;
        return (static_cast<int>((random >> 16) & 2047u) - 1024) / 32768.0f;
    };
    for (int token = 0; token < max_tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            float q_norm = 0.0f, k_norm = 0.0f;
            size_t base = ((size_t)token * heads + head) * key_dim;
            for (int dim = 0; dim < key_dim; ++dim) {
                float q = sample(), k = sample();
                host_q[base + dim] = float_to_bf16(q);
                host_k[base + dim] = float_to_bf16(k);
                host_v[base + dim] = float_to_bf16(sample());
                q_norm += q * q;
                k_norm += k * k;
            }
            float q_scale = 0.08838834764831845f / std::sqrt(q_norm + 1e-6f);
            float k_scale = 1.0f / std::sqrt(k_norm + 1e-6f);
            for (int dim = 0; dim < key_dim; ++dim) {
                host_q[base + dim] = float_to_bf16(
                    bf16_to_float(host_q[base + dim]) * q_scale);
                host_k[base + dim] = float_to_bf16(
                    bf16_to_float(host_k[base + dim]) * k_scale);
            }
            size_t gate = (size_t)token * heads + head;
            host_beta[gate] = float_to_bf16(0.25f + std::fabs(sample()) * 8.0f);
            host_g[gate] = float_to_bf16(-0.001f - std::fabs(sample()) * 0.25f);
        }
    }

    GpuBuffer<bf16> q(host_q.size(), queue), k(host_k.size(), queue),
                      v(host_v.size(), queue), beta(host_beta.size(), queue),
                      g(host_g.size(), queue), output(vectors, queue),
                      reference(vectors, queue);
    GpuBuffer<float> baseline_state(state_values, queue),
                     esimd_state(state_values, queue);
    q.upload(host_q.data(), host_q.size());
    k.upload(host_k.data(), host_k.size());
    v.upload(host_v.data(), host_v.size());
    beta.upload(host_beta.data(), host_beta.size());
    g.upload(host_g.data(), host_g.size());

    auto reset = [&](const std::string& kernel) {
        if (kernel == "baseline") baseline_state.zero();
        else if (kernel == "esimd") esimd_state.zero();
        else throw std::runtime_error("unknown kernel: " + kernel);
    };
    auto run_prefill = [&](const std::string& kernel, bf16* destination, int seq) {
        if (kernel == "baseline")
            qwen35_recurrent_delta(queue, q.data(), k.data(), v.data(), beta.data(),
                                   g.data(), baseline_state.data(), destination, seq,
                                   heads, key_dim, value_dim);
        else if (kernel == "esimd")
            qwen35_recurrent_delta_esimd(queue, q.data(), k.data(), v.data(),
                                         beta.data(), g.data(), esimd_state.data(),
                                         destination, seq, heads, key_dim, value_dim);
        else
            throw std::runtime_error("unknown kernel: " + kernel);
    };
    auto run_decode = [&](const std::string& kernel, bf16* destination, int tokens) {
        for (int token = 0; token < tokens; ++token) {
            size_t vector_offset = (size_t)token * heads * key_dim;
            size_t gate_offset = (size_t)token * heads;
            if (kernel == "baseline")
                qwen35_recurrent_delta(
                    queue, q.data() + vector_offset, k.data() + vector_offset,
                    v.data() + vector_offset, beta.data() + gate_offset,
                    g.data() + gate_offset, baseline_state.data(),
                    destination + vector_offset, 1, heads, key_dim, value_dim);
            else if (kernel == "esimd")
                qwen35_recurrent_delta_esimd(
                    queue, q.data() + vector_offset, k.data() + vector_offset,
                    v.data() + vector_offset, beta.data() + gate_offset,
                    g.data() + gate_offset, esimd_state.data(),
                    destination + vector_offset, 1, heads, key_dim, value_dim);
            else
                throw std::runtime_error("unknown kernel: " + kernel);
        }
    };

    auto correctness = [&](int tokens, bool decode) {
        reset("baseline");
        if (decode) run_decode("baseline", reference.data(), tokens);
        else run_prefill("baseline", reference.data(), tokens);
        queue.wait();
        reset("esimd");
        if (decode) run_decode("esimd", output.data(), tokens);
        else run_prefill("esimd", output.data(), tokens);
        queue.wait();
        size_t count = (size_t)tokens * heads * value_dim;
        std::vector<bf16> expected(count), actual(count);
        reference.download(expected.data(), count);
        output.download(actual.data(), count);
        float max_abs = 0.0f, max_rel = 0.0f;
        for (size_t index = 0; index < count; ++index) {
            float e = bf16_to_float(expected[index]);
            float a = bf16_to_float(actual[index]);
            float error = std::fabs(e - a);
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, error / std::max(1e-3f, std::fabs(e)));
        }
        return std::pair<float, float>{max_abs, max_rel};
    };

    std::printf("[bench] Qwen3.5 DeltaNet core: heads=48 K=128 V=128 state=FP32\n");
    auto benchmark = [&](const char* kind, int tokens, bool decode) {
        auto errors = correctness(tokens, decode);
        for (const auto& kernel : kernels) {
            for (int i = 0; i < warmup; ++i) {
                reset(kernel);
                if (decode) run_decode(kernel, output.data(), tokens);
                else run_prefill(kernel, output.data(), tokens);
            }
            queue.wait();
            std::vector<double> samples;
            for (int i = 0; i < runs; ++i) {
                reset(kernel);
                auto start = std::chrono::steady_clock::now();
                if (decode) run_decode(kernel, output.data(), tokens);
                else run_prefill(kernel, output.data(), tokens);
                queue.wait();
                samples.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            }
            Stat stat = aggregate(samples);
            double per_token = stat.mean / tokens;
            std::printf("qwen35_deltanet kind=%s kernel=%s tokens=%d mean_ms=%.3f "
                        "sd_ms=%.3f ms_per_token=%.6f max_abs=%.6f max_rel=%.6f\n",
                        kind, kernel.c_str(), tokens, stat.mean, stat.sd,
                        per_token, errors.first, errors.second);
        }
    };

    for (int seq : prefills) benchmark("prefill", seq, false);
    benchmark("decode", decode_tokens, true);
    return 0;
}

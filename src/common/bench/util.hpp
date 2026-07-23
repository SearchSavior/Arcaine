#pragma once

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace arcaine::bench {

struct Stat { double mean = 0.0, sd = 0.0; };

inline Stat aggregate(const std::vector<double>& values) {
    Stat s;
    if (values.empty()) return s;
    for (double v : values) s.mean += v;
    s.mean /= values.size();
    if (values.size() > 1) {
        for (double v : values) s.sd += (v - s.mean) * (v - s.mean);
        s.sd = std::sqrt(s.sd / (values.size() - 1));
    }
    return s;
}

inline std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

inline std::vector<int> parse_int_csv(const std::string& s) {
    std::vector<int> out;
    for (const auto& t : split_csv(s)) out.push_back(std::stoi(t));
    return out;
}

template <typename Fn>
double elapsed_ms(sycl::queue& q, int iterations, const Fn& fn) {
    q.wait();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) fn();
    q.wait();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           (double)iterations;
}

}  // namespace arcaine::bench

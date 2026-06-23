#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct TensorView {
    std::string          dtype;     // e.g. "BF16", "F32", "Q8_0"
    std::vector<int64_t> shape;
    const void*          data;      // pointer into mmap region (zero-copy)
    size_t               nbytes;

    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }
};

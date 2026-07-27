#pragma once

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace gpu_device_control {

inline bool valid_device_index(const std::string& value) {
    if (value.empty()) return false;
    for (unsigned char ch : value) {
        if (!std::isdigit(ch))
            return false;
    }
    return true;
}

inline void apply_device_index(const std::string& index) {
    if (!valid_device_index(index))
        throw std::runtime_error("--device expects a non-negative GPU index");

    if (::setenv("ZE_AFFINITY_MASK", index.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set ZE_AFFINITY_MASK: " +
                                 std::string(std::strerror(errno)));
    }
}

inline const char* active_gpus_spec() {
    const char* spec = std::getenv("ZE_AFFINITY_MASK");
    return (spec && *spec) ? spec : nullptr;
}

}  // namespace gpu_device_control

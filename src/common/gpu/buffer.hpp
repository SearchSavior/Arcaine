#pragma once
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "engine.hpp"

using bf16 = uint16_t;  // BF16 stored as raw bits

inline float bf16_to_float(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t float_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    // round to nearest even
    uint32_t rounding_bias = ((u >> 16) & 1) + 0x7FFFu;
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

template<typename T>
class GpuBuffer {
public:
    GpuBuffer() = default;

    // Allocates on the given queue's device.  Defaults to GPU 0.
    explicit GpuBuffer(size_t n, sycl::queue& q = GpuEngine::get().queue)
        : count_(n), q_(&q)
    {
        ptr_ = sycl::malloc_device<T>(n, q);
        if (!ptr_) throw std::runtime_error("GpuBuffer: device alloc failed");
    }

    ~GpuBuffer() {
        if (ptr_ && q_) sycl::free(ptr_, *q_);
    }

    // Non-copyable
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Movable
    GpuBuffer(GpuBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_), q_(o.q_) {
        o.ptr_ = nullptr; o.count_ = 0; o.q_ = nullptr;
    }
    GpuBuffer& operator=(GpuBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr_ && q_) sycl::free(ptr_, *q_);
            ptr_ = o.ptr_; count_ = o.count_; q_ = o.q_;
            o.ptr_ = nullptr; o.count_ = 0; o.q_ = nullptr;
        }
        return *this;
    }

    T*     data()  const { return ptr_; }
    size_t count() const { return count_; }
    bool   empty() const { return ptr_ == nullptr; }

    void upload(const T* host, size_t n) {
        q_->memcpy(ptr_, host, n * sizeof(T)).wait();
    }

    void download(T* host, size_t n) const {
        q_->memcpy(host, ptr_, n * sizeof(T)).wait();
    }

    void zero() {
        q_->memset(ptr_, 0, count_ * sizeof(T)).wait();
    }

    sycl::queue& queue() const { return *q_; }

private:
    T*           ptr_   = nullptr;
    size_t       count_ = 0;
    sycl::queue* q_     = nullptr;
};

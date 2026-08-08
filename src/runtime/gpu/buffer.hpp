#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "runtime/gpu/engine.hpp"

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

// Process-wide live device bytes allocated through GpuBuffer (all T).
// Used by model preflight checks to estimate free device memory.
inline std::atomic<size_t>& gpu_buffer_live_bytes() {
    static std::atomic<size_t> b{0};
    return b;
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
          gpu_buffer_live_bytes().fetch_add(n * sizeof(T), std::memory_order_relaxed);
      }

      ~GpuBuffer() {
          if (ptr_ && q_) {
              sycl::free(ptr_, *q_);
              gpu_buffer_live_bytes().fetch_sub(count_ * sizeof(T), std::memory_order_relaxed);
          }
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
              if (ptr_ && q_) {
                  sycl::free(ptr_, *q_);
                  gpu_buffer_live_bytes().fetch_sub(count_ * sizeof(T), std::memory_order_relaxed);
              }
              ptr_ = o.ptr_; count_ = o.count_; q_ = o.q_;
            o.ptr_ = nullptr; o.count_ = 0; o.q_ = nullptr;
        }
        return *this;
    }

    T*     data()  const { return ptr_; }
    size_t count() const { return count_; }
    bool   empty() const { return ptr_ == nullptr; }

    void upload(const T* host, size_t n) {
        copy_chunked(ptr_, host, n * sizeof(T));
    }

    void download(T* host, size_t n) const {
        copy_chunked(host, ptr_, n * sizeof(T));
    }

    void zero() {
        // memset takes the same path as memcpy in the driver, so it gets the
        // same treatment; a large KV cache is well over the limit.
        size_t remaining = count_ * sizeof(T);
        char* dst = reinterpret_cast<char*>(ptr_);
        sycl::event last;
        while (remaining) {
            size_t step = std::min(remaining, kMaxTransferBytes);
            last = q_->memset(dst, 0, step);
            dst += step;
            remaining -= step;
        }
        if (count_) last.wait();
    }

    sycl::queue& queue() const { return *q_; }

private:
    // Large single transfers can stall indefinitely on some kernel/driver
    // pairings: the copy is enqueued, its event never signals, and the runtime
    // spins in sched_yield() forever. It is a hang, not a slow path - nothing
    // times out and no error is reported.
    //
    // Observed loading Qwen3.6-27B on an Arc Pro B70 under kernel 7.0: the load
    // stops with device memory frozen at ~2.5 GB, which is the embedding
    // table's size, and the backtrace sits in urEventWait under
    // GpuBuffer::upload. The same binary and card under kernel 6.17 never
    // stalls, and swapping the GPU userspace across 26.18 / 26.22 / 26.27
    // changes nothing, so it is below the userspace.
    //
    // **The mechanism is not established.** A standalone SYCL reproducer with
    // no oneDNN and no model stalls only intermittently: a 2 GiB copy hung
    // once, 1 GiB on an in-order queue hung once, and a size sweep from 256 MiB
    // to 1 GiB then passed cleanly on the same machine minutes later. So the
    // trigger is not simply "copies above size N", and this constant is an
    // empirical mitigation, not a documented boundary. What is consistent is
    // that this engine reproduces it reliably where the probes do not, and this
    // engine differs by doing ~1968 allocate-then-copy pairs during load.
    // llama.cpp on the same host is unaffected and also splits its transfers.
    //
    // 256 MiB was the largest size that never stalled in any probe run. If a
    // load hangs again, lower it and re-test rather than assuming this value is
    // principled. Splitting costs nothing measurable - the pieces sustain the
    // same GB/s - and the queue is in-order, so ordering holds and only the
    // final event needs waiting on.
    static constexpr size_t kMaxTransferBytes = 256ull * 1024ull * 1024ull;

    void copy_chunked(void* dst, const void* src, size_t bytes) const {
        char* d = static_cast<char*>(dst);
        const char* s = static_cast<const char*>(src);
        sycl::event last;
        size_t remaining = bytes;
        while (remaining) {
            size_t step = std::min(remaining, kMaxTransferBytes);
            last = q_->memcpy(d, s, step);
            d += step;
            s += step;
            remaining -= step;
        }
        if (bytes) last.wait();
    }

    T*           ptr_   = nullptr;
    size_t       count_ = 0;
    sycl::queue* q_     = nullptr;
};

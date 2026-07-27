#pragma once

#include <atomic>

namespace arcaine::inference {

// Lightweight thread-safe cancellation flag. The transport sets it when the
// HTTP client disconnects, the SSE sink fails, or the request is explicitly
// cancelled. Each model session checks it at model-owned interruption points.
class CancellationToken {
public:
    void set() noexcept { flag_.store(true, std::memory_order_release); }
    bool is_set() const noexcept { return flag_.load(std::memory_order_acquire); }

    // Accessor for model engines that accept a raw atomic flag (e.g. the
    // diffusion denoising loop's per-step cancellation check).
    const std::atomic<bool>& flag() const noexcept { return flag_; }

private:
    std::atomic<bool> flag_{false};
};

}  // namespace arcaine::inference

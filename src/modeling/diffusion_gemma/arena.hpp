#pragma once
#include "../../common/gpu/buffer.hpp"
#include "../../common/gpu/engine.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Liveness-based device-memory arena (path A — the execution port of the
// mem_planner analysis).  Replaces the old "stable named pools" (diffscratch /
// ExpertWorkspace): instead of one persistent buffer per logical tensor (whose
// high-water marks all sum up), every activation is an *anonymous* allocation
// carved from a single per-GPU arena and returned to the free-list the moment
// its scoped handle dies.  Tensors with disjoint lifetimes — attention temps vs
// FFN temps, the MoE Xe→gu→act→Ye pipeline, the LM-head logits/probs — then
// reuse the same storage, so the arena's capacity is the *peak simultaneously
// live* set rather than the sum of every buffer.  That is exactly what
// memplan::plan() predicts, and it roughly halves the activation footprint,
// handing the difference back to the KV cache (more context).
//
// Safety: inference runs one in-order SYCL queue per GPU (oneDNN default
// stream), and the host issues kernels in program order.  A handle is only
// reset() after all kernels that read it have been *submitted*; any later
// allocation that reuses the freed offset is written by a kernel submitted
// afterwards, so the in-order queue serialises the consumer before the new
// producer.  This is the same guarantee the old pools relied on to reuse
// buffers across layers — here it also holds *within* a layer.
//
// DIFF_ARENA=off (or DISABLE_SCRATCH=1) drops to a non-pooled mode where every
// alloc is a fresh sycl::malloc_device freed on scope exit — the pre-pool
// baseline, kept for A/B measurement.
// ---------------------------------------------------------------------------
namespace diffarena {

inline bool disabled() {
    static const bool v = [] {
        const char* a = std::getenv("DIFF_ARENA");
        if (a && (!std::strcmp(a, "off") || !std::strcmp(a, "0") ||
                  !std::strcmp(a, "false") || !std::strcmp(a, "no")))
            return true;
        const char* e = std::getenv("DISABLE_SCRATCH");
        return e && (!std::strcmp(e, "1") || !std::strcmp(e, "true") ||
                     !std::strcmp(e, "TRUE") || !std::strcmp(e, "yes"));
    }();
    return v;
}

class Arena;

// Scoped, movable handle to one arena allocation.  Returns its storage to the
// arena on destruction (or on reset()).  Non-copyable, like GpuBuffer.
template <class T>
class Alloc {
public:
    Alloc() = default;
    Alloc(Arena* a, int chunk, size_t off, size_t bytes, T* ptr, size_t count)
        : arena_(a), chunk_(chunk), off_(off), bytes_(bytes), ptr_(ptr), count_(count) {}
    ~Alloc() { reset(); }

    Alloc(const Alloc&) = delete;
    Alloc& operator=(const Alloc&) = delete;
    Alloc(Alloc&& o) noexcept { steal(o); }
    Alloc& operator=(Alloc&& o) noexcept { if (this != &o) { reset(); steal(o); } return *this; }

    T*     data()  const { return ptr_; }
    size_t count() const { return count_; }
    bool   empty() const { return ptr_ == nullptr; }

    void reset();   // defined out-of-line below (needs Arena)

private:
    void steal(Alloc& o) {
        arena_ = o.arena_; chunk_ = o.chunk_; off_ = o.off_;
        bytes_ = o.bytes_; ptr_ = o.ptr_; count_ = o.count_;
        o.arena_ = nullptr; o.ptr_ = nullptr; o.count_ = 0; o.bytes_ = 0;
    }
    Arena* arena_ = nullptr;
    int    chunk_ = -1;
    size_t off_   = 0;
    size_t bytes_ = 0;
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};

// Per-GPU arena: a list of device "chunks" (never moved or freed, so live
// pointers stay valid) over which a best-fit, coalescing free-list hands out
// offsets.  One chunk normally covers the whole pass (sized by reserve() from
// the planner estimate); extra chunks are added only if a data-dependent spike
// (e.g. a hot MoE expert) outgrows it.
class Arena {
public:
    explicit Arena(int gpu) : gpu_(gpu) {}

    static constexpr size_t ALIGN = 256;
    static size_t aligned(size_t s) { return ((s ? s : 1) + ALIGN - 1) & ~(ALIGN - 1); }

    // Pre-size chunk 0 so the common case never needs to grow mid-pass.
    void reserve(size_t bytes) {
        if (disabled()) return;
        bytes = aligned(bytes);
        if (capacity_ >= bytes) return;
        add_chunk(bytes - capacity_);
    }

    template <class T>
    Alloc<T> alloc(size_t n) {
        size_t bytes = aligned(n * sizeof(T));
        if (disabled()) {
            T* p = sycl::malloc_device<T>(n ? n : 1, queue());
            if (!p) throw std::runtime_error("diffarena: standalone alloc failed");
            return Alloc<T>(this, -1, 0, bytes, p, n);
        }
        int ci; size_t off;
        if (!find_fit(bytes, ci, off)) {
            // Grow proportionally so overflow needs only a handful of chunks.
            add_chunk(std::max(bytes, capacity_ / 4));
            find_fit(bytes, ci, off);
        }
        carve(ci, off, bytes);
        live_ += bytes;
        peak_live_ = std::max(peak_live_, live_);
        T* p = reinterpret_cast<T*>(chunks_[ci].data() + off);
        return Alloc<T>(this, ci, off, bytes, p, n);
    }

    // Called by Alloc::reset(); ptr only used for the standalone (disabled) path.
    void free_block(int chunk, size_t off, size_t bytes, void* ptr) {
        if (chunk < 0) { if (ptr) sycl::free(ptr, queue()); return; }
        insert_free(chunk, off, bytes);
        live_ -= bytes;
    }

    size_t capacity()  const { return capacity_; }   // device memory held
    size_t peak_live() const { return peak_live_; }  // max simultaneously live

private:
    struct Block { int chunk; size_t off, size; };

    sycl::queue& queue() { return GpuEngine::get(gpu_).queue; }

    void add_chunk(size_t size) {
        size = aligned(size);
        chunks_.emplace_back(size, queue());
        free_.push_back({(int)chunks_.size() - 1, 0, size});
        capacity_ += size;
    }

    // Best-fit across all free blocks.
    bool find_fit(size_t bytes, int& ci, size_t& off) {
        int best = -1; size_t best_sz = SIZE_MAX;
        for (int i = 0; i < (int)free_.size(); ++i)
            if (free_[i].size >= bytes && free_[i].size < best_sz) { best = i; best_sz = free_[i].size; }
        if (best < 0) return false;
        ci = free_[best].chunk; off = free_[best].off;
        fit_idx_ = best;
        return true;
    }

    void carve(int /*ci*/, size_t /*off*/, size_t bytes) {
        Block& b = free_[fit_idx_];
        b.off += bytes; b.size -= bytes;
        if (b.size == 0) free_.erase(free_.begin() + fit_idx_);
    }

    // Insert (chunk, off, size) keeping free_ sorted by (chunk, off); coalesce
    // adjacent blocks within the same chunk.
    void insert_free(int chunk, size_t off, size_t size) {
        int i = 0;
        while (i < (int)free_.size() &&
               (free_[i].chunk < chunk || (free_[i].chunk == chunk && free_[i].off < off)))
            ++i;
        free_.insert(free_.begin() + i, {chunk, off, size});
        // coalesce with next
        if (i + 1 < (int)free_.size() && free_[i + 1].chunk == chunk &&
            free_[i].off + free_[i].size == free_[i + 1].off) {
            free_[i].size += free_[i + 1].size;
            free_.erase(free_.begin() + i + 1);
        }
        // coalesce with prev
        if (i > 0 && free_[i - 1].chunk == chunk &&
            free_[i - 1].off + free_[i - 1].size == free_[i].off) {
            free_[i - 1].size += free_[i].size;
            free_.erase(free_.begin() + i);
        }
    }

    int                          gpu_;
    std::vector<GpuBuffer<uint8_t>> chunks_;
    std::vector<Block>           free_;
    size_t                       capacity_  = 0;
    size_t                       live_      = 0;
    size_t                       peak_live_ = 0;
    int                          fit_idx_   = -1;
};

template <class T>
inline void Alloc<T>::reset() {
    if (arena_) arena_->free_block(chunk_, off_, bytes_, ptr_);
    arena_ = nullptr; ptr_ = nullptr; count_ = 0; bytes_ = 0; chunk_ = -1;
}

// Per-GPU arena singletons.  Intentionally leaked (like the old pools): the
// backing device chunks must NOT be freed during static destruction, after the
// SYCL queues are already gone.  The OS reclaims at process exit.
inline Arena& arena(int gpu) {
    static std::vector<Arena>* pool = [] {
        auto* v = new std::vector<Arena>();
        v->reserve(GpuEngine::count());
        for (int i = 0; i < GpuEngine::count(); ++i) v->emplace_back(i);
        return v;
    }();
    int n = (int)pool->size();
    return (*pool)[(gpu >= 0 && gpu < n) ? gpu : 0];
}

inline void reserve(int gpu, size_t bytes) { arena(gpu).reserve(bytes); }

// Device memory held by the arenas across all GPUs (high-water capacity).
inline size_t total_bytes() {
    size_t b = 0;
    for (int i = 0; i < GpuEngine::count(); ++i) b += arena(i).capacity();
    return b;
}

// Peak simultaneously-live bytes across all GPUs (what a perfect allocator would
// need; <= total_bytes()).
inline size_t peak_live_bytes() {
    size_t b = 0;
    for (int i = 0; i < GpuEngine::count(); ++i) b += arena(i).peak_live();
    return b;
}

}  // namespace diffarena

#pragma once
// Device-resident block-diffusion sampler + stopping kernels.
//
// These replace the per-step host round-trip in the denoising loop: instead of
// downloading argmax/entropy/denoiser every step, running the entropy-bound
// accept/renoise sampler and the stable-and-confident stop check on the CPU,
// and re-uploading the renoised canvas, the canvas / argmax / entropy / denoiser
// stay on the GPU and the sampler + stopping run as small kernels on the same
// in-order queue.  The only host sync per step is the 4-byte stop flag needed to
// break the loop (skipped entirely when DIFF_FORCE_DENOISE_STEPS is set).
//
// The device RNG is a deterministic counter-based hash; it does NOT match the
// host std::mt19937 path (kept under DIFF_HOST_SAMPLER) bit-for-bit.  Sampler
// RNG is not required to match HF exactly -- validation is at the per-step
// logits / argmax-canvas level (see notes/diffusion_gemma/...Architecture.md).
//
// All kernels target one canvas of `seq <= 256` positions (canvas_length = 256)
// and run as a single work-group of WG = 256 threads.
#include <cstdint>
#include <sycl/sycl.hpp>

namespace diffsamp {

inline constexpr int kMaxCanvas = 256;

// Counter-based device RNG: a well-mixed uint32 from (seed, a, b, c).
// Used for canvas init, per-step renoise, and sampling uniforms.
inline uint32_t rng_u32(uint64_t seed, uint32_t a, uint32_t b, uint32_t c) {
    uint64_t x = seed + 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t)a * 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t)b * 0xbf58476d1ce4e5b9ULL;
    x ^= (uint64_t)c * 0x94d049bb1335ebULL;
    // splitmix64 finalizer
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb1335ebULL;
    x ^= x >> 31;
    return (uint32_t)(x >> 32);
}

// Fill `canvas` (seq int32) with uniform-random tokens in [0, V).
// V must divide 2^32 for an unbiased % (true for V = 262144 = 2^18).
inline void init_canvas_random(sycl::queue& q, int32_t* canvas,
                               uint64_t seed, uint32_t block, int seq, int V) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(seq), [=](sycl::id<1> idx) {
            int i = (int)idx[0];
            canvas[i] = (int32_t)(rng_u32(seed, block, 0, (uint32_t)i) % (uint32_t)V);
        });
    });
}

// Fill `u` (seq float) with uniform [0,1) sampling variates for fused_logits_head.
inline void fill_uniform(sycl::queue& q, float* u,
                         uint64_t seed, uint32_t block, uint32_t step, int seq) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(seq), [=](sycl::id<1> idx) {
            int i = (int)idx[0];
            uint32_t r = rng_u32(seed, block, step, (uint32_t)i);
            u[i] = (float)(r >> 8) * (1.0f / 16777216.0f);   // [0,1)
        });
    });
}

// Entropy-bound accept + renoise, in place over the canvas.
//
// For each position i: sort-stable by (entropy, index); let prefix[i] be the
// sum of entropies of the positions that sort strictly before i.  Accept i iff
// prefix[i] <= entropy_bound (matches the host sampler's cumulative-bound rule).
// Accepted positions take the denoiser token; rejected positions are renoised
// with a fresh uniform-random token (staying on the uniform-noise manifold).
//
// O(seq^2) per position, one work-group of WG = kMaxCanvas; seq <= kMaxCanvas.
inline void entropy_bound_renoise(sycl::queue& q,
                                  int32_t* canvas, const int32_t* denoiser,
                                  const float* entropy, char* accepted,
                                  uint64_t seed, uint32_t block, uint32_t step,
                                  int seq, float entropy_bound, int V) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ent(sycl::range<1>(kMaxCanvas), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(kMaxCanvas), sycl::range<1>(kMaxCanvas)),
            [=](sycl::nd_item<1> it) {
                int i = (int)it.get_local_id(0);
                if (i < seq) ent[i] = entropy[i];
                it.barrier(sycl::access::fence_space::local_space);
                if (i < seq) {
                    float ei = ent[i];
                    float prefix = 0.0f;
                    for (int j = 0; j < seq; ++j) {
                        float ej = ent[j];
                        // strict sort key (entropy[j], j) < (entropy[i], i)
                        bool before = (ej < ei) || (ej == ei && j < i);
                        if (before) prefix += ej;
                    }
                    bool accept = (prefix <= entropy_bound);
                    int32_t tok = accept
                        ? denoiser[i]
                        : (int32_t)(rng_u32(seed, block, step, (uint32_t)i) % (uint32_t)V);
                    canvas[i] = tok;
                    accepted[i] = accept ? 1 : 0;
                }
            });
    });
}

// Stable-and-confident stopping, device-side.  Computes:
//   mean      = mean(entropy)
//   stable    = (stability_threshold == 0) || all `stability_threshold` history
//               slots equal the current argmax (i.e. stability_threshold+1
//               consecutive identical canvases -- matches the reference's
//               rolling-history criterion, not just the immediately-preceding
//               canvas)
//   confident = mean < confidence_threshold
//   stop      = stable && confident
// `history` is a (stability_threshold, seq) device buffer that the caller
// resets to a sentinel (any value no real token id can take, e.g. -1) at the
// start of each block; `slot` (0..stability_threshold-1, host-tracked and
// incremented mod stability_threshold every step) selects which history row
// this step's argmax rotates into.
// Writes mean -> *mean_out and stop -> *stop_out.
inline void stopping_check(sycl::queue& q,
                           const int32_t* argmax, int32_t* history,
                           const float* entropy,
                           float* mean_out, int32_t* stop_out,
                           float confidence_threshold, int stability_threshold,
                           int slot, int seq) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1>  sent(sycl::range<1>(kMaxCanvas), h);
        sycl::local_accessor<int32_t, 1> sarg(sycl::range<1>(kMaxCanvas), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(kMaxCanvas), sycl::range<1>(kMaxCanvas)),
            [=](sycl::nd_item<1> it) {
                int i = (int)it.get_local_id(0);
                if (i < seq) {
                    sent[i] = entropy[i];
                    sarg[i] = argmax[i];
                } else {
                    sent[i] = 0.0f; sarg[i] = 0;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (i == 0) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq; ++j) sum += sent[j];
                    float mean = sum / (float)seq;

                    bool stable = true;
                    for (int hstep = 0; hstep < stability_threshold && stable; ++hstep)
                        for (int j = 0; j < seq; ++j)
                            if (history[hstep * seq + j] != sarg[j]) { stable = false; break; }
                    bool confident = mean < confidence_threshold;
                    *mean_out = mean;
                    *stop_out = (stable && confident) ? 1 : 0;
                }
                // rotate this step's argmax into the history ring
                if (stability_threshold > 0 && i < seq)
                    history[slot * seq + i] = sarg[i];
            });
    });
}

// F5 + race fix.  Parallel mean (WG reduce) + parallel stability (each thread
// checks its canvas position across all history rows, AND-reduce) — replaces
// the original stopping_check's serial-on-thread-0 mean and stability loops.
//
// Correctness: the original stopping_check reads `history` (GLOBAL) in the
// i==0 block while every other thread concurrently rotates the CURRENT canvas
// into the SAME global `history` row (default stability_threshold=1 ->
// hstep==slot every step), with only a local fence between them -> RAW race
// that can make the stability check read the just-overwritten current canvas
// and trivially report "stable", biasing toward premature stop.  This version
// snapshots the history rows into SLM up front (global read, fenced + barrier-
// separated from the later global rotate-write) and runs the check from the
// SLM copy, eliminating the race.
// Enable: DIFF_STOP_FIX (default off -> original stopping_check).  Recommended
// for default-on after validating canvas-by-canvas stop parity vs DIFF_HOST_SAMPLER.
inline void stopping_check_fixed(sycl::queue& q,
                            const int32_t* argmax, int32_t* history,
                            const float* entropy,
                            float* mean_out, int32_t* stop_out,
                            float confidence_threshold, int stability_threshold,
                            int slot, int seq) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sent(sycl::range<1>(kMaxCanvas), h);
        sycl::local_accessor<int, 1>   sarg(sycl::range<1>(kMaxCanvas), h);
        int hist_rows = stability_threshold > 0 ? stability_threshold : 1;
        sycl::local_accessor<int, 1>   shist(sycl::range<1>((size_t)hist_rows * kMaxCanvas), h);
        sycl::local_accessor<float, 1> spart(sycl::range<1>(kMaxCanvas), h);
        sycl::local_accessor<int, 1>   sstop(sycl::range<1>(kMaxCanvas), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(kMaxCanvas), sycl::range<1>(kMaxCanvas)),
            [=](sycl::nd_item<1> it) {
                int i = (int)it.get_local_id(0);
                if (i < seq) { sent[i] = entropy[i]; sarg[i] = argmax[i]; }
                else { sent[i] = 0.0f; sarg[i] = 0; }
                // Snapshot history rows (GLOBAL read) before any rotate-write below.
                if (stability_threshold > 0)
                    for (int r = 0; r < stability_threshold; ++r)
                        if (i < seq) shist[(size_t)r * kMaxCanvas + i] = history[(size_t)r * seq + i];
                // global_and_local fence: orders these global reads before the
                // global rotate-write at the bottom (prevents the RAW race).
                it.barrier(sycl::access::fence_space::global_and_local);

                // Per-thread partial mean + per-position stable flag (from SLM snapshot).
                spart[i] = (i < seq) ? sent[i] : 0.0f;
                bool my_stable = true;
                if (stability_threshold > 0 && i < seq)
                    for (int r = 0; r < stability_threshold && my_stable; ++r)
                        if (shist[(size_t)r * kMaxCanvas + i] != sarg[i]) my_stable = false;
                sstop[i] = my_stable ? 1 : 0;
                it.barrier(sycl::access::fence_space::local_space);

                // One tree: sum spart (mean) AND sstop (stability).
                for (int o = kMaxCanvas / 2; o > 0; o >>= 1) {
                    if (i < o) {
                        spart[i] += spart[i + o];
                        if (sstop[i + o] == 0) sstop[i] = 0;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                if (i == 0) {
                    float mean = spart[0] / (float)seq;
                    bool stable = (stability_threshold == 0) || (sstop[0] != 0);
                    bool confident = mean < confidence_threshold;
                    *mean_out = mean;
                    *stop_out = (stable && confident) ? 1 : 0;
                }
                // Rotate current argmax into the history ring (GLOBAL write).  Safe:
                // the check read the SLM snapshot above; the global_and_local barrier
                // separates those reads from this write.
                if (stability_threshold > 0 && i < seq)
                    history[slot * seq + i] = sarg[i];
            });
    });
}

}  // namespace diffsamp

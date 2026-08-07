// Qwen3.5 DeltaNet conv-state layout contract.
//
// One conv-state buffer is written and read by two paths. The chunked path
// (qwen35_conv_causal + qwen35_update_conv_state) serves prefill and any batch
// wider than the fused decode window; the fused ESIMD decode path
// (qwen35_delta_decode_fused_esimd + qwen35_update_conv_state_time_major)
// serves single tokens. Prefill therefore always hands its state to the other
// path, and nothing converts between them, so they have to index it the same
// way.
//
// They did not. The chunked path used channel-major, state[channel*history +
// slot]; the fused path uses time-major, state[slot*channels + channel]. With
// conv_dim 10240 against history 3 those are unrelated addresses, so the first
// decode tokens after every prompt convolved over the wrong history and only
// recovered once `history` real tokens had shifted through. The failure is
// silent: text stays fluent, and the existing decode-fusion benchmark misses it
// because it gives each path its own state buffer and so never crosses the
// boundary.
//
// Needs a GPU; needs no model.

#include <cmath>
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/engine.hpp"
#include "modeling/qwen3_5/kernels.hpp"

using namespace qwen35_kernels;

namespace {

constexpr int kChannels = 64;
constexpr int kKernel = 4;
constexpr int kHistory = kKernel - 1;

std::vector<bf16> sample_vector(size_t n, uint32_t seed) {
    std::vector<bf16> out(n);
    uint32_t state = seed;
    for (size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        out[i] = float_to_bf16((static_cast<int>((state >> 16) & 2047u) - 1024) /
                               1024.0f);
    }
    return out;
}

std::vector<bf16> download(const GpuBuffer<bf16>& buffer) {
    std::vector<bf16> host(buffer.count());
    buffer.download(host.data(), host.size());
    return host;
}

// Reports one line per check rather than one per element: a layout mismatch
// differs in most of a 64-channel buffer, and thousands of identical failures
// bury the result.
struct Mismatch {
    int count = 0;
    size_t first = 0;
    float got = 0.0f, want = 0.0f;

    void observe(size_t index, bf16 a, bf16 b) {
        if (a == b) return;
        if (count == 0) {
            first = index;
            got = bf16_to_float(a);
            want = bf16_to_float(b);
        }
        ++count;
    }
    void report(const char* label, size_t total) const {
        if (count == 0) {
            std::printf("  %-28s ok (%zu values)\n", label, total);
            return;
        }
        std::fprintf(stderr,
                     "  %-28s %d/%zu differ; first at %zu: got %g want %g\n",
                     label, count, total, first, got, want);
    }
};

}  // namespace

int main() {
    std::printf("qwen3_5 conv state layout\n");
    sycl::queue& queue = GpuEngine::get(0).queue;
    const int tokens = 5;

    std::vector<bf16> host_input = sample_vector((size_t)tokens * kChannels, 0x51ce);
    GpuBuffer<bf16> input(host_input.size(), queue);
    input.upload(host_input.data(), host_input.size());

    // 1. The chunked writer must leave the state time-major. Written out as an
    //    explicit index rather than compared against another kernel, so the
    //    contract is readable from the test alone.
    {
        GpuBuffer<bf16> state((size_t)kHistory * kChannels, queue);
        state.zero();
        qwen35_update_conv_state(queue, input.data(), state.data(), tokens,
                                 kChannels, kKernel, /*had_state=*/false);
        queue.wait();
        std::vector<bf16> host_state = download(state);
        Mismatch m;
        for (int slot = 0; slot < kHistory; ++slot) {
            int source = tokens - kHistory + slot;
            for (int channel = 0; channel < kChannels; ++channel) {
                size_t at = (size_t)slot * kChannels + channel;
                m.observe(at, host_state[at],
                          host_input[(size_t)source * kChannels + channel]);
            }
        }
        m.report("writer is time-major", host_state.size());
        CHECK(m.count == 0);
    }

    // 2. Both writers must produce identical bytes from identical inputs. This
    //    is the check the layout split failed: each writer was self-consistent,
    //    so only comparing them against each other exposes the disagreement.
    {
        GpuBuffer<bf16> chunked((size_t)kHistory * kChannels, queue);
        GpuBuffer<bf16> fused((size_t)kHistory * kChannels, queue);
        std::vector<bf16> seed = sample_vector((size_t)kHistory * kChannels, 0xc0de);
        chunked.upload(seed.data(), seed.size());
        fused.upload(seed.data(), seed.size());

        // One token appended to an existing history, the prefill-to-decode step.
        const bf16* last = input.data() + (size_t)(tokens - 1) * kChannels;
        qwen35_update_conv_state(queue, last, chunked.data(), /*seq=*/1,
                                 kChannels, kKernel, /*had_state=*/true);
        qwen35_update_conv_state_time_major(queue, last, kChannels, fused.data(),
                                            kChannels);
        queue.wait();
        std::vector<bf16> host_chunked = download(chunked);
        std::vector<bf16> host_fused = download(fused);
        Mismatch m;
        for (size_t i = 0; i < host_chunked.size(); ++i)
            m.observe(i, host_chunked[i], host_fused[i]);
        m.report("writers agree", host_chunked.size());
        CHECK(m.count == 0);
    }

    // 3. Continuity across the boundary: convolving a whole sequence at once
    //    must equal convolving a prefix, carrying the state, and convolving the
    //    remainder. This is what a caller actually depends on, and it is the
    //    regression guard for the reader half of the fix.
    {
        std::vector<bf16> host_weight =
            sample_vector((size_t)kChannels * kKernel, 0x7a95);
        GpuBuffer<bf16> weight(host_weight.size(), queue);
        weight.upload(host_weight.data(), host_weight.size());

        GpuBuffer<bf16> whole((size_t)tokens * kChannels, queue);
        GpuBuffer<bf16> empty(kHistory * kChannels, queue);
        empty.zero();
        qwen35_conv_causal(queue, input.data(), weight.data(), empty.data(),
                           whole.data(), tokens, kChannels, kKernel,
                           /*has_state=*/false);

        const int split = tokens - 1;
        GpuBuffer<bf16> state((size_t)kHistory * kChannels, queue);
        GpuBuffer<bf16> prefix((size_t)split * kChannels, queue);
        GpuBuffer<bf16> tail(kChannels, queue);
        state.zero();
        qwen35_conv_causal(queue, input.data(), weight.data(), state.data(),
                           prefix.data(), split, kChannels, kKernel,
                           /*has_state=*/false);
        qwen35_update_conv_state(queue, input.data(), state.data(), split,
                                 kChannels, kKernel, /*had_state=*/false);
        qwen35_conv_causal(queue, input.data() + (size_t)split * kChannels,
                           weight.data(), state.data(), tail.data(), 1, kChannels,
                           kKernel, /*has_state=*/true);
        queue.wait();

        std::vector<bf16> host_whole = download(whole);
        std::vector<bf16> host_tail = download(tail);
        Mismatch m;
        for (int channel = 0; channel < kChannels; ++channel)
            m.observe(channel, host_tail[channel],
                      host_whole[(size_t)split * kChannels + channel]);
        m.report("prefill/decode continuity", kChannels);
        CHECK(m.count == 0);
    }

    RETURN_TESTS();
}

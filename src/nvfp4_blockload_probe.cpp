// src/nvfp4_blockload_probe.cpp
//
// Phase 0, Probe 2 — do the Intel block-2D-load primitives compile/link/run for
// intel_gpu_bmg_g31, and do they return the correct bytes of a known tensor?
//
// IMPORTANT (revised): the sycl-tla inline-asm path (lsc_load_block2d.ugm via
// `"=rw"`/`"rw.u"` constraints) does NOT compile under icpx 2025.3.3 — those
// constraint letters need a patched clang we lack. The SYCL-native substitute is
// ESIMD `sycl::ext::intel::experimental::esimd::lsc_load_2d`, which emits the
// same `lsc_load_block2d.ugm` VISA instruction. See memory file
// nvfp4_blockload_probe_findings.md for the full diagnosis.
//
// Two loads are exercised on the same 16x16 uint8 block (256 bytes, 64-B aligned):
//   nn  — normal 2D block load (Transformed=false).
//   nt  — VNNI 2D block load  (Transformed=true); packs 4 K per 32-bit word.
// One 16-wide subgroup issues the load; the HW replicates the full block to every
// thread (confirmed REPLICATED, not distributed). The host checks the multiset of
// loaded bytes == the known pattern and dumps lane-0's layout (+ the VNNI 32-bit
// packing) so the convention is documented.
//
// Build:
//   cmake --build build --target nvfp4_blockload_probe
// Run:
//   ZE_AFFINITY_MASK=0 ./build/nvfp4_blockload_probe

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include "common/gpu/engine.hpp"

namespace esimd_x = sycl::ext::intel::experimental::esimd;

static constexpr int W = 16;     // block width  (contiguous, bytes for uint8)
static constexpr int H = 16;     // block height (strided,  rows)
static constexpr int BLK = W * H; // W*H bytes

// One subgroup (16 lanes) loads the 16x16 block. ESIMD lsc_load_2d replicates the
// full block to every thread; we write lane 0's view to `out` (lanes 1..15 are
// identical).
template <bool Transformed>
static void run_load(sycl::queue& q, const uint8_t* in, uint8_t* out) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>{16}, sycl::range<1>{16}),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto v = esimd_x::lsc_load_2d<uint8_t, W, H,
                                               /*NBlocks*/1, /*Transposed*/false,
                                               /*Transformed*/Transformed>(
                    in, W - 1, H - 1, W - 1, 0, 0);
                constexpr int N = decltype(v)::length;
                if (it.get_local_id(0) == 0)
                    for (int j = 0; j < N; ++j) out[j] = v[j];
            });
    });
    q.wait();
}

int main() {
    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    std::string dev = q.get_device().get_info<sycl::info::device::name>();
    while (!dev.empty() && (unsigned char)dev.back() <= ' ') dev.pop_back();
    std::printf("[probe] device: %s | ESIMD lsc_load_2d block-load probe\n", dev.c_str());

    uint8_t host_in[BLK];
    for (int i = 0; i < BLK; ++i) host_in[i] = (uint8_t)i;

    uint8_t* in = (uint8_t*)sycl::aligned_alloc_device(64, BLK, q);   // 64-B aligned base
    uint8_t* out_nn = (uint8_t*)sycl::malloc_device(BLK, q);
    uint8_t* out_nt = (uint8_t*)sycl::malloc_device(BLK, q);
    if (!in || !out_nn || !out_nt) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    q.memcpy(in, host_in, BLK).wait();
    q.memset(out_nn, 0xAu, BLK).wait();
    q.memset(out_nt, 0xAu, BLK).wait();

    bool ok_nn = false, ok_nt = false;
    try { run_load</*Transformed*/false>(q, in, out_nn); ok_nn = true; }
    catch (const std::exception& e) { std::printf("[probe] nn load FAILED: %s\n", e.what()); }
    try { run_load</*Transformed*/true>(q, in, out_nt);  ok_nt = true; }
    catch (const std::exception& e) { std::printf("[probe] nt load FAILED: %s\n", e.what()); }
    uint8_t host_nn[BLK], host_nt[BLK];
    q.memcpy(host_nn, out_nn, BLK).wait();
    q.memcpy(host_nt, out_nt, BLK).wait();

    // Correctness = the loaded multiset equals the input multiset (the load reads
    // the right block; HW only redistributes/replicates it across lanes).
    auto multiset_ok = [&](const uint8_t* out) {
        uint8_t a[BLK], b[BLK];
        std::memcpy(a, out, BLK); std::memcpy(b, host_in, BLK);
        std::sort(a, a + BLK); std::sort(b, b + BLK);
        return std::memcmp(a, b, BLK) == 0;
    };
    bool nn_ok = ok_nn && multiset_ok(host_nn);
    bool nt_ok = ok_nt && multiset_ok(host_nt);

    std::printf("[probe] nn (normal)     multiset %s\n", nn_ok ? "PASS" : "FAIL");
    std::printf("[probe] nt (VNNI xform) multiset %s\n", nt_ok ? "PASS" : "FAIL");

    std::printf("[probe] nn lane-0 16x16 (row-major view):\n");
    for (int r = 0; r < H; ++r) {
        std::printf("  "); for (int c = 0; c < W; ++c) std::printf("%3u ", host_nn[r*W+c]);
        std::printf("\n");
    }
    std::printf("[probe] nt lane-0 16x16 (VNNI-packed view):\n");
    for (int r = 0; r < H; ++r) {
        std::printf("  "); for (int c = 0; c < W; ++c) std::printf("%3u ", host_nt[r*W+c]);
        std::printf("\n");
    }
    // VNNI packing: reinterpret lane-0's first 4 uint32 (4 bytes/word).
    std::printf("[probe] nt lane-0 as uint32 (first 8 words):\n  ");
    for (int w = 0; w < 8; ++w) {
        uint32_t v = (uint32_t)host_nt[w*4+0] |
                     ((uint32_t)host_nt[w*4+1] << 8) |
                     ((uint32_t)host_nt[w*4+2] << 16) |
                     ((uint32_t)host_nt[w*4+3] << 24);
        std::printf(" 0x%08x", v);
    }
    std::printf("\n");

    std::printf("[probe] SUMMARY: nn=%s nt=%s\n",
                nn_ok ? "PASS" : "FAIL", nt_ok ? "PASS" : "FAIL");
    std::printf("[probe] %s\n",
                nn_ok ? "Direction A (block-load) FEASIBLE via ESIMD lsc_load_2d — proceed to Phase 2."
                      : "Direction A load primitive NOT working. Inspect errors.");

    sycl::free(in, q); sycl::free(out_nn, q); sycl::free(out_nt, q);
    return 0;
}

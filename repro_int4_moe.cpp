// Synthetic repro for the qwen35 AWQ int4 grouped MoE dpas hang at pairs=64.
// Random small weights, same dims as the real model (H=2048, I=512, E=256,
// top_k=8, group=32). Exercises build_routes -> gateup_swiglu -> down ->
// combine at pairs=8 then pairs=64, mimicking the bench's cell sequence
// (including reallocation between cells) without the multi-minute model load.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>

#include "modeling/qwen3_5_moe/kernels/int4_grouped_moe.hpp"

int main(int argc, char** argv)
{
    const int H = 2048, I = 512, E = 256, TOPK = 8, G = 32;
    int max_S = argc > 1 ? atoi(argv[1]) : 8;
    bool distinct = argc > 2; // any 2nd arg: experts = p % E (no multi-pair)
    int max_pairs = max_S * TOPK;

    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property::queue::in_order{});

    // Per-expert weight tables (full-size allocations, memset pattern).
    std::vector<const uint8_t*> gu_w(E), dn_w(E);
    std::vector<const bf16*> gu_s(E), dn_s(E), gu_z(E), dn_z(E);
    size_t gu_bytes = (size_t)2 * I * (H / 2);
    size_t dn_bytes = (size_t)H * (I / 2);
    size_t gu_sc = (size_t)(H / G) * 2 * I;
    size_t dn_sc = (size_t)(I / G) * H;
    // Arena allocations: this driver/context allows only ~1024 device USM
    // allocation handles (probed: 1MB chunks fail at #1012), so per-expert
    // buffers are slices of one arena per tensor kind — matching how the
    // real loader stays under the limit.
    std::vector<bf16> scale_pat(gu_sc > dn_sc ? gu_sc : dn_sc);
    for (auto& v : scale_pat) v = float_to_bf16(0.01f);
    uint8_t* gu_w_arena = sycl::malloc_device<uint8_t>(E * gu_bytes, q);
    uint8_t* dn_w_arena = sycl::malloc_device<uint8_t>(E * dn_bytes, q);
    bf16* gu_s_arena = sycl::malloc_device<bf16>(E * gu_sc, q);
    bf16* dn_s_arena = sycl::malloc_device<bf16>(E * dn_sc, q);
    if (!gu_w_arena || !dn_w_arena || !gu_s_arena || !dn_s_arena) {
        printf("arena alloc failed\n");
        return 1;
    }
    q.memset(gu_w_arena, 0x11, E * gu_bytes);
    q.memset(dn_w_arena, 0x22, E * dn_bytes);
    q.memcpy(gu_s_arena, scale_pat.data(), gu_sc * sizeof(bf16));
    for (int e = 1; e < E; ++e)
        q.memcpy(gu_s_arena + e * gu_sc, gu_s_arena, gu_sc * sizeof(bf16));
    q.memcpy(dn_s_arena, scale_pat.data(), dn_sc * sizeof(bf16));
    for (int e = 1; e < E; ++e)
        q.memcpy(dn_s_arena + e * dn_sc, dn_s_arena, dn_sc * sizeof(bf16));
    for (int e = 0; e < E; ++e) {
        gu_w[e] = gu_w_arena + e * gu_bytes;
        dn_w[e] = dn_w_arena + e * dn_bytes;
        gu_s[e] = gu_s_arena + e * gu_sc;
        dn_s[e] = dn_s_arena + e * dn_sc;
        gu_z[e] = nullptr; dn_z[e] = nullptr; // symmetric (zp==8) path
    }
    auto d_gu_w = sycl::malloc_device<const uint8_t*>(E, q);
    auto d_dn_w = sycl::malloc_device<const uint8_t*>(E, q);
    auto d_gu_s = sycl::malloc_device<const bf16*>(E, q);
    auto d_dn_s = sycl::malloc_device<const bf16*>(E, q);
    auto d_gu_z = sycl::malloc_device<const bf16*>(E, q);
    auto d_dn_z = sycl::malloc_device<const bf16*>(E, q);
    q.memcpy(d_gu_w, gu_w.data(), E * sizeof(void*));
    q.memcpy(d_dn_w, dn_w.data(), E * sizeof(void*));
    q.memcpy(d_gu_s, gu_s.data(), E * sizeof(void*));
    q.memcpy(d_dn_s, dn_s.data(), E * sizeof(void*));
    q.memcpy(d_gu_z, gu_z.data(), E * sizeof(void*));
    q.memcpy(d_dn_z, dn_z.data(), E * sizeof(void*));

    bf16* hidden = sycl::malloc_device<bf16>((size_t)max_S * H, q);
    {
        std::vector<bf16> hh((size_t)max_S * H);
        srand(42);
        for (auto& v : hh) v = float_to_bf16(0.001f * (rand() % 100));
        q.memcpy(hidden, hh.data(), hh.size() * sizeof(bf16));
    }
    bf16* out = sycl::malloc_device<bf16>((size_t)max_S * H, q);

    for (int S = 1; S <= max_S; S *= 2) { // ramp pairs to find threshold
        int pairs = S * TOPK;
        std::vector<int> idx(pairs);
        std::vector<float> wgt(pairs);
        srand(S);
        for (int p = 0; p < pairs; ++p) {
            idx[p] = distinct ? (p % E) : (rand() % E);
            wgt[p] = 0.125f;
        }

        int* d_idx = sycl::malloc_device<int>(pairs, q);
        float* d_wgt = sycl::malloc_device<float>(pairs, q);
        int32_t* offsets = sycl::malloc_device<int32_t>(E + 1, q);
        int32_t* tokens = sycl::malloc_device<int32_t>(pairs, q);
        bf16* inter = sycl::malloc_device<bf16>((size_t)pairs * I, q);
        bf16* pout = sycl::malloc_device<bf16>((size_t)pairs * H, q);
        q.memcpy(d_idx, idx.data(), pairs * sizeof(int));
        q.memcpy(d_wgt, wgt.data(), pairs * sizeof(float));

        printf("[S=%d] build_routes...\n", S); fflush(stdout);
        qwen_int4_grouped_build_routes(q, d_idx, E, pairs, offsets, tokens);
        q.wait();
        std::vector<int32_t> off_h(E + 1);
        q.memcpy(off_h.data(), offsets, (E + 1) * sizeof(int32_t)).wait();
        int maxc = 0, active = 0;
        for (int e = 0; e < E; ++e) {
            int c = off_h[e + 1] - off_h[e];
            if (c > 0) ++active;
            if (c > maxc) maxc = c;
        }
        printf("[S=%d] routes total=%d active=%d max_per_expert=%d\n",
               S, off_h[E], active, maxc);
        fflush(stdout);

        printf("[S=%d] gateup...\n", S); fflush(stdout);
        qwen_int4_grouped_dpas_gateup_swiglu(
            q, hidden, H, d_gu_w, d_gu_s, d_gu_z, offsets, tokens, d_wgt,
            E, pairs, TOPK, I, G, inter);
        q.wait();
        printf("[S=%d] gateup done\n", S); fflush(stdout);

        printf("[S=%d] down...\n", S); fflush(stdout);
        qwen_int4_grouped_dpas_down(
            q, inter, I, d_dn_w, d_dn_s, d_dn_z, offsets, tokens,
            E, pairs, H, G, pout);
        q.wait();
        printf("[S=%d] down done\n", S); fflush(stdout);

        qwen_int4_grouped_combine(q, pout, S, TOPK, H, out);
        q.wait();
        std::vector<bf16> oh(8);
        q.memcpy(oh.data(), out, 8 * sizeof(bf16)).wait();
        printf("[S=%d] combine done, out[0..3]=%f %f %f %f\n", S,
               bf16_to_float(oh[0]), bf16_to_float(oh[1]),
               bf16_to_float(oh[2]), bf16_to_float(oh[3]));
        fflush(stdout);

        sycl::free(d_idx, q); sycl::free(d_wgt, q); sycl::free(offsets, q);
        sycl::free(tokens, q); sycl::free(inter, q); sycl::free(pout, q);
    }
    printf("ALL DONE\n");
    return 0;
}

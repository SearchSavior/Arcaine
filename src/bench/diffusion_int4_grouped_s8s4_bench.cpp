// Focused benchmark / numerical check for the ESIMD s8 x s4 grouped MoE
// kernels (DIFF_INT4_GROUPED_DPAS_MOE).  Builds synthetic AWQ-style weights
// (N-major packed s4 + per-group bf16 scales) and validates the grouped
// gate_up+GeGLU and down kernels against a host fp32 reference, then times
// them at the production DiffusionGemma MoE shapes.
//
// Registered as `diffusion-int4-grouped-s8s4` in arcaine_kbench.
//
// Run:
//   ./build/arcaine_kbench diffusion-int4-grouped-s8s4 \
//       --pairs 512 --experts 16 --hidden 2816 --inter 704 --iterations 50
#include "common/bench/registry.hpp"
#include "common/bench/util.hpp"

#include "common/gpu/buffer.hpp"
#include "common/gpu/int4_grouped_moe.hpp"
#include "common/gpu/q8_0.hpp"  // raw SPIR-V dpas intrinsic declaration

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using arcaine::bench::elapsed_ms;

namespace {

int s4v(int n, int k) { return ((n * 131 + k * 17) % 16) - 8; }

// Pack logical (N, K) s4 into N-major [N, K/2], low nibble = even k.
std::vector<uint8_t> pack_s4(int N, int K) {
    std::vector<uint8_t> p((size_t)N * K / 2, 0);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            uint8_t v = (uint8_t)(s4v(n, k) & 0xf);
            uint8_t& by = p[(size_t)n * (K / 2) + k / 2];
            if (k & 1) by |= (uint8_t)(v << 4); else by |= v;
        }
    return p;
}

std::vector<bf16> make_scales(int G, int N) {
    std::vector<bf16> s((size_t)G * N);
    for (int g = 0; g < G; ++g)
        for (int n = 0; n < N; ++n)
            s[(size_t)g * N + n] = float_to_bf16(0.004f + 0.0005f * ((g + n) % 5));
    return s;
}

struct Weights {
    std::vector<uint8_t> gu_w, dn_w;
    std::vector<bf16> gu_s, dn_s;
    int H = 0, inter = 0;
};

Weights make_weights(int H, int inter) {
    Weights w;
    w.H = H; w.inter = inter;
    w.gu_w = pack_s4(2 * inter, H);
    w.gu_s = make_scales(H / 32, 2 * inter);
    w.dn_w = pack_s4(H, inter);
    w.dn_s = make_scales(inter / 32, H);
    return w;
}

float s4_at(const std::vector<uint8_t>& p, int N, int K, int n, int k) {
    uint8_t by = p[(size_t)n * (K / 2) + k / 2];
    int v = (k & 1) ? (by >> 4) : (by & 0xf);
    return (float)(v >= 8 ? v - 16 : v);
}

// ---- correctness -----------------------------------------------------------
int correctness() {
    auto& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    const int H = 256, inter = 128, E = 5, pairs = 70, top_k = 1;
    Weights W = make_weights(H, inter);

    // Host activations (bf16) -> device; kernel quantizes internally.
    std::vector<bf16> A_h((size_t)pairs * H);
    for (size_t i = 0; i < A_h.size(); ++i)
        A_h[i] = float_to_bf16(0.05f * (float)((i * 17 + 11) % 41 - 20));

    bf16* dA = sycl::malloc_device<bf16>((size_t)pairs * H, q);
    int8_t* dAq = sycl::malloc_device<int8_t>((size_t)pairs * H, q);
    float* dAs = sycl::malloc_device<float>(pairs, q);
    q.memcpy(dA, A_h.data(), A_h.size() * sizeof(bf16)).wait();
    quantize_bf16_rows_to_s8(q, dA, pairs, H, dAq, dAs);
    q.wait();

    // Routes: expert e = p % E, token = p (top_k = 1).
    std::vector<int32_t> offs(E + 1, 0), tokens(pairs);
    for (int p = 0; p < pairs; ++p) { tokens[p] = p; }
    for (int p = 0; p < pairs; ++p) offs[1 + (p % E)]++;
    for (int e = 0; e < E; ++e) offs[e + 1] += offs[e];

    // Device pointer tables (all experts share the same synthetic weights).
    uint8_t *dgu_raw = sycl::malloc_device<uint8_t>(W.gu_w.size(), q);
    bf16 *dgs = sycl::malloc_device<bf16>(W.gu_s.size(), q);
    uint8_t *ddn_raw = sycl::malloc_device<uint8_t>(W.dn_w.size(), q);
    bf16 *dds = sycl::malloc_device<bf16>(W.dn_s.size(), q);
    q.memcpy(dgu_raw, W.gu_w.data(), W.gu_w.size()).wait();
    q.memcpy(dgs, W.gu_s.data(), W.gu_s.size() * sizeof(bf16)).wait();
    q.memcpy(ddn_raw, W.dn_w.data(), W.dn_w.size()).wait();
    q.memcpy(dds, W.dn_s.data(), W.dn_s.size() * sizeof(bf16)).wait();
    // One-time repack into DPAS B fragment order.
    uint8_t* dgu = sycl::malloc_device<uint8_t>(W.gu_w.size(), q);
    uint8_t* ddn = sycl::malloc_device<uint8_t>(W.dn_w.size(), q);
    repack_s4_to_dpas_fragments(q, dgu_raw, 2 * inter, H, dgu);
    repack_s4_to_dpas_fragments(q, ddn_raw, H, inter, ddn);
    q.wait();
    const uint8_t* hgu[E]; const bf16* hgs[E];
    const uint8_t* hdn[E]; const bf16* hds[E];
    for (int e = 0; e < E; ++e) { hgu[e] = dgu; hgs[e] = dgs; hdn[e] = ddn; hds[e] = dds; }
    auto* gu_w_dev = sycl::malloc_device<const uint8_t*>(E, q);
    auto* gu_s_dev = sycl::malloc_device<const bf16*>(E, q);
    auto* dn_w_dev = sycl::malloc_device<const uint8_t*>(E, q);
    auto* dn_s_dev = sycl::malloc_device<const bf16*>(E, q);
    q.memcpy(gu_w_dev, hgu, E * sizeof(void*)).wait();
    q.memcpy(gu_s_dev, hgs, E * sizeof(void*)).wait();
    q.memcpy(dn_w_dev, hdn, E * sizeof(void*)).wait();
    q.memcpy(dn_s_dev, hds, E * sizeof(void*)).wait();
    int32_t* offs_dev = sycl::malloc_device<int32_t>(E + 1, q);
    int32_t* tok_dev = sycl::malloc_device<int32_t>(pairs, q);
    q.memcpy(offs_dev, offs.data(), (E + 1) * sizeof(int32_t)).wait();
    q.memcpy(tok_dev, tokens.data(), pairs * sizeof(int32_t)).wait();

    bf16* act = sycl::malloc_device<bf16>((size_t)pairs * inter, q);
    bf16* ye  = sycl::malloc_device<bf16>((size_t)pairs * H, q);
    matmul_int4_grouped_s8s4_gateup_geglu(q, dAq, dAs, H,
        gu_w_dev, gu_s_dev, offs_dev, tok_dev, E, pairs, top_k, inter, act);
    int8_t* dIq = sycl::malloc_device<int8_t>((size_t)pairs * inter, q);
    float* dIs = sycl::malloc_device<float>(pairs, q);
    quantize_bf16_rows_to_s8(q, act, pairs, inter, dIq, dIs);
    matmul_int4_grouped_s8s4_down(q, dIq, dIs, inter,
        dn_w_dev, dn_s_dev, offs_dev, tok_dev, E, pairs, H, ye);
    q.wait();

    std::vector<bf16> act_h((size_t)pairs * inter), ye_h((size_t)pairs * H);
    q.memcpy(act_h.data(), act, act_h.size() * sizeof(bf16)).wait();
    q.memcpy(ye_h.data(), ye, ye_h.size() * sizeof(bf16)).wait();

    // Host reference: quantized A -> dequant, dequant s4 weights, fp32.
    std::vector<float> aq_h((size_t)pairs * H), as_h(pairs);
    std::vector<int8_t> aqi_h((size_t)pairs * H);
    q.memcpy(aqi_h.data(), dAq, aqi_h.size()).wait();
    q.memcpy(as_h.data(), dAs, pairs * sizeof(float)).wait();
    for (size_t i = 0; i < aq_h.size(); ++i)
        aq_h[i] = (float)aqi_h[i] * as_h[i / H];

    double num = 0, den = 0; float maxd = 0;
    for (int m = 0; m < pairs; ++m)
        for (int n = 0; n < inter; ++n) {
            float cg = 0, cu = 0;
            for (int k = 0; k < H; ++k) {
                float a = aq_h[(size_t)m * H + k];
                cg += a * (s4_at(W.gu_w, 2 * inter, H, n, k) *
                           bf16_to_float(W.gu_s[(size_t)(k / 32) * 2 * inter + n]));
                cu += a * (s4_at(W.gu_w, 2 * inter, H, inter + n, k) *
                           bf16_to_float(W.gu_s[(size_t)(k / 32) * 2 * inter + inter + n]));
            }
            float t = std::tanh(0.7978845608028654f * (cg + 0.044715f * cg * cg * cg));
            float ref = 0.5f * cg * (1.0f + t) * cu;
            float d = bf16_to_float(act_h[(size_t)m * inter + n]) - ref;
            maxd = std::max(maxd, std::fabs(d)); num += (double)d * d; den += (double)ref * ref;
        }
    std::printf("[check] gateup+geglu  max_abs %.5g  rel_l2 %.5g\n",
                maxd, den > 0 ? std::sqrt(num / den) : 0.0);

    // Down reference using the kernel's own quantized intermediate.
    std::vector<int8_t> iqi_h((size_t)pairs * inter); std::vector<float> is_h(pairs);
    q.memcpy(iqi_h.data(), dIq, iqi_h.size()).wait();
    q.memcpy(is_h.data(), dIs, pairs * sizeof(float)).wait();
    double num2 = 0, den2 = 0; float maxd2 = 0;
    for (int m = 0; m < pairs; ++m)
        for (int d = 0; d < H; ++d) {
            float ref = 0;
            for (int k = 0; k < inter; ++k) {
                float a = (float)iqi_h[(size_t)m * inter + k] * is_h[m];
                ref += a * (s4_at(W.dn_w, H, inter, d, k) *
                            bf16_to_float(W.dn_s[(size_t)(k / 32) * H + d]));
            }
            float dd = bf16_to_float(ye_h[(size_t)m * H + d]) - ref;
            maxd2 = std::max(maxd2, std::fabs(dd)); num2 += (double)dd * dd; den2 += (double)ref * ref;
        }
    std::printf("[check] down         max_abs %.5g  rel_l2 %.5g\n",
                maxd2, den2 > 0 ? std::sqrt(num2 / den2) : 0.0);
    return (maxd < 0.1f && maxd2 < 0.1f) ? 0 : 1;
}

// ---- perf ------------------------------------------------------------------
void perf(int pairs, int E, int H, int inter, int iterations) {
    auto& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    Weights W = make_weights(H, inter);
    std::vector<int32_t> offs(E + 1), tokens(pairs);
    for (int p = 0; p < pairs; ++p) tokens[p] = p;
    for (int p = 0; p < pairs; ++p) offs[1 + (p % E)]++;
    for (int e = 0; e < E; ++e) offs[e + 1] += offs[e];

    int8_t* Aq = sycl::malloc_device<int8_t>((size_t)pairs * H, q);
    float* As = sycl::malloc_device<float>(pairs, q);
    int8_t* Iq = sycl::malloc_device<int8_t>((size_t)pairs * inter, q);
    float* Is = sycl::malloc_device<float>(pairs, q);
    uint8_t* dgu = sycl::malloc_device<uint8_t>(W.gu_w.size(), q);
    bf16* dgs = sycl::malloc_device<bf16>(W.gu_s.size(), q);
    uint8_t* ddn = sycl::malloc_device<uint8_t>(W.dn_w.size(), q);
    bf16* dds = sycl::malloc_device<bf16>(W.dn_s.size(), q);
    bf16* act = sycl::malloc_device<bf16>((size_t)pairs * inter, q);
    bf16* ye = sycl::malloc_device<bf16>((size_t)pairs * H, q);
    q.memset(Aq, 3, (size_t)pairs * H); q.memset(Iq, 3, (size_t)pairs * inter);
    { std::vector<float> sc(pairs, 0.01f);
      q.memcpy(As, sc.data(), pairs * sizeof(float));
      q.memcpy(Is, sc.data(), pairs * sizeof(float)); }
    uint8_t* gu_raw = sycl::malloc_device<uint8_t>(W.gu_w.size(), q);
    uint8_t* dn_raw = sycl::malloc_device<uint8_t>(W.dn_w.size(), q);
    q.memcpy(gu_raw, W.gu_w.data(), W.gu_w.size());
    q.memcpy(dgs, W.gu_s.data(), W.gu_s.size() * sizeof(bf16));
    q.memcpy(dn_raw, W.dn_w.data(), W.dn_w.size());
    q.memcpy(dds, W.dn_s.data(), W.dn_s.size() * sizeof(bf16));
    repack_s4_to_dpas_fragments(q, gu_raw, 2 * inter, H, dgu);
    repack_s4_to_dpas_fragments(q, dn_raw, H, inter, ddn);
    int32_t* offs_dev = sycl::malloc_device<int32_t>(E + 1, q);
    int32_t* tok_dev = sycl::malloc_device<int32_t>(pairs, q);
    q.memcpy(offs_dev, offs.data(), (E + 1) * sizeof(int32_t));
    q.memcpy(tok_dev, tokens.data(), pairs * sizeof(int32_t));
    const uint8_t* gu_w[E]; const bf16* gu_s[E]; const uint8_t* dn_w[E]; const bf16* dn_s[E];
    for (int e = 0; e < E; ++e) { gu_w[e] = dgu; gu_s[e] = dgs; dn_w[e] = ddn; dn_s[e] = dds; }
    auto* gu_w_dev = sycl::malloc_device<const uint8_t*>(E, q);
    auto* gu_s_dev = sycl::malloc_device<const bf16*>(E, q);
    auto* dn_w_dev = sycl::malloc_device<const uint8_t*>(E, q);
    auto* dn_s_dev = sycl::malloc_device<const bf16*>(E, q);
    q.memcpy(gu_w_dev, gu_w, E * sizeof(void*));
    q.memcpy(gu_s_dev, gu_s, E * sizeof(void*));
    q.memcpy(dn_w_dev, dn_w, E * sizeof(void*));
    q.memcpy(dn_s_dev, dn_s, E * sizeof(void*));
    q.wait();

    auto run_gu = [&] {
        matmul_int4_grouped_s8s4_gateup_geglu(q, Aq, As, H, gu_w_dev, gu_s_dev,
            offs_dev, tok_dev, E, pairs, 1, inter, act);
    };
    auto run_dn = [&] {
        matmul_int4_grouped_s8s4_down(q, Iq, Is, inter, dn_w_dev, dn_s_dev,
            offs_dev, tok_dev, E, pairs, H, ye);
    };
    for (int i = 0; i < 3; ++i) { run_gu(); run_dn(); }
    q.wait();
    double gu_ms = elapsed_ms(q, iterations, run_gu);
    double dn_ms = elapsed_ms(q, iterations, run_dn);
    double gu_flop = 2.0 * pairs * (2.0 * inter) * H;
    double dn_flop = 2.0 * pairs * H * inter;
    std::printf("[perf] E=%d pairs=%d H=%d inter=%d | gateup %.3f ms (%.1f TF) | "
                "down %.3f ms (%.1f TF)\n",
                E, pairs, H, inter, gu_ms, gu_flop / (gu_ms * 1e9),
                dn_ms, dn_flop / (dn_ms * 1e9));
}

// ESIMD dpas throughput ceiling: independent accumulator chains over register
// constants, so the only work is the dpas itself (no memory traffic).  Tells
// us how much of the grouped-kernel gap is inherent to the ESIMD dpas path on
// this device vs. operand delivery.
void dpas_peak(int subgroups, int iters, int iterations) {
    using namespace igs8;
    auto& q = GpuEngine::get(0).queue;
    auto run = [&] {
        q.parallel_for(
            sycl::nd_range<1>((size_t)subgroups * kES, kES),
            [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                esimd::simd<uint8_t, kBB> b = esimd::simd<uint8_t, kBB>(0x31);
                esimd::simd<int8_t, kAF> a = esimd::simd<int8_t, kAF>(3);
                esimd::simd<int, kRES> c0 = 0, c1 = 0, c2 = 0, c3 = 0,
                    c4 = 0, c5 = 0, c6 = 0, c7 = 0;
                for (int i = 0; i < iters; ++i) {
                    c0 = xmx::dpas<kSD, kRC, int, int, uint8_t, int8_t,
                                   dpas_t::s4, dpas_t::s8, kRES, kBB, kAF>(c0, b, a);
                    c1 = xmx::dpas<kSD, kRC, int, int, uint8_t, int8_t,
                                   dpas_t::s4, dpas_t::s8, kRES, kBB, kAF>(c1, b, a);
                    c2 = xmx::dpas<kSD, kRC, int, int, uint8_t, int8_t,
                                   dpas_t::s4, dpas_t::s8, kRES, kBB, kAF>(c2, b, a);
                    c3 = xmx::dpas<kSD, kRC, int, int, uint8_t, int8_t,
                                   dpas_t::s4, dpas_t::s8, kRES, kBB, kAF>(c3, b, a);
                    c4 = xmx::dpas<kSD, kRC, int, int, uint8_t, int8_t,
                                   dpas_t::s4, dpas_t::s8, kRES, kBB, kAF>(c4, b, a);
                    c5 = xmx::dpas<kSD, kRC, int, int, uint8_t, int8_t,
                                   dpas_t::s4, dpas_t::s8, kRES, kBB, kAF>(c5, b, a);
                    c6 = xmx::dpas<kSD, kRC, int, int, uint8_t, int8_t,
                                   dpas_t::s4, dpas_t::s8, kRES, kBB, kAF>(c6, b, a);
                    c7 = xmx::dpas<kSD, kRC, int, int, uint8_t, int8_t,
                                   dpas_t::s4, dpas_t::s8, kRES, kBB, kAF>(c7, b, a);
                }
                if (c0[0]+c1[0]+c2[0]+c3[0]+c4[0]+c5[0]+c6[0]+c7[0] == 123456789) sycl::ext::oneapi::experimental::printf("x");
            });
    };
    for (int i = 0; i < 2; ++i) run();
    q.wait();
    double ms = elapsed_ms(q, iterations, run);
    double dpas_n = (double)subgroups * iters * 8;
    double macs = dpas_n * (double)kRC * kES * kKT;
    std::printf("[dpas] s8xs4 subgroups=%d iters=%d | %.3f ms | %.1f G dpas/s | %.1f TF\n",
                subgroups, iters, ms, dpas_n / (ms * 1e6), 2.0 * macs / (ms * 1e9));
}

// bf16 ESIMD dpas ceiling (the path the tree's existing kernels use), to tell
// whether the low s8xs4 number is the mixed-int path or ESIMD dpas in general.
void dpas_peak_bf16(int subgroups, int iters, int iterations) {
    namespace esimd = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    using dpas_t2 = xmx::dpas_argument_type;
    using bf = sycl::ext::oneapi::bfloat16;
    constexpr int SD = 8, RC = 8, ES = 16;
    constexpr int KT = 16, AF = RC * KT, BN = KT * ES, RES = RC * ES;
    auto& q = GpuEngine::get(0).queue;
    auto run = [&] {
        q.parallel_for(
            sycl::nd_range<1>((size_t)subgroups * ES, ES),
            [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                esimd::simd<bf, BN> b = esimd::simd<bf, BN>((bf)1.0f);
                esimd::simd<bf, AF> a = esimd::simd<bf, AF>((bf)1.0f);
                esimd::simd<float, RES> c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                for (int i = 0; i < iters; ++i) {
                    c0 = xmx::dpas<SD, RC, float, float, bf, bf,
                                   dpas_t2::bf16, dpas_t2::bf16, RES, BN, AF>(c0, b, a);
                    c1 = xmx::dpas<SD, RC, float, float, bf, bf,
                                   dpas_t2::bf16, dpas_t2::bf16, RES, BN, AF>(c1, b, a);
                    c2 = xmx::dpas<SD, RC, float, float, bf, bf,
                                   dpas_t2::bf16, dpas_t2::bf16, RES, BN, AF>(c2, b, a);
                    c3 = xmx::dpas<SD, RC, float, float, bf, bf,
                                   dpas_t2::bf16, dpas_t2::bf16, RES, BN, AF>(c3, b, a);
                }
                if (c0[0]+c1[0]+c2[0]+c3[0] == 123456789.0f) sycl::ext::oneapi::experimental::printf("x");
            });
    };
    for (int i = 0; i < 2; ++i) run();
    q.wait();
    double ms = elapsed_ms(q, iterations, run);
    double dpas_n = (double)subgroups * iters * 4;
    double macs = dpas_n * (double)RC * ES * KT;
    std::printf("[dpas] bf16    subgroups=%d iters=%d | %.3f ms | %.1f G dpas/s | %.1f TF\n",
                subgroups, iters, ms, dpas_n / (ms * 1e6), 2.0 * macs / (ms * 1e9));
}

// Raw SPIR-V intrinsic (__spirv_SubgroupMatrixMultiplyAccumulateINTEL, 0x3000
// bf16) dpas ceiling -- the path q8_0.hpp / nvfp4.hpp use.  Compares against
// the ESIMD dpas ceiling to isolate ESIMD codegen from the whole IGC path.
void dpas_peak_raw(int subgroups, int iters, int iterations) {
    auto& q = GpuEngine::get(0).queue;
    using v8s = short __attribute__((ext_vector_type(8)));
    using v8i = int   __attribute__((ext_vector_type(8)));
    using v8f = float __attribute__((ext_vector_type(8)));
    auto run = [&] {
        q.parallel_for(
            sycl::nd_range<1>((size_t)subgroups * 16, 16),
            [=](sycl::nd_item<1>) [[sycl::reqd_sub_group_size(16)]] {
                v8s a = {0,0,0,0,0,0,0,0};
                v8i b = {0x3f803f80,0x3f803f80,0x3f803f80,0x3f803f80,
                         0x3f803f80,0x3f803f80,0x3f803f80,0x3f803f80};
                v8f c0 = {0,0,0,0,0,0,0,0}, c1 = c0, c2 = c0, c3 = c0;
                for (int i = 0; i < iters; ++i) {
                    c0 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c0, 0x3000);
                    c1 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c1, 0x3000);
                    c2 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c2, 0x3000);
                    c3 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c3, 0x3000);
                }
                if (c0[0]+c1[0]+c2[0]+c3[0] == 123456789.0f) sycl::ext::oneapi::experimental::printf("x");
            });
    };
    for (int i = 0; i < 2; ++i) run();
    q.wait();
    double ms = elapsed_ms(q, iterations, run);
    double dpas_n = (double)subgroups * iters * 4;
    double macs = dpas_n * 8.0 * 16.0 * 16.0;
    std::printf("[dpas] raw bf16 subgroups=%d iters=%d | %.3f ms | %.1f G dpas/s | %.1f TF\n",
                subgroups, iters, ms, dpas_n / (ms * 1e6), 2.0 * macs / (ms * 1e9));
}

int run(int argc, char** argv) {
    int pairs = 4096, E = 128, H = 2816, inter = 704, iterations = 50;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--pairs") pairs = std::atoi(next().c_str());
        else if (a == "--experts") E = std::atoi(next().c_str());
        else if (a == "--hidden") H = std::atoi(next().c_str());
        else if (a == "--inter") inter = std::atoi(next().c_str());
        else if (a == "--iterations") iterations = std::atoi(next().c_str());
        else if (a == "--checks-only") { return correctness(); }
        else if (a == "--perf-only") { perf(pairs, E, H, inter, iterations); return 0; }
        else if (a == "--dpas-peak") {
            int sg = pairs > 0 ? pairs : 2048, it = E > 0 ? E : 2000;
            dpas_peak(sg, it, iterations);
            return 0;
        }
        else if (a == "--dpas-peak-raw") {
            int sg = pairs > 0 ? pairs : 2048, it = E > 0 ? E : 2000;
            dpas_peak_raw(sg, it, iterations);
            return 0;
        }
        else if (a == "--dpas-peak-bf16") {
            int sg = pairs > 0 ? pairs : 2048, it = E > 0 ? E : 2000;
            dpas_peak_bf16(sg, it, iterations);
            return 0;
        }
        else if (a == "-h" || a == "--help") {
            std::printf("Usage: %s [--pairs N] [--experts N] [--hidden N] "
                        "[--inter N] [--iterations N] [--checks-only|--perf-only]\n", argv[0]);
            return 0;
        } else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (int rc = correctness()) return rc;
    perf(pairs, E, H, inter, iterations);
    return 0;
}

}  // namespace

REGISTER_BENCH("diffusion-int4-grouped-s8s4",
    "DiffusionGemma ESIMD s8xs4 grouped MoE gate_up+GeGLU / down kernels",
    run)

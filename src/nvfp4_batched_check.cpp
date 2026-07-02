// src/nvfp4_batched_check.cpp
//
// Phase 1 numerical probe for matmul_nvfp4_packed_batched.
//
// GOAL: isolate two distinct questions about the oneDNN batched f4_e2m1 path:
//   (Q1) What f4_e2m1 dequant table does the batched primitive actually use?
//        (The earlier "uniform 1/sqrt(2)" guess was inconclusive; the mean
//        ratio on uniform data came out ~1/16, i.e. uncorrelated, not 0.5.)
//   (Q2) Apart from any table difference, is the batched LAYOUT correct
//        (weights K-contiguous tag::acb, scales [B,G,N]/[B,M,G], per-group)?
//        I.e. does the batched output match a host reference that uses
//        oneDNN's *measured* f4 table + the project's e4m3 scale table?
//
// Approach:
//   1. measure_table(): ones matmul (A=W=code c) for each code c in 0..7.
//      C = K * onednn(c)^2  =>  onednn(c) = sqrt(C/K). Prints the table and
//      the ratio onednn(c)/project(c) per code.
//   2. test_dst_scale(): ones with dst_scale = s, checks whether oneDNN
//      multiplies or divides by dst_scale.
//   3. verify(): random f4 + (uniform|random) f8 scales, compares the batched
//      kernel against a host reference that uses the MEASURED oneDNN f4 table
//      (so a match => layout is correct, only the dequant table differs).
//      Reports max_abs and a robust median ratio.
//
// Build: cmake --build build --target nvfp4_batched_check
// Run:   ZE_AFFINITY_MASK=0 ./build/nvfp4_batched_check
//        (set DIFF_NVFP4_VERBOSE=1 to print the chosen oneDNN impl string)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "common/gpu/buffer.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/nvfp4.hpp"

using bf16 = uint16_t;

static float project_f4(uint8_t c) { return nvfp4_e2m1_fast(c); }
static float project_f8(uint8_t b) { return nvfp4_e4m3_to_float(b); }

// ---- byte/nibble helpers for the host reference (project layout) ----
// A {B,M,K} K-contiguous (tag::abc); A_scale {B,M,G} G-contiguous (tag::abc)
// W {B,K,N} K-contiguous (tag::acb); W_scale {B,G,N} N-contiguous (tag::abc)
static float a_dequant(const std::vector<uint8_t>& A, const std::vector<uint8_t>& As,
                       int B, int M, int K, int G,
                       const float f4tab[8], int b, int m, int k) {
    size_t elem = (size_t)b * M * K + (size_t)m * K + k;
    uint8_t nib = (A[elem / 2] >> ((elem % 2) * 4)) & 0xF;
    int g = k / 16;
    size_t sidx = (size_t)b * M * G + (size_t)m * G + g;
    float a = f4tab[nib & 7]; if (nib & 8) a = -a;
    return a * project_f8(As[sidx]);
}
static float w_dequant(const std::vector<uint8_t>& W, const std::vector<uint8_t>& Ws,
                       int B, int K, int N, int G,
                       const float f4tab[8], int b, int k, int n) {
    size_t elem = (size_t)b * K * N + (size_t)n * K + k;  // K-contiguous
    uint8_t nib = (W[elem / 2] >> ((elem % 2) * 4)) & 0xF;
    int g = k / 16;
    size_t sidx = (size_t)b * G * N + (size_t)g * N + n;  // {B,G,N} N-contiguous
    float w = f4tab[nib & 7]; if (nib & 8) w = -w;
    return w * project_f8(Ws[sidx]);
}

static double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Fill a packed-f4 buffer entirely with one code c (both nibbles = c).
static void fill_uniform_code(std::vector<uint8_t>& buf, uint8_t c) {
    for (auto& b : buf) b = (uint8_t)(c | (c << 4));
}
static void fill_uniform_scale(std::vector<uint8_t>& buf, uint8_t s) {
    for (auto& b : buf) b = s;
}

// Measure oneDNN's f4_e2m1 dequant table via a ones matmul per code.
static void measure_table(GpuEngine& ctx, int E, int M, int K, int N,
                          float onednn_tab[8]) {
    int G = K / 16;
    auto& q = ctx.queue;
    std::vector<uint8_t> hA((size_t)E * M * K / 2), hAs((size_t)E * M * G);
    std::vector<uint8_t> hW((size_t)E * K * N / 2), hWs((size_t)E * G * N);
    float dst_scale = 1.0f;

    GpuBuffer<uint8_t> Abuf((size_t)E * M * K / 2, q);
    GpuBuffer<uint8_t> Asbuf((size_t)E * M * G, q);
    GpuBuffer<uint8_t> Wbuf((size_t)E * K * N / 2, q);
    GpuBuffer<uint8_t> Wsbuf((size_t)E * G * N, q);
    GpuBuffer<float> Dsbuf(1, q);
    GpuBuffer<bf16> Cbuf((size_t)E * M * N, q);
    Dsbuf.upload(&dst_scale, 1);

    std::printf("[batched-check] measuring oneDNN f4_e2m1 table (ones matmul, K=%d):\n", K);
    std::printf("[batched-check]   code | project | onednn | onednn/project\n");
    for (int c = 0; c < 8; ++c) {
        fill_uniform_code(hA, (uint8_t)c);
        fill_uniform_code(hW, (uint8_t)c);
        fill_uniform_scale(hAs, 0x38);  // f8 = 1.0
        fill_uniform_scale(hWs, 0x38);
        Abuf.upload(hA.data(), hA.size());
        Asbuf.upload(hAs.data(), hAs.size());
        Wbuf.upload(hW.data(), hW.size());
        Wsbuf.upload(hWs.data(), hWs.size());

        matmul_nvfp4_packed_batched(Abuf.data(), Asbuf.data(), E, M, K,
                                    Wbuf.data(), Wsbuf.data(), Dsbuf.data(),
                                    N, Cbuf.data(), ctx);
        q.wait();
        bf16 tmp = 0; Cbuf.download(&tmp, 1);
        float val = bf16_to_float(tmp);
        // C = K * onednn(c)^2 * dst_scale(=1)  => onednn(c) = sqrt(val/K)
        float onednn = (K > 0 && val >= 0) ? std::sqrt(val / (float)K) : 0.0f;
        onednn_tab[c] = onednn;
        float proj = project_f4((uint8_t)c);
        float ratio = (proj != 0.0f) ? onednn / proj : 0.0f;
        std::printf("[batched-check]     %d  | %6.4f | %6.4f | %7.4f   (C0=%g)\n",
                    c, proj, onednn, ratio, (double)val);
    }
}

// Test whether oneDNN multiplies or divides by dst_scale.
static void test_dst_scale(GpuEngine& ctx, int E, int M, int K, int N) {
    int G = K / 16;
    auto& q = ctx.queue;
    std::vector<uint8_t> hA((size_t)E * M * K / 2), hAs((size_t)E * M * G);
    std::vector<uint8_t> hW((size_t)E * K * N / 2), hWs((size_t)E * G * N);
    fill_uniform_code(hA, 2);  // project f4 code 2 = 1.0
    fill_uniform_code(hW, 2);
    fill_uniform_scale(hAs, 0x38);
    fill_uniform_scale(hWs, 0x38);
    GpuBuffer<uint8_t> Abuf((size_t)E * M * K / 2, q); Abuf.upload(hA.data(), hA.size());
    GpuBuffer<uint8_t> Asbuf((size_t)E * M * G, q);   Asbuf.upload(hAs.data(), hAs.size());
    GpuBuffer<uint8_t> Wbuf((size_t)E * K * N / 2, q); Wbuf.upload(hW.data(), hW.size());
    GpuBuffer<uint8_t> Wsbuf((size_t)E * G * N, q);     Wsbuf.upload(hWs.data(), hWs.size());
    GpuBuffer<bf16> Cbuf((size_t)E * M * N, q);

    float base = 0.0f, scaled = 0.0f;
    {
        float ds = 1.0f; GpuBuffer<float> Ds(1, q); Ds.upload(&ds, 1);
        matmul_nvfp4_packed_batched(Abuf.data(), Asbuf.data(), E, M, K,
                                    Wbuf.data(), Wsbuf.data(), Ds.data(),
                                    N, Cbuf.data(), ctx);
        q.wait();
        bf16 tmp = 0; Cbuf.download(&tmp, 1);
        base = bf16_to_float(tmp);
    }
    {
        float ds = 4.0f; GpuBuffer<float> Ds(1, q); Ds.upload(&ds, 1);
        matmul_nvfp4_packed_batched(Abuf.data(), Asbuf.data(), E, M, K,
                                    Wbuf.data(), Wsbuf.data(), Ds.data(),
                                    N, Cbuf.data(), ctx);
        q.wait();
        bf16 tmp = 0; Cbuf.download(&tmp, 1);
        scaled = bf16_to_float(tmp);
    }
    float ratio = (base != 0.0f) ? scaled / base : 0.0f;
    std::printf("[batched-check] dst_scale semantics: base(ds=1)=%g  scaled(ds=4)=%g  ratio=%.4f\n",
                (double)base, (double)scaled, (double)ratio);
    std::printf("[batched-check]   => oneDNN %s by dst_scale (ratio==4 => multiply, ==0.25 => divide)\n",
                (std::fabs(ratio - 4.0f) < 0.1f) ? "MULTIPLIES"
                : (std::fabs(ratio - 0.25f) < 0.1f) ? "DIVIDES" : "?? (unexpected)");
}

// Verify the batched kernel vs a host reference using the given f4 table.
static void verify(GpuEngine& ctx, const char* name, int E, int M, int K, int N,
                   bool rand_scales, const float f4tab[8], unsigned seed) {
    int G = K / 16;
    auto& q = ctx.queue;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_int_distribution<int> sc(40, 110);

    std::vector<uint8_t> hA((size_t)E * M * K / 2), hAs((size_t)E * M * G);
    std::vector<uint8_t> hW((size_t)E * K * N / 2), hWs((size_t)E * G * N);
    for (auto& b : hA) b = (uint8_t)(nib(rng) | (nib(rng) << 4));
    for (auto& b : hW) b = (uint8_t)(nib(rng) | (nib(rng) << 4));
    if (rand_scales) {
        for (auto& b : hAs) b = (uint8_t)sc(rng);
        for (auto& b : hWs) b = (uint8_t)sc(rng);
    } else {
        for (auto& b : hAs) b = 0x38;
        for (auto& b : hWs) b = 0x38;
    }
    float dst_scale = 1.0f;

    GpuBuffer<uint8_t> A((size_t)E * M * K / 2, q);   A.upload(hA.data(), hA.size());
    GpuBuffer<uint8_t> As((size_t)E * M * G, q);      As.upload(hAs.data(), hAs.size());
    GpuBuffer<uint8_t> W((size_t)E * K * N / 2, q);   W.upload(hW.data(), hW.size());
    GpuBuffer<uint8_t> Ws((size_t)E * G * N, q);      Ws.upload(hWs.data(), hWs.size());
    GpuBuffer<float> Ds(1, q);                         Ds.upload(&dst_scale, 1);
    GpuBuffer<bf16> C((size_t)E * M * N, q);

    matmul_nvfp4_packed_batched(A.data(), As.data(), E, M, K,
                                W.data(), Ws.data(), Ds.data(),
                                N, C.data(), ctx);
    q.wait();
    std::vector<bf16> hC((size_t)E * M * N);
    C.download(hC.data(), hC.size());

    double max_abs = 0.0;
    std::vector<double> ratios;
    for (int b = 0; b < E; ++b)
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k)
                    acc += a_dequant(hA, hAs, E, M, K, G, f4tab, b, m, k) *
                           w_dequant(hW, hWs, E, K, N, G, f4tab, b, k, n);
                float ref = acc / dst_scale;
                float got = bf16_to_float(hC[(size_t)(b * M + m) * N + n]);
                double d = std::fabs((double)got - (double)ref);
                if (d > max_abs) max_abs = d;
                if (std::fabs((double)ref) > 1.0)
                    ratios.push_back((double)got / (double)ref);
            }
    double med = median(ratios);
    std::printf("[%s] %s: E=%d M=%d K=%d N=%d scales=%s  max_abs=%.6g  median_ratio=%.4f (n=%zu)\n",
                "batched-check", name, E, M, K, N,
                rand_scales ? "random" : "uniform(1.0)",
                max_abs, med, ratios.size());
}

// Validate the host reference against the known-good SINGLE (non-batched) oneDNN
// kernel matmul_nvfp4_packed on expert-0 data. If this matches, the host ref is
// correct and any mismatch in the batched kernel is the batched impl's fault.
static void validate_single(GpuEngine& ctx, int M, int K, int N,
                            const std::vector<uint8_t>& hA,  // [E,M,K/2]
                            const std::vector<uint8_t>& hAs, // [E,M,G]
                            const std::vector<uint8_t>& hW,  // [E,K*N/2]
                            const std::vector<uint8_t>& hWs, // [E,G,N]
                            const float f4tab[8]) {
    int G = K / 16;
    auto& q = ctx.queue;
    float dst_scale = 1.0f;

    // Expert-0 slices are contiguous prefixes of the full batched buffers.
    GpuBuffer<uint8_t> A0((size_t)M * K / 2, q);   A0.upload(hA.data(), (size_t)M * K / 2);
    GpuBuffer<uint8_t> As0((size_t)M * G, q);      As0.upload(hAs.data(), (size_t)M * G);
    GpuBuffer<uint8_t> W0((size_t)K * N / 2, q);   W0.upload(hW.data(), (size_t)K * N / 2);
    GpuBuffer<uint8_t> Ws0((size_t)G * N, q);      Ws0.upload(hWs.data(), (size_t)G * N);
    GpuBuffer<float> Ds(1, q);                     Ds.upload(&dst_scale, 1);
    GpuBuffer<bf16> C0((size_t)M * N, q);

    Nvfp4Linear Wlin;
    Wlin.in_features = K;
    Wlin.out_features = N;
    Wlin.input_global_scale = 1.0f;
    Wlin.weight_packed = std::move(W0);
    Wlin.weight_scale = std::move(Ws0);
    Wlin.dst_scale = std::move(Ds);

    matmul_nvfp4_packed(A0.data(), As0.data(), M, K, Wlin, C0.data(), ctx);
    q.wait();
    std::vector<bf16> hC((size_t)M * N);
    C0.download(hC.data(), hC.size());

    double max_abs = 0.0;
    std::vector<double> ratios;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += a_dequant(hA, hAs, /*B*/1, M, K, G, f4tab, 0, m, k) *
                       w_dequant(hW, hWs, /*B*/1, K, N, G, f4tab, 0, k, n);
            float ref = acc / dst_scale;
            float got = bf16_to_float(hC[(size_t)m * N + n]);
            double d = std::fabs((double)got - (double)ref);
            if (d > max_abs) max_abs = d;
            if (std::fabs((double)ref) > 1.0)
                ratios.push_back((double)got / (double)ref);
        }
    std::printf("[batched-check] host-ref validation vs SINGLE oneDNN kernel (expert0): "
                "max_abs=%.6g  median_ratio=%.4f (n=%zu)\n",
                max_abs, median(ratios), ratios.size());
}

// Controlled layout probe: A depends ONLY on m, W depends ONLY on n (both uniform
// across k and b). With uniform scales + dst_scale=1, the correct result is
//   C[b,m,n] = K * f4(m&7) * f4(n&7)
// independent of the K-reduction layout. By inspecting how C varies with m, n,
// and b we can tell which dimensions the oneDNN batched reference impl honors.
static void test_layout_probe(GpuEngine& ctx, int E, int M, int K, int N) {
    int G = K / 16;
    auto& q = ctx.queue;
    std::vector<uint8_t> hA((size_t)E * M * K / 2, 0), hAs((size_t)E * M * G, 0x38);
    std::vector<uint8_t> hW((size_t)E * K * N / 2, 0), hWs((size_t)E * G * N, 0x38);
    // A[b,m,k] = code(m&7): set every nibble in row m to code(m&7).
    for (int b = 0; b < E; ++b)
        for (int m = 0; m < M; ++m) {
            uint8_t c = (uint8_t)(m & 7);
            uint8_t byte = (uint8_t)(c | (c << 4));
            size_t base = (size_t)b * M * K / 2 + (size_t)m * K / 2;
            for (int kk = 0; kk < K / 2; ++kk) hA[base + kk] = byte;
        }
    // W[b,k,n] = code(n&7) in K-contiguous (acb) storage: element (b,k,n) at
    // offset b*K*N + n*K + k; set every nibble in column n to code(n&7).
    for (int b = 0; b < E; ++b)
        for (int n = 0; n < N; ++n) {
            uint8_t c = (uint8_t)(n & 7);
            uint8_t byte = (uint8_t)(c | (c << 4));
            size_t base = (size_t)b * K * N / 2 + (size_t)n * K / 2;
            for (int kk = 0; kk < K / 2; ++kk) hW[base + kk] = byte;
        }
    float dst_scale = 1.0f;
    GpuBuffer<uint8_t> A((size_t)E * M * K / 2, q);  A.upload(hA.data(), (size_t)E * M * K / 2);
    GpuBuffer<uint8_t> As((size_t)E * M * G, q);     As.upload(hAs.data(), (size_t)E * M * G);
    GpuBuffer<uint8_t> W((size_t)E * K * N / 2, q);  W.upload(hW.data(), (size_t)E * K * N / 2);
    GpuBuffer<uint8_t> Ws((size_t)E * G * N, q);     Ws.upload(hWs.data(), (size_t)E * G * N);
    GpuBuffer<float> Ds(1, q);                        Ds.upload(&dst_scale, 1);
    GpuBuffer<bf16> C((size_t)E * M * N, q);
    matmul_nvfp4_packed_batched(A.data(), As.data(), E, M, K,
                                W.data(), Ws.data(), Ds.data(), N, C.data(), ctx);
    q.wait();
    std::vector<bf16> hC((size_t)E * M * N);
    C.download(hC.data(), hC.size());
    auto val = [&](int b, int m, int n) {
        return bf16_to_float(hC[(size_t)(b * M + m) * N + n]);
    };
    auto expf = [&](int idx) { return project_f4((uint8_t)(idx & 7)); };

    std::printf("[batched-check] layout probe (A=m-only, W=n-only; expect C=K*f(m)*f(n), K=%d):\n", K);
    std::printf("[batched-check]   C[0,m,2] vs K*f(m):  ");
    for (int m = 0; m < std::min(M, 8); ++m)
        std::printf("m%d:%g(%.0f) ", m, (double)val(0, m, 2), (double)(K * expf(m)));
    std::printf("\n");
    std::printf("[batched-check]   C[0,2,n] vs K*f(n):  ");
    for (int n = 0; n < std::min(N, 8); ++n)
        std::printf("n%d:%g(%.0f) ", n, (double)val(0, 2, n), (double)(K * expf(n)));
    std::printf("\n");
    std::printf("[batched-check]   C[b,2,2] vs K:       ");
    for (int b = 0; b < std::min(E, 8); ++b)
        std::printf("b%d:%g(%.0f) ", b, (double)val(b, 2, 2), (double)K);
    std::printf("\n");
}

int main() {
    GpuEngine& ctx = GpuEngine::get(0);
    // Allow overriding dims to probe larger sizes (where oneDNN may pick the
    // tuned jit impl instead of ocl:ref:any).
    int E = 2, M = 8, K = 32, N = 32;
    if (const char* e = std::getenv("CK_E")) E = std::atoi(e);
    if (const char* m = std::getenv("CK_M")) M = std::atoi(m);
    if (const char* k = std::getenv("CK_K")) K = std::atoi(k);
    if (const char* n = std::getenv("CK_N")) N = std::atoi(n);
    if (K % 16 != 0) { std::fprintf(stderr, "K must be divisible by 16\n"); return 1; }

    std::printf("[batched-check] dims: E=%d M=%d K=%d N=%d\n", E, M, K, N);

    float onednn_tab[8] = {};
    measure_table(ctx, E, M, K, N, onednn_tab);
    test_dst_scale(ctx, E, M, K, N);

    // Build random data once, reuse for both single validation and batched verify.
    int G = K / 16;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> nibd(0, 15);
    std::uniform_int_distribution<int> scd(40, 110);
    std::vector<uint8_t> hA((size_t)E * M * K / 2), hAs((size_t)E * M * G);
    std::vector<uint8_t> hW((size_t)E * K * N / 2), hWs((size_t)E * G * N);
    for (auto& b : hA) b = (uint8_t)(nibd(rng) | (nibd(rng) << 4));
    for (auto& b : hW) b = (uint8_t)(nibd(rng) | (nibd(rng) << 4));
    for (auto& b : hAs) b = (uint8_t)scd(rng);
    for (auto& b : hWs) b = (uint8_t)scd(rng);

    std::printf("[batched-check] validate host ref (project f4 table) vs single kernel:\n");
    float proj_tab[8];
    for (int c = 0; c < 8; ++c) proj_tab[c] = project_f4((uint8_t)c);
    validate_single(ctx, M, K, N, hA, hAs, hW, hWs, proj_tab);
    test_layout_probe(ctx, E, M, K, N);

    std::printf("[batched-check] verify vs host-ref (project e4m3 scale, measured oneDNN f4 table):\n");
    verify(ctx, "A-uniform-scales", E, M, K, N, /*rand_scales*/ false, onednn_tab, 42);
    verify(ctx, "B-random-scales",  E, M, K, N, /*rand_scales*/ true,  onednn_tab, 42);
    return 0;
}

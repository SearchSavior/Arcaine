#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "config.hpp"
#include "kernels.hpp"
#include "model.hpp"
#include "prefill.hpp"
#include "sampling.hpp"
#include "weights.hpp"

#include "../../common/preprocess/chat_template.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("[test] %-42s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

sycl::queue make_queue() {
    return sycl::queue(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                       sycl::property::queue::in_order{});
}

template <typename T>
T* dev_alloc(sycl::queue& q, size_t n) {
    T* p = sycl::malloc_device<T>(n, q);
    if (!p) throw std::runtime_error("alloc");
    return p;
}

template <typename T>
void upload(sycl::queue& q, T* dst, const std::vector<T>& src) {
    q.memcpy(dst, src.data(), src.size() * sizeof(T)).wait();
}

// ---------------------------------------------------------------- kernel tests

void test_q4_gemv(sycl::queue& q) {
    using namespace flint;
    const int n = 32, k = 128;
    std::vector<uint8_t> packed(n * (k / 2));
    std::vector<bf16> scales(n * (k / 64)), biases(n * (k / 64));
    std::vector<bf16> x(k);
    std::vector<int> codes((size_t)n * k);
    uint32_t rng = 12345;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (uint8_t)(next() & 0xff);
    for (size_t e = 0; e < (size_t)n * k; ++e) codes[e] = (int)((next() >> 4) & 15);
    for (size_t i = 0; i < packed.size(); ++i) {
        const int base = (int)i * 2;
        packed[i] = (uint8_t)(codes[base] | (codes[base + 1] << 4));
    }
    for (auto& v : scales) v = bf16(0.01f + 0.001f * (next() % 100));
    for (auto& v : biases) v = bf16(-0.05f + 0.0005f * (next() % 100));
    for (auto& v : x) v = bf16(-1.f + 2.f * (next() % 1000) / 1000.f);

    QTensor t;
    t.w = dev_alloc<uint8_t>(q, packed.size());
    t.s = dev_alloc<bf16>(q, scales.size());
    t.b = dev_alloc<bf16>(q, biases.size());
    t.n = n; t.k = k;
    upload(q, const_cast<uint8_t*>(t.w), packed);
    upload(q, const_cast<bf16*>(t.s), scales);
    upload(q, const_cast<bf16*>(t.b), biases);
    bf16* xd = dev_alloc<bf16>(q, k);
    bf16* yd = dev_alloc<bf16>(q, n);
    upload(q, xd, x);
    kernels::q4_gemv(q, t, xd, yd);
    q.wait();
    std::vector<bf16> y(n);
    q.memcpy(y.data(), yd, n * sizeof(bf16)).wait();

    std::vector<float> ref(n, 0.f), got(n);
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < k; ++i) {
            const int g = i / 64;
            ref[r] += (float(scales[r * (k / 64) + g]) * codes[(size_t)r * k + i] +
                       float(biases[r * (k / 64) + g])) * float(x[i]);
        }
    for (int r = 0; r < n; ++r) got[r] = float(y[r]);
    check(max_abs_diff(ref, got) < 0.02f, "q4_gemv (n=32,k=128)");
    sycl::free(const_cast<uint8_t*>(t.w), q); sycl::free(const_cast<flint::bf16*>(t.s), q); sycl::free(const_cast<flint::bf16*>(t.b), q);
    sycl::free(xd, q); sycl::free(yd, q);
}

void test_gemm_int8(sycl::queue& q) {
    using namespace flint;
    const int M = 3, N = 16, K = 128;
    std::vector<uint8_t> packed((size_t)N * (K / 2));
    std::vector<bf16> scales((size_t)N * (K / 64)), biases((size_t)N * (K / 64));
    std::vector<bf16> A((size_t)M * K);
    std::vector<int> codes((size_t)N * K);
    uint32_t rng = 707;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (auto& c : codes) c = (int)((next() >> 4) & 15);
    for (size_t i = 0; i < packed.size(); ++i)
        packed[i] = (uint8_t)(codes[i * 2] | (codes[i * 2 + 1] << 4));
    for (auto& v : scales) v = bf16(0.01f + 0.001f * (next() % 100));
    for (auto& v : biases) v = bf16(-0.05f + 0.0005f * (next() % 100));
    for (auto& v : A) v = bf16(-1.f + 2.f * (next() % 1000) / 1000.f);

    QTensor t;
    t.w = dev_alloc<uint8_t>(q, packed.size());
    t.s = dev_alloc<bf16>(q, scales.size());
    t.b = dev_alloc<bf16>(q, biases.size());
    t.n = N; t.k = K;
    upload(q, const_cast<uint8_t*>(t.w), packed);
    upload(q, const_cast<bf16*>(t.s), scales);
    upload(q, const_cast<bf16*>(t.b), biases);
    bf16* ad = dev_alloc<bf16>(q, A.size());
    int8_t* aq = dev_alloc<int8_t>(q, A.size());
    float* xs = dev_alloc<float>(q, (size_t)M * (K / 64));
    bf16* cd = dev_alloc<bf16>(q, (size_t)M * N);
    upload(q, ad, A);
    flint::prefill::quantize_int8(q, ad, aq, xs, M, K);
    q.wait_and_throw();
    std::vector<int8_t> hq(A.size());
    std::vector<float> hs((size_t)M * (K / 64));
    q.memcpy(hq.data(), aq, hq.size()).wait();
    q.memcpy(hs.data(), xs, hs.size() * sizeof(float)).wait();
    flint::prefill::gemm_q4<true>(q, t, nullptr, aq, xs, nullptr, cd, M, N);
    q.wait_and_throw();
    std::vector<bf16> C((size_t)M * N);
    q.memcpy(C.data(), cd, C.size() * sizeof(bf16)).wait();
    float worst = 0.f;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = 0.f;
            for (int k = 0; k < K; ++k) {
                const int g = k / 64;
                ref += (float(scales[n * (K / 64) + g]) * codes[n * K + k] +
                        float(biases[n * (K / 64) + g])) *
                       hs[m * (K / 64) + g] * float(hq[m * K + k]);
            }
            worst = std::fmax(worst, std::fabs(ref - float(C[m * N + n])));
        }
    check(worst < 0.06f, "prefill gemm_q4 int8 (M=3,N=16,K=128)");
    sycl::free(const_cast<uint8_t*>(t.w), q);
    sycl::free(const_cast<bf16*>(t.s), q);
    sycl::free(const_cast<bf16*>(t.b), q);
    sycl::free(ad, q); sycl::free(aq, q); sycl::free(xs, q); sycl::free(cd, q);
}

void test_rmsnorm(sycl::queue& q) {
    using namespace flint;
    const int H = 2048;
    std::vector<bf16> x(H), w(H);
    uint32_t rng = 999;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (auto& v : x) v = bf16(-1.f + 2.f * (next() % 1000) / 1000.f);
    for (auto& v : w) v = bf16(0.5f + (next() % 1000) / 1000.f);
    bf16 *xd = dev_alloc<bf16>(q, H), *wd = dev_alloc<bf16>(q, H),
         *yd = dev_alloc<bf16>(q, H);
    upload(q, xd, x); upload(q, wd, w);
    kernels::rmsnorm(q, xd, wd, yd, H, 1e-6f);
    q.wait();
    std::vector<bf16> y(H);
    q.memcpy(y.data(), yd, H * sizeof(bf16)).wait();
    float ss = 0.f;
    for (auto v : x) ss += float(v) * float(v);
    const float inv = 1.f / std::sqrt(ss / H + 1e-6f);
    std::vector<float> ref(H), got(H);
    for (int i = 0; i < H; ++i)
        ref[i] = float(w[i]) * float(bf16(float(x[i]) * inv));
    for (int i = 0; i < H; ++i) got[i] = float(y[i]);
    check(max_abs_diff(ref, got) < 0.01f, "rmsnorm (H=2048)");
    sycl::free(xd, q); sycl::free(wd, q); sycl::free(yd, q);
}

void test_rmsnorm_batch(sycl::queue& q) {
    using namespace flint;
    const int M = 3, H = 2048;
    std::vector<bf16> x((size_t)M * H), w(H);
    uint32_t rng = 5150;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (auto& v : x) v = bf16(-1.f + 2.f * (next() % 1000) / 1000.f);
    for (auto& v : w) v = bf16(0.5f + (next() % 1000) / 1000.f);
    bf16* xd = dev_alloc<bf16>(q, x.size());
    bf16* wd = dev_alloc<bf16>(q, H);
    bf16* yd = dev_alloc<bf16>(q, x.size());
    upload(q, xd, x);
    upload(q, wd, w);
    flint::prefill::rmsnorm_batch(q, xd, wd, yd, M, H, 1e-6f);
    q.wait_and_throw();
    std::vector<bf16> y(x.size());
    q.memcpy(y.data(), yd, y.size() * sizeof(bf16)).wait();
    float worst = 0.f;
    for (int m = 0; m < M; ++m) {
        float ss = 0.f;
        for (int i = 0; i < H; ++i) ss += float(x[m * H + i]) * float(x[m * H + i]);
        const float inv = 1.f / std::sqrt(ss / H + 1e-6f);
        for (int i = 0; i < H; ++i) {
            const float ref = float(w[i]) * float(bf16(float(x[m * H + i]) * inv));
            worst = std::fmax(worst, std::fabs(ref - float(y[m * H + i])));
        }
    }
    check(worst < 0.01f, "rmsnorm_batch (M=3, H=2048)");
    sycl::free(xd, q); sycl::free(wd, q); sycl::free(yd, q);
}

void test_embed_batch(sycl::queue& q) {
    using namespace flint;
    const int M = 3, H = 2048, V = 8;
    std::vector<uint8_t> w((size_t)V * H);
    std::vector<bf16> sc((size_t)V * (H / 64)), bi((size_t)V * (H / 64));
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t)(i * 37 + 11);
    for (auto& v : sc) v = bf16(0.01f);
    for (auto& v : bi) v = bf16(-0.02f);
    std::vector<int> toks = {1, 2, 3};
    QTensor t;
    t.w = dev_alloc<uint8_t>(q, w.size());
    t.s = dev_alloc<bf16>(q, sc.size());
    t.b = dev_alloc<bf16>(q, bi.size());
    t.n = V; t.k = H;
    upload(q, const_cast<uint8_t*>(t.w), w);
    upload(q, const_cast<bf16*>(t.s), sc);
    upload(q, const_cast<bf16*>(t.b), bi);
    int* td = dev_alloc<int>(q, M);
    bf16* hd = dev_alloc<bf16>(q, (size_t)M * H);
    bf16* nd = dev_alloc<bf16>(q, (size_t)M * H);
    std::vector<bf16> nw(H, bf16(1.f));
    bf16* nwd = dev_alloc<bf16>(q, H);
    upload(q, td, toks);
    upload(q, nwd, nw);
    flint::prefill::embed_batch(q, t, td, hd, M);
    q.wait_and_throw();
    flint::prefill::rmsnorm_batch(q, hd, nwd, nd, M, H, 1e-6f);
    q.wait_and_throw();
    std::vector<bf16> out((size_t)M * H);
    q.memcpy(out.data(), nd, out.size() * sizeof(bf16)).wait();
    float worst = 0.f;
    for (int m = 0; m < M; ++m) {
        const int tok = toks[m];
        float ss = 0.f;
        for (int i = 0; i < H; ++i) {
            const float ref = float(sc[tok * (H / 64) + i / 64]) * w[tok * H + i] +
                              float(bi[tok * (H / 64) + i / 64]);
            ss += ref * ref;
        }
        const float inv = 1.f / std::sqrt(ss / H + 1e-6f);
        for (int i = 0; i < H; ++i) {
            const float ref = float(sc[tok * (H / 64) + i / 64]) * w[tok * H + i] +
                              float(bi[tok * (H / 64) + i / 64]);
            worst = std::fmax(worst, std::fabs(float(bf16(ref * inv)) -
                                               float(out[m * H + i])));
        }
    }
    check(worst < 0.02f, "embed_batch + rmsnorm_batch (M=3)");
    sycl::free(const_cast<uint8_t*>(t.w), q);
    sycl::free(const_cast<bf16*>(t.s), q);
    sycl::free(const_cast<bf16*>(t.b), q);
    sycl::free(td, q); sycl::free(hd, q); sycl::free(nd, q); sycl::free(nwd, q);
}

void test_attention(sycl::queue& q) {
    using namespace flint;
    const int heads = 32, kvh = 4, cap = 16, pos = 15;
    std::vector<bf16> qbuf((size_t)heads * 128), sink(heads);
    std::vector<bf16> kc((size_t)kvh * cap * 128), vc((size_t)kvh * cap * 128);
    uint32_t rng = 4242;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (auto& v : qbuf) v = bf16(-2.f + 4.f * (next() % 1000) / 1000.f);
    for (auto& v : sink) v = bf16(-1.f + 2.f * (next() % 1000) / 1000.f);
    for (auto& v : kc) v = bf16(-2.f + 4.f * (next() % 1000) / 1000.f);
    for (auto& v : vc) v = bf16(-2.f + 4.f * (next() % 1000) / 1000.f);
    bf16 *qd = dev_alloc<bf16>(q, qbuf.size()), *sd = dev_alloc<bf16>(q, heads);
    bf16 *kd = dev_alloc<bf16>(q, kc.size()), *vd = dev_alloc<bf16>(q, vc.size());
    bf16* od = dev_alloc<bf16>(q, qbuf.size());
    upload(q, qd, qbuf); upload(q, sd, sink); upload(q, kd, kc); upload(q, vd, vc);
    kernels::attention(q, qd, kd, vd, sd, od, heads, kvh, /*global=*/true, pos, cap);
    q.wait();
    std::vector<bf16> out(qbuf.size());
    q.memcpy(out.data(), od, out.size() * sizeof(bf16)).wait();

    const float scale = 0.08838834764831845f;
    float worst = 0.f;
    for (int h = 0; h < heads; ++h) {
        const int kv = h / (heads / kvh);
        for (int d = 0; d < 128; ++d) {
            double hi = float(sink[h]), z = 1.0, acc = 0.0;
            for (int t = 0; t <= pos; ++t) {
                double s = 0.0;
                for (int e = 0; e < 128; ++e)
                    s += double(float(qbuf[h * 128 + e])) * scale *
                         double(float(kc[((size_t)kv * cap + t) * 128 + e]));
                const double m = std::fmax(hi, s);
                const double oldw = std::exp(hi - m), neww = std::exp(s - m);
                z = z * oldw + neww;
                acc = acc * oldw + neww * double(float(vc[((size_t)kv * cap + t) * 128 + d]));
                hi = m;
            }
            worst = std::fmax(worst, (float)std::fabs(acc / z - double(out[h * 128 + d])));
        }
    }
    check(worst < 0.03f, "attention decode (32 heads, ctx=16, sinks)");
    sycl::free(qd, q); sycl::free(sd, q); sycl::free(kd, q);
    sycl::free(vd, q); sycl::free(od, q);
}

void test_gemm(sycl::queue& q) {
    using namespace flint;
    const int M = 3, N = 16, K = 128;
    std::vector<uint8_t> packed((size_t)N * (K / 2));
    std::vector<bf16> scales((size_t)N * (K / 64)), biases((size_t)N * (K / 64));
    std::vector<bf16> A((size_t)M * K);
    uint32_t rng = 2024;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    std::vector<int> codes((size_t)N * K);
    for (auto& c : codes) c = (int)((next() >> 4) & 15);
    for (size_t i = 0; i < packed.size(); ++i)
        packed[i] = (uint8_t)(codes[i * 2] | (codes[i * 2 + 1] << 4));
    for (auto& v : scales) v = bf16(0.01f + 0.001f * (next() % 100));
    for (auto& v : biases) v = bf16(-0.05f + 0.0005f * (next() % 100));
    for (auto& v : A) v = bf16(-1.f + 2.f * (next() % 1000) / 1000.f);

    QTensor t;
    t.w = dev_alloc<uint8_t>(q, packed.size());
    t.s = dev_alloc<bf16>(q, scales.size());
    t.b = dev_alloc<bf16>(q, biases.size());
    t.n = N; t.k = K;
    upload(q, const_cast<uint8_t*>(t.w), packed);
    upload(q, const_cast<bf16*>(t.s), scales);
    upload(q, const_cast<bf16*>(t.b), biases);
    bf16* ad = dev_alloc<bf16>(q, A.size());
    bf16* cd = dev_alloc<bf16>(q, (size_t)M * N);
    upload(q, ad, A);
    flint::prefill::gemm_q4<false>(q, t, ad, nullptr, nullptr, nullptr, cd, M, N);
    q.wait_and_throw();
    std::vector<bf16> C((size_t)M * N);
    q.memcpy(C.data(), cd, C.size() * sizeof(bf16)).wait();
    float worst = 0.f;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = 0.f;
            for (int k = 0; k < K; ++k)
                ref += (float(scales[n * (K / 64) + k / 64]) *
                            codes[n * K + k] +
                        float(biases[n * (K / 64) + k / 64])) *
                       float(A[m * K + k]);
            worst = std::fmax(worst, std::fabs(ref - float(C[m * N + n])));
        }
    check(worst < 0.05f, "prefill gemm_q4 (M=3,N=16,K=128)");
    sycl::free(const_cast<uint8_t*>(t.w), q);
    sycl::free(const_cast<bf16*>(t.s), q);
    sycl::free(const_cast<bf16*>(t.b), q);
    sycl::free(ad, q);
    sycl::free(cd, q);
}

void test_router(sycl::queue& q) {
    using namespace flint;
    const int experts = 256, H = 256, top_k = 6;
    std::vector<bf16> w((size_t)experts * H), x(H);
    uint32_t rng = 77;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (auto& v : w) v = bf16(-0.1f + 0.2f * (next() % 1000) / 1000.f);
    for (auto& v : x) v = bf16(-1.f + 2.f * (next() % 1000) / 1000.f);
    bf16* wd = dev_alloc<bf16>(q, w.size());
    bf16* xd = dev_alloc<bf16>(q, H);
    int* idd = dev_alloc<int>(q, top_k);
    float* swd = dev_alloc<float>(q, top_k);
    upload(q, wd, w); upload(q, xd, x);
    kernels::router_topk(q, wd, xd, experts, H, top_k, idd, swd);
    q.wait();
    std::vector<int> ids(top_k);
    std::vector<float> sw(top_k);
    q.memcpy(ids.data(), idd, top_k * sizeof(int)).wait();
    q.memcpy(sw.data(), swd, top_k * sizeof(float)).wait();

    std::vector<float> score(experts);
    for (int e = 0; e < experts; ++e) {
        float a = 0.f;
        for (int i = 0; i < H; ++i) a += float(w[(size_t)e * H + i]) * float(x[i]);
        a = float(bf16(a));
        score[e] = 1.f / (1.f + std::exp(-a));
    }
    std::vector<float> work = score;
    std::vector<int> rid(top_k);
    std::vector<float> rw(top_k);
    for (int r = 0; r < top_k; ++r) {
        float best = -3.402823466e+38f;
        int idx = 0;
        for (int i = 0; i < experts; ++i)
            if (work[i] > best || (work[i] == best && i > idx)) {
                best = work[i];
                idx = i;
            }
        rid[r] = idx;
        rw[r] = best;
        work[idx] = -3.402823466e+38f;
    }
    float sum = 1e-20f;
    for (float v : rw) sum += v;
    for (float& v : rw) v /= sum;
    bool ok = true;
    for (int r = 0; r < top_k; ++r)
        ok = ok && ids[r] == rid[r] && std::fabs(sw[r] - rw[r]) < 1e-5f;
    check(ok, "router top-6 (256 experts)");
    sycl::free(wd, q); sycl::free(xd, q); sycl::free(idd, q); sycl::free(swd, q);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "/workspace/models/flint-int4";
    std::string prompt = "The capital of France is";
    int max_new = 16;
    int ctx = 4096;
    bool greedy = false;
    bool tests = true;
    bool tests_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--prompt") prompt = next();
        else if (a == "--max-new") max_new = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--greedy") greedy = true;
        else if (a == "--no-tests") tests = false;
        else if (a == "--tests-only") { tests = true; tests_only = true; }
        else if (a.rfind("--", 0) != 0) model_dir = a;
    }

    auto q = make_queue();
    std::printf("[flint] device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    if (tests) {
        test_q4_gemv(q);
        test_rmsnorm(q);
        test_rmsnorm_batch(q);
        test_embed_batch(q);
        test_attention(q);
        test_router(q);
        test_gemm(q);
        test_gemm_int8(q);
        if (failures) {
            std::printf("[test] %d kernel test(s) failed\n", failures);
            return 1;
        }
        if (tests_only) return 0;
    }

    FlintModel model(model_dir, ctx);
    auto in = model.prepare_input(prompt, {}, {}, "");
    std::printf("[flint] prompt tokens: %zu\n", in.tokens.size());
    std::printf("[flint] prompt: %s\n", prompt.c_str());

    std::printf("[flint] decoded prompt tokens: %s\n",
                decode_tokens(model_dir, in.tokens).c_str());
    auto t0 = std::chrono::steady_clock::now();
    auto logits = model.forward({in.tokens, 0});
    auto t1 = std::chrono::steady_clock::now();
    std::printf("[flint] prefill: %.1f ms (%zu tokens)\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                in.tokens.size());

    uint64_t rng = 0x9e3779b97f4a7c15ull;
    std::vector<int> generated;
    int past = (int)in.tokens.size();
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new; ++i) {
        const int tok = flint_sample(logits, greedy ? 0.f : 1.0f, 20, 0.95f, rng);
        if (tok < 0) break;
        generated.push_back(tok);
        if (model.info().is_eos(tok)) break;
        std::vector<int> one{tok};
        logits = model.forward({one, past});
        ++past;
    }
    auto t3 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    if (const char* trace = std::getenv("ARCAINE_FLINT_TRACE")) model.dump_trace(trace);
    std::printf("[flint] decode: %d tokens in %.1f ms (%.2f ms/token)\n",
                (int)generated.size(), ms, generated.empty() ? 0.0 : ms / generated.size());

    std::printf("[flint] tokens:");
    for (int t : generated) std::printf(" %d", t);
    std::printf("\n");
    try {
        std::printf("[flint] output: %s\n", decode_tokens(model_dir, generated).c_str());
    } catch (const std::exception& e) {
        std::printf("[flint] detokenize failed: %s\n", e.what());
    }
    return 0;
}

#include "common/bench/registry.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include "modeling/flint/kernels/awq_xmx.hpp"
#include "modeling/flint/kernels/awq_joint_matrix.hpp"
#include "modeling/flint/kernels/kernels.hpp"
#include "modeling/flint/kernels/prefill.hpp"
#include "modeling/flint/kernels/prefill_xmx.hpp"
#include "modeling/flint/weights.hpp"

using namespace flint;

namespace {

template <typename T>
T* dev(sycl::queue& q, size_t n) {
    T* p = sycl::malloc_device<T>(n, q);
    if (!p) { std::fprintf(stderr, "alloc failed\n"); std::exit(1); }
    return p;
}

QTensor make_q4(sycl::queue& q, int n, int k) {
    QTensor t;
    std::vector<uint8_t> w((size_t)n * (k / 2), 0x5a);
    std::vector<bf16> s((size_t)n * (k / 64), bf16(0.01f));
    std::vector<bf16> b((size_t)n * (k / 64), bf16(-0.005f));
    // 256-byte aligned packed bytes: the XMX B fragment load is a 256-byte
    // block_load and the repack kernel stages 16-byte vectors.
    uint8_t* raw = dev<uint8_t>(q, w.size() + 256);
    uint8_t* wa = raw + ((256 - ((uintptr_t)raw & 255)) & 255);
    t.w = wa;
    t.s = dev<bf16>(q, s.size());
    t.b = dev<bf16>(q, b.size());
    t.n = n; t.k = k; t.experts = 1;
    t.stride_w = (size_t)n * (k / 2);
    t.stride_s = (size_t)n * (k / 64);
    t.stride_b = t.stride_s;
    q.memcpy(const_cast<uint8_t*>(t.w), w.data(), w.size()).wait();
    q.memcpy(const_cast<bf16*>(t.s), s.data(), s.size() * sizeof(bf16)).wait();
    q.memcpy(const_cast<bf16*>(t.b), b.data(), b.size() * sizeof(bf16)).wait();
    return t;
}

struct Timer {
    std::chrono::steady_clock::time_point t0;
    void start() { t0 = std::chrono::steady_clock::now(); }
    double ms(int iters) const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count() / iters;
    }
};

void report(const char* name, int iters, double ms_per_iter, double bytes) {
    const double gbps = bytes > 0 ? bytes / (ms_per_iter * 1e6) : 0.0;
    std::printf("%-22s %8d iters  %10.4f ms/iter  %9.2f GB/s\n", name, iters,
                ms_per_iter, gbps);
}

int parse_bench_int(const char* text) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) return 0;
    return static_cast<int>(value);
}

struct AwqBuffers {
    sycl::queue& q;
    std::vector<void*> allocations;

    explicit AwqBuffers(sycl::queue& queue) : q(queue) { allocations.reserve(6); }
    ~AwqBuffers() {
        try { q.wait(); } catch (...) {}
        for (void* p : allocations) sycl::free(p, q);
    }

    template <typename T>
    T* allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::overflow_error("AWQ allocation size overflow");
        T* p = sycl::aligned_alloc_device<T>(64, count, q);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    }
};

struct AwqErrors {
    size_t checked = 0, failed = 0;
    double max_abs = 0, max_tol_ratio = 0;

    void add(float actual, float expected) {
        ++checked;
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++failed;
            max_abs = max_tol_ratio = std::numeric_limits<double>::infinity();
            return;
        }
        const double error = std::fabs(double(actual) - double(expected));
        const double limit = 1e-2 + 1e-2 * std::fabs(double(expected));
        max_abs = std::max(max_abs, error);
        max_tol_ratio = std::max(max_tol_ratio, error / limit);
        if (error > limit) ++failed;
    }

    bool report(const char* reference) const {
        std::printf("check %-11s checked=%zu failed=%zu max_abs=%.6g "
                    "max_tol_ratio=%.6g atol=0.01 rtol=0.01 %s\n",
                    reference, checked, failed, max_abs, max_tol_ratio,
                    failed ? "FAIL" : "PASS");
        return failed == 0;
    }
};

int bench_awq(const std::string& which, int M, int N, int K, int iters, bool check) {
    const bool gemv = which == "awq-gemv";
    const bool joint = which == "prefill-awq-joint-matrix";
    const bool xmx = which == "prefill-awq-xmx" || joint;
    if (M <= 0 || N <= 0 || K <= 0 || K % 64 || iters <= 0)
        throw std::invalid_argument("AWQ requires M,N > 0, K > 0 divisible by 64, --iters > 0");
    const uint64_t nk = uint64_t(N) * K, mk = uint64_t(M) * K, mn = uint64_t(M) * N;
    const uint64_t max_extent = uint64_t(std::numeric_limits<int>::max()) - 256;
    if (nk > max_extent || mk > max_extent || mn > max_extent)
        throw std::invalid_argument("AWQ dimensions exceed safe indexing limits");
    if (xmx && !joint && N % 16)
        throw std::invalid_argument("prefill-awq-xmx requires N divisible by 16; no fallback");
    if (joint && awq_joint_matrix::mode() == awq_joint_matrix::Mode::disabled)
        throw std::invalid_argument("prefill-awq-joint-matrix requires external "
                                    "ARCAINE_FLINT_AWQ_JOINT_MATRIX=dpas or dpas-sg32");
    if (xmx && !joint) {
        if (awq_joint_matrix::mode() != awq_joint_matrix::Mode::disabled)
            throw std::invalid_argument("prefill-awq-xmx requires ARCAINE_FLINT_AWQ_JOINT_MATRIX unset; "
                                        "use prefill-awq-joint-matrix for native kernels");
        const char* enabled = std::getenv("ARCAINE_FLINT_AWQ_XMX");
        const char* decompose = std::getenv("IGC_allowDecompose2DBlockFuncs");
        if (!enabled || std::strcmp(enabled, "1") || !decompose || std::strcmp(decompose, "0"))
            throw std::invalid_argument("prefill-awq-xmx requires external ARCAINE_FLINT_AWQ_XMX=1 "
                                        "and IGC_allowDecompose2DBlockFuncs=0; no fallback");
    }

    const char* profile_env = std::getenv("ARCAINE_FLINT_AWQ_JOINT_MATRIX_PROFILE");
    const bool profile = joint && profile_env && !std::strcmp(profile_env, "1");
    const sycl::property_list props = profile
        ? sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}
        : sycl::property_list{sycl::property::queue::in_order{}};
    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                  [](sycl::exception_list errors) {
                      for (const auto& error : errors) std::rethrow_exception(error);
                  }, props);
    const auto device = q.get_device();
    const bool vector_fits = uint64_t(K) * sizeof(bf16) <=
                            device.get_info<sycl::info::device::local_mem_size>();
    if (!gemv && !xmx && !vector_fits)
        throw std::invalid_argument("prefill-awq K exceeds device local memory capacity");
    std::printf("[kbench] %s  CUs=%u  local=%zuKB\n",
                device.get_info<sycl::info::device::name>().c_str(),
                device.get_info<sycl::info::device::max_compute_units>(),
                device.get_info<sycl::info::device::local_mem_size>() / 1024);
    if (joint)
        std::printf("[joint_matrix] mode=%s weight=W4A16 tile=%s stage=%s global_a_requested=%d grf=256\n",
                    awq_joint_matrix::mode() == awq_joint_matrix::Mode::dpas ? "dpas" : "dpas-sg32",
                    awq_joint_matrix::tile_name(), awq_joint_matrix::staging_name(),
                    int(awq_joint_matrix::global_a_enabled()));
    std::printf("BENCH_BEGIN %s\n", which.c_str());
    std::fflush(stdout);

    std::vector<uint8_t> hw(static_cast<size_t>(nk / 2));
    std::vector<sycl::half> hs(static_cast<size_t>(nk / 64));
    std::vector<uint8_t> hz(hs.size());
    std::vector<bf16> ha(static_cast<size_t>(mk));
    uint32_t rng = 0x6d2b79f5u ^ uint32_t(M) * 0x9e3779b9u ^ uint32_t(N) * 131u ^ uint32_t(K);
    auto rnd = [&] {
        rng = rng * 1664525u + 1013904223u;
        return rng >> 8;
    };
    for (auto& value : hw) value = uint8_t(rnd());
    for (size_t i = 0; i < hs.size(); ++i) {
        hs[i] = sycl::half(0.0025f + float(rnd() % 499) * 0.0001f);
        hz[i] = uint8_t(rnd() & 15);
    }
    for (auto& value : ha) value = bf16(float(int(rnd() % 2049) - 1024) / 1024.f);

    AwqBuffers buffers(q);
    uint8_t* w = buffers.allocate<uint8_t>(hw.size());
    sycl::half* s = buffers.allocate<sycl::half>(hs.size());
    uint8_t* z = buffers.allocate<uint8_t>(hz.size());
    bf16* A = buffers.allocate<bf16>(ha.size());
    bf16* C = buffers.allocate<bf16>(static_cast<size_t>(mn));
    QTensor t;
    t.format = WeightFormat::AWQ;
    t.w = w; t.awq_s = s; t.awq_z = z;
    t.n = N; t.k = K; t.experts = 1;
    t.stride_w = hw.size();
    t.stride_s = t.stride_b = hs.size();
    q.memcpy(w, hw.data(), hw.size());
    q.memcpy(s, hs.data(), hs.size() * sizeof(sycl::half));
    q.memcpy(z, hz.data(), hz.size());
    q.memcpy(A, ha.data(), ha.size() * sizeof(bf16));
    q.memset(C, 0xff, size_t(mn) * sizeof(bf16));
    q.wait_and_throw();

    sycl::event kernel_event;
    auto launch = [&] {
        if (gemv) {
            for (int token = 0; token < M; ++token)
                kernels::q4_gemv(q, t, A + size_t(token) * K, C + size_t(token) * N);
        } else if (xmx) {
            const bool accepted = joint ? awq_joint_matrix::gemm(q, t, A, nullptr, C, M, N,
                                                                  nullptr, nullptr, 0, 0,
                                                                  profile ? &kernel_event : nullptr)
                                        : awq_xmx::gemm(q, t, A, nullptr, C, M, N);
            if (!accepted)
                throw std::runtime_error("awq_xmx::gemm rejected M=" + std::to_string(M) +
                                         " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                         "; refusing vector fallback");
        } else {
            prefill::gemm_awq(q, t, A, nullptr, C, M, N);
        }
    };
    constexpr int warmup = 3;
    for (int i = 0; i < warmup; ++i) launch();
    q.wait_and_throw();
    Timer timer;
    std::vector<sycl::event> events;
    if (profile) events.reserve(iters);
    timer.start();
    for (int i = 0; i < iters; ++i) {
        launch();
        if (profile) events.push_back(kernel_event);
    }
    q.wait_and_throw();
    const double ms = timer.ms(iters);
    std::printf("%s M=%d N=%d K=%d warmup=%d iters=%d  %.6f ms/iter  "
                "%.3f effective TFLOP/s\n", which.c_str(), M, N, K, warmup, iters,
                ms, 2.0 * M * N * K / (ms * 1e9));
    if (profile) {
        double device_ns = 0;
        for (const auto& event : events)
            device_ns += double(event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                event.get_profiling_info<sycl::info::event_profiling::command_start>());
        const double device_ms = device_ns / (iters * 1e6);
        std::printf("joint_matrix device=%.6f ms/iter %.3f effective TFLOP/s\n",
                    device_ms, 2.0 * M * N * K / (device_ms * 1e9));
    }

    if (check) {
        std::vector<bf16> hc(static_cast<size_t>(mn));
        q.memcpy(hc.data(), C, hc.size() * sizeof(bf16));
        q.wait_and_throw();
        for (size_t i = 0; i < hc.size(); ++i) {
            if (!std::isfinite(float(hc[i]))) {
                std::fprintf(stderr, "AWQ nonfinite output at token=%zu row=%zu\n", i / N, i % N);
                return 1;
            }
        }
        const uint64_t work = mn * K;
        const bool full = work <= 16ull * 1024 * 1024;
        const int tokens = full ? M : std::min(M, 8);
        const int rows = full ? N : std::min(N, 32);
        AwqErrors cpu;
        for (int ti = 0; ti < tokens; ++ti) {
            const int token = full ? ti : (tokens == 1 ? 0 : int(uint64_t(ti) * (M - 1) / (tokens - 1)));
            for (int ri = 0; ri < rows; ++ri) {
                const int row = full ? ri : (rows == 1 ? 0 : int(uint64_t(ri) * (N - 1) / (rows - 1)));
                float acc = 0.f;
                for (int i = 0; i < K; ++i) {
                    const uint8_t packed = hw[size_t(row) * (K / 2) + i / 2];
                    const int code = (packed >> ((i & 1) * 4)) & 15;
                    const size_t param = size_t(i / 64) * N + row;
                    const float weight = float(bf16(float(code - int(hz[param])) * float(hs[param])));
                    acc = std::fma(weight, float(ha[size_t(token) * K + i]), acc);
                }
                cpu.add(float(hc[size_t(token) * N + row]), float(bf16(acc)));
            }
        }
        bool pass = cpu.report(full ? "cpu-full" : "cpu-sampled");
        if (xmx || gemv) {
            constexpr uint64_t vector_budget = 64ull * 1024 * 1024;
            int ref_m = vector_fits && nk <= vector_budget ? std::min(M, int(vector_budget / nk)) : 0;
            if (work > vector_budget) ref_m = std::min(ref_m, 8);
            if (ref_m) {
                std::vector<bf16> hv(size_t(ref_m) * N);
                bf16* Cv = buffers.allocate<bf16>(hv.size());
                q.memset(Cv, 0xff, hv.size() * sizeof(bf16));
                prefill::gemm_awq(q, t, A, nullptr, Cv, ref_m, N);
                q.memcpy(hv.data(), Cv, hv.size() * sizeof(bf16));
                q.wait_and_throw();
                AwqErrors vector;
                for (size_t i = 0; i < hv.size(); ++i)
                    vector.add(float(hc[i]), float(hv[i]));
                if (!vector.report("vector")) pass = false;
            } else {
                std::printf("check vector skipped (work/local-memory limit)\n");
            }
        }
        if (!pass) return 1;
    }
    std::printf("BENCH_END %s\n", which.c_str());
    return 0;
}

}  // namespace

namespace {
int run_flint_kernels(int argc, char** argv) {
    std::string which = "gemv";
    int iters = 200;
    int inner = 256;
    int groups = 1024;
    int n = 4096, k = 2048, depth = 8192, parts = 64, M = 512;
    bool check = false, m_explicit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--help" || a == "-h") {
            std::printf("Usage: %s --case CASE [--iters N] [--n N] [--k N] [--m N] [--depth N] [--parts N] [--inner N] [--groups N] [--check]\n"
                        "Cases: gemv, lmhead, qkv, attn-serial, attn-split, moe, moe-fused, dense, xmx-int8, xmx-bf16, alu-fp32, rmsnorm, router, prefill-bf16, prefill-int8, prefill-int8-xmx, prefill-xmx-check, awq-gemv, prefill-awq, prefill-awq-xmx, prefill-awq-joint-matrix\n", argv[0]);
            return 0;
        }
        if (a == "--case") {
            which = next();
            if (which == "prefill-xmx-check") {
                which = "prefill-int8-xmx";
                check = true;
            }
        }
        else if (a == "--iters") iters = parse_bench_int(next());
        else if (a == "--inner") inner = std::atoi(next());
        else if (a == "--groups") groups = std::atoi(next());
        else if (a == "--n") n = parse_bench_int(next());
        else if (a == "--k") k = parse_bench_int(next());
        else if (a == "--depth") depth = std::atoi(next());
        else if (a == "--parts") parts = std::atoi(next());
        else if (a == "--m") { M = parse_bench_int(next()); m_explicit = true; }
        else if (a == "--check") check = true;
    }

    if (which == "awq-gemv" || which == "prefill-awq" || which == "prefill-awq-xmx" || which == "prefill-awq-joint-matrix") {
        try {
            return bench_awq(which, which == "awq-gemv" && !m_explicit ? 1 : M,
                             n, k, iters, check);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", which.c_str(), e.what());
            return 1;
        }
    }

    sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
                  sycl::property::queue::in_order{});
    {
        const auto& dev = q.get_device();
        std::printf("[kbench] %s  CUs=%u  local=%zuKB  SGs=",
                    dev.get_info<sycl::info::device::name>().c_str(),
                    dev.get_info<sycl::info::device::max_compute_units>(),
                    dev.get_info<sycl::info::device::local_mem_size>() / 1024);
        for (size_t sg : dev.get_info<sycl::info::device::sub_group_sizes>())
            std::printf("%zu ", sg);
        std::printf("\n");
    }
    std::printf("BENCH_BEGIN %s\n", which.c_str());
    std::fflush(stdout);
    Timer timer;

    if (which == "gemv" || which == "lmhead") {
        if (which == "lmhead") { n = 150144; k = 2048; }
        QTensor t = make_q4(q, n, k);
        bf16* x = dev<bf16>(q, k);
        bf16* y = dev<bf16>(q, n);
        const size_t bytes = (size_t)n * (k / 2) + (size_t)n * (k / 64) * 4;
        kernels::q4_gemv(q, t, x, y);
        q.wait();
        timer.start();
        for (int i = 0; i < iters; ++i) kernels::q4_gemv(q, t, x, y);
        q.wait();
        report(which.c_str(), iters, timer.ms(iters), bytes);
    } else if (which == "qkv") {
        const int nq = 4096, nk = 512, kk = 2048;
        QTensor qw = make_q4(q, nq, kk), kw = make_q4(q, nk, kk), vw = make_q4(q, nk, kk);
        bf16* x = dev<bf16>(q, kk);
        bf16* qo = dev<bf16>(q, nq);
        bf16* ko = dev<bf16>(q, nk);
        bf16* vo = dev<bf16>(q, nk);
        const size_t bytes = (size_t)(nq + 2 * nk) * (kk / 2) +
                             (size_t)(nq + 2 * nk) * (kk / 64) * 4;
        kernels::qkv_gemv(q, qw, kw, vw, x, qo, ko, vo);
        q.wait();
        timer.start();
        for (int i = 0; i < iters; ++i)
            kernels::qkv_gemv(q, qw, kw, vw, x, qo, ko, vo);
        q.wait();
        report("qkv_gemv (fused)", iters, timer.ms(iters), bytes);
    } else if (which == "attn-serial" || which == "attn-split") {
        const int heads = 32, kvh = 4, cap = depth;
        bf16* qq = dev<bf16>(q, (size_t)heads * 128);
        bf16* sink = dev<bf16>(q, heads);
        bf16* kc = dev<bf16>(q, (size_t)kvh * cap * 128);
        bf16* vc = dev<bf16>(q, (size_t)kvh * cap * 128);
        bf16* out = dev<bf16>(q, (size_t)heads * 128);
        float* partial = dev<float>(q, (size_t)heads * 128 * 128);
        float* stats = dev<float>(q, (size_t)heads * 128 * 2);
        q.memset(qq, 0, (size_t)heads * 128 * 2);
        q.memset(sink, 0, heads * 2);
        q.memset(kc, 0, (size_t)kvh * cap * 128 * 2);
        q.memset(vc, 0, (size_t)kvh * cap * 128 * 2);
        q.wait();
        const size_t bytes = (size_t)kvh * cap * 128 * 2 * 2;
        if (which == "attn-serial") {
            kernels::attention(q, qq, kc, vc, sink, out, heads, kvh, true,
                               cap - 1, cap);
            q.wait();
            timer.start();
            for (int i = 0; i < iters; ++i)
                kernels::attention(q, qq, kc, vc, sink, out, heads, kvh, true,
                                   cap - 1, cap);
            q.wait();
        } else {
            kernels::attention_split(q, qq, kc, vc, sink, partial, stats, heads,
                                     kvh, true, cap - 1, cap, parts);
            kernels::attention_merge(q, partial, stats, out, heads, parts);
            q.wait();
            timer.start();
            for (int i = 0; i < iters; ++i) {
                kernels::attention_split(q, qq, kc, vc, sink, partial, stats, heads,
                                         kvh, true, cap - 1, cap, parts);
                kernels::attention_merge(q, partial, stats, out, heads, parts);
            }
            q.wait();
        }
        char name[64];
        std::snprintf(name, sizeof(name), "%s d=%d p=%d", which.c_str(), depth,
                      which == "attn-split" ? parts : 1);
        report(name, iters, timer.ms(iters), bytes);
    } else if (which == "moe" || which == "moe-fused") {
        const int H = 2048, moe_n = 512, shared_n = 1024, top_k = 6;
        FlintLayer L;
        L.egate = make_q4(q, moe_n, H);
        L.eup = make_q4(q, moe_n, H);
        L.edown = make_q4(q, H, moe_n);
        L.sgate = make_q4(q, shared_n, H);
        L.sup = make_q4(q, shared_n, H);
        L.sdown = make_q4(q, H, shared_n);
        bf16* x = dev<bf16>(q, H);
        bf16* act = dev<bf16>(q, (size_t)7 * shared_n);
        bf16* h = dev<bf16>(q, H);
        bf16* down = dev<bf16>(q, (size_t)7 * H);
        int* ids = dev<int>(q, top_k);
        float* weights = dev<float>(q, top_k);
        q.memset(ids, 0, top_k * sizeof(int));
        q.memset(weights, 0, top_k * sizeof(float));
        q.wait();
        const size_t bytes =
            (size_t)top_k * (moe_n * H / 2 + moe_n * (H / 64) * 4) * 2 +
            (size_t)top_k * (H * moe_n / 2 + H * (moe_n / 64) * 4) +
            shared_n * H / 2 + H * shared_n / 2;
        if (which == "moe") {
            kernels::expert_gateup(q, L, H, moe_n, shared_n, top_k, x, ids, act);
            kernels::expert_down(q, L, H, moe_n, shared_n, top_k, act, ids, down);
            kernels::moe_combine(q, h, down, weights, H, top_k);
            q.wait();
            timer.start();
            for (int i = 0; i < iters; ++i) {
                kernels::expert_gateup(q, L, H, moe_n, shared_n, top_k, x, ids, act);
                kernels::expert_down(q, L, H, moe_n, shared_n, top_k, act, ids, down);
                kernels::moe_combine(q, h, down, weights, H, top_k);
            }
            q.wait();
            report("moe (3 launches)", iters, timer.ms(iters), bytes);
        } else {
            kernels::expert_gateup(q, L, H, moe_n, shared_n, top_k, x, ids, act);
            kernels::moe_down_combine(q, L, H, moe_n, shared_n, top_k, act, ids,
                                      weights, h);
            q.wait();
            timer.start();
            for (int i = 0; i < iters; ++i) {
                kernels::expert_gateup(q, L, H, moe_n, shared_n, top_k, x, ids, act);
                kernels::moe_down_combine(q, L, H, moe_n, shared_n, top_k, act,
                                          ids, weights, h);
            }
            q.wait();
            report("moe (2 launches)", iters, timer.ms(iters), bytes);
        }
    } else if (which == "dense") {
        const int H = 2048, I = 6144;
        QTensor g = make_q4(q, I, H), u = make_q4(q, I, H), d = make_q4(q, H, I);
        bf16* x = dev<bf16>(q, H);
        bf16* act = dev<bf16>(q, I);
        bf16* h = dev<bf16>(q, H);
        const size_t bytes = (size_t)I * H / 2 * 2 + (size_t)H * I / 2 +
                             (size_t)(I + I + H) * 64 * 4;
        kernels::dense_gateup_act(q, g, u, x, act);
        kernels::q4_gemv(q, d, act, h, 0, h);
        q.wait();
        timer.start();
        for (int i = 0; i < iters; ++i) {
            kernels::dense_gateup_act(q, g, u, x, act);
            kernels::q4_gemv(q, d, act, h, 0, h);
        }
        q.wait();
        report("dense (2 launches)", iters, timer.ms(iters), bytes);
    } else if (which == "xmx-int8") {
        namespace esimd = sycl::ext::intel::esimd;
        namespace xmx = sycl::ext::intel::esimd::xmx;
        using dpas_t = xmx::dpas_argument_type;
        constexpr int SD = 8, RC = 8, ES = 16;
        constexpr int KT = 32;              // s8: 8 systolic depth * 4 VNNI bytes
        constexpr int AF = RC * KT;         // 8x32 int8 A fragment
        constexpr int BN = ES * KT;         // 16x32 int8 B fragment (N is fixed at 16)
        constexpr int RES = RC * ES;        // 8x16 int32 accumulator
        const int subgroups = groups;
        // Operands must live in memory, not be compile-time constants: with
        // constant a/b IGC folds the whole dpas loop away and this benchmark
        // then reports launch overhead as "dpas/s" (and XMX counts stay 0).
        int8_t* a_buf = dev<int8_t>(q, AF);
        int8_t* b_buf = dev<int8_t>(q, BN);
        int* c_buf = dev<int>(q, RES);
        q.fill(a_buf, (int8_t)3, AF);
        q.fill(b_buf, (int8_t)5, BN);
        q.fill(c_buf, 0, RES).wait();
        auto run = [&] {
            q.parallel_for(
                sycl::nd_range<1>((size_t)subgroups * 16, 16),
                sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::sub_group_size<16>},
                [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                    esimd::simd<int8_t, AF> a =
                        esimd::block_load<int8_t, AF>(a_buf, 0);
                    esimd::simd<int8_t, BN> b =
                        esimd::block_load<int8_t, BN>(b_buf, 0);
                    esimd::simd<int, RES> c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    esimd::simd<int, RES> c4 = 0, c5 = 0, c6 = 0, c7 = 0;
                    for (int i = 0; i < inner; ++i) {
                        c4 = xmx::dpas<SD, RC, int, int, int8_t, int8_t,
                                       dpas_t::s8, dpas_t::s8, RES, BN, AF>(
                            c4, b, a);
                        c5 = xmx::dpas<SD, RC, int, int, int8_t, int8_t,
                                       dpas_t::s8, dpas_t::s8, RES, BN, AF>(
                            c5, b, a);
                        c6 = xmx::dpas<SD, RC, int, int, int8_t, int8_t,
                                       dpas_t::s8, dpas_t::s8, RES, BN, AF>(
                            c6, b, a);
                        c7 = xmx::dpas<SD, RC, int, int, int8_t, int8_t,
                                       dpas_t::s8, dpas_t::s8, RES, BN, AF>(
                            c7, b, a);
                        c0 = xmx::dpas<SD, RC, int, int, int8_t, int8_t,
                                       dpas_t::s8, dpas_t::s8, RES, BN, AF>(
                            c0, b, a);
                        c1 = xmx::dpas<SD, RC, int, int, int8_t, int8_t,
                                       dpas_t::s8, dpas_t::s8, RES, BN, AF>(
                            c1, b, a);
                        c2 = xmx::dpas<SD, RC, int, int, int8_t, int8_t,
                                       dpas_t::s8, dpas_t::s8, RES, BN, AF>(
                            c2, b, a);
                        c3 = xmx::dpas<SD, RC, int, int, int8_t, int8_t,
                                       dpas_t::s8, dpas_t::s8, RES, BN, AF>(
                            c3, b, a);
                    }
                    esimd::block_store<int, RES>(
                        c_buf, 0, c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
                });
        };
        run();
        q.wait();
        const double dpas_per_launch =
            (double)subgroups * 16.0 * inner * 8;
        timer.start();
        for (int i = 0; i < iters; ++i) run();
        q.wait();
        const double ms = timer.ms(iters);
        const double dpas_s = dpas_per_launch / (ms * 1e-3);
        const double tops = dpas_s * (RC * ES * KT * 2.0) / 1e12;
        const char* env_rl = std::getenv("ARCAINE_XMX_INT8_TOPS");
        const double roofline = env_rl ? std::atof(env_rl) : 363.0;
        int chk = 0;
        q.copy(c_buf, &chk, 1).wait();
        std::printf("xmx_int8 s8xs8 subgroups=%d inner=%d iters=%d  %.3f ms  "
                    "%.1f G dpas/s  %.1f TOPS  (%.1f%% of %.0f TOPS roofline)  "
                    "checksum=%d expect=%d\n",
                    subgroups, inner, iters, ms, dpas_s / 1e9, tops,
                    100.0 * tops / roofline, roofline, chk, inner * 8 * 3 * 5 * KT);
    } else if (which == "xmx-bf16") {
        namespace esimd = sycl::ext::intel::esimd;
        namespace xmx = sycl::ext::intel::esimd::xmx;
        using dpas_t = xmx::dpas_argument_type;
        constexpr int SD = 8, RC = 8, ES = 16;
        constexpr int KT = 16;              // bf16: systolic depth * 2
        constexpr int AF = RC * KT;
        constexpr int BN = ES * KT;
        constexpr int RES = RC * ES;
        const int subgroups = groups;
        bf16* a_buf = dev<bf16>(q, AF);
        bf16* b_buf = dev<bf16>(q, BN);
        float* c_buf = dev<float>(q, RES);
        q.fill(a_buf, bf16(1.5f), AF);
        q.fill(b_buf, bf16(0.5f), BN);
        q.fill(c_buf, 0.0f, RES).wait();
        auto run = [&] {
            q.parallel_for(
                sycl::nd_range<1>((size_t)subgroups * 16, 16),
                [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                    esimd::simd<bf16, AF> a =
                        esimd::block_load<bf16, AF>(a_buf, 0);
                    esimd::simd<bf16, BN> b =
                        esimd::block_load<bf16, BN>(b_buf, 0);
                    esimd::simd<float, RES> c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    for (int i = 0; i < inner; ++i) {
                        c0 = xmx::dpas<SD, RC, float, float, bf16, bf16,
                                       dpas_t::bf16, dpas_t::bf16, RES, BN,
                                       AF>(c0, b, a);
                        c1 = xmx::dpas<SD, RC, float, float, bf16, bf16,
                                       dpas_t::bf16, dpas_t::bf16, RES, BN,
                                       AF>(c1, b, a);
                        c2 = xmx::dpas<SD, RC, float, float, bf16, bf16,
                                       dpas_t::bf16, dpas_t::bf16, RES, BN,
                                       AF>(c2, b, a);
                        c3 = xmx::dpas<SD, RC, float, float, bf16, bf16,
                                       dpas_t::bf16, dpas_t::bf16, RES, BN,
                                       AF>(c3, b, a);
                    }
                    esimd::block_store<float, RES>(c_buf, 0, c0 + c1 + c2 + c3);
                });
        };
        run();
        q.wait();
        const double dpas_per_launch =
            (double)subgroups * 16.0 * inner * 4;
        timer.start();
        for (int i = 0; i < iters; ++i) run();
        q.wait();
        const double ms = timer.ms(iters);
        const double dpas_s = dpas_per_launch / (ms * 1e-3);
        const double tflops = dpas_s * (RC * ES * KT * 2.0) / 1e12;
        const char* env_rl = std::getenv("ARCAINE_XMX_BF16_TFLOPS");
        const double roofline = env_rl ? std::atof(env_rl) : 181.5;
        float chk = 0;
        q.copy(c_buf, &chk, 1).wait();
        std::printf("xmx_bf16 s_bf16 subgroups=%d inner=%d iters=%d  %.3f ms  "
                    "%.1f G dpas/s  %.1f TFLOPS  (%.1f%% of %.1f TFLOPS roofline)"
                    "  checksum=%f expect=%f\n",
                    subgroups, inner, iters, ms, dpas_s / 1e9, tflops,
                    100.0 * tflops / roofline, roofline, chk,
                    inner * 4 * 1.5f * 0.5f * KT);
    } else if (which == "alu-fp32") {
        // Vector-pipe roofline: independent fp32 FMAs, no memory traffic in the
        // hot loop.  256 CUs x 16 lanes x 2 flop x ~2.8 GHz ~= 23 TFLOPS.
        const size_t nthreads = 1 << 20;
        float* x = dev<float>(q, nthreads);
        q.fill(x, 1.0f, nthreads).wait();
        auto run = [&] {
            q.parallel_for(sycl::nd_range<1>(nthreads, 256),
                           [=](sycl::nd_item<1> it) {
                               const size_t i = it.get_global_id(0);
                               const float m = x[i];
                               // 8 independent chains: keeps the FMA pipes fed
                               // instead of measuring dependency latency.
                               float v0 = m, v1 = m + 1, v2 = m + 2, v3 = m + 3;
                               float v4 = m, v5 = m + 1, v6 = m + 2, v7 = m + 3;
                               for (int j = 0; j < inner; ++j) {
                                   v0 = v0 * m + 0.5f;
                                   v1 = v1 * m + 0.5f;
                                   v2 = v2 * m + 0.5f;
                                   v3 = v3 * m + 0.5f;
                                   v4 = v4 * m + 0.5f;
                                   v5 = v5 * m + 0.5f;
                                   v6 = v6 * m + 0.5f;
                                   v7 = v7 * m + 0.5f;
                               }
                               x[i] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
                           });
        };
        run();
        q.wait();
        timer.start();
        for (int i = 0; i < iters; ++i) run();
        q.wait();
        const double ms = timer.ms(iters);
        const double flops = 2.0 * (double)nthreads * inner * 8;
        const double tflops = flops / (ms * 1e-3) / 1e12;
        std::printf("alu_fp32 threads=%zu inner=%d iters=%d  %.3f ms  "
                    "%.2f TFLOPS  (%.1f%% of 23 TFLOPS roofline)\n",
                    nthreads, inner, iters, ms, tflops, 100.0 * tflops / 23.0);
    } else if (which == "rmsnorm") {
        const int H = 2048;
        bf16* x = dev<bf16>(q, H);
        bf16* w = dev<bf16>(q, H);
        bf16* y = dev<bf16>(q, H);
        kernels::rmsnorm(q, x, w, y, H, 1e-6f);
        q.wait();
        timer.start();
        for (int i = 0; i < iters; ++i) kernels::rmsnorm(q, x, w, y, H, 1e-6f);
        q.wait();
        report("rmsnorm H=2048", iters, timer.ms(iters), (size_t)H * 2 * 2);
    } else if (which == "router") {
        const int experts = 256, H = 2048, top_k = 6;
        bf16* w = dev<bf16>(q, (size_t)experts * H);
        bf16* x = dev<bf16>(q, H);
        int* ids = dev<int>(q, top_k);
        float* sc = dev<float>(q, top_k);
        kernels::router_topk(q, w, x, experts, H, top_k, ids, sc);
        q.wait();
        timer.start();
        for (int i = 0; i < iters; ++i)
            kernels::router_topk(q, w, x, experts, H, top_k, ids, sc);
        q.wait();
        report("router 256x2048", iters, timer.ms(iters),
               (size_t)experts * H * 2);
    } else if (which == "prefill-bf16" || which == "prefill-int8") {
        QTensor t = make_q4(q, n, k);
        bf16* A = dev<bf16>(q, (size_t)M * k);
        bf16* C = dev<bf16>(q, (size_t)M * n);
        int8_t* Aq = dev<int8_t>(q, (size_t)M * k);
        float* xs = dev<float>(q, (size_t)M * (k / 64));
        const size_t bytes = (size_t)((M + 7) / 8) * (n * k / 2 + n * (k / 64) * 4);
        if (which == "prefill-bf16") {
            prefill::gemm_q4<false>(q, t, A, nullptr, nullptr, nullptr, C, M, n);
            q.wait();
            timer.start();
            for (int i = 0; i < iters; ++i)
                prefill::gemm_q4<false>(q, t, A, nullptr, nullptr, nullptr, C, M, n);
            q.wait();
        } else {
            prefill::quantize_int8(q, A, Aq, xs, M, k);
            prefill::gemm_q4<true>(q, t, nullptr, Aq, xs, nullptr, C, M, n);
            q.wait();
            timer.start();
            for (int i = 0; i < iters; ++i)
                prefill::gemm_q4<true>(q, t, nullptr, Aq, xs, nullptr, C, M, n);
            q.wait();
        }
        char name[64];
        std::snprintf(name, sizeof(name), "%s M=%d N=%d K=%d", which.c_str(), M, n, k);
        report(name, iters, timer.ms(iters), bytes);
    } else if (which == "prefill-int8-xmx") {
        // XMX (DPAS) int8 prefill GEMM against the vector-pipe int8 path.
        //  * random MLX-affine int4 weights (make_q4() only fills 0x5a codes)
        //  * random bf16 activations, quantized per 64-group by quantize_int8()
        //  * xi (per-group int sums) precomputed by prefill_xmx::group_sums
        // --check runs both kernels on identical data and compares them; the
        // existing vector path prefill::gemm_q4<true> is the reference.
        auto q4_rand = [&](int nn, int kk) {
            uint32_t rng = 0x9e3779b9u + (uint32_t)(nn * 131 + kk);
            auto rnd = [&] {
                rng = rng * 1664525u + 1013904223u;
                return rng >> 9;
            };
            std::vector<uint8_t> hw((size_t)nn * (kk / 2));
            for (auto& v : hw) v = (uint8_t)rnd();
            std::vector<bf16> hs((size_t)nn * (kk / 64));
            for (auto& v : hs) v = bf16(0.001f + 1e-6f * (float)(rnd() % 4096));
            std::vector<bf16> hb((size_t)nn * (kk / 64));
            for (auto& v : hb) v = bf16(-0.0005f + 1e-7f * (float)(rnd() % 4096));
            QTensor t = make_q4(q, nn, kk);
            q.memcpy(const_cast<uint8_t*>(t.w), hw.data(), hw.size());
            q.memcpy(const_cast<bf16*>(t.s), hs.data(), hs.size() * sizeof(bf16));
            q.memcpy(const_cast<bf16*>(t.b), hb.data(), hb.size() * sizeof(bf16));
            return t;
        };
        // Same tensor contents in DPAS fragment order: the XMX kernel reads the
        // repacked copy, the vector kernel stays the reference on the original
        // row-major layout.
        auto to_dpas = [&](const QTensor& src) {
            const size_t bytes = (size_t)src.n * (src.k / 2);
            uint8_t* raw = dev<uint8_t>(q, bytes + 256);
            uint8_t* w = raw + ((256 - ((uintptr_t)raw & 255)) & 255);
            uint8_t* tmp = dev<uint8_t>(q, bytes);
            q.memcpy(w, src.w, bytes).wait();
            QTensor t = src;
            t.w = w;
            prefill_xmx::repack_q4_dpas(q, t, tmp);
            q.wait();
            sycl::free(tmp, q);
            return t;
        };
        auto rand_act = [&](int mm, int kk) {
            bf16* a = dev<bf16>(q, (size_t)mm * kk);
            std::vector<bf16> h((size_t)mm * kk);
            uint32_t rng = 0x1234567u + (uint32_t)(mm * 7919 + kk);
            for (auto& v : h) {
                rng = rng * 1664525u + 1013904223u;
                v = bf16(-1.f + 2e-9f * (float)((rng >> 9) % 1000000000u));
            }
            q.memcpy(a, h.data(), h.size() * sizeof(bf16));
            return a;
        };
        // Both paths on identical data; returns the max relative difference.
        auto parity = [&](int mm, int nn, int kk, bool scatter, bool resid,
                          int extra_stride) {
            QTensor wt = q4_rand(nn, kk);   // original layout: reference
            QTensor wx = to_dpas(wt);       // DPAS fragment order: XMX
            bf16* a = rand_act(mm, kk);
            int8_t* aq = dev<int8_t>(q, (size_t)mm * kk);
            float* xsx = dev<float>(q, (size_t)mm * (kk / 64));
            int* xix = dev<int>(q, (size_t)mm * (kk / 64));
            prefill::quantize_int8(q, a, aq, xsx, mm, kk);
            prefill_xmx::group_sums(q, aq, xix, mm, kk);
            const int stride = extra_stride ? nn + extra_stride : 0;
            const size_t span = stride ? (size_t)stride : (size_t)nn;
            bf16* cv = dev<bf16>(q, (size_t)mm * span);
            bf16* cx = dev<bf16>(q, (size_t)mm * span);
            q.fill(cv, bf16(0.f), (size_t)mm * span);
            q.fill(cx, bf16(0.f), (size_t)mm * span).wait();
            int* ridx = nullptr;
            if (scatter) {
                ridx = dev<int>(q, mm);
                std::vector<int> h(mm);
                for (int i = 0; i < mm; ++i) h[i] = mm - 1 - i;
                q.memcpy(ridx, h.data(), mm * sizeof(int));
            }
            bf16* res = nullptr;
            if (resid) res = rand_act(mm, (int)span);
            prefill::gemm_q4<true>(q, wt, nullptr, aq, xsx, ridx, cv, mm, nn,
                                   res, nullptr, stride);
            if (!prefill_xmx::gemm_q4_xmx(q, wx, aq, xsx, xix, ridx, cx, mm, nn,
                                          res, nullptr, stride)) {
                std::fprintf(stderr, "gemm_q4_xmx rejected M=%d N=%d K=%d\n", mm,
                             nn, kk);
                return 1.0;
            }
            std::vector<bf16> hv((size_t)mm * span), hx((size_t)mm * span);
            q.copy(cv, hv.data(), hv.size()).wait();
            q.copy(cx, hx.data(), hx.size()).wait();
            double max_abs = 0, max_rel = 0, ref_max = 0;
            int shown = 0;
            for (int i = 0; i < mm; ++i) {
                for (int r = 0; r < nn; ++r) {
                    const size_t e = (size_t)i * span + r;
                    const double va = (double)float(hv[e]);
                    const double vb = (double)float(hx[e]);
                    const double d = std::fabs(va - vb);
                    max_abs = std::fmax(max_abs, d);
                    max_rel = std::fmax(max_rel, d / std::fmax(std::fabs(va), 1e-3));
                    ref_max = std::fmax(ref_max, std::fabs(va));
                    if (d > 1e-3 + 1e-2 * std::fabs(va) && shown++ < 4)
                        std::printf("  t=%d r=%d ref=%.6g xmx=%.6g\n", i, r, va, vb);
                }
            }
            std::printf("parity M=%d N=%d K=%d%s%s  max_abs=%.6g  max_rel=%.6g  "
                        "ref_max=%.6g  %s\n",
                        mm, nn, kk, scatter ? " scatter" : "",
                        resid ? " residual" : "", max_abs, max_rel, ref_max,
                        max_rel < 1e-2 ? "PASS" : "FAIL");
            return max_rel;
        };

        QTensor t = q4_rand(n, k);   // original layout (vector reference)
        QTensor tx = to_dpas(t);     // DPAS fragment order (XMX)
        bf16* A = rand_act(M, k);
        bf16* Cv = dev<bf16>(q, (size_t)M * n);
        bf16* Cx = dev<bf16>(q, (size_t)M * n);
        int8_t* Aq = dev<int8_t>(q, (size_t)M * k);
        float* xs = dev<float>(q, (size_t)M * (k / 64));
        int* xi = dev<int>(q, (size_t)M * (k / 64));
        prefill::quantize_int8(q, A, Aq, xs, M, k);
        prefill_xmx::group_sums(q, Aq, xi, M, k);
        if (!prefill_xmx::gemm_q4_xmx(q, tx, Aq, xs, xi, nullptr, Cx, M, n)) {
            std::fprintf(stderr, "gemm_q4_xmx rejected M=%d N=%d K=%d\n", M, n, k);
            return 1;
        }
        prefill::gemm_q4<true>(q, t, nullptr, Aq, xs, nullptr, Cv, M, n);
        q.wait();

        // Weight bytes each path re-reads (one pass per token tile).
        const size_t wrow = n * (size_t)(k / 2) + n * (size_t)(k / 64) * 4;
        const size_t vbytes = (size_t)((M + 7) / 8) * wrow;
        const size_t xbytes = (size_t)((M + prefill_xmx::kBM - 1) /
                                       prefill_xmx::kBM) * wrow;
        const double flops = 2.0 * (double)M * n * k;
        timer.start();
        for (int i = 0; i < iters; ++i)
            prefill::gemm_q4<true>(q, t, nullptr, Aq, xs, nullptr, Cv, M, n);
        q.wait();
        const double ms_vec = timer.ms(iters);
        timer.start();
        for (int i = 0; i < iters; ++i)
            prefill_xmx::gemm_q4_xmx(q, tx, Aq, xs, xi, nullptr, Cx, M, n);
        q.wait();
        const double ms_xmx = timer.ms(iters);
        char name[64];
        std::snprintf(name, sizeof(name), "prefill-int8 vec M=%d N=%d K=%d", M, n,
                      k);
        report(name, iters, ms_vec, (double)vbytes);
        std::printf("%-22s %8s%14.2f TFLOPS\n", "", "",
                    flops / (ms_vec * 1e-3) / 1e12);
        std::snprintf(name, sizeof(name), "prefill-int8 xmx M=%d N=%d K=%d", M, n,
                      k);
        report(name, iters, ms_xmx, (double)xbytes);
        std::printf("%-22s %8s%14.2f TFLOPS  (%.2fx vs vector)\n", "", "",
                    flops / (ms_xmx * 1e-3) / 1e12, ms_vec / ms_xmx);

        if (check) {
            std::vector<bf16> hv((size_t)M * n), hx((size_t)M * n);
            q.copy(Cv, hv.data(), hv.size()).wait();
            q.copy(Cx, hx.data(), hx.size()).wait();
            double max_abs = 0, max_rel = 0, ref_max = 0;
            for (size_t i = 0; i < hv.size(); ++i) {
                const double va = (double)float(hv[i]);
                const double vb = (double)float(hx[i]);
                const double d = std::fabs(va - vb);
                max_abs = std::fmax(max_abs, d);
                max_rel = std::fmax(max_rel, d / std::fmax(std::fabs(va), 1e-3));
                ref_max = std::fmax(ref_max, std::fabs(va));
            }
            std::printf("parity M=%d N=%d K=%d  max_abs=%.6g  max_rel=%.6g  "
                        "ref_max=%.6g  %s\n",
                        M, n, k, max_abs, max_rel, ref_max,
                        max_rel < 1e-2 ? "PASS" : "FAIL");
            if (max_rel >= 1e-2) return 1;
            // Contract edge cases: partial 32-token tile, activation row gather
            // (row_index), residual add, out_stride wider than N.
            if (parity(51, 64, 192, true, true, 32) >= 1e-2) {
                std::fprintf(stderr, "edge-case parity FAILED\n");
                return 1;
            }
        }
    } else {
        std::fprintf(stderr, "unknown case %s\n", which.c_str());
        return 1;
    }

    std::printf("BENCH_END %s\n", which.c_str());
    return 0;
}

}  // namespace

REGISTER_BENCH("flint-kernels", "Flint projection, attention, MoE and prefill kernel benchmarks", run_flint_kernels)

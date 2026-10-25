#include "../common/bench/registry.hpp"
#include "../modeling/flint/gpu_prefill.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace flint_attention_bench {
using namespace flint;
struct Buffers {
    sycl::queue& q;
    std::vector<void*> allocations;
    explicit Buffers(sycl::queue& queue) : q(queue) {}
    ~Buffers() {
        try { q.wait(); } catch (...) {}
        for (auto* p : allocations) sycl::free(p, q);
    }
    template <class T> T* alloc(size_t n) {
        T* p = sycl::malloc_device<T>(n, q);
        if (!p) throw std::bad_alloc{};
        allocations.push_back(p);
        return p;
    }
};

int run(int argc, char** argv) {
    try {
        int M = 512, depth = 0, heads = 32, kvh = 4, cap = 513, iters = 10;
        bool global = false, check = false, decode = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--global") { global = true; continue; }
            if (arg == "--decode") { decode = true; continue; }
            if (arg == "--check") { check = true; continue; }
            if (arg == "--help") {
                std::printf("flint-prefill-attention --m 512 --depth 0 --heads 32 "
                            "--kv-heads 4 --cap 513 --iters 10 [--global] [--decode] [--check]\n"
                            "ARCAINE_FLINT_GPU_PREFILL=1 selects GPU token scheduling; "
                            "unset/0 selects serial host scheduling. Inputs are normalized K/V.\n");
                return 0;
            }
            if (++i >= argc) throw std::invalid_argument("missing value");
            size_t end = 0;
            const int n = std::stoi(argv[i], &end);
            if (end != std::strlen(argv[i])) throw std::invalid_argument("invalid integer");
            if (arg == "--m") M = n;
            else if (arg == "--depth") depth = n;
            else if (arg == "--heads") heads = n;
            else if (arg == "--kv-heads") kvh = n;
            else if (arg == "--cap") cap = n;
            else if (arg == "--iters") iters = n;
            else throw std::invalid_argument("unknown option " + arg);
        }
        if (M <= 0 || M > 4096 || depth < 0 || depth > 32768 ||
            heads <= 0 || heads > 32 || kvh <= 0 || heads % kvh ||
            cap <= 0 || cap > 36864 || iters <= 0 ||
            (global && depth + M > cap) || (decode && M != 1))
            throw std::invalid_argument("invalid attention geometry or global capacity");
        sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
            [](sycl::exception_list errors) {
                for (const auto& e : errors) std::rethrow_exception(e);
            }, sycl::property::queue::in_order{});
        Buffers b(q);
        const size_t nq = size_t(M) * heads * 128;
        const size_t nk = size_t(M) * kvh * 128, nc = size_t(kvh) * cap * 128;
        auto* query = b.alloc<bf16>(nq);
        auto* k = b.alloc<bf16>(nk); auto* v = b.alloc<bf16>(nk);
        auto* kc = b.alloc<bf16>(nc); auto* vc = b.alloc<bf16>(nc);
        auto* legacy_kc = b.alloc<bf16>(nc); auto* legacy_vc = b.alloc<bf16>(nc);
        auto* sink = b.alloc<bf16>(heads);
        auto* output = b.alloc<bf16>(nq); auto* reference = b.alloc<bf16>(nq);
        const int max_parts = std::max(decode ? gpu_prefill::decode_min_parts(global) : 1,
            gpu_prefill::parts_for(decode ? cap : (global ? depth + M : std::min(depth + M, cap))));
        int* device_position = decode ? b.alloc<int>(1) : nullptr;
        auto* partial = b.alloc<float>(size_t(M) * heads * max_parts * 128);
        auto* stats = b.alloc<float>(size_t(M) * heads * max_parts * 2);
        uint32_t state = 17;
        auto signal = [&](size_t n, float amplitude) {
            std::vector<bf16> a(n);
            for (auto& value : a) {
                state = state * 1664525u + 1013904223u;
                value = bf16(float(int((state >> 8) % 2049) - 1024) / 1024.f * amplitude);
            }
            return a;
        };
        const auto hq = signal(nq, 1), hk = signal(nk, 1), hv = signal(nk, 2);
        const auto hkc = signal(nc, 1), hvc = signal(nc, 2), hsink = signal(heads, 2);
        q.memcpy(query, hq.data(), nq * sizeof(bf16));
        q.memcpy(k, hk.data(), nk * sizeof(bf16));
        q.memcpy(v, hv.data(), nk * sizeof(bf16));
        q.memcpy(kc, hkc.data(), nc * sizeof(bf16));
        q.memcpy(vc, hvc.data(), nc * sizeof(bf16));
        q.memcpy(legacy_kc, hkc.data(), nc * sizeof(bf16));
        q.memcpy(legacy_vc, hvc.data(), nc * sizeof(bf16));
        q.memcpy(sink, hsink.data(), size_t(heads) * sizeof(bf16)).wait_and_throw();
        if (decode) {
            q.memcpy(device_position, &depth, sizeof(int));
            gpu_prefill::commit(q, k, v, kc, vc, 1, kvh, global, depth, cap);
            gpu_prefill::commit(q, k, v, legacy_kc, legacy_vc, 1, kvh, global, depth, cap);
        }
        auto gpu = [&] {
            gpu_prefill::attention(q, query, k, v, kc, vc, sink, output,
                                   partial, stats, M, heads, kvh, global, depth, cap, device_position);
            if (!decode) gpu_prefill::commit(q, k, v, kc, vc, M, kvh, global, depth, cap);
        };
        auto legacy = [&] {
            for (int t = 0; t < M; ++t) {
                if (!decode) gpu_prefill::commit(q, k + size_t(t) * kvh * 128,
                    v + size_t(t) * kvh * 128, legacy_kc, legacy_vc,
                    1, kvh, global, depth + t, cap);
                const int count = global ? depth + t + 1 : std::min(depth + t + 1, cap);
                const int parts = kernels::attention_parts(count);
                if (parts == 1)
                    kernels::attention(q, query + size_t(t) * heads * 128,
                        legacy_kc, legacy_vc, sink, reference + size_t(t) * heads * 128,
                        heads, kvh, global, depth + t, cap);
                else {
                    kernels::attention_split(q, query + size_t(t) * heads * 128,
                        legacy_kc, legacy_vc, sink, partial, stats, heads, kvh,
                        global, depth + t, cap, parts);
                    kernels::attention_merge(q, partial, stats,
                        reference + size_t(t) * heads * 128, heads, parts);
                }
            }
        };
        if (check) {
            legacy();
            q.memset(output, 0xff, nq * sizeof(bf16));
            gpu(); q.wait_and_throw();
            std::vector<bf16> actual(nq), expected(nq), ak(nc), av(nc), ek(nc), ev(nc);
            q.memcpy(actual.data(), output, nq * sizeof(bf16));
            q.memcpy(expected.data(), reference, nq * sizeof(bf16));
            q.memcpy(ak.data(), kc, nc * sizeof(bf16));
            q.memcpy(av.data(), vc, nc * sizeof(bf16));
            q.memcpy(ek.data(), legacy_kc, nc * sizeof(bf16));
            q.memcpy(ev.data(), legacy_vc, nc * sizeof(bf16)).wait_and_throw();
            size_t bad = 0, cache_bad = 0, unequal = 0;
            double worst = 0;
            for (size_t i = 0; i < nq; ++i) {
                const float a = float(actual[i]), e = float(expected[i]);
                const double error = std::fabs(a - e);
                worst = std::max(worst, error);
                if (a != e) ++unequal;
                if (!std::isfinite(a) || error > 0.002 + 0.01 * std::fabs(e)) ++bad;
            }
            for (size_t i = 0; i < nc; ++i)
                if (float(ak[i]) != float(ek[i]) || float(av[i]) != float(ev[i])) ++cache_bad;
            std::printf("check outputs=%zu failed=%zu unequal=%zu max_abs=%.6g "
                        "cache_bad=%zu %s\n", nq, bad, unequal, worst, cache_bad,
                        bad || cache_bad ? "FAIL" : "PASS");
            if (bad || cache_bad) return 1;
        }
        // Inputs may overwrite historical ring slots between timing repetitions;
        // arithmetic and launch counts stay identical. Correctness is checked
        // above from identical initial caches, outside timing.
        const bool use_gpu = gpu_prefill::enabled();
        auto once = [&] { if (use_gpu) gpu(); else legacy(); };
        once(); q.wait_and_throw();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) once();
        q.wait_and_throw();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / iters;
        std::printf("flint-attention decode=%d split=%d path=%s M=%d depth=%d global=%d "
                    "cap=%d heads=%d kv_heads=%d iters=%d ms=%.4f\n",
            decode, gpu_prefill::decode_min_parts(global), use_gpu ? "gpu" : "host", M, depth, global, cap, heads, kvh, iters, ms);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "flint-prefill-attention: %s\n", e.what());
        return 1;
    }
}
}
REGISTER_BENCH("flint-prefill-attention",
              "Flint causal/sink attention and KV commit with GPU token scheduling", flint_attention_bench::run)

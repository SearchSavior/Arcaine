// Flint's routed W4A16 prefill branch: device routes -> gate/up+SiLU -> down.
// Includes host routing copies in the legacy timing; excludes setup/readback.
#include "../common/bench/registry.hpp"
#include "../modeling/flint/awq_xmx.hpp"
#include "../modeling/flint/grouped_moe.hpp"
#include "../modeling/flint/prefill.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace flint_grouped_moe_bench {
using namespace flint;

struct Buffers {
    sycl::queue& q;
    std::vector<void*> all;
    explicit Buffers(sycl::queue& queue) : q(queue) {}
    ~Buffers() {
        try { q.wait(); } catch (...) {}
        for (auto* p : all) sycl::free(p, q);
    }
    template <class T> T* alloc(size_t n) {
        T* p = sycl::aligned_alloc_device<T>(64, n, q);
        if (!p) throw std::bad_alloc{};
        all.push_back(p);
        return p;
    }
};

uint32_t hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

int run(int argc, char** argv) {
    try {
        int M = 512, H = 2048, I = 512, E = 256, top = 6, iters = 10;
        bool check = false;
        std::string routing = "balanced";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::printf("flint-awq-grouped-moe [--m 512] [--hidden 2048] "
                            "[--inter 512] [--experts 256] [--top-k 6] [--iters 10] "
                            "[--routing balanced|skewed|concentrated] [--check]\n"
                            "ARCAINE_FLINT_AWQ_GROUPED_MOE=1 selects grouped timing; "
                            "unset/0 selects the existing host/per-expert path.\n"
                            "ARCAINE_FLINT_AWQ_XMX=1 selects XMX for the legacy GEMMs.\n");
                return 0;
            }
            if (arg == "--check") { check = true; continue; }
            if (++i >= argc) throw std::invalid_argument("missing value: " + arg);
            if (arg == "--routing") { routing = argv[i]; continue; }
            size_t used = 0;
            const int n = std::stoi(argv[i], &used);
            if (used != std::strlen(argv[i]) || n <= 0)
                throw std::invalid_argument("invalid positive integer");
            if (arg == "--m") M = n;
            else if (arg == "--hidden") H = n;
            else if (arg == "--inter") I = n;
            else if (arg == "--experts") E = n;
            else if (arg == "--top-k") top = n;
            else if (arg == "--iters") iters = n;
            else throw std::invalid_argument("unknown option: " + arg);
        }
        if (H % 64 || I % 64 || E > 256 || top > E || M > 16384 ||
            H > 16384 || I > 16384)
            throw std::invalid_argument("require H,I multiples of 64 <=16384, M<=16384, top-k<=E<=256");
        if (routing != "balanced" && routing != "skewed" && routing != "concentrated")
            throw std::invalid_argument("unknown routing pattern");
        sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
            [](sycl::exception_list errors) {
                for (const auto& error : errors) std::rethrow_exception(error);
            }, sycl::property::queue::in_order{});
        Buffers b(q);
        auto tensor = [&](int N, int K, uint32_t seed) {
            QTensor t;
            t.format = WeightFormat::AWQ; t.n = N; t.k = K; t.experts = E;
            t.stride_w = size_t(N) * (K / 2);
            t.stride_s = t.stride_b = size_t(N) * (K / 64);
            auto* w = b.alloc<uint8_t>(size_t(E) * t.stride_w);
            auto* s = b.alloc<sycl::half>(size_t(E) * t.stride_s);
            auto* z = b.alloc<uint8_t>(size_t(E) * t.stride_b);
            t.w = w; t.awq_s = s; t.awq_z = z;
            q.parallel_for(sycl::range<1>(size_t(E) * t.stride_w), [=](sycl::id<1> id) {
                w[id] = uint8_t(hash(uint32_t(id[0]) ^ seed));
            });
            q.parallel_for(sycl::range<1>(size_t(E) * t.stride_s), [=](sycl::id<1> id) {
                const auto r = hash(uint32_t(id[0]) ^ (seed + 17));
                s[id] = sycl::half(0.001f + float(r % 200) * 0.00005f);
                z[id] = uint8_t((r >> 16) & 15);
            });
            return t;
        };
        const auto gate = tensor(I, H, 11), up = tensor(I, H, 37), down = tensor(H, I, 71);
        if (!grouped_moe::supported(q, gate, up, down))
            throw std::runtime_error("grouped W4A16 unsupported on this device");
        const int P = M * top;
        auto* A = b.alloc<bf16>(size_t(M) * H);
        auto* ids = b.alloc<int>(P);
        auto* act = b.alloc<bf16>(size_t(P) * I);
        auto* output = b.alloc<bf16>(size_t(P) * H);
        auto* reference = b.alloc<bf16>(size_t(P) * H);
        auto* g = b.alloc<bf16>(size_t(P) * I);
        auto* u = b.alloc<bf16>(size_t(P) * I);
        auto* rows = b.alloc<int>(P);
        auto* pairs = b.alloc<int>(P);
        auto* metadata = b.alloc<int>(grouped_moe::metadata_size(E));
        q.parallel_for(sycl::range<1>(size_t(M) * H), [=](sycl::id<1> id) {
            A[id] = bf16(float(int(hash(uint32_t(id[0]) + 79) % 2049) - 1024) / 1024.f);
        });
        std::vector<int> host_ids(P), host_rows(P), host_pairs(P);
        std::vector<int> counts(E), offsets(E), cursors(E);
        for (int t = 0; t < M; ++t)
            for (int r = 0; r < top; ++r) {
                int base = t * top;
                if (routing == "concentrated" || (routing == "skewed" && t % 8))
                    base = 0;
                host_ids[t * top + r] = (base + r) % E;
            }
        q.memcpy(ids, host_ids.data(), size_t(P) * sizeof(int)).wait_and_throw();
        auto grouped = [&] {
            grouped_moe::run(q, gate, up, down, A, ids, act, output, M, top, metadata, pairs);
        };
        auto gemm = [&](const QTensor& t, const bf16* a, const int* ri,
                        bf16* c, int n, const int* oi, int e) {
            if (!awq_xmx::gemm(q, t, a, ri, c, n, t.n, nullptr, oi, t.n, e))
                prefill::gemm_awq(q, t, a, ri, c, n, t.n, nullptr, oi, t.n, e);
        };
        auto legacy = [&] {
            q.memcpy(host_ids.data(), ids, size_t(P) * sizeof(int)).wait_and_throw();
            std::fill(counts.begin(), counts.end(), 0);
            for (const int id : host_ids) ++counts[id];
            int total = 0;
            for (int e = 0; e < E; ++e) {
                offsets[e] = cursors[e] = total;
                total += counts[e];
            }
            for (int p = 0; p < P; ++p) {
                const int at = cursors[host_ids[p]]++;
                host_rows[at] = p / top;
                host_pairs[at] = p;
            }
            q.memcpy(rows, host_rows.data(), size_t(P) * sizeof(int)).wait_and_throw();
            q.memcpy(pairs, host_pairs.data(), size_t(P) * sizeof(int)).wait_and_throw();
            for (int e = 0; e < E; ++e) {
                const int n = counts[e], off = offsets[e];
                if (!n) continue;
                gemm(gate, A, rows + off, g, n, nullptr, e);
                gemm(up, A, rows + off, u, n, nullptr, e);
                prefill::silu_mul_batch(q, g, u, n * I);
                gemm(down, g, nullptr, reference, n, pairs + off, e);
            }
        };
        if (check) {
            legacy();
            q.memset(output, 0xff, size_t(P) * H * sizeof(bf16));
            grouped();
            q.wait_and_throw();
            std::vector<bf16> actual(size_t(P) * H), expected(actual.size());
            std::vector<int> meta(grouped_moe::metadata_size(E)), perm(P), seen(P);
            q.memcpy(actual.data(), output, actual.size() * sizeof(bf16));
            q.memcpy(expected.data(), reference, expected.size() * sizeof(bf16));
            q.memcpy(meta.data(), metadata, meta.size() * sizeof(int));
            q.memcpy(perm.data(), pairs, size_t(P) * sizeof(int)).wait_and_throw();
            size_t failures = 0;
            double max_error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                const double error = std::fabs(float(actual[i]) - float(expected[i]));
                max_error = std::max(max_error, error);
                if (!std::isfinite(float(actual[i])) ||
                    error > 0.01 + 0.01 * std::fabs(float(expected[i]))) ++failures;
            }
            int routes_bad = 0;
            const int* off = meta.data() + E;
            const int* tiles = meta.data() + 3 * E + 1;
            if (off[0] || off[E] != P || tiles[0]) ++routes_bad;
            for (int e = 0; e < E; ++e) {
                if (meta[e] != counts[e] || off[e + 1] - off[e] != counts[e] ||
                    tiles[e + 1] - tiles[e] != (counts[e] + 7) / 8) ++routes_bad;
                for (int j = off[e]; j < off[e + 1]; ++j) {
                    const int p = perm[j];
                    if (p < 0 || p >= P || host_ids[p] != e) ++routes_bad;
                    else ++seen[p];
                }
            }
            for (int count : seen) if (count != 1) ++routes_bad;
            std::printf("check outputs=%zu failed=%zu max_abs=%.6g routes_bad=%d %s\n",
                actual.size(), failures, max_error, routes_bad,
                failures || routes_bad ? "FAIL" : "PASS");
            if (failures || routes_bad) return 1;
        }
        const bool use_grouped = grouped_moe::enabled();
        auto once = [&] { if (use_grouped) grouped(); else legacy(); };
        once(); q.wait_and_throw();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) once();
        q.wait_and_throw();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / iters;
        std::printf("flint-awq-grouped-moe path=%s M=%d H=%d I=%d E=%d top=%d "
                    "routing=%s iters=%d ms=%.4f routing_host_bytes=%zu\n",
            use_grouped ? "grouped" : "legacy", M, H, I, E, top,
            routing.c_str(), iters, ms, use_grouped ? size_t(0) : size_t(P) * 3 * sizeof(int));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "flint-awq-grouped-moe: %s\n", e.what());
        return 1;
    }
}
}  // namespace
REGISTER_BENCH("flint-awq-grouped-moe",
              "Flint W4A16 routed prefill: device grouping, fused gate/up+SiLU, down", flint_grouped_moe_bench::run)

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
        bool global = false, check = false, decode = false, profile=false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--global") { global = true; continue; }
            if (arg == "--decode") { decode = true; continue; }
            if (arg == "--profile") { profile=true; continue; }
            if (arg == "--check") { check = true; continue; }
            if (arg == "--help") {
                std::printf("flint-prefill-attention --m 512 --depth 0 --heads 32 "
                            "--kv-heads 4 --cap 513 --iters 10 [--global] [--decode] [--check] [--profile]\n"
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
        if (M <= 0 || M > 4096 || depth < 0 || depth > 262144 ||
            heads <= 0 || heads > 32 || kvh <= 0 || heads % kvh ||
            cap <= 0 || cap > 262272 || iters <= 0 ||
            (global && depth + M > cap) || (decode && M != 1))
            throw std::invalid_argument("invalid attention geometry or global capacity");
        sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
            [](sycl::exception_list errors) {
                for (const auto& e : errors) std::rethrow_exception(e);
            }, {sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
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
        auto* decode_scores=flash_decode_twopass::enabled()?b.alloc<float>(size_t(heads)*cap):nullptr;
        uint32_t* packed_v=nullptr;
        const char* packed_env=std::getenv("ARCAINE_FLINT_KBENCH_PACKED_V");
        if(packed_env&&!std::strcmp(packed_env,"1")){
            if(!decode||!global||!decode_scores)throw std::invalid_argument("packed V requires global two-pass decode");
            const int blocks=(cap+15)/16;
            const size_t words=size_t(kvh)*blocks*8*16*8;
            packed_v=b.alloc<uint32_t>(words);
            q.parallel_for(sycl::range<1>(words),[=](sycl::id<1> id){
                const size_t i=id[0];const int pair=int(i%8),lane=int(i/8%16),dblock=int(i/(8*16)%8);
                const int block=int(i/(8*16*8)%blocks),kv=int(i/(size_t(blocks)*8*16*8));
                const int pos0=block*16+pair*2,pos1=pos0+1,d=dblock*16+lane;
                const uint16_t lo=pos0<cap?sycl::bit_cast<uint16_t>(vc[(size_t(kv)*cap+pos0)*128+d]):0;
                const uint16_t high=pos1<cap?sycl::bit_cast<uint16_t>(vc[(size_t(kv)*cap+pos1)*128+d]):0;
                packed_v[i]=uint32_t(lo)|(uint32_t(high)<<16);
            });
        }
        sycl::event partial_event,merge_event;
        if(profile&&(!decode||max_parts==1||!gpu_prefill::enabled()))
            throw std::invalid_argument("--profile requires GPU split decode");
        auto gpu = [&] {
            gpu_prefill::attention(q, query, k, v, kc, vc, sink, output,
                                   partial, stats, M, heads, kvh, global, depth, cap, device_position,decode_scores,packed_v,profile?&partial_event:nullptr,profile?&merge_event:nullptr);
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
        std::vector<sycl::event> partial_events,merge_events;
        for (int i = 0; i < iters; ++i) {
            once();
            if(profile){partial_events.push_back(partial_event);merge_events.push_back(merge_event);}
        }
        q.wait_and_throw();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / iters;
        std::printf("flint-attention decode=%d split=%d path=%s M=%d depth=%d global=%d "
                    "cap=%d heads=%d kv_heads=%d iters=%d ms=%.4f\n",
            decode, gpu_prefill::decode_min_parts(global), use_gpu ? "gpu" : "host", M, depth, global, cap, heads, kvh, iters, ms);
        if(profile){
            auto duration=[](const sycl::event& e){return double(
                e.get_profiling_info<sycl::info::event_profiling::command_end>()-
                e.get_profiling_info<sycl::info::event_profiling::command_start>())/1e6;};
            double p=0,m=0,first=0;
            for(size_t i=0;i<partial_events.size();++i)p+=double(
                merge_events[i].get_profiling_info<sycl::info::event_profiling::command_start>()-
                partial_events[i].get_profiling_info<sycl::info::event_profiling::command_start>())/1e6;
            for(const auto& e:merge_events)m+=duration(e);
            for(const auto& e:partial_events)first+=duration(e);
            std::printf("attention GPU stages partial_ms=%.4f first_kernel_ms=%.4f merge_ms=%.4f total_ms=%.4f\n",p/iters,first/iters,m/iters,(p+m)/iters);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "flint-prefill-attention: %s\n", e.what());
        return 1;
    }
}
}
REGISTER_BENCH("flint-prefill-attention",
              "Flint causal/sink attention and KV commit with GPU token scheduling", flint_attention_bench::run)

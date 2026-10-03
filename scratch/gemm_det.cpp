#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using dt = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;

static uint16_t f32_to_bf16_bits(float v) {
    uint32_t x; std::memcpy(&x, &v, 4);
    uint32_t y = x & 0xFFFF0000u;
    uint32_t rem = x & 0xFFFFu;
    y += (rem > 0x8000u || (rem == 0x8000u && (y & 0x10000u))) ? 0x10000u : 0;
    return (uint16_t)(y >> 16);
}

static void report(const char* name, const std::vector<uint16_t>& a,
                   const std::vector<uint16_t>& b) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            if (bad < 5)
                std::printf("  %s idx=%zu 0x%04x vs 0x%04x\n", name, i, a[i], b[i]);
            ++bad;
        }
    std::printf("%s mismatched=%zu/%zu\n", name, bad, a.size());
    std::fflush(stdout);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        auto eng = dnnl::engine(dnnl::engine::kind::gpu, 0);
        auto dev = dnnl::sycl_interop::get_device(eng);
        auto sctx = dnnl::sycl_interop::get_context(eng);
        sycl::queue q(sctx, dev, sycl::property::queue::in_order{});
        auto stream = dnnl::sycl_interop::make_stream(eng, q);
        std::srand(42);

        // int4 w4a16 diffusion decode shape
        const int M = 256, K = 2816, N = 4096, G = 32;
        std::printf("[1] creating int4 pd...\n");
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {G, 1}, dt::bf16);
        attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
        dnnl::matmul::primitive_desc pd(eng,
            dnnl::memory::desc({M, K}, dt::bf16, tag::ab),
            dnnl::memory::desc({K, N}, dt::s4, tag::ba),
            dnnl::memory::desc({M, N}, dt::bf16, tag::ab), attr);
        auto prim = dnnl::matmul(pd);
        std::printf("[1] int4 pd ok impl=%s\n", pd.impl_info_str());

        uint16_t* A  = sycl::malloc_device<uint16_t>((size_t)M * K, q);
        uint8_t*  W  = sycl::malloc_device<uint8_t>((size_t)K * N / 2, q);
        uint16_t* S  = sycl::malloc_device<uint16_t>((size_t)G * N, q);
        uint16_t* C0 = sycl::malloc_device<uint16_t>((size_t)M * N, q);
        uint16_t* C1 = sycl::malloc_device<uint16_t>((size_t)M * N, q);
        std::vector<uint16_t> Ah((size_t)M * K), Sh((size_t)G * N);
        std::vector<uint8_t> Wh((size_t)K * N / 2);
        for (auto& v : Ah) v = f32_to_bf16_bits((float)(std::rand() % 2000 - 1000) / 511.0f);
        for (auto& v : Sh) v = f32_to_bf16_bits(0.001f + (float)(std::rand() % 1000) / 1000.0f);
        for (auto& v : Wh) v = (uint8_t)(std::rand() % 256);
        q.memcpy(A, Ah.data(), Ah.size() * 2).wait();
        q.memcpy(S, Sh.data(), Sh.size() * 2).wait();
        q.memcpy(W, Wh.data(), Wh.size()).wait();
        q.wait();
        std::printf("[1] host init done\n");

        auto run = [&](uint16_t* dst) {
            prim.execute(stream, {
                {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({M, K}, dt::bf16, tag::ab), eng,
                    dnnl::sycl_interop::memory_kind::usm, A)},
                {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({K, N}, dt::s4, tag::ba), eng,
                    dnnl::sycl_interop::memory_kind::usm, W)},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({G, N}, dt::bf16, tag::ab), eng,
                    dnnl::sycl_interop::memory_kind::usm, S)},
                {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({M, N}, dt::bf16, tag::ab), eng,
                    dnnl::sycl_interop::memory_kind::usm, dst)}});
            stream.wait();
        };
        std::printf("[1] warm runs...\n");
        run(C0); run(C0); run(C0);
        std::vector<uint16_t> r0((size_t)M * N), r1((size_t)M * N);
        q.memcpy(r0.data(), C0, (size_t)M * N * 2).wait();
        run(C1);
        q.memcpy(r1.data(), C1, (size_t)M * N * 2).wait();
        report("int4 w4a16 run4-vs-run5", r0, r1);
        sycl::free(A, q); sycl::free(W, q); sycl::free(S, q);
        sycl::free(C0, q); sycl::free(C1, q);

        // bf16 batched strided (gqa score shape)
        const int B = 8, Mm = 512, Kk = 1024, Nn = 768;
        std::printf("[2] creating batched pd...\n");
        dnnl::matmul::primitive_desc pd2(eng,
            dnnl::memory::desc({B, Mm, Kk}, dt::bf16,
                dnnl::memory::dims{(long long)Mm * Kk, Kk, 1}),
            dnnl::memory::desc({B, Kk, Nn}, dt::bf16,
                dnnl::memory::dims{1024, 1, 8 * 1024}),
            dnnl::memory::desc({B, Mm, Nn}, dt::bf16,
                dnnl::memory::dims{(long long)Mm * Nn, Nn, 1}));
        auto prim2 = dnnl::matmul(pd2);
        std::printf("[2] batched pd ok impl=%s\n", pd2.impl_info_str());
        uint16_t* A2 = sycl::malloc_device<uint16_t>((size_t)B * Mm * Kk, q);
        uint16_t* W2 = sycl::malloc_device<uint16_t>((size_t)B * Kk * Nn, q);
        uint16_t* D0 = sycl::malloc_device<uint16_t>((size_t)B * Mm * Nn, q);
        uint16_t* D1 = sycl::malloc_device<uint16_t>((size_t)B * Mm * Nn, q);
        std::vector<uint16_t> Ah2((size_t)B * Mm * Kk), Wh2((size_t)B * Kk * Nn);
        for (auto& v : Ah2) v = f32_to_bf16_bits((float)(std::rand() % 2000 - 1000) / 999.0f);
        for (auto& v : Wh2) v = f32_to_bf16_bits((float)(std::rand() % 2000 - 1000) / 999.0f);
        q.memcpy(A2, Ah2.data(), Ah2.size() * 2).wait();
        q.memcpy(W2, Wh2.data(), Wh2.size() * 2).wait();
        std::printf("[2] host init done\n");
        auto run2 = [&](uint16_t* dst) {
            prim2.execute(stream, {
                {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({B, Mm, Kk}, dt::bf16,
                        dnnl::memory::dims{(long long)Mm * Kk, Kk, 1}), eng,
                    dnnl::sycl_interop::memory_kind::usm, A2)},
                {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({B, Kk, Nn}, dt::bf16,
                        dnnl::memory::dims{1024, 1, 8 * 1024}), eng,
                    dnnl::sycl_interop::memory_kind::usm, W2)},
                {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
                    dnnl::memory::desc({B, Mm, Nn}, dt::bf16,
                        dnnl::memory::dims{(long long)Mm * Nn, Nn, 1}), eng,
                    dnnl::sycl_interop::memory_kind::usm, dst)}});
            stream.wait();
        };
        std::printf("[2] warm runs...\n");
        run2(D0); run2(D0); run2(D0);
        std::vector<uint16_t> s0((size_t)B * Mm * Nn), s1((size_t)B * Mm * Nn);
        q.memcpy(s0.data(), D0, s0.size() * 2).wait();
        run2(D1);
        q.memcpy(s1.data(), D1, s1.size() * 2).wait();
        report("bf16 batched run4-vs-run5", s0, s1);
        sycl::free(A2, q); sycl::free(W2, q); sycl::free(D0, q); sycl::free(D1, q);
        std::printf("PROBE DONE\n");
    } catch (const std::exception& e) {
        std::printf("PROBE EXCEPTION: %s\n", e.what());
        return 1;
    }
    return 0;
}

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

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        auto eng = dnnl::engine(dnnl::engine::kind::gpu, 0);
        auto dev = dnnl::sycl_interop::get_device(eng);
        auto sctx = dnnl::sycl_interop::get_context(eng);
        sycl::queue q(sctx, dev, sycl::property::queue::in_order{});
        auto stream = dnnl::sycl_interop::make_stream(eng, q);
        std::srand(7);
        struct Shape { int M, K, N, G; const char* name; };
        Shape shapes[] = {
            {256, 2816, 16384, 32, "sliding fused qkv"},
            {256, 2816, 8192, 32,  "full q"},
            {256, 2816, 4096, 32,  "full k (and sliding k/v)"},
            {256, 4096, 2816, 32,  "sliding o_proj"},
            {256, 8192, 2816, 32,  "full o_proj"},
        };
        for (auto& sh : shapes) {
            dnnl::primitive_attr attr;
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {sh.G, 1}, dt::bf16);
            attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
            dnnl::matmul::primitive_desc pd(eng,
                dnnl::memory::desc({sh.M, sh.K}, dt::bf16, tag::ab),
                dnnl::memory::desc({sh.K, sh.N}, dt::s4, tag::ba),
                dnnl::memory::desc({sh.M, sh.N}, dt::bf16, tag::ab), attr);
            auto prim = dnnl::matmul(pd);
            size_t MN = (size_t)sh.M * sh.N;
            uint16_t* A  = sycl::malloc_device<uint16_t>((size_t)sh.M * sh.K, q);
            uint8_t*  W  = sycl::malloc_device<uint8_t>((size_t)sh.K * sh.N / 2, q);
            uint16_t* S  = sycl::malloc_device<uint16_t>((size_t)(sh.K / sh.G) * sh.N, q);
            uint16_t* C0 = sycl::malloc_device<uint16_t>(MN, q);
            uint16_t* C1 = sycl::malloc_device<uint16_t>(MN, q);
            std::vector<uint16_t> Ah((size_t)sh.M * sh.K), Sh((size_t)(sh.K / sh.G) * sh.N);
            std::vector<uint8_t> Wh((size_t)sh.K * sh.N / 2);
            for (auto& v : Ah) v = f32_to_bf16_bits((float)(std::rand() % 2000 - 1000) / 511.0f);
            for (auto& v : Sh) v = f32_to_bf16_bits(0.001f + (float)(std::rand() % 1000) / 1000.0f);
            for (auto& v : Wh) v = (uint8_t)(std::rand() % 256);
            q.memcpy(A, Ah.data(), Ah.size() * 2).wait();
            q.memcpy(S, Sh.data(), Sh.size() * 2).wait();
            q.memcpy(W, Wh.data(), Wh.size()).wait();
            auto run = [&](uint16_t* dst) {
                prim.execute(stream, {
                    {DNNL_ARG_SRC, dnnl::sycl_interop::make_memory(
                        dnnl::memory::desc({sh.M, sh.K}, dt::bf16, tag::ab), eng,
                        dnnl::sycl_interop::memory_kind::usm, A)},
                    {DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
                        dnnl::memory::desc({sh.K, sh.N}, dt::s4, tag::ba), eng,
                        dnnl::sycl_interop::memory_kind::usm, W)},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::sycl_interop::make_memory(
                        dnnl::memory::desc({(dnnl_dim_t)(sh.K / sh.G), sh.N}, dt::bf16, tag::ab), eng,
                        dnnl::sycl_interop::memory_kind::usm, S)},
                    {DNNL_ARG_DST, dnnl::sycl_interop::make_memory(
                        dnnl::memory::desc({sh.M, sh.N}, dt::bf16, tag::ab), eng,
                        dnnl::sycl_interop::memory_kind::usm, dst)}});
                stream.wait();
            };
            run(C0); run(C0); run(C0); run(C0);
            std::vector<uint16_t> r0(MN), r1(MN);
            q.memcpy(r0.data(), C0, MN * 2).wait();
            run(C1);
            q.memcpy(r1.data(), C1, MN * 2).wait();
            size_t bad = 0;
            for (size_t i = 0; i < MN; ++i) if (r0[i] != r1[i]) {
                if (bad < 3) std::printf("   idx=%zu 0x%04x vs 0x%04x\n", i, r0[i], r1[i]);
                ++bad;
            }
            std::printf("%-24s K=%5d N=%5d impl=%s scratch=%zu mismatched=%zu/%zu\n",
                        sh.name, sh.K, sh.N, pd.impl_info_str(),
                        pd.scratchpad_desc().get_size(), bad, MN);
            sycl::free(A, q); sycl::free(W, q); sycl::free(S, q);
            sycl::free(C0, q); sycl::free(C1, q);
        }
        std::printf("PROBE DONE\n");
    } catch (const std::exception& e) {
        std::printf("PROBE EXCEPTION: %s\n", e.what());
        return 1;
    }
    return 0;
}

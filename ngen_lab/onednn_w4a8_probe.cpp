// oneDNN W4A8 (s8 activations x s4 weights -> f32/bf16) throughput probe.
//
// Answers: does the container's oneDNN 3.13 have a working s8 x s4 matmul
// primitive, does it hit the int-pipe DPAS fast path, and WHICH activation-scale
// granularities gemmstone accepts (per-tensor is trivial; the production g128
// path needs per-(row,128-group), which oneDNN may reject at pd.cpp:373).
//
//   C[M,N] = (A_s8 * A_scale) @ (W_s4 * W_scale[g,n])
//
// src=s8 {M,K} tag::ab, wei=s4 {K,N} tag::ba (Arcaine nibble-packed layout),
// dst=f32 or bf16 {M,N} tag::ab. Weight scales are per-K-group (group=32),
// mirroring matmul_int4.
//
// W4A8_SRC_SCALE (activation scale granularity):
//   0 = per-tensor      mask 0  shape {}
//   1 = per-row         mask 1  shape {M}
//   2 = per-row-bcast   mask 3  shape {M,1}     (vLLM w4a8 shim style)
//   3 = per-K-group     mask 2  shape {G128}    (one scale per 128-K block)
//   4 = per-row-group   mask 3  shape {M,G128}  (g128 production target)
//   G128 = K/128 (must divide K)
//
// Env:
//   W4A8_M / W4A8_N / W4A8_K   GEMM shape (default 1024/16384/5120)
//   W4A8_SRC_SCALE             see above (default 0)
//   W4A8_DST                   f32 (default) | bf16
//   W4A8_REPS                  timed samples (best kept, default 20)
//
// Build (inside container):
//   icpx -fsycl -I/opt/onednn/include onednn_w4a8_probe.cpp \
//        -L/opt/onednn/lib -ldnnl -o onednn_w4a8_probe

#include <dnnl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static uint32_t envU32(const char *name, uint32_t dflt) {
    const char *v = std::getenv(name);
    return v ? uint32_t(std::atoi(v)) : dflt;
}

int main() {
    using namespace dnnl;
    using dt = memory::data_type;
    using tag = memory::format_tag;

    const int M = int(envU32("W4A8_M", 1024));
    const int N = int(envU32("W4A8_N", 16384));
    const int K = int(envU32("W4A8_K", 5120));
    const int group = int(envU32("W4A8_GROUP", 32));
    const int mode = int(envU32("W4A8_SRC_SCALE", 0));
    const int wlayout = int(envU32("W4A8_WLAYOUT", 0));   // 0=ba (raw), 1=any (blocked)
    const std::string dst_t = std::getenv("W4A8_DST") ? std::getenv("W4A8_DST") : "f32";
    const int reps = int(envU32("W4A8_REPS", 20));

    if (K % group != 0 || K % 2 != 0 || K % 128 != 0) {
        std::fprintf(stderr, "need K%%group==0 and K%%2==0 and K%%128==0\n");
        return 1;
    }
    const int G = K / group;     // weight scale groups (group=32)
    const int G128 = K / 128;    // activation scale groups (g128)

    const char *mode_name[] = {"per-tensor", "per-row", "per-row-bcast",
            "per-K-group", "per-row-group"};

    std::printf("onednn %d.%d.%d  s8 x s4 -> %s  M=%d N=%d K=%d  src_scale=%s(mode %d)\n",
            DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH,
            dst_t.c_str(), M, N, K, mode_name[mode], mode);

    engine eng(engine::kind::gpu, 0);
    stream s(eng);

    dt dst_dt = (dst_t == "bf16") ? dt::bf16
              : (dst_t == "f16") ? dt::f16
              : dt::f32;

    auto src_md = memory::desc({M, K}, dt::s8, tag::ab);
    auto wei_md = (wlayout == 1)
            ? memory::desc({K, N}, dt::s4, tag::any)
            : memory::desc({K, N}, dt::s4, tag::ba);
    auto dst_md = memory::desc({M, N}, dst_dt, tag::ab);

    // Activation scales (mask, shape, element count) per mode.
    int as_mask = 0;
    std::vector<int64_t> as_shape;
    size_t nAS = 1;
    switch (mode) {
        case 0: as_mask = 0; as_shape = {};            nAS = 1;      break;
        case 1: as_mask = 1; as_shape = {M};           nAS = M;      break;
        case 2: as_mask = 3; as_shape = {M, 1};        nAS = M;      break;
        case 3: as_mask = 2; as_shape = {G128};        nAS = G128;   break;
        case 4: as_mask = 3; as_shape = {M, G128};     nAS = (size_t)M * G128; break;
    }

    primitive_attr attr;
    attr.set_scales(DNNL_ARG_SRC, as_mask, as_shape, dt::f32);
    // Per-group weight scales along K (group along dim 0, per-channel on N).
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {group, 1}, dt::f32);

    auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
    auto prim = matmul(pd);

    // ---- host data ---------------------------------------------------------
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> distA(-127, 127);   // activations s8
    std::uniform_int_distribution<int> nib(0, 15);          // weights s4
    std::uniform_real_distribution<float> distAS(0.005f, 0.02f);
    std::uniform_real_distribution<float> distWS(5e-4f, 2e-3f);

    std::vector<int8_t> hA((size_t)M * K);
    for (auto &v : hA) v = int8_t(distA(rng));

    std::vector<uint8_t> hW(pd.weights_desc().get_size());
    for (auto &b : hW) b = uint8_t(nib(rng) | (nib(rng) << 4));

    std::vector<float> hAS(nAS);
    for (auto &v : hAS) v = distAS(rng);

    std::vector<float> hWS((size_t)G * N);
    for (auto &v : hWS) v = distWS(rng);

    // ---- upload ------------------------------------------------------------
    memory src(pd.src_desc(), eng), wei(pd.weights_desc(), eng),
            dst(pd.dst_desc(), eng);
    {
        auto *p = src.map_data<int8_t>();
        std::memcpy(p, hA.data(), hA.size());
        src.unmap_data(p);
    }
    {
        auto *p = wei.map_data<uint8_t>();
        std::memcpy(p, hW.data(), hW.size());
        wei.unmap_data(p);
    }

    memory::dims as_dims;
    if (as_shape.empty()) as_dims = {1};
    else as_dims = as_shape;
    memory as(memory::desc(as_dims, dt::f32, as_dims.size() == 1 ? tag::a : tag::ab), eng);
    {
        auto *p = as.map_data<float>();
        std::memcpy(p, hAS.data(), hAS.size() * 4);
        as.unmap_data(p);
    }
    auto ws_md = memory::desc({G, N}, dt::f32, tag::ab);
    memory ws(ws_md, eng);
    {
        auto *p = ws.map_data<float>();
        std::memcpy(p, hWS.data(), hWS.size() * 4);
        ws.unmap_data(p);
    }

    // ---- execute + time ----------------------------------------------------
    const double flops = 2.0 * double(M) * N * K;
    double best = 1e30;
    for (int r = 0; r < reps + 2; r++) {
        auto t0 = std::chrono::steady_clock::now();
        prim.execute(s, {{DNNL_ARG_SRC, src},
                                {DNNL_ARG_WEIGHTS, wei},
                                {DNNL_ARG_DST, dst},
                                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, as},
                                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, ws}});
        s.wait();
        auto t1 = std::chrono::steady_clock::now();
        if (r >= 2)
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    std::printf("[perf] w4a8 M=%d N=%d K=%d dst=%s  %8.3f ms  %7.2f TFLOP/s\n",
            M, N, K, dst_t.c_str(), best * 1e3, flops / best / 1e12);
    return 0;
}

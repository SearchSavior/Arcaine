// src/nvfp4_roofline_bench.cpp
//
// NVFP4 expert-kernel roofline benchmark (random weights, no model).
//
// Measures the production NVFP4 expert GEMM kernels against a device roofline
// across the expert M regime, using random NVFP4 weights (no model file, no
// loader, no diffusion forward pass). Default kernel = onednn-loop, the hybrid
// baseline (E separate matmul_nvfp4_packed calls). The hand-rolled xe2 / xe2v2 /
// custom grouped kernels and a dense oneDNN reference are wired behind --kernels
// so they can be A/B'd later with no code change.
//
// Kernels are called DIRECTLY (not via run_shard), so the production dispatch
// env knobs (DIFF_NVFP4_GROUPED_GEMM, DIFF_NVFP4_EXPERT_KERNEL,
// DIFF_NVFP4_GPU_LAYOUT) do NOT select kernels here — --kernels does. The
// --kernels default is taken from DIFF_NVFP4_ROOFLINE_KERNELS if set, else
// onednn-loop. DIFF_NVFP4_WEIGHT_LAYOUT must stay unset (Raw) so random weights
// are not oneDNN-reordered.
//
// Build:
//   cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx \
//         -DARCAINE_SYCL_TARGETS=intel_gpu_bmg_g31
//   cmake --build build --target nvfp4_roofline_bench
//
// Reference roofline (default), both shapes, standard sweep:
//   ZE_AFFINITY_MASK=0 ./build/nvfp4_roofline_bench -p 512,1024,2048,8192
//
// A/B the hand-rolled kernels against the reference:
//   ZE_AFFINITY_MASK=0 ./build/nvfp4_roofline_bench \
//       --kernels onednn-loop,xe2v2,xe2,custom -p 512,1024,2048,8192
//
// Correctness wiring check (loose; random weights — catches pointer/stride bugs):
//   ZE_AFFINITY_MASK=0 ./build/nvfp4_roofline_bench \
//       --kernels onednn-loop,xe2v2,custom --check -p 512 --experts 4 --shapes gateup

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/gpu/buffer.hpp"
#include "common/gpu/device_select.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/nvfp4.hpp"

namespace {

using bf16 = uint16_t;  // mirrors buffer.hpp

enum class Kernel { OnednnLoop, OnednnBatched, OnednnDense, Xe2, Xe2v2, Xe2v3, Xe2v4, Xe2v5, Xe2v6, Xe2v7, Custom };

Kernel parse_kernel(const std::string& s) {
    if (s == "onednn-loop")    return Kernel::OnednnLoop;
    if (s == "onednn-batched") return Kernel::OnednnBatched;
    if (s == "onednn-dense")   return Kernel::OnednnDense;
    if (s == "xe2")            return Kernel::Xe2;
    if (s == "xe2v2")          return Kernel::Xe2v2;
    if (s == "xe2v3")          return Kernel::Xe2v3;
    if (s == "xe2v4")          return Kernel::Xe2v4;
    if (s == "xe2v5")          return Kernel::Xe2v5;
    if (s == "xe2v6")          return Kernel::Xe2v6;
    if (s == "xe2v7")          return Kernel::Xe2v7;
    if (s == "custom")         return Kernel::Custom;
    throw std::runtime_error("unknown kernel '" + s +
        "' (use: onednn-loop, onednn-batched, onednn-dense, xe2, xe2v2, xe2v3, xe2v4, xe2v5, xe2v6, xe2v7, custom)");
}
const char* kernel_name(Kernel k) {
    switch (k) {
        case Kernel::OnednnLoop:    return "onednn-loop";
        case Kernel::OnednnBatched: return "onednn-batched";
        case Kernel::OnednnDense:   return "onednn-dense";
        case Kernel::Xe2:           return "xe2";
        case Kernel::Xe2v2:         return "xe2v2";
        case Kernel::Xe2v3:         return "xe2v3";
        case Kernel::Xe2v4:         return "xe2v4";
        case Kernel::Xe2v5:         return "xe2v5";
        case Kernel::Xe2v6:         return "xe2v6";
        case Kernel::Xe2v7:         return "xe2v7";
        case Kernel::Custom:         return "custom";
    }
    return "?";
}

struct Shape { std::string name; int K, N; };

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out; size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}
std::vector<int> parse_int_csv(const std::string& s) {
    std::vector<int> out; for (const auto& t : split_csv(s)) out.push_back(std::stoi(t)); return out;
}
std::vector<Shape> parse_shapes(const std::string& s) {
    std::vector<Shape> out;
    for (const auto& tok : split_csv(s)) {
        if (tok == "gateup") out.push_back({"gateup", 2816, 1408});
        else if (tok == "down") out.push_back({"down", 704, 2816});
        else {
            auto c = tok.find_first_of(":x");
            if (c == std::string::npos)
                throw std::runtime_error("bad shape '" + tok + "' (use gateup, down, or K:N)");
            out.push_back({tok, std::stoi(tok.substr(0, c)), std::stoi(tok.substr(c + 1))});
        }
    }
    return out;
}

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -p <csv>          total rows across experts (default 512,1024,2048,8192)\n"
        "  --kernels <csv>   onednn-loop|onednn-dense|xe2|xe2v2|custom (default onednn-loop,\n"
        "                    or $DIFF_NVFP4_ROOFLINE_KERNELS if set)\n"
        "  --shapes <csv>    gateup|down|K:N (default gateup,down)\n"
        "  --experts <E>     expert count (default 128)\n"
        "  --iters <N>       timed iterations after warmup (default 50)\n"
        "  --warmup <N>      warmup iterations (default 1)\n"
        "  --peak-tflops <F> device peak TFLOP/s for %%peak (default 160)\n"
        "  --peak-gbps <F>   device peak GB/s for %%peak (default 456)\n"
        "  --check           correctness vs onednn-loop ref (loose; random weights)\n"
        "  --md              markdown table output\n"
        "  --csv             CSV output\n"
        "  --seed <N>        RNG seed (default 42)\n"
        "  -h, --help        show this help\n", p);
}

struct Row {
    std::string kernel, shape; int K, N, m_per, total;
    double ms, tflops, gbps, pct_tf, pct_bw;
};

}  // namespace

int main(int argc, char** argv) {
    std::string p_csv = "512,1024,2048,8192";
    std::string kernels_csv;
    std::string shapes_csv = "gateup,down";
    int E = 128;
    int iters = 50, warmup = 1;
    double peak_tflops = 160.0, peak_gbps = 456.0;
    bool check = false, md = false, csv = false;
    unsigned seed = 42;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if      (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
            else if (a == "-p")              p_csv = next();
            else if (a == "--kernels")       kernels_csv = next();
            else if (a == "--shapes")        shapes_csv = next();
            else if (a == "--experts")       E = std::stoi(next());
            else if (a == "--iters")         iters = std::stoi(next());
            else if (a == "--warmup")        warmup = std::stoi(next());
            else if (a == "--peak-tflops")   peak_tflops = std::stod(next());
            else if (a == "--peak-gbps")     peak_gbps = std::stod(next());
            else if (a == "--seed")          seed = (unsigned)std::stoul(next());
            else if (a == "--check")         check = true;
            else if (a == "--md")            md = true;
            else if (a == "--csv")           csv = true;
            else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
        }
    } catch (std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    if (iters < 1) iters = 1;
    if (warmup < 0) warmup = 0;

    if (kernels_csv.empty()) {
        if (const char* env = std::getenv("DIFF_NVFP4_ROOFLINE_KERNELS"))
            kernels_csv = env;
        else
            kernels_csv = "onednn-loop";
    }
    if (const char* lay = std::getenv("DIFF_NVFP4_WEIGHT_LAYOUT"))
        if (std::string(lay) == "any" || std::string(lay) == "xe" || std::string(lay) == "reorder")
            std::fprintf(stderr,
                "[nvfp4-roofline] WARNING: DIFF_NVFP4_WEIGHT_LAYOUT=%s is set; random weights\n"
                "                will be oneDNN-reordered. Unset it for a clean Raw roofline.\n",
                lay);

    std::vector<Kernel> selected;
    for (const auto& k : split_csv(kernels_csv)) selected.push_back(parse_kernel(k));
    auto p_list = parse_int_csv(p_csv);
    auto shapes = parse_shapes(shapes_csv);

    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;

    std::string dev_name = ctx.queue.get_device().get_info<sycl::info::device::name>();
    while (!dev_name.empty() && std::isspace((unsigned char)dev_name.back())) dev_name.pop_back();

    std::printf("[nvfp4-roofline] device: %s | E=%d | peaks: %.1f TFLOP/s | %.1f GB/s\n",
                dev_name.c_str(), E, peak_tflops, peak_gbps);
    if (const char* aff = gpu_device_control::active_gpus_spec())
        std::printf("[nvfp4-roofline] ZE_AFFINITY_MASK=%s\n", aff);
    std::printf("[nvfp4-roofline] kernels: %s | shapes: %s | iters:%d warmup:%d | check:%s\n",
                kernels_csv.c_str(), shapes_csv.c_str(), iters, warmup, check ? "on" : "off");

    if (md) {
        std::printf("| kernel | shape | K | N | M/per | total | ms/iter | TFLOP/s | GB/s | %%peakTF | %%peakBW |\n");
        std::printf("|---|---|---|---|---|---|---|---|---|---|---|\n");
    } else if (csv) {
        std::printf("kernel,shape,K,N,M_per,total,ms_iter,tflops,gbps,pct_peak_tf,pct_peak_bw\n");
    } else {
        std::printf("%-12s %-7s %5s %5s %6s %6s %9s %9s %8s %8s %8s\n",
                    "kernel", "shape", "K", "N", "M/per", "total",
                    "ms/iter", "TFLOP/s", "GB/s", "%peakTF", "%peakBW");
    }

    bool need_grouped = false, need_coal = false, need_batched = false;
    for (auto k : selected) {
        if (k == Kernel::Xe2 || k == Kernel::Xe2v2 || k == Kernel::Xe2v3 || k == Kernel::Xe2v4 || k == Kernel::Xe2v5 || k == Kernel::Xe2v6 || k == Kernel::Xe2v7 || k == Kernel::Custom) need_grouped = true;
        if (k == Kernel::Xe2 || k == Kernel::Xe2v3 || k == Kernel::Xe2v4 || k == Kernel::Xe2v5 || k == Kernel::Xe2v6 || k == Kernel::Xe2v7) need_coal = true;
        if (k == Kernel::OnednnBatched) need_batched = true;
    }

    for (const auto& sh : shapes) {
        const int K = sh.K, N = sh.N;
        const int G = K / 16;

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> nib(0, 15), sc(40, 110);
        std::uniform_real_distribution<float> xf(-1.0f, 1.0f);

        // ---- E random Nvfp4Linear weights (Raw layout) ----
        std::vector<Nvfp4Linear> W(E);
        const size_t wp = (size_t)N * K / 2;
        const size_t ws = (size_t)G * (size_t)N;
        std::vector<uint8_t> hwp(wp), hws(ws);
        for (int e = 0; e < E; ++e) {
            for (auto& b : hwp) b = (uint8_t)(nib(rng) | (nib(rng) << 4));
            for (auto& b : hws) b = (uint8_t)sc(rng);
            W[e].in_features = K;
            W[e].out_features = N;
            W[e].input_global_scale = 1.0f;
            W[e].weight_global_scale = 1.0f;
            W[e].weight_packed = GpuBuffer<uint8_t>(wp, q);
            W[e].weight_packed.upload(hwp.data(), wp);
            W[e].weight_scale = GpuBuffer<uint8_t>(ws, q);
            W[e].weight_scale.upload(hws.data(), ws);
            W[e].dst_scale = GpuBuffer<float>(1, q);
            const float one = 1.0f;
            W[e].dst_scale.upload(&one, 1);
        }

        // ---- per-expert device pointer arrays (grouped kernels) ----
        GpuBuffer<const uint8_t*> W_packed_dev, W_scale_dev, W_coal_dev;
        GpuBuffer<const float*> dst_scale_dev;
        if (need_grouped) {
            std::vector<const uint8_t*> wp_p(E), ws_p(E);
            std::vector<const float*> ds_p(E);
            for (int e = 0; e < E; ++e) {
                wp_p[e] = W[e].weight_packed.data();
                ws_p[e] = W[e].weight_scale.data();
                ds_p[e] = W[e].dst_scale.data();
            }
            W_packed_dev = GpuBuffer<const uint8_t*>(E, q); W_packed_dev.upload(wp_p.data(), E);
            W_scale_dev  = GpuBuffer<const uint8_t*>(E, q); W_scale_dev.upload(ws_p.data(), E);
            dst_scale_dev = GpuBuffer<const float*>(E, q);  dst_scale_dev.upload(ds_p.data(), E);
            if (need_coal) {
                std::vector<const uint8_t*> wc_p(E);
                for (int e = 0; e < E; ++e)
                    wc_p[e] = nvfp4_coalesced_weight(W[e], K, N, ctx);
                W_coal_dev = GpuBuffer<const uint8_t*>(E, q); W_coal_dev.upload(wc_p.data(), E);
            }
        }

        // ---- batched weight buffers [E,K,N] f4 (tag::acb, K-contiguous per
        // batch, matching the single kernel's tag::ba on {K,N}) + [E,G,N] f8
        // (tag::abc, N-contiguous). Built by concatenating the per-expert packed
        // weights (each [N,K] f4, K-contiguous) end-to-end. Shared dst_scale =
        // W[0].dst_scale. ----
        GpuBuffer<uint8_t> W_batch, W_scale_batch;
        if (need_batched) {
            W_batch = GpuBuffer<uint8_t>((size_t)E * wp, q);
            W_scale_batch = GpuBuffer<uint8_t>((size_t)E * ws, q);
            for (int e = 0; e < E; ++e) {
                W[e].weight_packed.download(hwp.data(), wp);
                W[e].weight_scale.download(hws.data(), ws);
                q.memcpy(W_batch.data() + (size_t)e * wp, hwp.data(), wp);
                q.memcpy(W_scale_batch.data() + (size_t)e * ws, hws.data(), ws);
            }
            q.wait();
        }

        for (int total_rows : p_list) {
            if (total_rows < E) {
                std::fprintf(stderr,
                    "[nvfp4-roofline] p=%d < E=%d; skipped (need p>=E so every expert gets >=1 row)\n",
                    total_rows, E);
                continue;
            }

            // ---- even, contiguous-per-expert row distribution ----
            std::vector<int32_t> rows_per_expert(E), expert_offsets(E);
            const int base = total_rows / E, rem = total_rows % E;
            int run = 0;
            for (int e = 0; e < E; ++e) {
                rows_per_expert[e] = base + (e < rem ? 1 : 0);
                expert_offsets[e] = run;
                run += rows_per_expert[e];
            }
            std::vector<int32_t> row_expert(total_rows);
            for (int e = 0; e < E; ++e)
                for (int r = 0; r < rows_per_expert[e]; ++r)
                    row_expert[expert_offsets[e] + r] = e;

            // ---- random bf16 activations ----
            std::vector<bf16> hX((size_t)total_rows * K);
            for (auto& v : hX) v = float_to_bf16(xf(rng));
            GpuBuffer<bf16> X((size_t)total_rows * K, q);
            X.upload(hX.data(), hX.size());

            std::vector<float> hinp(E, 1.0f);
            GpuBuffer<float> input_scale_dev(E, q); input_scale_dev.upload(hinp.data(), E);
            GpuBuffer<int32_t> row_expert_dev(total_rows, q);
            row_expert_dev.upload(row_expert.data(), total_rows);
            GpuBuffer<int32_t> expert_offsets_dev(E, q);
            expert_offsets_dev.upload(expert_offsets.data(), E);
            GpuBuffer<int32_t> rows_per_expert_dev(E, q);
            rows_per_expert_dev.upload(rows_per_expert.data(), E);

            // ---- pack activations (excluded from timing) ----
            GpuBuffer<uint8_t> A_packed((size_t)total_rows * K / 2, q);
            GpuBuffer<uint8_t> A_scale((size_t)total_rows * G, q);
            pack_bf16_to_nvfp4_grouped_rows(q, X.data(), K, row_expert_dev.data(),
                                            total_rows, input_scale_dev.data(),
                                            A_packed.data(), A_scale.data());
            q.wait();

            GpuBuffer<bf16> C((size_t)total_rows * N, q);

            int max_mt = 1;
            for (int e = 0; e < E; ++e) {
                int mt = (rows_per_expert[e] + 7) / 8;
                if (mt > max_mt) max_mt = mt;
            }

            auto launch = [&](Kernel k) {
                switch (k) {
                    case Kernel::OnednnLoop:
                        for (int e = 0; e < E; ++e) {
                            if (rows_per_expert[e] == 0) continue;
                            const int off = expert_offsets[e];
                            matmul_nvfp4_packed(A_packed.data() + (size_t)off * (K / 2),
                                                A_scale.data() + (size_t)off * G,
                                                rows_per_expert[e], K, W[e],
                                                C.data() + (size_t)off * N, ctx);
                        }
                        break;
                    case Kernel::OnednnDense:
                        matmul_nvfp4_packed(A_packed.data(), A_scale.data(),
                                            total_rows, K, W[0], C.data(), ctx);
                        break;
                    case Kernel::OnednnBatched: {
                        // oneDNN batched matmul needs a uniform M per batch. The bench's
                        // even row split gives that only when total_rows is divisible by E;
                        // the standard sweep (-p 512,1024,2048,8192, E=128) always is.
                        if (total_rows % E != 0)
                            throw std::runtime_error(
                                "onednn-batched requires total rows divisible by --experts "
                                "(uniform M per batch)");
                        const int M_batch = total_rows / E;
                        matmul_nvfp4_packed_batched(A_packed.data(), A_scale.data(),
                                                    E, M_batch, K,
                                                    W_batch.data(), W_scale_batch.data(),
                                                    W[0].dst_scale.data(),
                                                    N, C.data(), ctx);
                        break;
                    }
                    case Kernel::Xe2:
                        matmul_nvfp4_grouped_rows_xe2(ctx, A_packed.data(), A_scale.data(), K,
                                                      row_expert_dev.data(), total_rows,
                                                      W_coal_dev.data(), W_scale_dev.data(),
                                                      dst_scale_dev.data(), N, C.data());
                        break;
                    case Kernel::Xe2v2:
                        matmul_nvfp4_grouped_rows_xe2_v2(ctx, A_packed.data(), A_scale.data(), K,
                                                         expert_offsets_dev.data(),
                                                         rows_per_expert_dev.data(), E,
                                                         W_packed_dev.data(), W_scale_dev.data(),
                                                         dst_scale_dev.data(), N, C.data());
                        break;
                    case Kernel::Xe2v3:
                        matmul_nvfp4_grouped_rows_xe2_v3(ctx, A_packed.data(), A_scale.data(), K,
                                                         expert_offsets_dev.data(),
                                                         rows_per_expert_dev.data(), E,
                                                         W_coal_dev.data(), W_scale_dev.data(),
                                                         dst_scale_dev.data(), N, C.data());
                        break;
                    case Kernel::Xe2v4:
                        matmul_nvfp4_grouped_rows_xe2_v4(ctx, A_packed.data(), A_scale.data(), K,
                                                         expert_offsets_dev.data(),
                                                         rows_per_expert_dev.data(), E,
                                                         W_coal_dev.data(), W_scale_dev.data(),
                                                         dst_scale_dev.data(), N, C.data());
                        break;
                    case Kernel::Xe2v5:
                        matmul_nvfp4_grouped_rows_xe2_v5(ctx, A_packed.data(), A_scale.data(), K,
                                                         expert_offsets_dev.data(),
                                                         rows_per_expert_dev.data(), E,
                                                         W_coal_dev.data(), W_scale_dev.data(),
                                                         dst_scale_dev.data(), N, C.data());
                        break;
                    case Kernel::Xe2v6:
                        matmul_nvfp4_grouped_rows_xe2_v6(ctx, A_packed.data(), A_scale.data(), K,
                                                         expert_offsets_dev.data(),
                                                         rows_per_expert_dev.data(), E,
                                                         max_mt,
                                                         W_coal_dev.data(), W_scale_dev.data(),
                                                         dst_scale_dev.data(), N, C.data());
                        break;
                    case Kernel::Xe2v7:
                        matmul_nvfp4_grouped_rows_xe2_v7(ctx, A_packed.data(), A_scale.data(), K,
                                                         expert_offsets_dev.data(),
                                                         rows_per_expert_dev.data(), E,
                                                         W_coal_dev.data(), W_scale_dev.data(),
                                                         dst_scale_dev.data(), N, C.data());
                        break;
                    case Kernel::Custom:
                        matmul_nvfp4_grouped_rows_custom(q, A_packed.data(), A_scale.data(), K,
                                                         row_expert_dev.data(), total_rows,
                                                         W_packed_dev.data(), W_scale_dev.data(),
                                                         dst_scale_dev.data(), N, C.data());
                        break;
                }
            };

            auto timeit = [&](Kernel k) -> double {
                for (int w = 0; w < warmup; ++w) launch(k);
                q.wait();
                auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < iters; ++it) launch(k);
                q.wait();
                auto t1 = std::chrono::steady_clock::now();
                return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
            };

            auto roofline = [&](Kernel k, double ms) -> Row {
                Row r;
                const double flops = 2.0 * (double)total_rows * K * N;
                const int we = (k == Kernel::OnednnDense) ? 1 : E;
                const double w_bytes = (double)we * ((double)N * K / 2 + (double)G * N);
                const double a_bytes = (double)total_rows * ((double)K / 2 + (double)G);
                const double o_bytes = (double)total_rows * N * 2;
                const double bytes = w_bytes + a_bytes + o_bytes;
                const double sec = ms / 1000.0;
                r.kernel = kernel_name(k); r.shape = sh.name;
                r.K = K; r.N = N; r.m_per = total_rows / E; r.total = total_rows;
                r.ms = ms;
                r.tflops = sec > 0 ? flops / sec / 1e12 : 0.0;
                r.gbps   = sec > 0 ? bytes / sec / 1e9 : 0.0;
                r.pct_tf = peak_tflops > 0 ? 100.0 * r.tflops / peak_tflops : 0.0;
                r.pct_bw = peak_gbps > 0 ? 100.0 * r.gbps / peak_gbps : 0.0;
                return r;
            };

            auto print_row = [&](const Row& r) {
                if (md)
                    std::printf("| %s | %s | %d | %d | %d | %d | %.4f | %.2f | %.2f | %.2f | %.2f |\n",
                        r.kernel.c_str(), r.shape.c_str(), r.K, r.N, r.m_per, r.total,
                        r.ms, r.tflops, r.gbps, r.pct_tf, r.pct_bw);
                else if (csv)
                    std::printf("%s,%s,%d,%d,%d,%d,%.4f,%.2f,%.2f,%.2f,%.2f\n",
                        r.kernel.c_str(), r.shape.c_str(), r.K, r.N, r.m_per, r.total,
                        r.ms, r.tflops, r.gbps, r.pct_tf, r.pct_bw);
                else
                    std::printf("%-12s %-7s %5d %5d %6d %6d %9.4f %9.2f %8.2f %8.2f %8.2f\n",
                        r.kernel.c_str(), r.shape.c_str(), r.K, r.N, r.m_per, r.total,
                        r.ms, r.tflops, r.gbps, r.pct_tf, r.pct_bw);
            };

            std::vector<Row> rows;
            std::vector<std::string> check_lines;
            std::vector<bf16> host_ref, host_cur;

            if (check) {
                rows.push_back(roofline(Kernel::OnednnLoop, timeit(Kernel::OnednnLoop)));
                host_ref.resize((size_t)total_rows * N);
                C.download(host_ref.data(), (size_t)total_rows * N);
                for (Kernel k : selected) {
                    if (k == Kernel::OnednnLoop) continue;  // already the reference
                    rows.push_back(roofline(k, timeit(k)));
                    if (k != Kernel::OnednnDense) {
                        host_cur.resize((size_t)total_rows * N);
                        C.download(host_cur.data(), (size_t)total_rows * N);
                        double max_abs = 0.0, max_rel = 0.0;
                        for (size_t i = 0; i < host_cur.size(); ++i) {
                            const float a = bf16_to_float(host_cur[i]);
                            const float b = bf16_to_float(host_ref[i]);
                            const float d = std::fabs(a - b);
                            if ((double)d > max_abs) max_abs = d;
                            const float denom = std::fmax(std::fabs(b), 1e-3f);
                            const double rel = (double)(d / denom);
                            if (rel > max_rel) max_rel = rel;
                        }
                        char buf[256];
                        std::snprintf(buf, sizeof(buf), "[check] %-12s %-7s p=%-6d max_abs=%.4g max_rel=%.4g",
                                      kernel_name(k), sh.name.c_str(), total_rows, max_abs, max_rel);
                        check_lines.push_back(buf);
                    }
                }
            } else {
                for (Kernel k : selected)
                    rows.push_back(roofline(k, timeit(k)));
            }

            for (const auto& r : rows) print_row(r);
            for (const auto& line : check_lines) std::printf("%s\n", line.c_str());
        }  // p
    }  // shape
    return 0;
}

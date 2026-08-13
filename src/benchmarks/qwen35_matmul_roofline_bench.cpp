// src/benchmarks/qwen35_matmul_roofline_bench.cpp
//
// qwen3_5 (dense 27B, AWQ checkpoint) GEMM roofline benchmark (random weights,
// no model). Measures the production oneDNN-backed GEMM paths the architecture
// takes through operators.hpp:
//   int4  -> matmul_int4 (runtime/quantization/int4.hpp), including the
//            asymmetric zero-point side-path (rowsum + bf16 correction GEMM +
//            subtract) that is active on the cyankiwi AWQ checkpoint
//            (group_size=32, symmetric:false).
//   bf16  -> matmul_bf16 (runtime/gpu/ops.hpp) for the bf16-dense projections
//            (in_proj_ba, lm_head on the AWQ checkpoint).
// This is the baseline the proposed ngen s8x4 DPAS port of matmul_int4 is
// measured against (B70 ceilings: ~170 TFLOPS bf16 pipe, ~363 TFLOPS int pipe,
// ~456 GB/s).
//
// Kernels are called DIRECTLY (not via a model), so production dispatch env
// knobs do not select kernels here -- --kernels does. DIFF_INT4_WEIGHT_LAYOUT
// must stay unset (Raw) so random weights are not oneDNN-reordered.
//
// Registered as `qwen35-matmul-roofline` in the unified kernel_bench binary.
//
// Standard sweep (mbench prefill sizes + decode):
//   ZE_AFFINITY_MASK=0 ./build/arcaine_kbench qwen35-matmul-roofline --md
//
// Isolate the asymmetric-ZP side-path cost:
//   ZE_AFFINITY_MASK=0 ./build/arcaine_kbench qwen35-matmul-roofline \
//       --kernels int4 --symmetric

#include "benchmarks/registry.hpp"
#include "benchmarks/util.hpp"

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

#include "runtime/gpu/buffer.hpp"
#include "runtime/gpu/device_select.hpp"
#include "runtime/gpu/engine.hpp"
#include "runtime/gpu/ops.hpp"
#include "runtime/quantization/int4.hpp"

using arcaine::bench::parse_int_csv;
using arcaine::bench::split_csv;

namespace {

enum class Kernel { Int4, Bf16, W4a8 };

Kernel parse_kernel(const std::string& s) {
    if (s == "int4") return Kernel::Int4;
    if (s == "bf16") return Kernel::Bf16;
    if (s == "w4a8") return Kernel::W4a8;
    throw std::runtime_error("unknown kernel '" + s + "' (use: int4, bf16, w4a8)");
}
const char* kernel_name(Kernel k) {
    switch (k) {
        case Kernel::Int4: return "int4";
        case Kernel::Bf16: return "bf16";
        case Kernel::W4a8: return "w4a8";
    }
    return "?";
}

// Production weight shapes of cyankiwi_Qwen3.6-27B-AWQ-INT4
// (H=5120, I=17408, V=248320, Hq=24*dh=256, Hkv=4, Hk=16*dk=128, Hv=48*dv=128).
struct Shape { std::string name; int K, N; };

std::vector<Shape> parse_shapes(const std::string& s) {
    std::vector<Shape> out;
    for (const auto& tok : split_csv(s)) {
        if (tok == "qkv")         out.push_back({"qkv",    5120, 14336});   // fused 2*Hq*dh + 2*Hkv*dh
        else if (tok == "o")      out.push_back({"o",      6144,  5120});   // Hq*dh -> H
        else if (tok == "qkvz")   out.push_back({"qkvz",   5120, 16384});   // conv_dim + Hv*dv
        else if (tok == "out")    out.push_back({"out",    6144,  5120});   // Hv*dv -> H
        else if (tok == "gateup") out.push_back({"gateup", 5120, 34816});   // H -> 2I
        else if (tok == "down")   out.push_back({"down",  17408,  5120});   // I -> H
        else if (tok == "ba")     out.push_back({"ba",     5120,    96});   // H -> 2*Hv (bf16-dense)
        else if (tok == "lmhead") out.push_back({"lmhead", 5120, 248320});  // H -> V (bf16-dense)
        else {
            auto c = tok.find_first_of(":x");
            if (c == std::string::npos)
                throw std::runtime_error("bad shape '" + tok +
                    "' (use qkv, o, qkvz, out, gateup, down, ba, lmhead, or K:N)");
            out.push_back({tok, std::stoi(tok.substr(0, c)), std::stoi(tok.substr(c + 1))});
        }
    }
    return out;
}

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -p <csv>          M values (default 1,512,1024,2048,4096)\n"
        "  --kernels <csv>   int4|bf16|w4a8 (default int4,bf16,w4a8)\n"
        "  --shapes <csv>    qkv|o|qkvz|out|gateup|down|ba|lmhead|K:N (default all)\n"
        "  --group-size <G>  int4 group size (default 32, matching the AWQ ckpt)\n"
        "  --symmetric       int4 without the zp_offset side-path (A/B; ckpt is asymmetric)\n"
        "  --iters <N>       timed iterations after warmup (default 50)\n"
        "  --warmup <N>      warmup iterations (default 5; first call builds the\n"
        "                    oneDNN primitive)\n"
        "  --peak-tflops <F> device peak TFLOP/s for %%peak (default 160)\n"
        "  --peak-gbps <F>   device peak GB/s for %%peak (default 456)\n"
        "  --check           int4 vs dequant+bf16 reference (loose; random weights)\n"
        "  --md              markdown table output\n"
        "  --csv             CSV output\n"
        "  --seed <N>        RNG seed (default 42)\n"
        "  -h, --help        show this help\n", p);
}

struct Row {
    std::string kernel, shape; int K, N, M;
    double ms, tflops, gbps, pct_tf, pct_bw;
};

int run(int argc, char** argv) {
    std::string p_csv = "1,512,1024,2048,4096";
    std::string kernels_csv = "int4,bf16";
    std::string shapes_csv = "qkv,o,qkvz,out,gateup,down,ba,lmhead";
    int group_size = 32;
    bool symmetric = false;
    int iters = 50, warmup = 5;
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
            else if (a == "-p")            p_csv = next();
            else if (a == "--kernels")     kernels_csv = next();
            else if (a == "--shapes")      shapes_csv = next();
            else if (a == "--group-size")  group_size = std::stoi(next());
            else if (a == "--iters")       iters = std::stoi(next());
            else if (a == "--warmup")      warmup = std::stoi(next());
            else if (a == "--peak-tflops") peak_tflops = std::stod(next());
            else if (a == "--peak-gbps")   peak_gbps = std::stod(next());
            else if (a == "--seed")        seed = (unsigned)std::stoul(next());
            else if (a == "--symmetric")   symmetric = true;
            else if (a == "--check")       check = true;
            else if (a == "--md")          md = true;
            else if (a == "--csv")         csv = true;
            else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
        }
    } catch (std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    if (iters < 1) iters = 1;
    if (warmup < 0) warmup = 0;
    if (group_size <= 0) throw std::runtime_error("--group-size must be > 0");

    std::vector<Kernel> selected;
    for (const auto& k : split_csv(kernels_csv)) selected.push_back(parse_kernel(k));
    auto p_list = parse_int_csv(p_csv);
    auto shapes = parse_shapes(shapes_csv);

    if (const char* lay = std::getenv("DIFF_INT4_WEIGHT_LAYOUT"))
        if (std::string(lay) == "any")
            std::fprintf(stderr,
                "[qwen35-matmul-roofline] WARNING: DIFF_INT4_WEIGHT_LAYOUT=%s is set; random\n"
                "                weights will be oneDNN-reordered. Unset it for a clean Raw roofline.\n",
                lay);

    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;

    std::string dev_name = ctx.queue.get_device().get_info<sycl::info::device::name>();
    while (!dev_name.empty() && std::isspace((unsigned char)dev_name.back())) dev_name.pop_back();

    std::printf("[qwen35-matmul-roofline] device: %s | peaks: %.1f TFLOP/s | %.1f GB/s\n",
                dev_name.c_str(), peak_tflops, peak_gbps);
    if (const char* aff = gpu_device_control::active_gpus_spec())
        std::printf("[qwen35-matmul-roofline] ZE_AFFINITY_MASK=%s\n", aff);
    std::printf("[qwen35-matmul-roofline] kernels: %s | shapes: %s | group_size:%d | %s | iters:%d warmup:%d | check:%s\n",
                kernels_csv.c_str(), shapes_csv.c_str(), group_size,
                symmetric ? "symmetric" : "asymmetric(zp)", iters, warmup,
                check ? "on" : "off");

    if (md) {
        std::printf("| kernel | shape | K | N | M | ms/iter | TFLOP/s | GB/s | %%peakTF | %%peakBW |\n");
        std::printf("|---|---|---|---|---|---|---|---|---|---|\n");
    } else if (csv) {
        std::printf("kernel,shape,K,N,M,ms_iter,tflops,gbps,pct_peak_tf,pct_peak_bw\n");
    } else {
        std::printf("%-6s %-7s %6s %7s %5s %10s %9s %9s %8s %8s\n",
                    "kernel", "shape", "K", "N", "M", "ms/iter", "TFLOP/s", "GB/s",
                    "%peakTF", "%peakBW");
    }

    for (const auto& sh : shapes) {
        const int K = sh.K, N = sh.N;
        if (K % 2 != 0 || K % group_size != 0)
            throw std::runtime_error("shape " + sh.name + ": K=" + std::to_string(K) +
                " must be even and divisible by group_size=" + std::to_string(group_size));
        const int G = K / group_size;

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> nib(0, 15);
        std::uniform_real_distribution<float> xf(-1.0f, 1.0f), sf(0.01f, 0.05f),
            zf(-0.02f, 0.02f);

        // ---- random weights ----
        // int4: s4 nibbles (N,K) low-nibble-first + bf16 scales (G,N) + zp (G,N)
        const size_t wp = (size_t)N * K / 2;
        const size_t ws = (size_t)G * N;
        std::vector<uint8_t> hwp(wp);
        for (auto& b : hwp) b = (uint8_t)(nib(rng) | (nib(rng) << 4));
        std::vector<bf16> hws(ws), hzp(ws);
        for (size_t i = 0; i < ws; ++i) {
            hws[i] = float_to_bf16(sf(rng));
            hzp[i] = float_to_bf16(zf(rng));
        }
        Int4Linear W4;
        W4.in_features = K;
        W4.out_features = N;
        W4.group_size = group_size;
        W4.weight_packed = GpuBuffer<uint8_t>(wp, q);
        W4.weight_packed.upload(hwp.data(), wp);
        W4.weight_scale = GpuBuffer<bf16>(ws, q);
        W4.weight_scale.upload(hws.data(), ws);
        if (!symmetric) {
            W4.zp_offset = GpuBuffer<bf16>(ws, q);
            W4.zp_offset.upload(hzp.data(), ws);
        }

        // bf16: dense (N,K) weights
        const size_t wb = (size_t)N * K;
        std::vector<bf16> hwb(wb);
        for (auto& v : hwb) v = float_to_bf16(sf(rng));
        GpuBuffer<bf16> Wbf(wb, q);
        Wbf.upload(hwb.data(), wb);

        for (int M : p_list) {
            std::vector<bf16> hX((size_t)M * K);
            for (auto& v : hX) v = float_to_bf16(xf(rng));
            GpuBuffer<bf16> X((size_t)M * K, q);
            X.upload(hX.data(), hX.size());
            GpuBuffer<bf16> C((size_t)M * N, q);

            auto launch = [&](Kernel k) {
                switch (k) {
                    case Kernel::Int4:
                        matmul_int4(X.data(), M, K, W4, C.data(), ctx);
                        break;
                    case Kernel::Bf16:
                        matmul_bf16(X.data(), M, K, Wbf.data(), N, C.data(), ctx);
                        break;
                    case Kernel::W4a8:
                        matmul_int4_w4a8(X.data(), M, K, W4, C.data(), ctx);
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
                const double flops = 2.0 * (double)M * K * N;
                // Weight-side bytes are the floor; A/C traffic is exact. The int4
                // zp side-path re-reads zp_offset once more and touches small
                // (M,G)/(M,N) temporaries -- counted approximately here.
                double w_bytes;
                if (k == Kernel::Int4) {
                    w_bytes = (double)N * K / 2 + (double)G * N * 2;
                    if (!symmetric) w_bytes += 2.0 * G * N * 2;
                } else {
                    w_bytes = (double)N * K * 2;
                }
                const double bytes = w_bytes
                    + (double)M * K * 2 + (double)M * N * 2;
                const double sec = ms / 1000.0;
                r.kernel = kernel_name(k); r.shape = sh.name;
                r.K = K; r.N = N; r.M = M;
                r.ms = ms;
                r.tflops = sec > 0 ? flops / sec / 1e12 : 0.0;
                r.gbps   = sec > 0 ? bytes / sec / 1e9 : 0.0;
                r.pct_tf = peak_tflops > 0 ? 100.0 * r.tflops / peak_tflops : 0.0;
                r.pct_bw = peak_gbps > 0 ? 100.0 * r.gbps / peak_gbps : 0.0;
                return r;
            };

            auto print_row = [&](const Row& r) {
                if (md)
                    std::printf("| %s | %s | %d | %d | %d | %.4f | %.2f | %.2f | %.2f | %.2f |\n",
                        r.kernel.c_str(), r.shape.c_str(), r.K, r.N, r.M,
                        r.ms, r.tflops, r.gbps, r.pct_tf, r.pct_bw);
                else if (csv)
                    std::printf("%s,%s,%d,%d,%d,%.4f,%.2f,%.2f,%.2f,%.2f\n",
                        r.kernel.c_str(), r.shape.c_str(), r.K, r.N, r.M,
                        r.ms, r.tflops, r.gbps, r.pct_tf, r.pct_bw);
                else
                    std::printf("%-6s %-7s %6d %7d %5d %10.4f %9.2f %9.2f %8.2f %8.2f\n",
                        r.kernel.c_str(), r.shape.c_str(), r.K, r.N, r.M,
                        r.ms, r.tflops, r.gbps, r.pct_tf, r.pct_bw);
            };

            // Optional loose check: int4 vs dequantized bf16 reference.
            // ref = A @ (s4*scale)^T  -  rowsum_g(A) @ zp_offset   (host math on
            // top of a device bf16 GEMM over host-dequantized weights).
            if (check) {
                std::vector<bf16> hwd(wb);
                for (int n = 0; n < N; ++n)
                    for (int k = 0; k < K; ++k) {
                        const uint8_t byte = hwp[(size_t)n * (K / 2) + k / 2];
                        const int nibv = (byte >> (4 * (k % 2))) & 0xF;
                        const int s4 = nibv >= 8 ? nibv - 16 : nibv;
                        hwd[(size_t)n * K + k] = float_to_bf16(
                            (float)s4 * bf16_to_float(hws[(size_t)(k / group_size) * N + n]));
                    }
                GpuBuffer<bf16> Wdeq(wb, q);
                Wdeq.upload(hwd.data(), wb);
                GpuBuffer<bf16> Cref((size_t)M * N, q);
                matmul_bf16(X.data(), M, K, Wdeq.data(), N, Cref.data(), ctx);
                q.wait();
                std::vector<bf16> href((size_t)M * N);
                Cref.download(href.data(), href.size());
                if (!symmetric) {
                    std::vector<float> hrow((size_t)M * G, 0.0f);
                    for (int m = 0; m < M; ++m)
                        for (int g = 0; g < G; ++g) {
                            float acc = 0.0f;
                            for (int k = 0; k < group_size; ++k)
                                acc += bf16_to_float(hX[(size_t)m * K + g * group_size + k]);
                            hrow[(size_t)m * G + g] = acc;
                        }
                    for (int m = 0; m < M; ++m)
                        for (int n = 0; n < N; ++n) {
                            float corr = 0.0f;
                            for (int g = 0; g < G; ++g)
                                corr += hrow[(size_t)m * G + g] *
                                        bf16_to_float(hzp[(size_t)g * N + n]);
                            href[(size_t)m * N + n] = float_to_bf16(
                                bf16_to_float(href[(size_t)m * N + n]) - corr);
                        }
                }
                matmul_int4(X.data(), M, K, W4, C.data(), ctx);
                q.wait();
                std::vector<bf16> hcur((size_t)M * N);
                C.download(hcur.data(), hcur.size());
                double max_abs = 0.0, max_rel = 0.0;
                for (size_t i = 0; i < hcur.size(); ++i) {
                    const float a = bf16_to_float(hcur[i]);
                    const float b = bf16_to_float(href[i]);
                    const double d = std::fabs(a - b);
                    if (d > max_abs) max_abs = d;
                    const double rel = d / std::fmax(std::fabs(b), 1e-3);
                    if (rel > max_rel) max_rel = rel;
                }
                std::printf("[check] int4 %-7s M=%-5d max_abs=%.4g max_rel=%.4g\n",
                            sh.name.c_str(), M, max_abs, max_rel);
            }

            for (Kernel k : selected)
                print_row(roofline(k, timeit(k)));
        }  // M
    }  // shape
    return 0;
}

}  // namespace

REGISTER_BENCH("qwen35-matmul-roofline",
    "qwen3_5 27B GEMM roofline (random weights; int4=oneDNN s4+zp, bf16=oneDNN dense)",
    run)

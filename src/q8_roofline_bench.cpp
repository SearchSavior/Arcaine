// q8_roofline_bench
//
// Standalone roofline micro-benchmark for the Q8_0 (W8A16) matmul that the
// DiffusionGemma decode path actually executes (matmul_q8_0 — attention
// projections, dense MLP, LM head). Uses SYNTHETIC weights at the real decode
// shapes, so it needs no model file.
//
// Purpose: determine empirically whether decode (M = canvas_length = 256) is
// compute-bound or bandwidth-bound on this GPU.
//   - achieved TFLOP/s FLAT as M grows  (time ~ M)   => compute-bound  (healthy)
//   - achieved TFLOP/s RISING ~linearly with M (time ~flat) => bandwidth-bound (bug)
// The crossover M* is where arithmetic intensity AI(M) = FLOPs/bytes crosses the
// machine ridge (peak_tflops / peak_gbs). The roofline predicts compute-bound
// for M > M*. If the MEASURED curve stays bandwidth-bound past M*, the Q8_0
// oneDNN path is not realizing compute-bound throughput (config / layout bug).
//
// Run the default (oneDNN) backend, then `DIFF_Q8_BACKEND=custom` for the
// bandwidth-bound reference kernel (re-reads weights per 8-row tile by design).
//
//   ZE_AFFINITY_MASK=0 ./q8_roofline_bench [-p <M-csv>] [-r <runs>] [-w <warmup>]
//                         [--peak-tflops F] [--peak-gbs F] [--md]
//
// Defaults: M sweep "8,16,32,64,128,256,512,1024,2048" (includes the decode
// operating point 256 and straddles the crossover). Peaks default to Arc B580
// (160 TFLOP/s bf16, 456 GB/s) — override for other GPUs.

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/gpu/q8_0.hpp"

namespace {

struct Shape { const char* name; int K; int N; };

// Real DiffusionGemma decode linear shapes (H=2816, intermediate=2112, V=262144).
// K is the contracting (input) dim, N the output dim, matching matmul_q8_0(M,K,W[N,K]).
static const Shape kShapes[] = {
    { "qkv_proj",   2816,  2816   },  // one of Q/K/V/O (N = H)
    { "ffn_up",     2816,  2112   },  // dense MLP up   (feed_forward_length)
    { "ffn_down",   2112,  2816   },
    { "expert_mlp", 2816,   704   },  // expert FFN (note: real path uses grouped kernel)
    { "lm_head",   2816, 262144   },  // tied embedding / LM head (N = V)
};

struct Args {
    std::string m_csv   = "8,16,32,64,128,256,512,1024,2048";
    int   runs   = 5;
    int   warmup = 3;
    double peak_tflops = 160.0;   // Arc B580 bf16 XMX estimate
    double peak_gbs    = 456.0;   // Arc B580 GDDR6 estimate
    bool  md = false;
};

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -p <csv>          M dimension sweep      (default: 8,16,32,64,128,256,512,1024,2048)\n"
        "  -r <N>            timed runs per cell    (default: 5)\n"
        "  -w <N>            warmup runs per cell   (default: 3)\n"
        "  --peak-tflops F   peak bf16 TFLOP/s      (default: 160, Arc B580)\n"
        "  --peak-gbs F      peak bandwidth GB/s    (default: 456, Arc B580)\n"
        "  --md              emit a markdown table\n"
        "  -h                help\n"
        "Backend is selected by DIFF_Q8_BACKEND (unset=onednn, custom/dpas=custom).\n", p);
}

std::vector<int> parse_int_csv(const std::string& s) {
    std::vector<int> out; std::string t;
    for (char c : s) { if (c == ',') { if (!t.empty()) out.push_back(std::atoi(t.c_str())); t.clear(); } else t += c; }
    if (!t.empty()) out.push_back(std::atoi(t.c_str()));
    return out;
}

// Deterministic LCG so every run / cell sees identical data (no timing variance
// from content; values kept small to avoid overflow / denormals).
uint32_t lcg_state = 0x1234u;
inline uint32_t rng() { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }
inline float frand(float lo, float hi) { return lo + (hi - lo) * (rng() / 4294967296.0f); }

double now_s() {
    using Clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clk::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    return v[v.size() / 2];
}

// Build a synthetic Q8Linear [N,K] (out=N, in=K) with per-32-group f32 scales.
Q8Linear build_synthetic_q8(int K, int N, sycl::queue& q) {
    int G = K / 32;
    std::vector<int8_t> wq((size_t)N * K);
    std::vector<float>  sc((size_t)G * N);
    for (size_t i = 0; i < wq.size(); ++i) wq[i] = (int8_t)(frand(-127.0f, 127.0f));
    for (size_t i = 0; i < sc.size();  ++i) sc[i]  = frand(0.001f, 0.01f);

    Q8Linear W;
    W.in_features  = K;
    W.out_features = N;
    W.group_size   = 32;
    W.weight_qs    = GpuBuffer<int8_t>((size_t)N * K, q);
    W.weight_scale = GpuBuffer<float>((size_t)G * N, q);
    W.weight_qs.upload(wq.data(), wq.size());
    W.weight_scale.upload(sc.data(), sc.size());
    return W;
}

const char* backend_name() {
    return (q8_backend() == Q8Backend::Custom) ? "custom/dpas" : "onednn";
}

void print_header_md() {
    std::printf("| shape | M | K | N | steady us | serial us | TFLOP/s | GB/s | AI | "
                "roof_compute us | roof_bw us | roof_pred us | meas/pred | regime |\n");
    std::printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
}

void print_row_md(const char* sh, int M, int K, int N,
                  double steady_us, double serial_us, double tflops, double gbs,
                  double ai, double rc, double rb, double rp, double ratio, const char* reg) {
    std::printf("| %s | %d | %d | %d | %.2f | %.2f | %.1f | %.0f | %.1f | %.2f | %.2f | %.2f | %.2f | %s |\n",
                sh, M, K, N, steady_us, serial_us, tflops, gbs, ai, rc, rb, rp, ratio, reg);
}

void print_header_txt() {
    std::printf("%-10s %6s %7s %9s  %10s %10s  %9s %9s %7s  %11s %11s %11s %9s  %-12s\n",
        "shape", "M", "K", "N", "steady_us", "serial_us", "TFLOP/s", "GB/s", "AI",
        "roof_comp", "roof_bw", "roof_pred", "meas/pred", "regime");
    std::printf("%s\n", std::string(125, '-').c_str());
}

void print_row_txt(const char* sh, int M, int K, int N,
                   double steady_us, double serial_us, double tflops, double gbs,
                   double ai, double rc, double rb, double rp, double ratio, const char* reg) {
    std::printf("%-10s %6d %7d %9d  %10.2f %10.2f  %9.1f %9.0f %7.1f  %11.2f %11.2f %11.2f %9.2f  %-12s\n",
        sh, M, K, N, steady_us, serial_us, tflops, gbs, ai, rc, rb, rp, ratio, reg);
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(1); } return argv[++i]; };
        if      (s == "-h" || s == "--help") { usage(argv[0]); return 0; }
        else if (s == "-p")               a.m_csv       = next();
        else if (s == "-r")               a.runs        = std::atoi(next().c_str());
        else if (s == "-w")               a.warmup      = std::atoi(next().c_str());
        else if (s == "--peak-tflops")    a.peak_tflops = std::atof(next().c_str());
        else if (s == "--peak-gbs")       a.peak_gbs    = std::atof(next().c_str());
        else if (s == "--md")             a.md = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    std::vector<int> ms = parse_int_csv(a.m_csv);

    GpuEngine& ctx = GpuEngine::get(0);
    auto& q = ctx.queue;
    std::string dev_name = q.get_device().get_info<sycl::info::device::name>();
    double ridge = (a.peak_gbs > 0) ? a.peak_tflops * 1e12 / (a.peak_gbs * 1e9) : 0.0;

    std::printf("[q8_roofline] device: %s\n", dev_name.c_str());
    std::printf("[q8_roofline] backend: %s  (DIFF_Q8_BACKEND=%s)\n", backend_name(),
                std::getenv("DIFF_Q8_BACKEND") ? std::getenv("DIFF_Q8_BACKEND") : "<unset>");
    std::printf("[q8_roofline] peaks: %.0f TFLOP/s  %.0f GB/s  ridge=%.0f FLOP/byte\n",
                a.peak_tflops, a.peak_gbs, ridge);
    std::printf("[q8_roofline] sweep M={%s}  warmup=%d runs=%d\n\n",
                a.m_csv.c_str(), a.warmup, a.runs);
    std::printf("Decode operating point: M = canvas_length = 256.\n");
    std::printf("Read the achieved-TFLOP/s column: flat across M => compute-bound;\n");
    std::printf("rising ~linearly with M (serial/steady us ~flat) => bandwidth-bound.\n\n");

    if (a.md) print_header_md(); else print_header_txt();

    GpuBuffer<bf16> A, C;  // reused across M of the same shape (reallocated on growth)

    for (const auto& sh : kShapes) {
        int K = sh.K, N = sh.N;
        int G = K / 32;
        Q8Linear W = build_synthetic_q8(K, N, q);

        for (int M : ms) {
            if (M % 8 != 0 || N % 16 != 0) {
                std::fprintf(stderr, "skip M=%d N=%d (custom kernel needs M%%8==0 N%%16==0)\n", M, N);
                continue;
            }
            // (Re)allocate A,C for this M.
            A = GpuBuffer<bf16>((size_t)M * K, q);
            C = GpuBuffer<bf16>((size_t)M * N, q);
            {
                std::vector<bf16> ha((size_t)M * K);
                for (size_t i = 0; i < ha.size(); ++i) ha[i] = float_to_bf16(frand(-1.0f, 1.0f));
                A.upload(ha.data(), ha.size());
            }
            C.zero();

            // Warmup (steady-state, drained).
            for (int w = 0; w < a.warmup; ++w) matmul_q8_0(A.data(), M, K, W, C.data(), ctx);
            q.wait();

            // Steady-state: R submits, one drain. Models the decode in-order queue
            // (each linear waits on the previous; host does not wait per kernel).
            std::vector<double> steady, serial;
            for (int r = 0; r < a.runs; ++r) {
                double t0 = now_s();
                for (int k = 0; k < a.runs; ++k) matmul_q8_0(A.data(), M, K, W, C.data(), ctx);
                q.wait();
                double t1 = now_s();
                steady.push_back((t1 - t0) / a.runs * 1e6);

                double s0 = now_s();
                matmul_q8_0(A.data(), M, K, W, C.data(), ctx);
                q.wait();
                double s1 = now_s();
                serial.push_back((s1 - s0) * 1e6);
            }
            double steady_us = median(steady);
            double serial_us  = median(serial);

            double flops   = 2.0 * (double)M * K * N;
            double bytes   = (double)M * K * 2 + (double)N * K * 1 + (double)G * N * 4 + (double)M * N * 2;
            double ai      = flops / bytes;
            double tflops  = flops / (steady_us * 1e-6) / 1e12;
            double gbs     = bytes / (steady_us * 1e-6) / 1e9;
            double roof_c  = (a.peak_tflops > 0) ? flops / (a.peak_tflops * 1e12) * 1e6 : 0.0;
            double roof_b  = (a.peak_gbs > 0)    ? bytes / (a.peak_gbs * 1e9) * 1e6    : 0.0;
            double roof_p  = std::max(roof_c, roof_b);
            double ratio   = (roof_p > 0) ? steady_us / roof_p : 0.0;
            const char* reg = (ridge > 0)
                ? (ai >= ridge ? "compute" : "bandwidth")
                : (roof_c >= roof_b ? "compute*" : "bandwidth*");

            if (a.md) print_row_md(sh.name, M, K, N, steady_us, serial_us, tflops, gbs, ai, roof_c, roof_b, roof_p, ratio, reg);
            else      print_row_txt(sh.name, M, K, N, steady_us, serial_us, tflops, gbs, ai, roof_c, roof_b, roof_p, ratio, reg);
        }
    }

    std::printf("\nregime: '*' = inferred from max(roof_compute, roof_bw) since peak=0; "
                "otherwise AI vs ridge.\n");
    std::printf("meas/pred ~1.0 => operating at roofline; >>1 => inefficient (not reaching roof).\n");
    return 0;
}

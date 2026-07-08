// diffusion_bench — llama-bench style throughput benchmark for DiffusionGemma.
//
// Loads the model ONCE and sweeps a matrix of {kernel} x {-p prompt tokens} x
// {-n new tokens}, running warmup + timed runs per cell. In-process kernel
// switching (set_nvfp4_kernel) avoids the ~9 s model reload per config.
// Prompts are synthetic (deterministic pseudo-random token ids of the requested
// length), like llama-bench's -p tests — content-independent prefill timing.
//
//   ./build/diffusion_bench --model <dir> [options]
//     -p 128,512        prompt token counts to sweep   (default 512)
//     -n 128,256        new-token counts to sweep      (default 128)
//     -ds 48            denoising steps                 (default 48)
//     -w 1              warmup runs per cell            (default 1)
//     -r 5              timed runs per cell             (default 5)
//     --kernels hybrid  expert kernels to sweep         (default hybrid)
//     --layers split:15  layer placement override       (default auto)
//     --experts shard    expert placement override      (default auto)
//     --gpus all         GPU selection validation       (default all)
//     --device 0        visible GPU via ZE_AFFINITY_MASK (default env)
//     --seed 42 --md
//
// Per run it reports prefill throughput, actual output throughput, canvas
// positions/s, and the diffusion metrics (forward passes/s, tokens/forward,
// denoising passes).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

#include "modeling/diffusion_gemma/model.hpp"
#include "utils/chat.hpp"
#include "common/gpu/expert_parallel.hpp"
#include "common/gpu/mem_planner.hpp"
#include "common/gpu/placement.hpp"
#include "common/gpu/device_select.hpp"

namespace {

Nvfp4Kernel parse_kernel(const std::string& s) {
    if (s == "default") return Nvfp4Kernel::Default;
    if (s == "hybrid")  return Nvfp4Kernel::Hybrid;
    if (s == "custom") return Nvfp4Kernel::Custom;
    if (s == "grouped-dpas" || s == "dpas" || s == "geglu-pack") return Nvfp4Kernel::GroupedDpas;
    throw std::runtime_error("unknown kernel '" + s + "' (use: default, hybrid, custom, grouped-dpas)");
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out; size_t i = 0;
    while (i <= s.size()) { size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j + 1; }
    return out;
}
std::vector<int> parse_int_csv(const std::string& s) {
    std::vector<int> out; for (auto& t : split_csv(s)) out.push_back(std::stoi(t)); return out;
}

void unsupported_placement_value(const std::string& flag, const std::string& value) {
    throw std::runtime_error(flag + " value '" + value + "' is not implemented by this runtime yet");
}

void apply_layers_spec(const std::string& value, DiffPlacementOptions& placement) {
    if (value == "auto") {
        placement.layer_mode = DiffLayerPlacementMode::Auto;
        placement.layer_split = -1;
    } else if (value == "single") {
        placement.layer_mode = DiffLayerPlacementMode::Single;
        placement.layer_split = -1;
    } else if (value.rfind("split:", 0) == 0) {
        placement.layer_mode = DiffLayerPlacementMode::Split;
        placement.layer_split = std::stoi(value.substr(6));
    } else if (value.rfind("ranges:", 0) == 0 || value.rfind("gpus:", 0) == 0) {
        unsupported_placement_value("--layers", value);
    } else {
        throw std::runtime_error("--layers must be one of: auto, single, split:N, ranges:N,N,..., gpus:N");
    }
}

void apply_experts_spec(const std::string& value, DiffPlacementOptions& placement) {
    if (value == "auto") {
        placement.expert_mode = DiffExpertPlacementMode::Auto;
    } else if (value == "layer-owner") {
        placement.expert_mode = DiffExpertPlacementMode::LayerOwner;
    } else if (value == "shard") {
        placement.expert_mode = DiffExpertPlacementMode::Shard;
    } else if (value == "local") {
        throw std::runtime_error("--experts local was renamed; use --experts layer-owner");
    } else if (value == "replicate" || value.rfind("ranges:", 0) == 0 || value.rfind("gpus:", 0) == 0) {
        unsupported_placement_value("--experts", value);
    } else {
        throw std::runtime_error("--experts must be one of: auto, layer-owner, replicate, shard, ranges:N,N,..., gpus:N");
    }
}

void apply_gpus_spec(const std::string& value) {
    if (value != "all")
        unsupported_placement_value("--gpus", value);
}

// Deterministic synthetic prompt of `p` valid token ids (content-independent
// prefill workload). Spread across a safe low vocab range.
std::vector<int> synth_prompt(int p) {
    std::vector<int> ids((size_t)p);
    uint32_t s = 0x9e3779b9u;
    for (int i = 0; i < p; ++i) { s = s * 1664525u + 1013904223u; ids[i] = 16 + (int)(s % 48000u); }
    return ids;
}

struct Stat { double mean = 0, sd = 0; };
Stat aggregate(const std::vector<double>& v) {
    Stat s; if (v.empty()) return s;
    for (double x : v) s.mean += x; s.mean /= v.size();
    if (v.size() > 1) { double a = 0; for (double x : v) a += (x - s.mean) * (x - s.mean);
        s.sd = std::sqrt(a / (v.size() - 1)); }
    return s;
}

struct Row {
    std::string kernel; int n_prompt, n_gen, ds, runs;
    Stat prefill, decode, canvas, fwd; // throughputs (output t/s, canvas positions/s, passes/s)
    double tok_per_fwd, passes;        // diffusion metrics (means)
};

double output_tps_for(const DiffPerfStats& s) {
    return s.decode_s > 0 ? s.output_tokens / s.decode_s : 0.0;
}

double tokens_per_forward_for(const DiffPerfStats& s) {
    return s.decode_passes > 0 ? (double)s.output_tokens / s.decode_passes : 0.0;
}

std::string escaped_result(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    static constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c >= 0x20 && c != 0x7f) out.push_back((char)c);
        else {
            out += "\\x";
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 0xf]);
        }
    }
    return out;
}

void print_bench_result(TokenizerBridge& tokenizer,
                        const std::vector<int>& ids,
                        const std::string& kernel,
                        int prompt_tokens, int gen_tokens, int run, int runs) {
    std::string decoded = tokenizer.decode(ids);
    std::string text = escaped_result(decoded);
    std::printf("[bench-result] %-8s p%-5d n%-5d run %d/%d: "
                "output_tokens=%zu decoded_bytes=%zu escaped_bytes=%zu\n",
                kernel.c_str(), prompt_tokens, gen_tokens, run, runs,
                ids.size(), decoded.size(), text.size());
    std::printf("[bench-result] text=\"%s\"\n", text.c_str());
}

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s --model <dir> [options]\n"
        "  -p <csv>          prompt token counts to sweep   (default: 512)\n"
        "  -n <csv>          new-token counts to sweep       (default: 128)\n"
        "                    NOTE: block diffusion generates whole 256-token canvases,\n"
        "                    so -n sets the block count ceil(n/256); n<=256 = 1 block.\n"
        "  -ds <N>           denoising steps                 (default: 48)\n"
        "  -w <N>            warmup runs per cell            (default: 1)\n"
        "  -r <N>            timed runs per cell             (default: 5)\n"
        "  --kernels <csv>   expert kernels: default,hybrid,custom  (default: hybrid)\n"
        "  --layers <spec>   layer placement: auto, single, split:N, ranges:N,N,..., gpus:N\n"
        "  --experts <spec>  expert placement: auto, layer-owner, replicate, shard, ranges:N,N,..., gpus:N\n"
        "  --gpus <spec>     GPU selection: all, or future comma-list (default: all)\n"
        "  --device <N>      run with one visible Level Zero GPU\n"
        "  --seed <S>        base RNG seed                   (default: 42)\n"
        "  --md              emit a markdown table\n"
        "  --print-result    print decoded generated output for each timed run\n"
        "  --mem-plan        report arena planner peak vs retained capacity\n", p);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered: progress survives a kill/pipe
    std::string model_dir, kernels = "hybrid", p_csv = "512", n_csv = "128";
    std::string device_index;
    int ds = 48, warmup = 1, runs = 5;
    unsigned seed = 42;
    bool md = false;
    bool mem_plan = false;
    bool print_result = false;
    bool device_index_set = false;
    DiffPlacementOptions placement;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i]; };
            if      (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
            else if (a == "--model")   model_dir = next();
            else if (a == "-p")        p_csv = next();
            else if (a == "-n")        n_csv = next();
            else if (a == "-ds")       ds = std::stoi(next());
            else if (a == "-w")        warmup = std::stoi(next());
            else if (a == "-r")        runs = std::stoi(next());
            else if (a == "--kernels") kernels = next();
            else if (a == "--layers")  apply_layers_spec(next(), placement);
            else if (a == "--experts") apply_experts_spec(next(), placement);
            else if (a == "--gpus")    apply_gpus_spec(next());
            else if (a == "--device")  { device_index = next(); device_index_set = true; }
            else if (a == "--seed")    seed = (unsigned)std::stoul(next());
            else if (a == "--md")      md = true;
            else if (a == "--print-result" || a == "--print-output") print_result = true;
            else if (a == "--mem-plan") mem_plan = true;
            else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
        }
        if (model_dir.empty()) { usage(argv[0]); return 1; }
        if (device_index_set) gpu_device_control::apply_device_index(device_index);

        std::vector<std::string> kernel_list = split_csv(kernels);
        std::vector<int> p_list = parse_int_csv(p_csv), n_list = parse_int_csv(n_csv);

        // Size the KV cache for the largest prompt+generation in the matrix.
        int max_p = 0, max_n = 0;
        for (int p : p_list) max_p = std::max(max_p, p);
        for (int n : n_list) max_n = std::max(max_n, n);
        int max_seq = max_p + max_n + 512;

        std::printf("[bench] loading model from %s ...\n", model_dir.c_str());
        if (const char* active_gpus = gpu_device_control::active_gpus_spec())
            std::printf("[bench] device control: ZE_AFFINITY_MASK=%s\n", active_gpus);
        DiffusionGemmaModel model(model_dir, max_seq, placement, /*print_placement=*/false);
        std::unique_ptr<TokenizerBridge> result_tokenizer;
        if (print_result)
            result_tokenizer = std::make_unique<TokenizerBridge>(model_dir);
        std::printf("[bench] matrix: kernels={%s} p={%s} n={%s} ds=%d | warmup=%d runs=%d | "
                    "cells=%zu\n", kernels.c_str(), p_csv.c_str(), n_csv.c_str(), ds,
                    warmup, runs, kernel_list.size() * p_list.size() * n_list.size());

        double kv_gb       = model.kv_cache_bytes() / (1024.0 * 1024.0 * 1024.0);
        double kv_mb_per_t = model.kv_cache_bytes_per_token() / (1024.0 * 1024.0);
        std::printf("[bench] KV cache: %.2f GB allocated (max_seq=%d) | %.3f MB/token\n",
                    kv_gb, model.kv_cache_max_seq(), kv_mb_per_t);
        auto arena_env_off = [](const char* e) {
            return e && (!std::strcmp(e, "off") || !std::strcmp(e, "0") ||
                         !std::strcmp(e, "false") || !std::strcmp(e, "no"));
        };
        auto scratch_env_on = [](const char* e) {
            return e && (!std::strcmp(e, "1") || !std::strcmp(e, "true") ||
                         !std::strcmp(e, "TRUE") || !std::strcmp(e, "yes"));
        };
        bool arena_off = arena_env_off(std::getenv("DIFF_ARENA")) ||
                         scratch_env_on(std::getenv("DISABLE_SCRATCH"));
        std::printf("[bench] activation arena: %s\n",
                    arena_off ? "DISABLED (fresh alloc per op)" : "enabled (planner-sized)");
        std::vector<Row> rows;
        for (auto& kname : kernel_list) {
            set_nvfp4_kernel(parse_kernel(kname));
            for (int p : p_list) {
                std::vector<int> prompt = synth_prompt(p);
                for (int n : n_list) {
                    // Fixed seed + fixed prompt: every run does identical work,
                    // so the reported stddev is pure timing jitter (not content
                    // variation in EOS / output-token counts).
                    for (int w = 0; w < warmup; ++w)
                        model.generate(prompt, n, ds, seed, false);

                    std::vector<double> pre, dec, canvas, fwd; double tpf = 0, pass = 0;
                    for (int r = 0; r < runs; ++r) {
                        std::vector<int> out = model.generate(prompt, n, ds, seed, false);
                        const DiffPerfStats& s = model.stats();
                        double output_tps = output_tps_for(s);
                        double tok_per_fwd = tokens_per_forward_for(s);
                        if (result_tokenizer)
                            print_bench_result(*result_tokenizer, out, kname, p, n, r + 1, runs);
                        pre.push_back(s.prefill_tps());
                        dec.push_back(output_tps);
                        fwd.push_back(s.decode_passes_ps());
                        canvas.push_back(s.decode_passes_ps() * model.config().canvas_length);
                        tpf += tok_per_fwd;
                        pass += s.decode_passes;
                        double arena_gb = model.scratch_bytes() / (1024.0 * 1024.0 * 1024.0);
                        std::printf("[bench] %-8s p%-5d n%-5d run %d/%d: "
                                    "prefill %.0f t/s | output %.1f t/s | canvas %.0f pos/s | %.2f fwd/s | %d passes "
                                    "| kv %.3f MB/token | arena %.2f GB\n",
                                    kname.c_str(), p, n, r + 1, runs,
                                    s.prefill_tps(), output_tps,
                                    s.decode_passes_ps() * model.config().canvas_length,
                                    s.decode_passes_ps(), s.decode_passes, kv_mb_per_t, arena_gb);
                    }
                    rows.push_back({kname, p, n, ds, runs, aggregate(pre), aggregate(dec),
                                    aggregate(canvas), aggregate(fwd), tpf / runs, pass / runs});
                }
            }
        }

        // ---- report ----
        std::printf("\n");
        if (md) {
            std::printf("| kernel | n_prompt | n_gen | ds | prefill t/s | output t/s | canvas pos/s | fwd/s | tok/fwd | passes |\n");
            std::printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
            for (auto& r : rows)
                std::printf("| %s | %d | %d | %d | %.0f ± %.0f | %.1f ± %.1f | %.0f ± %.0f | %.2f ± %.2f | %.1f | %.0f |\n",
                    r.kernel.c_str(), r.n_prompt, r.n_gen, r.ds,
                    r.prefill.mean, r.prefill.sd, r.decode.mean, r.decode.sd,
                    r.canvas.mean, r.canvas.sd, r.fwd.mean, r.fwd.sd,
                    r.tok_per_fwd, r.passes);
        } else {
            std::printf("%-8s %8s %6s %4s  %14s  %15s  %15s  %13s  %7s %7s\n",
                        "kernel", "n_prompt", "n_gen", "ds", "prefill t/s",
                        "output t/s", "canvas pos/s", "fwd/s", "tok/fwd", "passes");
            std::printf("%s\n", std::string(120, '-').c_str());
            for (auto& r : rows)
                std::printf("%-8s %8d %6d %4d  %7.0f ±%-5.0f  %8.1f ±%-5.1f  %8.0f ±%-5.0f  %6.2f ±%-5.2f  %7.1f %7.0f\n",
                    r.kernel.c_str(), r.n_prompt, r.n_gen, r.ds,
                    r.prefill.mean, r.prefill.sd, r.decode.mean, r.decode.sd,
                    r.canvas.mean, r.canvas.sd, r.fwd.mean, r.fwd.sd,
                    r.tok_per_fwd, r.passes);
        }

        // ---- graph-allocator memory analysis (additive; no execution change) ----
        if (mem_plan) {
            const DiffConfig& cfg = model.config();
            int chunk = [] { const char* e = std::getenv("DIFF_PREFILL_CHUNK");
                             return e ? std::atoi(e) : 2048; }();
            if (chunk <= 0) chunk = max_p;
            int pf_chunk   = std::min(chunk, max_p);
            int pf_pastlen = std::max(0, max_p - pf_chunk);   // worst-case last chunk
            int dec_seq    = cfg.canvas_length;

            auto pf  = memplan::build_prefill_graph(cfg, pf_chunk, pf_pastlen);
            auto dec = memplan::build_decode_graph(cfg, dec_seq, max_p);
            double pf_peak  = memplan::plan(pf)  / (1024.0 * 1024.0 * 1024.0);
            double dec_peak = memplan::plan(dec) / (1024.0 * 1024.0 * 1024.0);
            double pf_sum   = memplan::no_reuse_bytes(pf)  / (1024.0 * 1024.0 * 1024.0);
            double dec_sum  = memplan::no_reuse_bytes(dec) / (1024.0 * 1024.0 * 1024.0);
            double measured = model.scratch_bytes() / (1024.0 * 1024.0 * 1024.0);

            std::printf("\n[mem-plan] liveness-planned peak-live activations (arena planner)\n");
            std::printf("  prefill chunk=%-6d (past_len=%d): peak-live %.2f GB  (every-tensor-distinct %.2f GB)\n",
                        pf_chunk, pf_pastlen, pf_peak, pf_sum);
            std::printf("  decode  seq=%-6d (enc_len=%d): peak-live %.2f GB  (every-tensor-distinct %.2f GB)\n",
                        dec_seq, max_p, dec_peak, dec_sum);
            std::printf("  current activation arena capacity (all GPUs): %.2f GB\n", measured);
            std::printf("  -> prefill/decode run sequentially, so each per-GPU arena targets ~%.2f GB\n",
                        std::max(pf_peak, dec_peak));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

// diffusion_gemma — model-owned benchmark (llama-bench approach).
// Drives the module's concrete DiffusionGemmaModel engine directly via
// generate(): synthetic encoder prompt, output-length sweep, denoising-step
// sweep, and KV-cache-depth (encoder-length) sweep, reported as llama-bench-
// style table rows with diffusion domain columns. For block-diffusion the KV
// cache IS the encoder context, so -p (prompt token counts) and -d (KV depths)
// both specify encoder length; the bench sweeps the union. The model's
// generate() performs its own bounded-chunk prefill (DIFF_PREFILL_CHUNK, mapped
// from -pchunk) and resets per call, so the bench does not reimplement cache
// semantics — it only resolves encoder length, output length, steps, and
// placement, then measures.
//
// Engine/implementation knobs (kernel selection, sampler modes, rope/softmax/
// arena modes, EOS commit trimming) stay env-only (DIFF_*) so they are shared
// with arcaine_server and are NOT CLI flags.
//
// Registered with the central arcaine_mbench dispatcher as "diffusion_gemma".
//
//   ./build/arcaine_mbench -m <dir> [options]
//   (see arcaine_mbench -m <dir> --help for the flag list)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "modeling/diffusion_gemma/model.hpp"
#include "modeling/diffusion_gemma/inference/placement.hpp"
#include "runtime/gpu/device_select.hpp"
#include "benchmarks/model_bench_registry.hpp"
#include "benchmarks/model_bench_util.hpp"

namespace {

// Verbatim diffusion-specific usage block appended to the shared help text.
const char* EXTRA_USAGE =
    "\n"
    "DifusionGemma:\n"
    "\n"
    "-ds, --denoising-steps\n"
    "  Default: 48\n"
    "  Behavior: Simulates computation required for convergence, which translates to performance. "
    "https://arxiv.org/abs/2606.20560 reports average of 12-16 steps; "
    "https://arxiv.org/pdf/2606.14620 reports 3.3 - 17 steps. Default of 48 is therefore worst case, "
    "expected only on hard enough problems.";

// Deterministic synthetic prompt of `p` valid token ids (content-independent
// prefill workload). Spread across a safe low vocab range.
std::vector<int> synth_prompt(int p) {
    std::vector<int> ids((size_t)p);
    uint32_t s = 0x9e3779b9u;
    for (int i = 0; i < p; ++i) { s = s * 1664525u + 1013904223u; ids[i] = 16 + (int)(s % 48000u); }
    return ids;
}

double output_tps_for(const DiffPerfStats& s) {
    return s.decode_s > 0 ? s.output_tokens / s.decode_s : 0.0;
}

double tokens_per_forward_for(const DiffPerfStats& s) {
    return s.decode_passes > 0 ? (double)s.output_tokens / s.decode_passes : 0.0;
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
        throw std::runtime_error("--experts value '" + value + "' is not implemented by this runtime yet");
    } else {
        throw std::runtime_error("--experts must be one of: auto, layer-owner, replicate, shard, ranges:N,N,..., gpus:N");
    }
}

}  // namespace

static int run(int argc, char* argv[]) {
    using namespace arcaine::bench;

    std::vector<int> ds_vals;
    bool ds_set = false;
    const BenchCsvIntFlag extras[] = {
        {"--denoising-steps", "-ds", &ds_vals, &ds_set},
    };

    BenchArgs args;
    int r = parse_bench_args(argc, argv, extras, 1, EXTRA_USAGE, args);
    if (r == 1) return 0;  // help shown
    if (r != 0) return 1;  // parse error

    if (!ds_set) ds_vals = {48};

    DiffPlacementOptions placement;
    if (args.experts_set) {
        try { apply_experts_spec(args.experts, placement); }
        catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    }
    if (args.device_set) {
        try { gpu_device_control::apply_device_index(args.device); }
        catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    }
    if (args.pchunk_set)
        setenv("DIFF_PREFILL_CHUNK", std::to_string(args.pchunk).c_str(), 1);

    // Encoder-length (KV-depth) sweep: union of -p and -d (diffusion's KV
    // cache IS the encoder context). Drop lengths < 1 (diffusion needs a
    // non-empty encoder). Dedup + sort.
    std::vector<int> encoder_lengths;
    for (int v : args.n_prompt) if (v >= 1) encoder_lengths.push_back(v);
    for (int v : args.n_depth)  if (v >= 1) encoder_lengths.push_back(v);
    if (encoder_lengths.empty()) {
        std::fprintf(stderr, "error: no encoder lengths >= 1 in -p/-d\n");
        return 1;
    }
    std::sort(encoder_lengths.begin(), encoder_lengths.end());
    encoder_lengths.erase(std::unique(encoder_lengths.begin(), encoder_lengths.end()),
                          encoder_lengths.end());

    // Size the KV cache for the largest prompt+generation in the matrix.
    int max_p = *std::max_element(encoder_lengths.begin(), encoder_lengths.end());
    int max_n = 0; for (int n : args.n_gen) max_n = std::max(max_n, n);
    int max_seq = max_p + max_n + 512;

    std::printf("loading model from %s ...\n", args.model.c_str());
    double t0 = now_ms();
    DiffusionGemmaModel model(args.model, max_seq, placement, /*print_placement=*/false);
    double load_s = (now_ms() - t0) * 0.001;

    const DiffConfig& cfg = model.config();
    std::string backend_line = "SYCL + oneDNN | GPUs: " +
                               std::to_string(GpuEngine::count()) +
                               " | max_seq: " + std::to_string(max_seq);
    if (const char* gpus = gpu_device_control::active_gpus_spec())
        backend_line += std::string(" | ZE_AFFINITY_MASK=") + gpus;

    if (!cfg.quantization_format.empty())
        std::printf("model   : %s (%s, canvas %d)\n",
                    cfg.model_type.c_str(), cfg.quantization_format.c_str(), cfg.canvas_length);
    else
        std::printf("model   : %s (canvas %d)\n",
                    cfg.model_type.c_str(), cfg.canvas_length);
    std::printf("backend : %s\n", backend_line.c_str());
    std::printf("load    : %.1f s\n", load_s);

    const bool force_full_canvas = [] {
        const char* e = std::getenv("DIFF_BENCH_FORCE_FULL_CANVAS");
        return e && std::strcmp(e, "0") && std::strcmp(e, "off") &&
               std::strcmp(e, "false") && std::strcmp(e, "no");
    }();

    MdTable table({{"test", 18, false},
                   {"prefill_t/s", 12, true},
                   {"decode_t/s", 15, true},
                   {"canvas pos/s", 12, true},
                   {"fwd/s", 12, true},
                   {"tok/fwd", 9, true},
                   {"passes", 7, true}});
    table.header();

    for (int enc : encoder_lengths) {
        std::vector<int> prompt = synth_prompt(enc);
        for (int n : args.n_gen) {
            for (int ds : ds_vals) {
                if (!args.no_warmup)
                    model.generate(prompt, n, ds, 42, false, nullptr, force_full_canvas);

                std::vector<double> pre, out_tps, canvas, fwd;
                double tpf = 0, pass = 0;
                for (int r = 0; r < args.reps; ++r) {
                    model.generate(prompt, n, ds, 42, false, nullptr, force_full_canvas);
                    const DiffPerfStats& s = model.stats();
                    double output_tps = output_tps_for(s);
                    pre.push_back(s.prefill_tps());
                    out_tps.push_back(output_tps);
                    fwd.push_back(s.decode_passes_ps());
                    canvas.push_back(s.decode_passes_ps() * cfg.canvas_length);
                    tpf += tokens_per_forward_for(s);
                    pass += s.decode_passes;
                }
                Stat o = aggregate(out_tps), p = aggregate(pre),
                     c = aggregate(canvas), f = aggregate(fwd);

                char prefill[32], decode[32], canvas_s[32], fwd_s[32], tokfwd[32], passes[32];
                std::snprintf(prefill, sizeof prefill, "%.0f ± %.0f", p.mean, p.sd);
                std::snprintf(decode, sizeof decode, "%.2f ± %.2f", o.mean, o.sd);
                std::snprintf(canvas_s, sizeof canvas_s, "%.0f ± %.0f", c.mean, c.sd);
                std::snprintf(fwd_s, sizeof fwd_s, "%.2f ± %.2f", f.mean, f.sd);
                std::snprintf(tokfwd, sizeof tokfwd, "%.1f", tpf / (args.reps ? args.reps : 1));
                std::snprintf(passes, sizeof passes, "%.0f", pass / (args.reps ? args.reps : 1));

                std::string name = test_name(enc, n, 0) + "+ds" + std::to_string(ds);
                table.row({name, prefill, decode, canvas_s, fwd_s, tokfwd, passes});
            }
        }
    }
    std::printf("\n");
    return 0;
}

REGISTER_MODEL_BENCH("diffusion_gemma", "DiffusionGemma block-diffusion MoE (encoder/output/steps sweep)", run)

// DiffusionGemma 26B-A4B driver.
//   ./build/diffusion_gemma --model <dir> --prompt "..." [options]
//
// Visualization:
//   --viz out.html   records every denoising step and writes a self-contained
//                    HTML replay (slider/autoplay; per-token entropy heatmap).
//   --stream         live draft canvas in the terminal (secondary; full-screen
//                    redraw per step).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cctype>
#include <unistd.h>
#include <stdexcept>

#include "modeling/diffusion_gemma/model.hpp"
#include "utils/chat.hpp"
#include "utils/viz.hpp"
#include "common/gpu/placement.hpp"
#include "common/gpu/engine.hpp"

// Device-reported VRAM usage for every GPU the engine runs on.
static std::vector<DiffGpuMem> query_gpu_mem() {
    std::vector<DiffGpuMem> out;
    for (int i = 0; i < GpuEngine::count(); ++i) {
        auto dev = GpuEngine::get(i).queue.get_device();
        DiffGpuMem m;
        m.total_gb = dev.get_info<sycl::info::device::global_mem_size>() / 1e9;
        if (dev.has(sycl::aspect::ext_intel_free_memory)) {
            double free_gb = dev.get_info<sycl::ext::intel::info::device::free_memory>() / 1e9;
            m.used_gb = m.total_gb - free_gb;
        }
        out.push_back(m);
    }
    return out;
}


static bool is_chat_exit(const std::string& line) {
    return line == "/exit" || line == "/quit" || line == "/q";
}

static bool is_blank(const std::string& line) {
    for (unsigned char c : line)
        if (!std::isspace(c)) return false;
    return true;
}

static void print_bench_metrics(const char* tag, int turn, int prompt_tokens,
                                const DiffusionGemmaModel& model, double wall_s) {
    const DiffPerfStats& st = model.stats();
    double kv_mb_per_t = model.kv_cache_bytes_per_token() / (1024.0 * 1024.0);
    double arena_gb = model.scratch_bytes() / (1024.0 * 1024.0 * 1024.0);
    double canvas_ps = st.decode_passes_ps() * model.config().canvas_length;
    if (turn > 0) {
        std::printf("[%s] turn %d: prompt_tokens=%d output_tokens=%d wall %.2f s | "
                    "prefill %.0f t/s | effective %.1f t/s | canvas %.0f pos/s | %.2f fwd/s | "
                    "%.1f tok/fwd | %d passes | kv %.3f MB/token | arena %.2f GB\n",
                    tag, turn, prompt_tokens, st.output_tokens, wall_s,
                    st.prefill_tps(), st.effective_tps(), canvas_ps, st.decode_passes_ps(),
                    st.tokens_per_forward(), st.decode_passes, kv_mb_per_t, arena_gb);
    } else {
        std::printf("[%s] prompt_tokens=%d output_tokens=%d wall %.2f s | "
                    "prefill %.0f t/s | effective %.1f t/s | canvas %.0f pos/s | %.2f fwd/s | "
                    "%.1f tok/fwd | %d passes | kv %.3f MB/token | arena %.2f GB\n",
                    tag, prompt_tokens, st.output_tokens, wall_s,
                    st.prefill_tps(), st.effective_tps(), canvas_ps, st.decode_passes_ps(),
                    st.tokens_per_forward(), st.decode_passes, kv_mb_per_t, arena_gb);
    }
}

static void unsupported_placement_value(const std::string& flag, const std::string& value) {
    throw std::runtime_error(flag + " value '" + value + "' is not implemented by this runtime yet");
}


static void apply_layers_spec(const std::string& value, DiffPlacementOptions& placement) {
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

static void apply_experts_spec(const std::string& value, DiffPlacementOptions& placement) {
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

static void apply_gpus_spec(const std::string& value) {
    if (value != "all")
        unsupported_placement_value("--gpus", value);
}

static void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s --model <dir> (--prompt <text> | --chat) [options]\n"
        "  --help             show this help text\n"
        "  --model <dir>      model directory containing config/tokenizer/weights\n"
        "  --prompt <text>    user prompt; with --chat, sends the first turn\n"
        "  --chat             interactive stdin chat; streams committed output chunks\n"
        "  --max-tokens <N>   max new tokens     (default: 256)\n"
        "  --steps <N>        denoising steps    (default: model config, 48)\n"
        "  --seed <S>         RNG seed           (default: 42)\n"
        "  --max-seq <N>      KV cache capacity  (default: 2048)\n"
        "  --viz <path>       write HTML denoising replay\n"
        "  --stream           live denoising canvas in terminal for single prompt\n"
        "  --no-stream        disable terminal streaming\n"
        "  --layers <spec>    layer placement: auto, single, split:N, ranges:N,N,..., gpus:N\n"
        "  --experts <spec>   expert placement: auto, layer-owner, replicate, shard, ranges:N,N,..., gpus:N\n"
        "  --gpus <spec>      GPU selection: all, or future comma-list (default: all)\n"
        "  --print-placement  print resolved layer/expert placement (default)\n"
        "  --no-print-placement suppress placement report during model load\n"
        "  --dry-run-placement print placement report and exit before loading weights\n"
        "  --verbose          per-step entropy logging\n", p);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model_dir, prompt, viz_path;
    int max_tokens = 256, steps = -1, max_seq = 2048;
    unsigned seed = 42;
    bool verbose = false, stream = false, chat = false, print_placement = true, dry_run_placement = false;
    DiffPlacementOptions placement;

    try {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else if (a == "--model")      model_dir = next();
        else if (a == "--prompt")     prompt = next();
        else if (a == "--max-tokens") max_tokens = std::stoi(next());
        else if (a == "--steps")      steps = std::stoi(next());
        else if (a == "--seed")       seed = (unsigned)std::stoul(next());
        else if (a == "--max-seq")    max_seq = std::stoi(next());
        else if (a == "--viz")        viz_path = next();
        else if (a == "--placement")  {
            std::fprintf(stderr, "--placement was removed; use --layers and --experts instead\n");
            usage(argv[0]);
            return 1;
        }
        else if (a == "--layers")     apply_layers_spec(next(), placement);
        else if (a == "--experts")    apply_experts_spec(next(), placement);
        else if (a == "--gpus")       apply_gpus_spec(next());
        else if (a == "--chat")       chat = true;
        else if (a == "--stream")     stream = true;
        else if (a == "--no-stream")  stream = false;
        else if (a == "--print-placement") print_placement = true;
        else if (a == "--no-print-placement") print_placement = false;
        else if (a == "--dry-run-placement") { dry_run_placement = true; print_placement = true; }
        else if (a == "--verbose")    verbose = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        usage(argv[0]);
        return 1;
    }

    if (model_dir.empty() || (!dry_run_placement && prompt.empty() && !chat)) { usage(argv[0]); return 1; }

    if (dry_run_placement) {
        DiffConfig cfg = DiffConfig::from_dir(model_dir);
        DiffPlacementOptions resolved_placement = resolve_diffusion_placement(cfg, placement);
        int split_layer = resolve_diffusion_split_layer(cfg, resolved_placement);
        std::printf("[model] %d GPU(s); %d layers, split at %d\n",
                    GpuEngine::count(), cfg.text.num_hidden_layers, split_layer);
        print_diffusion_placement(cfg, split_layer, resolved_placement);
        return 0;
    }

    std::printf("[main] starting tokenizer worker...\n");
    TokenizerBridge tok(model_dir);
    std::vector<int> prompt_ids = tok.build_prompt(prompt);
    std::printf("[main] prompt tokens: %zu\n", prompt_ids.size());

    std::printf("[main] loading model from %s ...\n", model_dir.c_str());
    auto t0 = std::chrono::steady_clock::now();
    DiffusionGemmaModel model(model_dir, max_seq, placement, print_placement);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("[main] load: %.1f s\n",
                std::chrono::duration<double>(t1 - t0).count());

    int n_steps = (steps > 0) ? steps : model.config().gen.max_denoising_steps;
    std::printf("[main] generating: max_tokens=%d denoising_steps=%d canvas=%d%s%s%s\n",
                max_tokens, n_steps, model.config().canvas_length,
                chat ? "  [chat]" : "", stream ? "  [stream]" : "",
                viz_path.empty() ? "" : "  [viz]");

    if (chat) {
        std::vector<ChatTemplateMessage> history;
        std::string queued_prompt = prompt;
        int turn = 0;
        std::printf("[chat] ready. Type /exit or press Ctrl-D to stop.\n");

        while (true) {
            std::string user_text;
            if (!queued_prompt.empty()) {
                user_text = queued_prompt;
                queued_prompt.clear();
                std::printf("\nuser> %s\n", user_text.c_str());
            } else {
                std::printf("\nuser> ");
                std::fflush(stdout);
                if (!std::getline(std::cin, user_text)) {
                    std::printf("\n");
                    break;
                }
            }

            if (is_chat_exit(user_text)) break;
            if (is_blank(user_text)) continue;

            history.push_back({"user", user_text});
            std::vector<int> prompt_ids = tok.build_prompt(history);
            ++turn;
            if ((int)prompt_ids.size() + max_tokens > model.kv_cache_max_seq()) {
                std::printf("[chat] warning: prompt_tokens(%zu)+max_tokens(%d) exceeds max_seq(%d)\n",
                            prompt_ids.size(), max_tokens, model.kv_cache_max_seq());
            }

            std::vector<int> streamed_ids;
            std::string streamed_text;
            DiffStreamCallback on_step = [&](const DiffStepEvent& ev) {
                if (!ev.committed || !ev.canvas) return;
                streamed_ids.insert(streamed_ids.end(), ev.canvas->begin(), ev.canvas->end());
                std::string full = tok.decode(streamed_ids);
                std::string delta;
                if (full.size() >= streamed_text.size() &&
                    full.compare(0, streamed_text.size(), streamed_text) == 0) {
                    delta = full.substr(streamed_text.size());
                } else {
                    delta = tok.decode(*ev.canvas);
                }
                if (!delta.empty()) {
                    std::printf("%s", delta.c_str());
                    std::fflush(stdout);
                }
                streamed_text = full;
            };

            std::printf("[chat] turn %d prompt tokens: %zu\nassistant> ", turn, prompt_ids.size());
            std::fflush(stdout);
            auto g0 = std::chrono::steady_clock::now();
            std::vector<int> out = model.generate(prompt_ids, max_tokens, n_steps,
                                                  seed + (unsigned)(turn - 1), false, on_step);
            auto g1 = std::chrono::steady_clock::now();

            std::string decoded = tok.decode(out);
            if (decoded.size() > streamed_text.size() &&
                decoded.compare(0, streamed_text.size(), streamed_text) == 0) {
                std::printf("%s", decoded.substr(streamed_text.size()).c_str());
            } else if (streamed_text.empty() && !decoded.empty()) {
                std::printf("%s", decoded.c_str());
            }
            std::printf("\n");
            if (!decoded.empty())
                history.push_back({"assistant", decoded});

            double wall_s = std::chrono::duration<double>(g1 - g0).count();
            print_bench_metrics("chat", turn, (int)prompt_ids.size(), model, wall_s);
        }
        return 0;
    }

    DiffVizRecorder recorder;
    std::string committed_text;
    bool tty = isatty(STDOUT_FILENO);

    DiffStreamCallback on_step;
    if (stream || !viz_path.empty()) {
        on_step = [&](const DiffStepEvent& ev) {
            if (!viz_path.empty()) recorder.record(ev);
            if (!stream) return;

            std::string draft = tok.decode(*ev.canvas);
            if (ev.committed) {
                committed_text += draft;
                if (!tty) std::printf("[blk %d committed] %s\n", ev.block, draft.c_str());
                return;
            }
            if (tty) {
                // Full clear per frame: no stale wrapped-line fragments.
                std::printf("\033[2J\033[H");
                std::printf("── block %d · step %2d · temp %.2f · entropy %.4f ──\n\n",
                            ev.block, ev.cur_step, ev.temperature, ev.mean_entropy);
                std::printf("%s\033[2m%s\033[0m\n", committed_text.c_str(), draft.c_str());
                std::fflush(stdout);
            } else {
                std::printf("[blk %d step %2d  H=%.4f] %.120s...\n",
                            ev.block, ev.cur_step, ev.mean_entropy, draft.c_str());
            }
        };
    }

    auto g0 = std::chrono::steady_clock::now();
    std::vector<int> out = model.generate(prompt_ids, max_tokens, n_steps, seed,
                                          verbose && !stream, on_step);
    auto g1 = std::chrono::steady_clock::now();

    if (stream && tty) std::printf("\n");
    const DiffPerfStats& st = model.stats();
    double wall_s = std::chrono::duration<double>(g1 - g0).count();
    std::printf("[main] generated %zu tokens in %.1f s\n", out.size(), wall_s);
    print_bench_metrics("perf", 0, (int)prompt_ids.size(), model, wall_s);

    std::vector<DiffGpuMem> gpu_mem = query_gpu_mem();
    for (size_t i = 0; i < gpu_mem.size(); ++i)
        std::printf("[perf] GPU%zu    : %5.1f / %.1f GB VRAM\n",
                    i, gpu_mem[i].used_gb, gpu_mem[i].total_gb);

    if (!viz_path.empty()) {
        recorder.write_html(viz_path, prompt, tok, st, gpu_mem);
        std::printf("[main] wrote denoising replay: %s\n", viz_path.c_str());
    }
    if (!stream) {
        std::string text = tok.decode(out);
        std::printf("\n===== OUTPUT =====\n%s\n==================\n", text.c_str());
    }
    return 0;
}

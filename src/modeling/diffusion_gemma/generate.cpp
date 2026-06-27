#include "model.hpp"
#include "sampler.hpp"
#include "../../utils/profile.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <vector>

std::vector<int> DiffusionGemmaModel::generate(
    const std::vector<int>& prompt_ids, int max_new_tokens, int max_denoising_steps,
    unsigned seed, bool verbose, const DiffStreamCallback& on_step)
{
    enc_kv_.reset();
    stats_ = DiffPerfStats{};
    diffprof::reset();
    std::mt19937 rng(seed);

    using Clk = std::chrono::steady_clock;
    auto secs = [](Clk::time_point a, Clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    int C = cfg_.canvas_length;
    int V = cfg_.text.vocab_size;
    int N = max_denoising_steps;
    int max_canvases = (max_new_tokens + C - 1) / C;

    DiffStopping stopping(cfg_.gen.stability_threshold, cfg_.gen.confidence_threshold);
    bool force_denoise_steps = std::getenv("DIFF_FORCE_DENOISE_STEPS") != nullptr;
    // DIFF_SKIP_LAST_SOFT_NEXT: the final scheduled step (step==1) has no
    // successor to consume its self-conditioning signal, so skip producing it.
    bool skip_last_soft_next = std::getenv("DIFF_SKIP_LAST_SOFT_NEXT") != nullptr;
    std::uniform_int_distribution<int> uni(0, V - 1);

    std::vector<int> output;
    int enc_len = 0;
    std::vector<int> prev_canvas;  // last appended canvas, re-encoded next round

    for (int blk = 0; blk < max_canvases; ++blk) {
        // 1. Encode: prompt (first block) or the previous canvas (continuation).
        {
            auto t0 = Clk::now();
            if (blk == 0) { encode(prompt_ids, 0); enc_len = (int)prompt_ids.size(); }
            else          { encode(prev_canvas, enc_len); enc_len += C; }
            stats_.prefill_s += secs(t0, Clk::now());
            stats_.prefill_tokens += (blk == 0) ? (int)prompt_ids.size() : C;
        }

        // 2. Init canvas with uniform-random tokens.
        std::vector<int> current(C);
        for (int& t : current) t = uni(rng);
        std::optional<GpuBuffer<bf16>> soft;
        stopping.reset();
        std::vector<int> argmax = current, denoiser;
        std::vector<float> entropy;

        // 3. Denoising loop (cur_step counts down N..1).
        for (int step = N; step >= 1; --step) {
            float temp = diff_temperature(step, N, cfg_.gen.t_min, cfg_.gen.t_max);
            GpuBuffer<bf16> soft_next;
            bool want_soft_next = !(skip_last_soft_next && step == 1);
            auto td0 = Clk::now();
            decode_step(current, soft ? soft->data() : nullptr, enc_len, temp,
                        argmax, entropy, denoiser, soft_next, rng, want_soft_next);
            stats_.decode_s += secs(td0, Clk::now());
            stats_.decode_passes += 1;

            std::vector<char> accepted;
            current = entropy_bound_accept_renoise(current, denoiser, entropy,
                                                   cfg_.gen.entropy_bound, V, rng,
                                                   &accepted);
            if (want_soft_next) soft.emplace(std::move(soft_next));

            bool stop = stopping.update(argmax, entropy);
            double me = 0; for (float e : entropy) me += e; me /= entropy.size();
            if (verbose)
                std::printf("[blk %d step %2d] temp=%.3f mean_entropy=%.4f%s\n",
                            blk, step, temp, me, stop ? "  [stop]" : "");
            if (on_step) {
                DiffStepEvent ev;
                ev.block = blk; ev.cur_step = step; ev.temperature = temp;
                ev.mean_entropy = (float)me; ev.committed = false;
                ev.canvas = &argmax; ev.entropy = &entropy; ev.accepted = &accepted;
                on_step(ev);
            }
            if (stop && !force_denoise_steps) break;
        }

        // 4. Find the first EOS in the denoised canvas; commit up to it.
        int keep = C;   // tokens of this block to keep
        for (int i = 0; i < C && keep == C; ++i)
            for (int e : cfg_.gen.eos_token_ids)
                if (argmax[i] == e) { keep = i; break; }
        bool hit_eos = (keep < C);

        std::vector<int> committed(argmax.begin(), argmax.begin() + keep);
        output.insert(output.end(), committed.begin(), committed.end());

        if (on_step) {
            DiffStepEvent ev;
            ev.block = blk; ev.cur_step = 0; ev.committed = true; ev.canvas = &committed;
            on_step(ev);
        }
        if (hit_eos) break;

        prev_canvas = argmax;
    }
    stats_.output_tokens = (int)output.size();
    diffprof::report();
    return output;
}

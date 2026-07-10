#include "model.hpp"
#include "sampler.hpp"
#include "device_sampler.hpp"
#include "../../utils/profile.hpp"
#include "../../common/gpu/engine.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <vector>

std::vector<int> DiffusionGemmaModel::generate(
    const std::vector<int>& prompt_ids, int max_new_tokens, int max_denoising_steps,
    unsigned seed, bool verbose, const DiffStreamCallback& on_step,
    bool ignore_eos)
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
    // DIFF_HOST_SAMPLER: keep the original fully host-side sampler + stopping
    // (host canvas, host entropy-bound accept/renoise, host stable-and-confident
    // check) for A/B against the device-resident default path.
    bool host_sampler = std::getenv("DIFF_HOST_SAMPLER") != nullptr;
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

        // 2-3. Denoise the canvas.  Two paths share the commit tail below:
        //   * device-resident (default): canvas / argmax / entropy / denoiser
        //     stay on the GPU; accept/renoise + stable-and-confident stop run as
        //     small device kernels on the same in-order queue.  The only host
        //     sync per step is the optional callback payload + 4-byte stop flag.
        //   * host sampler (DIFF_HOST_SAMPLER): the original path — host canvas,
        //     host entropy_bound_accept_renoise, host DiffStopping.  Kept for A/B.
        // Both leave the final denoised canvas in `argmax` (host).
        std::vector<int> argmax;

        if (host_sampler) {
            std::vector<int> current(C);
            for (int& t : current) t = uni(rng);
            std::optional<GpuBuffer<bf16>> soft;
            stopping.reset();
            argmax = current;
            std::vector<int> denoiser;
            std::vector<float> entropy;

            for (int step = N; step >= 1; --step) {
                float temp = diff_temperature(step, N, cfg_.gen.t_min, cfg_.gen.t_max);
                GpuBuffer<bf16> soft_next;
                bool want_soft_next = !(skip_last_soft_next && step == 1);
                auto td0 = Clk::now();
                decode_step(current, soft ? soft->data() : nullptr, enc_len, temp,
                            argmax, entropy, denoiser, soft_next, rng, want_soft_next,
                            (uint64_t)seed, (uint32_t)blk, (uint32_t)step);
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
        } else {
            ensure_device_buffers(C);
            auto& q0 = GpuEngine::get(0).queue;
            diffsamp::init_canvas_random(q0, canvas_dev_.data(), (uint64_t)seed,
                                         (uint32_t)blk, C, V);
            if (cfg_.gen.stability_threshold > 0)
                q0.fill(argmax_history_dev_.data(), (int32_t)-1,
                       (size_t)cfg_.gen.stability_threshold * C);
            int history_slot = 0;
            std::optional<GpuBuffer<bf16>> soft;
            GpuBuffer<bf16> soft_next_buf;
            std::vector<float> entropy_h;
            std::vector<char>  accepted_h;

            for (int step = N; step >= 1; --step) {
                float temp = diff_temperature(step, N, cfg_.gen.t_min, cfg_.gen.t_max);
                bool want_soft_next = !(skip_last_soft_next && step == 1);
                auto td0 = Clk::now();

                if (!diff_use_gumbel_sample())
                    diffsamp::fill_uniform(q0, u_dev_.data(), (uint64_t)seed,
                                           (uint32_t)blk, (uint32_t)step, C);
                decode_forward(canvas_dev_.data(), soft ? soft->data() : nullptr,
                               enc_len, C, temp, u_dev_.data(),
                               argmax_dev_.data(), entropy_dev_.data(),
                               denoiser_dev_.data(), soft_next_buf, want_soft_next,
                               (uint64_t)seed, (uint32_t)blk, (uint32_t)step);
                diffsamp::entropy_bound_renoise(q0, canvas_dev_.data(),
                                                denoiser_dev_.data(),
                                                entropy_dev_.data(), accepted_dev_.data(),
                                                (uint64_t)seed, (uint32_t)blk, (uint32_t)step,
                                                C, cfg_.gen.entropy_bound, V);
                if (diff_use_stop_fix())
                    diffsamp::stopping_check_fixed(q0, argmax_dev_.data(),
                                                    argmax_history_dev_.data(),
                                                    entropy_dev_.data(), mean_dev_.data(),
                                                    stop_dev_.data(),
                                                    cfg_.gen.confidence_threshold,
                                                    cfg_.gen.stability_threshold,
                                                    history_slot, C);
                else
                    diffsamp::stopping_check(q0, argmax_dev_.data(), argmax_history_dev_.data(),
                                             entropy_dev_.data(), mean_dev_.data(),
                                             stop_dev_.data(),
                                             cfg_.gen.confidence_threshold,
                                             cfg_.gen.stability_threshold,
                                             history_slot, C);
                if (cfg_.gen.stability_threshold > 0)
                    history_slot = (history_slot + 1) % cfg_.gen.stability_threshold;
                if (want_soft_next) soft.emplace(std::move(soft_next_buf));

                stats_.decode_s += secs(td0, Clk::now());
                stats_.decode_passes += 1;

                if (on_step || verbose || !force_denoise_steps) {
                    bool stop = false;
                    if (!force_denoise_steps) {
                        int32_t s = 0; stop_dev_.download(&s, 1); stop = (s != 0);
                    }
                    double me = 0.0;
                    if (on_step || verbose) {
                        entropy_h.resize(C); entropy_dev_.download(entropy_h.data(), C);
                        for (float e : entropy_h) me += e; me /= C;
                    }
                    if (verbose)
                        std::printf("[blk %d step %2d] temp=%.3f mean_entropy=%.4f%s\n",
                                    blk, step, temp, me, stop ? "  [stop]" : "");
                    if (on_step) {
                        argmax.resize(C); argmax_dev_.download(argmax.data(), C);
                        accepted_h.resize(C); accepted_dev_.download(accepted_h.data(), C);
                        DiffStepEvent ev;
                        ev.block = blk; ev.cur_step = step; ev.temperature = temp;
                        ev.mean_entropy = (float)me; ev.committed = false;
                        ev.canvas = &argmax; ev.entropy = &entropy_h; ev.accepted = &accepted_h;
                        on_step(ev);
                    }
                    if (stop && !force_denoise_steps) break;
                }
            }
            if (argmax.empty()) { argmax.resize(C); argmax_dev_.download(argmax.data(), C); }
        }

        // 4. Find the first EOS in the denoised canvas; commit up to it.
        // Throughput benchmarks may ignore EOS so one-block -n 256 runs always
        // count the complete finalized pool. This does not mask EOS logits or
        // alter the 48-step denoising trajectory; it only disables commit-time
        // truncation/stopping.
        int keep = C;   // tokens of this block to keep
        if (!ignore_eos) {
            for (int i = 0; i < C && keep == C; ++i)
                for (int e : cfg_.gen.eos_token_ids)
                    if (argmax[i] == e) { keep = i; break; }
        }
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
    nvfp4_sycl_graph_report();
    return output;
}

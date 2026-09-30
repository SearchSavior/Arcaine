// Run isolated structured reads. Do not capture graphs in this path.
#include "model.hpp"
#include "decision_validation.hpp"
#include "fusions/logits.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/gpu_watchdog.hpp"
#include "../../common/gpu/nvfp4.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct KvResetGuard {
    DiffKvCache& kv;
    ~KvResetGuard() { kv.reset(); }
};

// Draw a uniform token without modulo bias.
int32_t draw_noise_token(uint64_t stream_seed, uint32_t read_index, int vocab) {
    const uint64_t range = uint64_t{1} << 32;
    const uint64_t limit = range - (range % (uint64_t)vocab);
    for (uint32_t ctr = 0;; ++ctr) {
        uint32_t r = difflogits::rng_u32(stream_seed, read_index, ctr, 0);
        if ((uint64_t)r < limit) return (int32_t)(r % (uint64_t)vocab);
    }
}

void require(bool ok, const std::string& msg) {
    if (!ok) throw std::runtime_error("read_decisions: " + msg);
}

// Hold the current read aggregate.
struct ReadAggregate {
    std::vector<double> mean_q;   // averaged conditional label probabilities
    double mean_mass = 0.0;
    bool   latest_argmax_is_label = false;
};

ReadAggregate aggregate_reads(const std::vector<DecisionSlotSample>& reads,
                              int K, const std::vector<int32_t>& label_ids) {
    ReadAggregate agg;
    agg.mean_q.assign(K, 0.0);
    for (const auto& s : reads) {
        for (int j = 0; j < K; ++j)
            agg.mean_q[j] += std::exp((double)s.label_logprobs[j] - (double)s.log_label_mass);
        agg.mean_mass += std::exp((double)s.log_label_mass);
    }
    double n = (double)reads.size();
    for (int j = 0; j < K; ++j) agg.mean_q[j] /= n;
    agg.mean_mass /= n;
    const auto& last = reads.back();
    agg.latest_argmax_is_label =
        std::find(label_ids.begin(), label_ids.end(), last.argmax_id) != label_ids.end();
    return agg;
}

}  // namespace

DecisionReadResult DiffusionGemmaModel::read_decisions(
    const std::vector<int>& prompt_ids,
    const CompiledDecisionTemplate& compiled,
    const DecisionReadOptions& options)
{
    const int V = cfg_.text.vocab_size;

    // Validate before any GPU submission. Size checks run on the original
    // size_t values, so an oversized vector cannot narrow to a small count.
    validate_decision_read_request(V, kv_cache_max_seq(), canvas_capacity(),
                                   prompt_ids, compiled, options);

    const int C = (int)compiled.canvas.size();
    const int K = (int)compiled.label_ids.size();
    const int L = cfg_.text.num_hidden_layers;

    const int max_reads = options.auto_mode ? options.auto_max : options.reads;

    auto& ctx0 = GpuEngine::get(0);
    auto& q0 = ctx0.queue;

    // Disable graph capture for this read.
    Nvfp4EagerScope eager_scope;

    // Limit the time this read can hold the model lock. If the GPU stops, the
    // watchdog stops the process and writes a message.
    GpuWatchdog watchdog("systemone-read", options.watchdog_s);
    watchdog.beat("validation");

    enc_kv_.reset();
    KvResetGuard kv_guard{enc_kv_};

    DecisionReadResult result;
    result.prompt_tokens = (int)prompt_ids.size();

    using Clk = std::chrono::steady_clock;
    auto secs = [](Clk::time_point a, Clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    // Keep request buffers alive until error cleanup is complete.
    GpuBuffer<int32_t> canvas_dev(C, q0);
    GpuBuffer<int32_t> rows_dev(1, q0);
    GpuBuffer<int32_t> labels_dev(K, q0);
    GpuBuffer<float>   label_logp_dev(K, q0);
    GpuBuffer<float>   scalars_dev(2, q0);   // [0]=log label mass, [1]=argmax logprob
    GpuBuffer<float>   vocab_ent_dev(1, q0);
    GpuBuffer<int32_t> argmax_id_dev(1, q0);
    GpuBuffer<int32_t> nonfinite_dev(1, q0);
    {   int32_t row = compiled.slot_position;
        rows_dev.upload(&row, 1);
        labels_dev.upload(compiled.label_ids.data(), K);
    }

    // Error-cleanup test hooks. ARCAINE_SYSTEMONE_FAULT selects one point.
    static const std::string fault_mode = [] {
        const char* e = std::getenv("ARCAINE_SYSTEMONE_FAULT");
        return e ? std::string(e) : std::string();
    }();
    static std::atomic<bool> fault_fired{false};

    try {
        {   auto t0 = Clk::now();
            watchdog.beat("prefill");
            encode(prompt_ids, 0);
            result.prefill_s = secs(t0, Clk::now());
        }
        result.prefill_calls = 1;
        watchdog.beat("prefill done");

        if (fault_mode == "after_prefill" && !fault_fired.exchange(true))
            throw std::runtime_error(
                "read_decisions: injected fault after prefill (test hook)");

        for (int r = 0; r < max_reads; ++r) {
            auto td0 = Clk::now();

            std::vector<int32_t> canvas = compiled.canvas;
            canvas[compiled.slot_position] = draw_noise_token(options.stream_seed, (uint32_t)r, V);
            canvas_dev.upload(canvas.data(), C);

            DecisionScoreTarget target;
            target.rows           = rows_dev.data();
            target.label_ids      = labels_dev.data();
            target.num_rows       = 1;
            target.num_labels     = K;
            target.label_logprobs = label_logp_dev.data();
            target.log_label_mass = scalars_dev.data();
            target.vocab_entropy  = vocab_ent_dev.data();
            target.argmax_id      = argmax_id_dev.data();
            target.argmax_logprob = scalars_dev.data() + 1;
            target.nonfinite      = nonfinite_dev.data();

            GpuBuffer<bf16> soft_next;   // unused in structured mode
            watchdog.beat("decode");
            decode_forward(canvas_dev.data(), /*soft_or_null=*/nullptr,
                           /*enc_len=*/result.prompt_tokens, /*seq=*/C,
                           /*temp=*/1.0f, /*u_dev=*/nullptr,
                           /*argmax_dev=*/nullptr, /*entropy_dev=*/nullptr,
                           /*denoiser_dev=*/nullptr, soft_next,
                           /*want_soft_next=*/false,
                           /*rng_seed=*/0, /*rng_block=*/0, /*rng_step=*/0,
                           &target);

            if (fault_mode == "after_first_decode_submit" && r == 0 &&
                !fault_fired.exchange(true))
                throw std::runtime_error(
                    "read_decisions: injected fault after decode submission (test hook)");

            // Download only the compact result.
            watchdog.beat("download");
            DecisionSlotSample sample;
            sample.label_logprobs.resize(K);
            label_logp_dev.download(sample.label_logprobs.data(), K);
            float scalars[2];
            scalars_dev.download(scalars, 2);
            vocab_ent_dev.download(&sample.vocab_entropy, 1);
            argmax_id_dev.download(&sample.argmax_id, 1);
            int32_t nonfinite = 0;
            nonfinite_dev.download(&nonfinite, 1);
            sample.log_label_mass = scalars[0];
            sample.argmax_logprob = scalars[1];
            result.decode_s += secs(td0, Clk::now());
            result.decode_calls += 1;
            watchdog.beat("read done");

            // Reject nonfinite scores.
            require(nonfinite == 0,
                    "nonfinite raw logits from the model at the scored slot (NaN/inf)");
            for (float v : sample.label_logprobs)
                require(std::isfinite(v), "nonfinite label logprob from the model");
            require(std::isfinite(sample.log_label_mass) &&
                    std::isfinite(sample.vocab_entropy) &&
                    std::isfinite(sample.argmax_logprob),
                    "nonfinite slot statistics from the model");

            result.reads.push_back(std::move(sample));

            // Stop when the auto policy is satisfied.
            if (options.auto_mode && r + 1 < max_reads) {
                ReadAggregate agg = aggregate_reads(result.reads, K, compiled.label_ids);
                double h = 0.0;
                for (int j = 0; j < K; ++j)
                    if (agg.mean_q[j] > 0.0) h -= agg.mean_q[j] * std::log(agg.mean_q[j]);
                double h_norm = h / std::log((double)K);
                bool escalate = h_norm > options.auto_entropy_threshold ||
                                agg.mean_mass < options.auto_min_label_mass ||
                                !agg.latest_argmax_is_label;
                if (!escalate) break;
            }
        }
    } catch (...) {
        // Drain queues before buffers are released. Keep the original error.
        try { q0.wait(); } catch (...) {}
        if (split_layer_ < L) { try { GpuEngine::get(1).queue.wait(); } catch (...) {} }
        throw;
    }
    return result;
}

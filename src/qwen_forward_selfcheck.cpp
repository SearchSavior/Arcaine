// Milestone-6 build/run smoke for the Qwen3.5-MoE end-to-end forward loop
// (modeling/qwen3_5_moe/model.cpp forward_tokens).
//
// Constructs a QwenModel capped to a few layers (ARCAINE_QWEN_MAX_LAYERS, or 4
// by default in this harness) so the full 40-layer / 22GB load is avoided, then
// drives forward_tokens directly with synthetic token ids — bypassing the BPE
// tokenizer (which still has the null-bos_token chat_template bug; see notes).
// Verifies:
//   * logits size == vocab_size
//   * prefill: finite, deterministic (bit-exact across 2 runs), nonzero,
//     not-all-equal (forward isn't degenerate)
//   * decode (prefill then +1 token): finite
//   * reset_cache then re-prefill == first prefill (cache reset works)
//
//   build : arcaine-dev-1
//   run   : arcaine-dev-run-*  ARCAINE_QWEN_MODEL_DIR=<qwen checkpoint dir>
// Numerical correctness vs HF is Phase 6 (deferred).
#include "common/gpu/engine.hpp"
#include "common/gpu/buffer.hpp"
#include "modeling/qwen3_5_moe/model.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using bf16 = uint16_t;

int main() {
    const char* dir_env = std::getenv("ARCAINE_QWEN_MODEL_DIR");
    std::string model_dir = dir_env ? dir_env
                                    : "/workspace/models/Qwen-AgentWorld-35B-A3B-NVFP4";
    // Cap layers for a fast smoke (override via the existing env var).
    if (!std::getenv("ARCAINE_QWEN_MAX_LAYERS"))
        setenv("ARCAINE_QWEN_MAX_LAYERS", "4", 1);

    int fails = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("[fwd-selfcheck] %-26s %s\n", name, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    };

    try {
        QwenModel model(model_dir);
        const QwenConfig& cfg = model.config();
        int V = cfg.vocab_size;
        auto& q = GpuEngine::get(0).queue;
        std::printf("[fwd-selfcheck] device: %s  V=%d  loaded_layers=%zu\n",
            q.get_device().get_info<sycl::info::device::name>().c_str(),
            V, model.weights().layers.size());

        // Synthetic token ids in a sane range (vocab is 248320). Deterministic.
        auto synth_ids = [&](int n, std::vector<int>& out) {
            out.resize(n);
            unsigned int s = 0xABCDEF01u;
            for (int i = 0; i < n; ++i) {
                s = s * 1103515245u + 12345u;
                out[i] = (int)((s >> 8) % (unsigned)V);
            }
        };

        // --- Prefill (8 tokens, past_len=0) ---
        std::vector<int> ids8; synth_ids(8, ids8);
        auto run = [&](const std::vector<int>& ids, int past) -> std::vector<float> {
            ForwardInput in{ids, past, nullptr, nullptr, nullptr};
            return model.forward(in);
        };
        // Model starts with clean (zeroed) caches, so the first prefill is from a
        // fresh state. Determinism is over reset->run (a prefill MUST start from a
        // reset cache — the linear-attn SSM state accumulates across calls by
        // design, so two prefills without a reset are NOT expected to match).
        model.reset_cache();
        std::vector<float> l1 = run(ids8, 0);
        bool finite = true, nonzero = false;
        float first = l1.empty() ? 0.0f : l1[0];
        bool all_eq = true;
        for (float f : l1) {
            if (!std::isfinite(f)) finite = false;
            if (f != 0.0f) nonzero = true;
            if (f != first) all_eq = false;
        }
        check("prefill:size", (int)l1.size() == V);
        check("prefill:finite", finite);
        check("prefill:nonzero", nonzero);
        check("prefill:not_degenerate", !all_eq);

        model.reset_cache();
        std::vector<float> l2 = run(ids8, 0);
        bool det = (l1.size() == l2.size());
        for (size_t i = 0; det && i < l1.size(); ++i)
            if (l1[i] != l2[i]) det = false;
        check("prefill:deterministic", det);

        // --- Decode (reset -> prefill 8 -> +1 token at past_len=8) ---
        std::vector<int> id1; synth_ids(1, id1);
        model.reset_cache();
        run(ids8, 0);                 // prime cache with 8 tokens
        std::vector<float> ld = run(id1, 8);
        bool dfinite = (int)ld.size() == V;
        for (float f : ld) if (!std::isfinite(f)) dfinite = false;
        check("decode:size+finite", dfinite);

        // --- reset_cache then re-prefill == first prefill ---
        model.reset_cache();
        std::vector<float> l3 = run(ids8, 0);
        bool reset_ok = (l3.size() == l1.size());
        for (size_t i = 0; reset_ok && i < l1.size(); ++i)
            if (l3[i] != l1[i]) reset_ok = false;
        check("reset_cache:reproducible", reset_ok);

        std::printf("[fwd-selfcheck] %s (%d failures)\n",
                    fails ? "FAILED" : "ALL PASSED", fails);
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("[fwd-selfcheck] EXCEPTION: %s\n", e.what());
        return 2;
    }
}

// Milestone-4 build/run smoke for the Qwen3.5-MoE block forward
// (modeling/qwen3_5_moe/moe.hpp).
//
// Loads one layer's full MoE block (router BF16, 256 routed NVFP4 experts, shared
// NVFP4 expert, shared_expert_gate BF16) straight from the checkpoint — no
// QwenWeights / no attention — then runs qwen_moe_forward on a synthetic hidden
// and checks output finiteness, determinism (bit-exact across two runs),
// nonzeroness, and zero-input -> zero-output. Exercises the prefill (S>1) and
// decode (S==1) paths.
//
//   build : arcaine-dev-1
//   run   : arcaine-dev-run-*  ARCAINE_QWEN_MODEL_DIR=<qwen checkpoint dir>
//                                ARCAINE_QWEN_MOE_LAYER=<layer idx; default 0>
//
// Numerical correctness vs the HF reference is Phase 6 (deferred); this is the
// milestone-4 build/run smoke (analogous to qwen_attn_selfcheck).
#include "common/io/quant_loader.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/buffer.hpp"
#include "modeling/qwen3_5_moe/config.hpp"
#include "modeling/qwen3_5_moe/weights.hpp"
#include "modeling/qwen3_5_moe/moe.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using bf16 = uint16_t;

static const char* kPrefix = "model.language_model.";

int main() {
    const char* dir_env = std::getenv("ARCAINE_QWEN_MODEL_DIR");
    std::string model_dir = dir_env ? dir_env
                                    : "/workspace/models/Qwen-AgentWorld-35B-A3B-NVFP4";
    int layer = 0;
    if (const char* e = std::getenv("ARCAINE_QWEN_MOE_LAYER")) layer = std::atoi(e);

    int fails = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("[moe-selfcheck] %-24s %s\n", name, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    };

    try {
        QwenConfig cfg = QwenConfig::from_dir(model_dir);
        auto& q = GpuEngine::get(0).queue;
        std::printf("[moe-selfcheck] device: %s  layer=%d  experts=%d top%d inter=%d\n",
            q.get_device().get_info<sycl::info::device::name>().c_str(), layer,
            cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size);

        ShardedSafetensors sf(model_dir);
        const std::string mp = std::string(kPrefix) + "layers." +
                               std::to_string(layer) + ".mlp.";

        QwenMoE w;
        w.router_gate = upload(sf.get(mp + "gate.weight"), q,
                               (mp + "gate.weight").c_str());
        w.experts_gate_up.reserve(cfg.num_experts);
        w.experts_down.reserve(cfg.num_experts);
        for (int e = 0; e < cfg.num_experts; ++e) {
            const std::string ep = mp + "experts." + std::to_string(e) + ".";
            w.experts_gate_up.push_back(
                upload_nvfp4_linear_pair(sf, ep + "gate_proj", ep + "up_proj", q));
            w.experts_down.push_back(upload_nvfp4_linear(sf, ep + "down_proj", q));
        }
        const std::string sep = mp + "shared_expert.";
        w.shared_gate_up = upload_nvfp4_linear_pair(sf, sep + "gate_proj",
                                                   sep + "up_proj", q);
        w.shared_down    = upload_nvfp4_linear(sf, sep + "down_proj", q);
        w.shared_expert_gate =
            upload(sf.get(mp + "shared_expert_gate.weight"), q,
                   (mp + "shared_expert_gate.weight").c_str());
        std::printf("[moe-selfcheck] loaded %d experts + shared\n", cfg.num_experts);

        const int H = cfg.hidden_size;
        auto synth_hidden = [&](int S, std::vector<bf16>& out) {
            out.assign((size_t)S * H, 0);
            unsigned int seed = 0xC0FFEEu;
            for (auto& v : out) {
                seed = seed * 1103515245u + 12345u;
                float r = ((seed >> 16) & 0x7fffu) / 32768.0f;   // [0,1)
                v = float_to_bf16(r * 0.2f - 0.1f);
            }
        };

        auto run = [&](int S, const std::vector<bf16>& hhost, std::vector<bf16>& ohost) {
            GpuBuffer<bf16> dh((size_t)S * H, q); dh.upload(hhost.data(), (size_t)S * H);
            GpuBuffer<bf16> dout((size_t)S * H, q);
            q.wait();
            qwen_moe_forward(GpuEngine::get(0), w, dh.data(), dout.data(), S, cfg);
            q.wait();
            ohost.assign((size_t)S * H, 0);
            dout.download(ohost.data(), (size_t)S * H);
        };

        // --- Prefill (S>1) ---
        {
            const int S = 4;
            std::vector<bf16> hhost; synth_hidden(S, hhost);
            std::vector<bf16> o1, o2;
            run(S, hhost, o1);
            run(S, hhost, o2);
            bool finite = true, nonzero = false;
            for (bf16 b : o1) {
                float f = bf16_to_float(b);
                if (!std::isfinite(f)) finite = false;
                if (b != 0) nonzero = true;
            }
            check("prefill:finite", finite);
            check("prefill:deterministic", o1 == o2);
            check("prefill:nonzero", nonzero);

            // zero-in -> zero-out (router uniform; experts(0)=0; shared(0)=0; sigmoid(0)*0=0)
            std::vector<bf16> zh((size_t)S * H, float_to_bf16(0.0f)), oz;
            run(S, zh, oz);
            bool zok = true;
            for (bf16 b : oz) if (b != 0) { zok = false; break; }
            check("prefill:zero_in_zero_out", zok);
        }

        // --- Decode (S==1) ---
        {
            const int S = 1;
            std::vector<bf16> hhost; synth_hidden(S, hhost);
            std::vector<bf16> o1, o2;
            run(S, hhost, o1);
            run(S, hhost, o2);
            bool finite = true;
            for (bf16 b : o1) if (!std::isfinite(bf16_to_float(b))) finite = false;
            check("decode:finite", finite);
            check("decode:deterministic", o1 == o2);
        }

        std::printf("[moe-selfcheck] %s (%d failures)\n",
                    fails ? "FAILED" : "ALL PASSED", fails);
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("[moe-selfcheck] EXCEPTION: %s\n", e.what());
        return 2;
    }
}

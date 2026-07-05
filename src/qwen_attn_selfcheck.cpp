// Milestone-3 build/run smoke for the Qwen3.5 full-attention forward
// (modeling/qwen3_5_moe/attention.hpp).
//
// Loads one full-attention layer's real NVFP4 (q/k/v/o_proj) + BF16 (q/k_norm,
// with +1 baked) weights straight from the checkpoint — no MoE, no full
// QwenWeights — then runs qwen_full_attention_forward on a synthetic hidden and
// checks output finiteness, determinism (bit-exact across two runs), and the
// zero-input -> zero-output structural sanity. Both the prefill (seq>1, causal
// mask) and decode (seq==1, skip_mask) code paths are exercised.
//
//   build : arcaine-dev-1   (SPIR-V codegen needs no physical GPU)
//   run   : arcaine-dev-run-*  ARCAINE_QWEN_MODEL_DIR=<qwen checkpoint dir>
//                              ARCAINE_QWEN_ATTN_LAYER=<full-attn layer idx; default 3>
//
// Numerical correctness vs the HF reference is Phase 6 (deferred); this is the
// milestone-3 build/run smoke (analogous to qwen_kernels_selfcheck for kernels.hpp).
#include "common/io/quant_loader.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/buffer.hpp"
#include "modeling/qwen3_5_moe/config.hpp"
#include "modeling/qwen3_5_moe/weights.hpp"
#include "modeling/qwen3_5_moe/attention.hpp"

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
    int layer = 3;
    if (const char* e = std::getenv("ARCAINE_QWEN_ATTN_LAYER")) layer = std::atoi(e);

    int fails = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("[attn-selfcheck] %-24s %s\n", name, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    };

    try {
        QwenConfig cfg = QwenConfig::from_dir(model_dir);
        if (!cfg.is_full_attn(layer))
            throw std::runtime_error("layer " + std::to_string(layer) +
                                     " is not full_attention (pick a full-attn layer)");
        auto& q = GpuEngine::get(0).queue;
        std::printf("[attn-selfcheck] device: %s  layer=%d\n",
            q.get_device().get_info<sycl::info::device::name>().c_str(), layer);

        ShardedSafetensors sf(model_dir);
        const std::string ap = std::string(kPrefix) + "layers." +
                               std::to_string(layer) + ".self_attn.";

        QwenFullAttn w;
        w.q_proj = upload_nvfp4_linear(sf, ap + "q_proj", q);
        w.k_proj = upload_nvfp4_linear(sf, ap + "k_proj", q);
        w.v_proj = upload_nvfp4_linear(sf, ap + "v_proj", q);
        w.o_proj = upload_nvfp4_linear(sf, ap + "o_proj", q);
        w.q_norm = upload_plus_one(sf.get(ap + "q_norm.weight"), q,
                                  (ap + "q_norm.weight").c_str());
        w.k_norm = upload_plus_one(sf.get(ap + "k_norm.weight"), q,
                                  (ap + "k_norm.weight").c_str());

        const int max_seq = 512;
        const int H = cfg.hidden_size;
        auto make_kv = [&]() -> QwenKvLayer {
            QwenKvLayer kv;
            kv.max_seq = max_seq;
            size_t n = (size_t)max_seq * cfg.num_key_value_heads * cfg.head_dim;
            kv.k = GpuBuffer<bf16>(n, q);
            kv.v = GpuBuffer<bf16>(n, q);
            kv.filled = 0;
            return kv;
        };

        // Seeded synthetic hidden in ~[-0.1, 0.1) (deterministic, nonzero).
        auto synth_hidden = [&](int S, std::vector<bf16>& out) {
            out.assign((size_t)S * H, 0);
            unsigned int seed = 0xC0FFEEu;
            for (auto& v : out) {
                seed = seed * 1103515245u + 12345u;
                float r = ((seed >> 16) & 0x7fffu) / 32768.0f;   // [0,1)
                v = float_to_bf16(r * 0.2f - 0.1f);
            }
        };

        // --- Prefill path (seq>1, causal mask, skip_mask=false) ---
        {
            const int S = 4;
            std::vector<bf16> hhost; synth_hidden(S, hhost);
            GpuBuffer<bf16> dh((size_t)S * H, q); dh.upload(hhost.data(), (size_t)S * H);
            GpuBuffer<bf16> dout((size_t)S * H, q);

            auto run = [&]() -> std::vector<bf16> {
                QwenKvLayer kv = make_kv();
                q.wait();
                qwen_full_attention_forward(GpuEngine::get(0), w, kv, dh.data(),
                                            dout.data(), S, /*past_len=*/0, cfg);
                q.wait();
                std::vector<bf16> out((size_t)S * H);
                dout.download(out.data(), (size_t)S * H);
                return out;
            };
            auto o1 = run();
            auto o2 = run();

            bool finite = true, nonzero = false;
            for (bf16 b : o1) {
                float f = bf16_to_float(b);
                if (!std::isfinite(f)) finite = false;
                if (b != 0) nonzero = true;
            }
            check("prefill:finite", finite);
            check("prefill:deterministic", o1 == o2);
            check("prefill:nonzero", nonzero);

            // zero-input -> zero-output (Q=K=V=0; softmax(uniform)@0=0; *sigmoid(0)=0; o_proj(0)=0)
            std::vector<bf16> zhost((size_t)S * H, float_to_bf16(0.0f));
            GpuBuffer<bf16> dz((size_t)S * H, q); dz.upload(zhost.data(), (size_t)S * H);
            GpuBuffer<bf16> doz((size_t)S * H, q);
            QwenKvLayer kv0 = make_kv();
            q.wait();
            qwen_full_attention_forward(GpuEngine::get(0), w, kv0, dz.data(),
                                        doz.data(), S, 0, cfg);
            q.wait();
            std::vector<bf16> oz((size_t)S * H); doz.download(oz.data(), (size_t)S * H);
            bool zok = true;
            for (bf16 b : oz) if (b != 0) { zok = false; break; }
            check("prefill:zero_in_zero_out", zok);
        }

        // --- Decode path (seq==1, skip_mask=true) ---
        {
            const int S = 1;
            std::vector<bf16> hhost; synth_hidden(S, hhost);
            GpuBuffer<bf16> dh((size_t)S * H, q); dh.upload(hhost.data(), (size_t)S * H);
            GpuBuffer<bf16> dout((size_t)S * H, q);

            auto run = [&]() -> std::vector<bf16> {
                QwenKvLayer kv = make_kv();
                q.wait();
                qwen_full_attention_forward(GpuEngine::get(0), w, kv, dh.data(),
                                            dout.data(), S, /*past_len=*/0, cfg);
                q.wait();
                std::vector<bf16> out((size_t)S * H);
                dout.download(out.data(), (size_t)S * H);
                return out;
            };
            auto o1 = run();
            auto o2 = run();
            bool finite = true;
            for (bf16 b : o1) if (!std::isfinite(bf16_to_float(b))) finite = false;
            check("decode:finite", finite);
            check("decode:deterministic", o1 == o2);
        }

        std::printf("[attn-selfcheck] %s (%d failures)\n",
                    fails ? "FAILED" : "ALL PASSED", fails);
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("[attn-selfcheck] EXCEPTION: %s\n", e.what());
        return 2;
    }
}

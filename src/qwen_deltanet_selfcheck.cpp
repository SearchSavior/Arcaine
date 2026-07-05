// Milestone-5 build/run smoke for the Qwen3.5-MoE Gated DeltaNet forward
// (modeling/qwen3_5_moe/gated_deltanet.hpp).
//
// Loads one linear-attn layer's real BF16 (in_proj_qkv/z/a/b, conv1d, A_log,
// dt_bias, norm — plain upload, ones-init) + NVFP4 (out_proj) weights straight
// from the checkpoint — no MoE, no QwenWeights — then runs
// qwen_linear_attn_forward on a synthetic hidden and checks:
//   * prefill (S>1): finite, deterministic, nonzero, zero-in->zero-out
//   * decode  (S==1, post-prefill): finite, deterministic
//   * EQUIVALENCE: prefill(N)[last] == prefill(N-1) + decode(last token)
//     This cross-validates the two independent core implementations (host
//     chunked vs device recurrent) — the strongest self-check we have short of
//     a golden reference (Phase 6, deferred).
//
//   build : arcaine-dev-1   (SPIR-V codegen needs no physical GPU)
//   run   : arcaine-dev-run-*  ARCAINE_QWEN_MODEL_DIR=<qwen checkpoint dir>
//                              ARCAINE_QWEN_DELTANET_LAYER=<linear-attn idx; default 0>
//
// Numerical correctness vs the HF reference is Phase 6 (deferred); this is the
// milestone-5 build/run smoke (analogous to qwen_attn_selfcheck / qwen_moe_selfcheck).
#include "common/io/quant_loader.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/buffer.hpp"
#include "modeling/qwen3_5_moe/config.hpp"
#include "modeling/qwen3_5_moe/weights.hpp"
#include "modeling/qwen3_5_moe/gated_deltanet.hpp"

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
    int layer = 0;
    if (const char* e = std::getenv("ARCAINE_QWEN_DELTANET_LAYER")) layer = std::atoi(e);

    int fails = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("[deltanet-selfcheck] %-28s %s\n", name, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    };

    try {
        QwenConfig cfg = QwenConfig::from_dir(model_dir);
        if (cfg.is_full_attn(layer))
            throw std::runtime_error("layer " + std::to_string(layer) +
                                     " is full_attention (pick a linear-attn layer)");
        auto& q = GpuEngine::get(0).queue;
        std::printf("[deltanet-selfcheck] device: %s  layer=%d (linear-attn)\n",
            q.get_device().get_info<sycl::info::device::name>().c_str(), layer);

        ShardedSafetensors sf(model_dir);
        const std::string ap = std::string("model.language_model.layers.") +
                               std::to_string(layer) + ".linear_attn.";

        QwenLinearAttn w;
        w.in_proj_qkv = upload(sf.get(ap + "in_proj_qkv.weight"), q,
                              (ap + "in_proj_qkv.weight").c_str());
        w.in_proj_z   = upload(sf.get(ap + "in_proj_z.weight"), q,
                              (ap + "in_proj_z.weight").c_str());
        w.in_proj_a   = upload(sf.get(ap + "in_proj_a.weight"), q,
                              (ap + "in_proj_a.weight").c_str());
        w.in_proj_b   = upload(sf.get(ap + "in_proj_b.weight"), q,
                              (ap + "in_proj_b.weight").c_str());
        w.conv1d      = upload(sf.get(ap + "conv1d.weight"), q,
                              (ap + "conv1d.weight").c_str());
        // A_log / dt_bias are bare parameters (no .weight suffix).
        w.A_log       = upload(sf.get(ap + "A_log"), q, (ap + "A_log").c_str());
        w.dt_bias     = upload(sf.get(ap + "dt_bias"), q, (ap + "dt_bias").c_str());
        // linear-attn gated norm: ones-init, PLAIN scale (no +1 baked).
        w.norm        = upload(sf.get(ap + "norm.weight"), q,
                              (ap + "norm.weight").c_str());
        w.out_proj    = upload_nvfp4_linear(sf, ap + "out_proj", q);

        const int H = cfg.hidden_size;
        int n_v = cfg.linear_num_value_heads;
        int d_k = cfg.linear_key_head_dim;
        int d_v = cfg.linear_value_head_dim;
        int kk  = cfg.linear_conv_kernel_dim;
        int key_dim   = cfg.linear_num_key_heads * d_k;
        int value_dim = n_v * d_v;
        int conv_dim  = 2 * key_dim + value_dim;

        auto make_cache = [&]() -> QwenLinearAttnCache {
            QwenLinearAttnCache c;
            c.conv_state = GpuBuffer<bf16>((size_t)conv_dim * (kk - 1), q);
            c.ssm_state  = GpuBuffer<float>((size_t)n_v * d_k * d_v, q);
            q.memset(c.conv_state.data(), 0, (size_t)conv_dim * (kk - 1) * sizeof(bf16));
            q.memset(c.ssm_state.data(),  0, (size_t)n_v * d_k * d_v * sizeof(float));
            c.has_state = false;
            q.wait();
            return c;
        };

        auto synth_hidden = [&](int S, std::vector<bf16>& out, unsigned int seed0 = 0xC0FFEEu) {
            out.assign((size_t)S * H, 0);
            unsigned int seed = seed0;
            for (auto& v : out) {
                seed = seed * 1103515245u + 12345u;
                float r = ((seed >> 16) & 0x7fffu) / 32768.0f;   // [0,1)
                v = float_to_bf16(r * 0.2f - 0.1f);
            }
        };

        // run prefill (sets has_state); downloads [S,H] output.
        auto run = [&](QwenLinearAttnCache& cache, const std::vector<bf16>& hhost,
                       int S, int past_len, std::vector<bf16>& ohost) {
            GpuBuffer<bf16> dh((size_t)S * H, q); dh.upload(hhost.data(), (size_t)S * H);
            GpuBuffer<bf16> dout((size_t)S * H, q);
            q.wait();
            qwen_linear_attn_forward(GpuEngine::get(0), w, cache, dh.data(),
                                     dout.data(), S, past_len, cfg);
            q.wait();
            ohost.assign((size_t)S * H, 0);
            dout.download(ohost.data(), (size_t)S * H);
        };

        // --- Prefill (S>1): finite / deterministic / nonzero / zero-in->zero-out ---
        {
            const int S = 4;
            std::vector<bf16> hhost; synth_hidden(S, hhost);
            std::vector<bf16> o1, o2;
            { QwenLinearAttnCache c = make_cache(); run(c, hhost, S, 0, o1); }
            { QwenLinearAttnCache c = make_cache(); run(c, hhost, S, 0, o2); }
            bool finite = true, nonzero = false;
            for (bf16 b : o1) {
                float f = bf16_to_float(b);
                if (!std::isfinite(f)) finite = false;
                if (b != 0) nonzero = true;
            }
            check("prefill:finite", finite);
            check("prefill:deterministic", o1 == o2);
            check("prefill:nonzero", nonzero);

            // zero-in -> zero-out: in_proj(0)=0; conv(silu(0))=0; q=k=v=0; beta=0.5;
            //   g=-exp(A_log)*softplus(0+dt_bias) (generally !=0) but q=0 -> core=0;
            //   gated_rmsnorm(0,z)*silu(z): rmsnorm(0)=0 -> 0; out_proj(0)=0.
            std::vector<bf16> zh((size_t)S * H, float_to_bf16(0.0f)), oz;
            { QwenLinearAttnCache c = make_cache(); run(c, zh, S, 0, oz); }
            bool zok = true;
            for (bf16 b : oz) if (b != 0) { zok = false; break; }
            check("prefill:zero_in_zero_out", zok);
        }

        // --- Decode (S==1, post-prefill): finite / deterministic ---
        {
            const int Spre = 3;
            std::vector<bf16> hpre; synth_hidden(Spre + 1, hpre);  // 4 tokens
            std::vector<bf16> hpre3(hpre.begin(), hpre.begin() + (size_t)Spre * H);
            std::vector<bf16> hdec(hpre.begin() + (size_t)Spre * H, hpre.end());

            auto decode_run = [&](std::vector<bf16>& out_dec) {
                QwenLinearAttnCache c = make_cache();
                std::vector<bf16> opre; run(c, hpre3, Spre, 0, opre);  // prime cache
                run(c, hdec, 1, Spre, out_dec);                        // decode token
            };
            std::vector<bf16> o1, o2;
            decode_run(o1);
            decode_run(o2);
            bool finite = true;
            for (bf16 b : o1) if (!std::isfinite(bf16_to_float(b))) finite = false;
            check("decode:finite", finite);
            check("decode:deterministic", o1 == o2);
        }

        // --- EQUIVALENCE: prefill(N)[last] == prefill(N-1) + decode(last) ---
        // Stress-test the chunked-vs-recurrent equivalence across several seeds and
        // lengths (incl. N>chunk_size so inter-chunk state passing is exercised).
        // Same in_proj/conv/l2norm/scale for the last token in both paths -> the only
        // divergence is the core delta-rule numerics (host chunked vs device recurrent)
        // + BF16. Report per-combo max_abs/max_rel; require nonzero + small rel error.
        {
            struct Case { int N; unsigned int seed; };
            Case cases[] = {{8, 0xC0FFEEu}, {12, 0x1234u}, {128, 0xBEEFu}, {200, 0x5A5Au}};
            bool all_ok = true;
            for (const auto& cs : cases) {
                const int N = cs.N;
                std::vector<bf16> hh; synth_hidden(N, hh, cs.seed);
                std::vector<bf16> hpre7(hh.begin(), hh.begin() + (size_t)(N - 1) * H);
                std::vector<bf16> hlast(hh.begin() + (size_t)(N - 1) * H, hh.end());

                std::vector<bf16> oA;
                { QwenLinearAttnCache c = make_cache(); run(c, hh, N, 0, oA); }
                std::vector<bf16> oA_last(oA.begin() + (size_t)(N - 1) * H, oA.end());

                std::vector<bf16> oB_last;
                {
                    QwenLinearAttnCache c = make_cache();
                    std::vector<bf16> opre; run(c, hpre7, N - 1, 0, opre);
                    run(c, hlast, 1, N - 1, oB_last);
                }

                float max_abs = 0.0f, max_rel = 0.0f, a_max = 0.0f, b_max = 0.0f;
                bool finite = true;
                for (int i = 0; i < H; ++i) {
                    float a = bf16_to_float(oA_last[i]);
                    float b = bf16_to_float(oB_last[i]);
                    if (!std::isfinite(a) || !std::isfinite(b)) { finite = false; continue; }
                    if (std::fabs(a) > a_max) a_max = std::fabs(a);
                    if (std::fabs(b) > b_max) b_max = std::fabs(b);
                    float d = std::fabs(a - b);
                    if (d > max_abs) max_abs = d;
                    float rel = d / (1e-3f + std::fabs(a));
                    if (rel > max_rel) max_rel = rel;
                }
                bool nonzero = (a_max > 0.0f) && (b_max > 0.0f);
                bool ok = finite && nonzero && max_rel < 0.10f;
                std::printf("[deltanet-selfcheck] equiv[N=%3d seed=%#x]: max_abs=%.4g max_rel=%.4g |A|<=%.4g |B|<=%.4g -> %s\n",
                            N, cs.seed, max_abs, max_rel, a_max, b_max, ok ? "OK" : "FAIL");
                if (!ok) all_ok = false;
            }
            check("equiv:prefill_vs_decode", all_ok);
        }

        std::printf("[deltanet-selfcheck] %s (%d failures)\n",
                    fails ? "FAILED" : "ALL PASSED", fails);
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("[deltanet-selfcheck] EXCEPTION: %s\n", e.what());
        return 2;
    }
}

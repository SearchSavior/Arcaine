// W4A8 accuracy probe for the planned ngen s8x4 matmul_int4 port.
//
// Loads a qwen3_5 AWQ checkpoint once and compares the baseline int4 forward
// against the s8-activation-quantized simulation (ARCAINE_INT4_ACT_QUANT_S8,
// toggled in-process via int4_act_quant_set). Reports prefill-logit metrics
// (cosine, rms_rel, KL, top-k overlap) and greedy-decode token divergence.
// With --stats, the baseline prefill additionally runs in stats mode, printing
// per-matmul error lines (call order follows layer order).
//
//   ./build/qwen35_actquant_probe -m <model_dir> [--prompt <text>]
//       [--gen <n>] [--stats] [--max-seq <n>]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "modeling/qwen3_5/model.hpp"
#include "runtime/quantization/int4.hpp"

namespace {

struct RunResult {
    std::vector<float> logits;   // prefill logits (last prompt position)
    std::vector<int> generated;  // greedy continuation
};

int argmax(const std::vector<float>& v) {
    return (int)std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

// Greedy continuation from the current cache state; does not reset the cache
// or touch `logits`.
void decode_continue(Qwen35Model& model, const std::vector<float>& logits,
                     int past, int gen, std::vector<int>& out) {
    std::vector<float> current = logits;
    for (int step = 0; step < gen; ++step) {
        int id = argmax(current);
        out.push_back(id);
        std::vector<int> next{id};
        current = model.forward(ForwardInput{next, past, nullptr, nullptr, nullptr});
        past += 1;
    }
}

RunResult run(Qwen35Model& model, const PreparedInput& prep, int gen) {
    model.reset_cache();
    const std::vector<int32_t>* mm =
        prep.mm_token_type_ids.empty() ? nullptr : &prep.mm_token_type_ids;
    RunResult out;
    out.logits = model.forward(ForwardInput{prep.tokens, 0, nullptr, nullptr, mm});
    decode_continue(model, out.logits, (int)prep.tokens.size(), gen, out.generated);
    return out;
}

void compare_logits(const std::vector<float>& base, const std::vector<float>& test,
                    int topk) {
    double dot = 0, nb = 0, nt = 0, sd = 0;
    float max_abs = 0;
    for (size_t i = 0; i < base.size(); ++i) {
        dot += (double)base[i] * test[i];
        nb += (double)base[i] * base[i];
        nt += (double)test[i] * test[i];
        double d = (double)base[i] - test[i];
        sd += d * d;
        max_abs = std::max(max_abs, std::fabs(base[i] - test[i]));
    }
    double cos = dot / (std::sqrt(nb) * std::sqrt(nt) + 1e-30);
    double rms_rel = std::sqrt(sd / base.size()) / (std::sqrt(nb / base.size()) + 1e-12);

    // KL(base || test) in fp64 over softmaxes.
    auto log_softmax = [](const std::vector<float>& v) {
        float mx = *std::max_element(v.begin(), v.end());
        double sum = 0;
        for (float x : v) sum += std::exp((double)x - mx);
        double lse = std::log(sum) + mx;
        std::vector<double> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = (double)v[i] - lse;
        return out;
    };
    auto lb = log_softmax(base), lt = log_softmax(test);
    double kl = 0;
    for (size_t i = 0; i < base.size(); ++i) kl += std::exp(lb[i]) * (lb[i] - lt[i]);

    auto top_ids = [&](const std::vector<float>& v) {
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&](int a, int b) { return v[a] > v[b]; });
        idx.resize(topk);
        std::sort(idx.begin(), idx.end());
        return idx;
    };
    auto tb = top_ids(base), tt = top_ids(test);
    int overlap = 0;
    for (int id : tb)
        if (std::binary_search(tt.begin(), tt.end(), id)) ++overlap;

    std::printf("[logits] cosine=%.8f rms_rel=%.6g max_abs=%.6g KL=%.6g "
                "argmax base=%d test=%d top-%d overlap=%d/%d\n",
                cos, rms_rel, (double)max_abs, kl, argmax(base), argmax(test),
                topk, overlap, topk);
}

void compare_decode(const std::vector<int>& base, const std::vector<int>& test) {
    int diverge = -1;
    for (size_t i = 0; i < base.size() && i < test.size(); ++i)
        if (base[i] != test[i]) { diverge = (int)i; break; }
    std::printf("[decode] gen=%zu match=%s", base.size(),
                diverge < 0 ? "full" : "diverged");
    if (diverge >= 0)
        std::printf(" first_divergence=%d (base=%d test=%d)", diverge,
                    base[diverge], test[diverge]);
    std::printf("\n[decode] base tokens:");
    for (int id : base) std::printf(" %d", id);
    std::printf("\n[decode] test tokens:");
    for (int id : test) std::printf(" %d", id);
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir, prompt =
        "The Intel Arc Pro B70 is a discrete GPU. Explain in a few sentences "
        "how systolic matrix engines accelerate transformer inference.";
    std::vector<float> clips{1.0f};
    int gen = 16, max_seq = 2048, topk = 10;
    bool stats = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_dir = next("--model");
        else if (a == "--prompt") prompt = next("--prompt");
        else if (a == "--gen") gen = std::atoi(next("--gen"));
        else if (a == "--max-seq") max_seq = std::atoi(next("--max-seq"));
        else if (a == "--topk") topk = std::atoi(next("--topk"));
        else if (a == "--stats") stats = true;
        else if (a == "--clip") {
            clips.clear();
            std::string v = next("--clip");
            size_t pos = 0;
            while (pos <= v.size()) {
                size_t comma = v.find(',', pos);
                clips.push_back(std::stof(v.substr(pos, comma - pos)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }
    if (model_dir.empty()) {
        std::fprintf(stderr, "usage: qwen35_actquant_probe -m <model_dir> "
                             "[--prompt <text>] [--gen <n>] [--stats]\n");
        return 1;
    }

    Qwen35Model model(model_dir, max_seq);
    auto prep = model.prepare_input(prompt, {}, {}, "");
    std::printf("[probe] prompt tokens=%zu, gen=%d, stats=%s\n",
                prep.tokens.size(), gen, stats ? "on" : "off");

    RunResult base;
    {
        model.reset_cache();
        const std::vector<int32_t>* mm =
            prep.mm_token_type_ids.empty() ? nullptr : &prep.mm_token_type_ids;
        int4_act_quant_set(stats ? Int4ActQuantMode::Stats : Int4ActQuantMode::Off);
        base.logits = model.forward(ForwardInput{prep.tokens, 0, nullptr, nullptr, mm});
        int4_act_quant_set(Int4ActQuantMode::Off);  // stats pass = prefill only
        decode_continue(model, base.logits, (int)prep.tokens.size(), gen,
                        base.generated);
    }

    for (float clip : clips) {
        std::printf("[apply] clip=%.3f\n", (double)clip);
        int4_act_quant_set_clip(clip);
        int4_act_quant_set(Int4ActQuantMode::Apply);
        RunResult test = run(model, prep, gen);
        int4_act_quant_set(Int4ActQuantMode::Off);
        compare_logits(base.logits, test.logits, topk);
        compare_decode(base.generated, test.generated);
    }
    return 0;
}

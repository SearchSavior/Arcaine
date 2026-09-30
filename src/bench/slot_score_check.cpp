// slot_score_check — numerical validation of the structured-read scoring
// kernel (fusions/logits.hpp slot_score_rows) against an FP64 CPU reference
// over the exact same synthetic logits.
//
// Cases per the implementation spec: a label outside the top-20, extreme
// logits, exact ties, very low label mass, 255 labels, true FP32 label-mass
// underflow, and NaN/+inf/-inf raw logits (softcap on and off).  The check
// covers the conditional label distribution q_j = exp(logp_j - log_mass) that
// the endpoint actually returns, its normalization, vocabulary entropy, and
// the nonfinite flag.  Targets: abs error <= 1e-4 for logprobs/entropy,
// <= 1e-5 for probabilities.
//
// Build & run (inside the arcaine-dev-1 container):
//   docker exec arcaine-dev-1 sh -c 'cd /workspace && icpx -fsycl -O2 \
//     -fsycl-targets=intel_gpu_bmg_g31 \
//     src/bench/slot_score_check.cpp \
//     -o /tmp/slot_score_check -I src -L/opt/onednn/lib -ldnnl \
//     -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate && \
//     ONEAPI_DEVICE_SELECTOR=level_zero:0 /tmp/slot_score_check'
#include "../modeling/diffusion_gemma/fusions/logits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(const char* name, bool ok, double err, double tol) {
    std::printf("[%s] %-40s err=%.3e tol=%.0e\n", ok ? "PASS" : "FAIL", name, err, tol);
    if (!ok) ++failures;
}

struct Ref {
    std::vector<double> label_logp;
    std::vector<double> cond;      // exp(label_logp - log_mass)
    double log_mass = 0, vocab_entropy = 0, argmax_logp = 0;
    int argmax_id = -1;
};

// FP64 reference over the same bf16-rounded logits.
Ref reference(const std::vector<bf16>& logits, int row, int V, float softcap,
              const std::vector<int32_t>& labels) {
    auto proc = [&](int c) {
        double v = bf16_to_float(logits[(size_t)row * V + c]);
        return softcap > 0.0f ? std::tanh(v / softcap) * softcap : v;
    };
    double m = -1e300;
    int am = 0;
    std::vector<double> z(V);
    for (int c = 0; c < V; ++c) {
        z[c] = proc(c);
        if (z[c] > m) { m = z[c]; am = c; }
    }
    double zs = 0.0;
    for (int c = 0; c < V; ++c) zs += std::exp(z[c] - m);
    double L = m + std::log(zs);
    double h = 0.0;
    for (int c = 0; c < V; ++c) {
        double p = std::exp(z[c] - L);
        if (p > 0) h -= p * (z[c] - L);
    }
    Ref r;
    r.argmax_id = am;
    r.argmax_logp = z[am] - L;
    r.vocab_entropy = h;
    double lm = -1e300;
    for (int32_t id : labels) {
        double lp = z[id] - L;
        r.label_logp.push_back(lp);
        lm = (lp > lm) ? lp + std::log1p(std::exp(lm - lp))
                       : lm + std::log1p(std::exp(lp - lm));
    }
    r.log_mass = lm;
    for (double lp : r.label_logp) r.cond.push_back(std::exp(lp - lm));
    return r;
}

// Expects the device nonfinite flag to be set; the numeric reference is not
// meaningful for nonfinite inputs.
void run_nonfinite_case(sycl::queue& q, const char* name, int seq, int V,
                        float softcap, const std::vector<bf16>& logits, int row,
                        const std::vector<int32_t>& labels) {
    GpuBuffer<bf16> logits_dev(logits.size(), q);
    logits_dev.upload(logits.data(), logits.size());
    GpuBuffer<int32_t> rows_dev(1, q), labels_dev(labels.size(), q);
    {   int32_t r = row; rows_dev.upload(&r, 1);
        labels_dev.upload(labels.data(), labels.size()); }
    int K = (int)labels.size();
    GpuBuffer<float> logp_dev(K, q), scal_dev(2, q), vent_dev(1, q);
    GpuBuffer<int32_t> amax_dev(1, q), nf_dev(1, q);
    slot_score_rows(q, logits_dev.data(), softcap, rows_dev.data(),
                    labels_dev.data(), 1, K, logp_dev.data(), scal_dev.data(),
                    vent_dev.data(), amax_dev.data(), scal_dev.data() + 1,
                    nf_dev.data(), V);
    int32_t nf = 0;
    nf_dev.download(&nf, 1);
    check(name, nf == 1, nf == 1 ? 0.0 : 1.0, 0.5);
}

void run_case(sycl::queue& q, const char* name, int seq, int V, float softcap,
              const std::vector<bf16>& logits, int row,
              const std::vector<int32_t>& labels) {
    GpuBuffer<bf16> logits_dev(logits.size(), q);
    logits_dev.upload(logits.data(), logits.size());
    GpuBuffer<int32_t> rows_dev(1, q), labels_dev(labels.size(), q);
    {   int32_t r = row; rows_dev.upload(&r, 1);
        labels_dev.upload(labels.data(), labels.size()); }
    int K = (int)labels.size();
    GpuBuffer<float> logp_dev(K, q), scal_dev(2, q), vent_dev(1, q);
    GpuBuffer<int32_t> amax_dev(1, q), nf_dev(1, q);

    slot_score_rows(q, logits_dev.data(), softcap, rows_dev.data(),
                    labels_dev.data(), 1, K, logp_dev.data(), scal_dev.data(),
                    vent_dev.data(), amax_dev.data(), scal_dev.data() + 1,
                    nf_dev.data(), V);

    std::vector<float> logp(K);
    float scal[2], vent;
    int32_t amax, nf = 0;
    logp_dev.download(logp.data(), K);
    scal_dev.download(scal, 2);
    vent_dev.download(&vent, 1);
    amax_dev.download(&amax, 1);
    nf_dev.download(&nf, 1);

    Ref ref = reference(logits, row, V, softcap, labels);

    check((std::string(name) + " finite-flag").c_str(), nf == 0, nf, 0.5);

    double lp_err = 0.0, p_err = 0.0, q_err = 0.0;
    double q_sum = 0.0;
    for (int j = 0; j < K; ++j) {
        lp_err = std::max(lp_err, std::fabs((double)logp[j] - ref.label_logp[j]));
        p_err = std::max(p_err, std::fabs(std::exp((double)logp[j]) -
                                          std::exp(ref.label_logp[j])));
        double q = std::exp((double)logp[j] - (double)scal[0]);
        q_err = std::max(q_err, std::fabs(q - ref.cond[j]));
        q_sum += q;
    }
    check((std::string(name) + " label_logp").c_str(), lp_err <= 1e-4, lp_err, 1e-4);
    check((std::string(name) + " label_prob").c_str(), p_err <= 1e-5, p_err, 1e-5);
    check((std::string(name) + " conditional_prob").c_str(), q_err <= 1e-5, q_err, 1e-5);
    check((std::string(name) + " cond_normalized").c_str(),
          std::fabs(q_sum - 1.0) <= 1e-4, std::fabs(q_sum - 1.0), 1e-4);
    check((std::string(name) + " log_mass").c_str(),
          std::fabs(scal[0] - ref.log_mass) <= 1e-4, std::fabs(scal[0] - ref.log_mass), 1e-4);
    check((std::string(name) + " vocab_entropy").c_str(),
          std::fabs(vent - ref.vocab_entropy) <= 1e-4, std::fabs(vent - ref.vocab_entropy), 1e-4);
    check((std::string(name) + " argmax").c_str(),
          amax == ref.argmax_id && std::fabs(scal[1] - ref.argmax_logp) <= 1e-4,
          std::fabs(scal[1] - ref.argmax_logp), 1e-4);
}

constexpr bf16 kInf  = 0x7F80;
constexpr bf16 kNegInf = 0xFF80;
constexpr bf16 kNaN  = 0x7FC0;

}  // namespace

int main() {
    // In-order queue, matching the engine's queues: GpuBuffer download()
    // relies on queue ordering against the submitted kernels.
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("[device] %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    const float cap = 30.0f;
    std::mt19937 rng(1234);

    {
        // Case A: random logits, one label deliberately outside the top-20.
        const int seq = 32, V = 300000;
        std::vector<bf16> logits((size_t)seq * V);
        std::normal_distribution<float> nd(0.0f, 4.0f);
        for (auto& v : logits) v = float_to_bf16(nd(rng));
        int row = 17;
        std::vector<float> rowvals(V);
        for (int c = 0; c < V; ++c) rowvals[c] = bf16_to_float(logits[(size_t)row * V + c]);
        std::vector<int> order(V);
        for (int c = 0; c < V; ++c) order[c] = c;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return rowvals[a] > rowvals[b]; });
        std::vector<int32_t> labels = {order[0], order[3], order[19], order[1000],
                                       order[V / 2], order[V - 1]};
        run_case(q, "A random+tail-label", seq, V, cap, logits, row, labels);
    }
    {
        // Case B: extreme logits (pre-softcap magnitudes far beyond the cap)
        // and exact ties among the top labels.
        const int seq = 16, V = 262144;
        std::vector<bf16> logits((size_t)seq * V, float_to_bf16(-1e4f));
        int row = 5;
        auto set = [&](int c, float v) { logits[(size_t)row * V + c] = float_to_bf16(v); };
        set(11, 1e4f); set(22, 1e4f);          // exact tie at the top
        set(33, 9000.0f); set(44, -9000.0f);
        std::vector<int32_t> labels = {11, 22, 33, 44};
        run_case(q, "B extremes+ties", seq, V, cap, logits, row, labels);
    }
    {
        // Case C: very low label mass — one hot token, labels deep below.
        const int seq = 16, V = 262144;
        std::vector<bf16> logits((size_t)seq * V, float_to_bf16(-20.0f));
        int row = 3;
        logits[(size_t)row * V + 7] = float_to_bf16(25.0f);   // dominant
        std::vector<int32_t> labels = {100, 200};             // both at -20
        run_case(q, "C low-label-mass", seq, V, cap, logits, row, labels);
    }
    {
        // Case D: softcap disabled (identity) with moderate logits.
        const int seq = 32, V = 65536;
        std::vector<bf16> logits((size_t)seq * V);
        std::uniform_real_distribution<float> ud(-12.0f, 12.0f);
        for (auto& v : logits) v = float_to_bf16(ud(rng));
        std::vector<int32_t> labels = {1, 5000, 65535};
        run_case(q, "D no-softcap", seq, V, 0.0f, logits, 31, labels);
    }
    {
        // Case E: 255 labels, all retained, conditional distribution normalizes.
        // (The vLLM example's 128-token cap is deliberately not inherited.)
        const int seq = 16, V = 262144;
        std::vector<bf16> logits((size_t)seq * V);
        std::normal_distribution<float> nd(0.0f, 3.0f);
        for (auto& v : logits) v = float_to_bf16(nd(rng));
        int row = 9;
        std::vector<int32_t> labels;
        for (int j = 0; j < 255; ++j) labels.push_back(3 + j * 1000);
        run_case(q, "E 255-labels", seq, V, cap, logits, row, labels);
    }
    {
        // Case F: true FP32 label-probability underflow.  Softcap off, raw span
        // 160 nats -> exp(logp) underflows float32, but the log-space label mass
        // and conditional q_j must stay finite and correct.
        const int seq = 8, V = 262144;
        std::vector<bf16> logits((size_t)seq * V, float_to_bf16(-80.0f));
        int row = 2;
        logits[(size_t)row * V + 1] = float_to_bf16(80.0f);
        std::vector<int32_t> labels = {100, 200};             // both at -80
        run_case(q, "F fp32-underflow", seq, V, 0.0f, logits, row, labels);
    }

    // Case G: nonfinite raw logits must be flagged, softcap on and off.  With
    // softcap these would otherwise become finite tanh values.
    {
        const int seq = 4, V = 4096;
        std::vector<int32_t> labels = {5, 7};
        for (float sc : {cap, 0.0f}) {
            const std::string tag = sc > 0 ? "G inf softcap" : "G inf no-softcap";
            std::vector<bf16> logits((size_t)seq * V, float_to_bf16(0.5f));
            logits[(size_t)1 * V + 5] = kInf;
            run_nonfinite_case(q, (tag + " (+inf)").c_str(), seq, V, sc, logits, 1, labels);

            logits.assign((size_t)seq * V, float_to_bf16(0.5f));
            logits[(size_t)1 * V + 7] = kNegInf;
            run_nonfinite_case(q, (tag + " (-inf)").c_str(), seq, V, sc, logits, 1, labels);

            logits.assign((size_t)seq * V, float_to_bf16(0.5f));
            logits[(size_t)1 * V + 5] = kNaN;
            run_nonfinite_case(q, (tag + " (NaN)").c_str(), seq, V, sc, logits, 1, labels);
        }
    }

    std::printf(failures ? "\n%d FAILURES\n" : "\nall numerical checks passed\n",
                failures);
    return failures ? 1 : 0;
}

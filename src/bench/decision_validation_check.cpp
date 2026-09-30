// decision_validation_check — exercises the direct model-API input validation
// (decision_validation.hpp) without loading a model or touching a GPU.
//
// Build & run (inside the arcaine-dev-1 container):
//   docker exec arcaine-dev-1 sh -c 'cd /workspace && icpx -fsycl -O2 \
//     src/bench/decision_validation_check.cpp \
//     -I src -I third_party -I/opt/onednn/include \
//     -o /tmp/decision_validation_check && /tmp/decision_validation_check'
#include "../modeling/diffusion_gemma/decision_validation.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

CompiledDecisionTemplate tmpl(int C, int K, int slot = 2) {
    CompiledDecisionTemplate t;
    t.canvas.assign(C, 5);
    t.slot_position = slot;
    t.label_ids.resize(K);
    for (int j = 0; j < K; ++j) t.label_ids[j] = 100 + j;
    return t;
}

void expect_ok(const char* name, const std::vector<int>& prompt,
               const CompiledDecisionTemplate& t, const DecisionReadOptions& o,
               int V = 1000, int max_seq = 64, int cap = 16) {
    try {
        validate_decision_read_request(V, max_seq, cap, prompt, t, o);
        std::printf("[PASS] %s\n", name);
    } catch (const std::exception& e) {
        std::printf("[FAIL] %s (unexpected throw: %s)\n", name, e.what());
        ++failures;
    }
}

void expect_throw(const char* name, const std::vector<int>& prompt,
                  const CompiledDecisionTemplate& t, const DecisionReadOptions& o,
                  int V = 1000, int max_seq = 64, int cap = 16) {
    try {
        validate_decision_read_request(V, max_seq, cap, prompt, t, o);
        std::printf("[FAIL] %s (expected a throw)\n", name);
        ++failures;
    } catch (const std::exception&) {
        std::printf("[PASS] %s\n", name);
    }
}

}  // namespace

int main() {
    DecisionReadOptions ok;  // reads=1, defaults

    // Valid: short canvas 16, boundary prompt exactly fills max_seq with canvas.
    expect_ok("valid: short canvas + boundary prompt",
              std::vector<int>(48, 1), tmpl(16, 2), ok, 1000, 64, 16);
    expect_ok("valid: minimum canvas width 1",
              std::vector<int>{1}, tmpl(1, 2, 0), ok, 1000, 64, 16);
    expect_ok("valid: K=255 labels",
              std::vector<int>{1, 2}, tmpl(16, 255), ok);

    // Prompt ids.
    expect_throw("reject: negative prompt id",
                 std::vector<int>{1, -1, 2}, tmpl(16, 2), ok);
    expect_throw("reject: prompt id == vocab",
                 std::vector<int>{1, 1000}, tmpl(16, 2), ok, 1000);
    expect_throw("reject: empty prompt", {}, tmpl(16, 2), ok);

    // Capacity / overflow-safe sizes.
    expect_throw("reject: prompt + canvas > KV capacity",
                 std::vector<int>(49, 1), tmpl(16, 2), ok, 1000, 64, 16);
    expect_throw("reject: canvas width above planned capacity",
                 std::vector<int>{1}, tmpl(17, 2), ok, 1000, 64, 16);
    expect_throw("reject: zero-width canvas",
                 std::vector<int>{1}, tmpl(0, 2, 0), ok);
    expect_throw("reject: prompt alone exceeds KV capacity",
                 std::vector<int>(65, 1), tmpl(1, 2, 0), ok, 1000, 64, 16);

    // Template / labels.
    expect_throw("reject: slot position outside canvas",
                 std::vector<int>{1}, tmpl(16, 2, 16), ok);
    expect_throw("reject: K=1", std::vector<int>{1}, tmpl(16, 1), ok);
    expect_throw("reject: K=257", std::vector<int>{1}, tmpl(16, 257), ok);
    {
        auto t = tmpl(16, 2);
        t.label_ids = {100, 100};
        expect_throw("reject: duplicate label id", std::vector<int>{1}, t, ok);
    }
    {
        auto t = tmpl(16, 2);
        t.label_ids = {100, 1000};
        expect_throw("reject: label id out of vocab", std::vector<int>{1}, t, ok, 1000);
    }
    {
        auto t = tmpl(16, 2);
        t.canvas[5] = 1000;
        expect_throw("reject: canvas token out of vocab", std::vector<int>{1}, t, ok, 1000);
    }

    // Read policy.
    {
        DecisionReadOptions o; o.reads = 0;
        expect_throw("reject: reads=0", std::vector<int>{1}, tmpl(16, 2), o);
    }
    {
        DecisionReadOptions o; o.reads = 33;
        expect_throw("reject: reads=33", std::vector<int>{1}, tmpl(16, 2), o);
    }
    {
        DecisionReadOptions o; o.auto_max = 0;
        expect_throw("reject: auto_max=0", std::vector<int>{1}, tmpl(16, 2), o);
    }
    {
        DecisionReadOptions o; o.auto_entropy_threshold = 1.5f;
        expect_throw("reject: entropy threshold > 1", std::vector<int>{1}, tmpl(16, 2), o);
    }
    {
        DecisionReadOptions o; o.auto_min_label_mass = -0.1f;
        expect_throw("reject: negative label mass", std::vector<int>{1}, tmpl(16, 2), o);
    }
    {
        DecisionReadOptions o; o.watchdog_s = 30.0f;
        expect_ok("valid: watchdog timeout 30s", std::vector<int>{1}, tmpl(16, 2), o);
    }
    {
        DecisionReadOptions o; o.watchdog_s = -1.0f;
        expect_throw("reject: negative watchdog timeout", std::vector<int>{1}, tmpl(16, 2), o);
    }
    {
        DecisionReadOptions o; o.watchdog_s = 3601.0f;
        expect_throw("reject: watchdog timeout > 3600", std::vector<int>{1}, tmpl(16, 2), o);
    }

    std::printf(failures ? "\n%d FAILURES\n" : "\nall validation checks passed\n",
                failures);
    return failures ? 1 : 0;
}

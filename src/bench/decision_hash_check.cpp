// decision_hash_check — properties of the structured-read content stream.
//
// The stream seed must depend only on canonical question content, never on the
// external question ID, the caller's question order, or unrelated questions.
// The field encoding is length-prefixed with an explicit little-endian 64-bit
// length, so fields containing separators or control characters cannot merge.
//
// Build & run (inside arcaine-dev-1):
//   docker exec arcaine-dev-1 sh -c 'cd /workspace && icpx -fsycl -O2 \
//     -fsycl-targets=intel_gpu_bmg_g31 \
//     src/bench/decision_hash_check.cpp \
//     src/utils/decision_schema.cpp src/utils/chat.cpp \
//     src/common/preprocess/tokenizer.cpp src/common/preprocess/unicode.cpp \
//     src/common/preprocess/unicode-data.cpp src/common/preprocess/chat_template.cpp \
//     -I src -I third_party -I third_party/minja/include \
//     -o /tmp/decision_hash_check && /tmp/decision_hash_check'
#include "utils/decision_schema.hpp"

#include <cstdio>
#include <string>
#include <vector>

using nlohmann::ordered_json;
using arcaine::diffusion_gemma::DecisionRequestSettings;

static int failures = 0;
static void check(const char* name, bool ok, const std::string& detail = {}) {
    std::printf("[%s] %s%s\n", ok ? "PASS" : "FAIL", name,
                (!ok && !detail.empty()) ? ("  -- " + detail).c_str() : "");
    if (!ok) ++failures;
}

static uint64_t seed_for(const std::string& state, const ordered_json& q,
                         uint64_t base_seed = 42) {
    DecisionRequestSettings s;
    s.base_seed = base_seed;
    ordered_json body = {{"model", "x"}, {"state", state},
                         {"questions", {{"qid", q}}}};
    auto req = arcaine::diffusion_gemma::parse_decision_request(body, s);
    return req.questions[0].stream_seed;
}

int main() {
    const std::string state = "The ticket is down.";
    ordered_json q1 = {{"type", "choice"}, {"instructions", "I"},
                       {"criteria", {{"x", "d"}, {"y", "e"}, {"z", "f"}}}};
    ordered_json q2 = {{"type", "choice"}, {"instructions", "I\nx=\"d\""},
                       {"criteria", {{"y", "e"}, {"z", "f"}}}};

    // The historical concatenation collision must now hash differently.
    const uint64_t s1 = seed_for(state, q1);
    const uint64_t s2 = seed_for(state, q2);
    check("concatenation-collision example hashes differently", s1 != s2,
          std::to_string(s1) + " vs " + std::to_string(s2));
    std::printf("  q1 seed=%llu  q2 seed=%llu\n",
                (unsigned long long)s1, (unsigned long long)s2);

    // Fields containing separators/control characters stay distinct.
    ordered_json c1 = {{"type", "choice"}, {"instructions", std::string("a\x1f") + "b"},
                       {"criteria", {{"x", "d"}, {"y", "e"}}}};
    ordered_json c2 = {{"type", "choice"}, {"instructions", "a"},
                       {"criteria", {{std::string("\x1f") + "b", "d"}, {"y", "e"}}}};
    check("control-character fields hash differently",
          seed_for(state, c1) != seed_for(state, c2));

    // A length-prefix example: "ab"+"c" must differ from "a"+"bc".
    ordered_json d1 = {{"type", "noul"}, {"instructions", "ab"},
                       {"criteria", {{"true", "c"}, {"false", "d"}}}};
    ordered_json d2 = {{"type", "noul"}, {"instructions", "a"},
                       {"criteria", {{"true", "bc"}, {"false", "d"}}}};
    check("field-boundary example hashes differently",
          seed_for(state, d1) != seed_for(state, d2));

    // External ID and question order must not change the seed or execution order.
    DecisionRequestSettings s;
    s.base_seed = 42;
    ordered_json A = {{"model", "x"}, {"state", state},
        {"questions", {{"urgent", q1}, {"route", q2}}}};
    ordered_json B = {{"model", "x"}, {"state", state},
        {"questions", {{"zzz", q2}, {"aaa", q1}}}};   // renamed + reordered
    auto ra = arcaine::diffusion_gemma::parse_decision_request(A, s);
    auto rb = arcaine::diffusion_gemma::parse_decision_request(B, s);
    std::vector<uint64_t> sa, sb;
    std::vector<std::string> ca, cb;
    for (auto& q : ra.questions) { sa.push_back(q.stream_seed); ca.push_back(q.canonical_content); }
    for (auto& q : rb.questions) { sb.push_back(q.stream_seed); cb.push_back(q.canonical_content); }
    // Compare by canonical content, ignoring order.
    bool same_seed = true;
    for (size_t i = 0; i < ca.size(); ++i) {
        // find the same canonical content in B
        size_t j = 0; while (j < cb.size() && cb[j] != ca[i]) ++j;
        if (j == cb.size() || sb[j] != sa[i]) same_seed = false;
    }
    check("rename/reorder keeps canonical-content seeds equal", same_seed);
    check("rename/reorder keeps canonical execution order equal", ca == cb,
          "A=[" + (ca.size() > 0 ? ca[0] : "") + ",...] B=[" +
          (cb.size() > 0 ? cb[0] : "") + ",...]");

    // Unrelated-question invariance: adding a distractor must not change seeds.
    ordered_json C = {{"model", "x"}, {"state", state},
        {"questions", {{"urgent", q1},
                       {"distractor", {{"type", "noul"}, {"instructions", "green sky?"}}},
                       {"route", q2}}}};
    auto rc = arcaine::diffusion_gemma::parse_decision_request(C, s);
    bool invariant = true;
    for (auto& q : rc.questions)
        if (q.canonical_content == ca[0] && q.stream_seed != sa[0]) invariant = false;
    check("unrelated question does not change another question's seed", invariant);

    // Golden vectors (documented FNV-1a-64 over LE-length-prefixed fields,
    // stream version "arcaine-systemone-v1"). A change here is a stream-version
    // change and must be documented.
    check("golden seed q1", s1 == 17734819047203551209ULL,
          std::to_string(s1));
    check("golden seed q2", s2 == 9788230640868820864ULL,
          std::to_string(s2));
    std::printf("  GOLDEN q1=%llu q2=%llu\n",
                (unsigned long long)s1, (unsigned long long)s2);

    std::printf("\n%s\n", failures ? "HASH CHECKS FAILED" : "all hash checks passed");
    return failures ? 1 : 0;
}

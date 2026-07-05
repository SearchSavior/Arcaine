#include "preprocess/tokenizer.hpp"
#include <cstdio>
#include <string>
#include <vector>

// Escape control chars for display.
static std::string esc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if      (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c == '\r') o += "\\r";
        else if (c < 32 || c == 127) { char b[8]; std::snprintf(b, sizeof(b), "\\x%02X", c); o += b; }
        else o += (char)c;
    }
    return o;
}

static void print_ids(const std::vector<int>& v) {
    for (int id : v) std::printf("%d ", id);
}

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1]
        : "/workspace/models/Qwen-AgentWorld-35B-A3B-NVFP4";
    auto tok = Tokenizer::from_json(model_dir + "/tokenizer.json");
    std::printf("Vocab size: %d\n\n", tok.vocab_size());

    // 6 HF reference encodings for Qwen2 byte-level BPE (add_bos=false, no EOS).
    struct Case { std::string text; std::vector<int> expected; };
    std::vector<Case> cases = {
        {"Hello!",                    {9419, 0}},
        {"Hello! How are you today?",  {9419, 0, 2500, 513, 488, 3242, 30}},
        {"I'm doing well, thank you.", {40, 2688, 3604, 1575, 11, 9414, 488, 13}},
        {"  double  spaces",          {220, 1923, 220, 12258}},
        {"line1\nline2",              {1021, 16, 198, 1021, 17}},
        {"numbers 12345 here",        {36142, 220, 16, 17, 18, 19, 20, 1532}},
    };

    int fails = 0;
    for (const auto& c : cases) {
        auto ids = tok.encode(c.text, /*add_bos=*/false);
        bool ok = (ids == c.expected);
        if (!ok) ++fails;
        std::printf("%s  encode(\"%s\")\n", ok ? "PASS" : "FAIL", esc(c.text).c_str());
        if (!ok) {
            std::printf("    expected: "); print_ids(c.expected); std::printf("\n");
            std::printf("    actual:   "); print_ids(ids);       std::printf("\n");
        }

        // Round-trip decode (no leading-space strip — matches HF byte-level).
        auto dec = tok.decode(ids, /*skip_special=*/false, /*strip_leading_space=*/false);
        bool rt = (dec == c.text);
        if (!rt) ++fails;
        std::printf("%s  decode -> \"%s\"\n\n", rt ? "PASS" : "FAIL", esc(dec).c_str());
    }

    std::printf("=== %d failure(s) ===\n", fails);
    return fails ? 1 : 0;
}

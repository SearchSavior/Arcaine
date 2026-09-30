// tokenizer_parity_check — prints the native Tokenizer's BOS-free encodings for
// a JSON array of fixture strings read from stdin, so a reference tokenizer can
// be compared against it.  Used to verify that the in-process tokenizer (the
// one structured-read template compilation relies on) matches the checkpoint's
// HF tokenizer.json on representative prompts and answer templates.
//
// Build & run (inside the arcaine-dev-1 container):
//   docker exec arcaine-dev-1 sh -c 'cd /workspace && icpx -O2 \
//     src/bench/tokenizer_parity_check.cpp \
//     src/common/preprocess/tokenizer.cpp src/common/preprocess/unicode.cpp \
//     src/common/preprocess/unicode-data.cpp -I src -o /tmp/tokenizer_parity_check && \
//     echo "[\"answer: A\"]" | /tmp/tokenizer_parity_check \
//       models/diffusiongemma-26B-A4B-it-AWQ-INT4/tokenizer.json'
#include "common/preprocess/tokenizer.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: tokenizer_parity_check <tokenizer.json> < fixtures.json\n";
        return 2;
    }
    Tokenizer tok = Tokenizer::from_json(argv[1]);
    nlohmann::json fixtures;
    std::cin >> fixtures;
    if (!fixtures.is_array()) {
        std::cerr << "input must be a JSON array of strings\n";
        return 2;
    }
    nlohmann::json out = nlohmann::json::array();
    for (const auto& f : fixtures) {
        std::vector<int> ids = tok.encode(f.get<std::string>(), /*add_bos=*/false);
        out.push_back(ids);
    }
    std::cout << out.dump() << std::endl;
    return 0;
}

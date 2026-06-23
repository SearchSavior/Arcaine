#include "preprocess/tokenizer.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "/workspace/models/gemma-4-12B-it";
    auto tok = Tokenizer::from_json(model_dir + "/tokenizer.json");
    std::printf("Vocab size: %d\n", tok.vocab_size());

    // Check specific vocab entries
    struct Check { std::string text; };

    auto test = [&](const std::string& prompt) {
        auto ids = tok.encode(prompt, true);
        std::printf("encode(%s):\n  ids: ", prompt.c_str());
        for (int id : ids) std::printf("%d ", id);
        std::printf("\n  decoded: %s\n", tok.decode(ids).c_str());
    };

    test("The capital of France is");
    test("Hello world");
    test("Paris");

    return 0;
}

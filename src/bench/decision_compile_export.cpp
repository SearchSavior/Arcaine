// decision_compile_export — run the real structured-read compiler and emit its
// artifacts as JSON, so an independent checkpoint-tokenizer construction can be
// compared against the tokens Arcaine actually uses. This is not a reimplemented
// tokenizer: it calls parse_decision_request + compile_decision_questions, the
// same path the HTTP route uses.
//
// Usage:
//   decision_compile_export <model_dir> <fixtures.json> [--seed N] [--max-seq N]
//       [--canvas-step N] [--canvas-max N] [--pad-token-id N] [--turn-close-id N]
//
// <fixtures.json> is an array of request bodies (same schema as /v1/systemone).
// Output is a JSON object:
//   {"settings": {...}, "results": [{"index":i,"questions":[...]}|{"index":i,"error":"..."}]}
// Each compiled question reports prompt_ids, canvas, slot_position, label_ids,
// label_texts, prompt_tokens, stream_seed, canonical_content, generation_suffix.
#include "utils/decision_schema.hpp"
#include "utils/chat.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using nlohmann::ordered_json;

namespace {

int arg_int(int argc, char** argv, const std::string& name, int fallback) {
    for (int i = 3; i + 1 < argc; ++i)
        if (name == argv[i]) return std::atoi(argv[i + 1]);
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: decision_compile_export <model_dir> <fixtures.json> "
                     "[--seed N] [--max-seq N] [--canvas-step N] [--canvas-max N] "
                     "[--pad-token-id N] [--turn-close-id N]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    std::ifstream in(argv[2]);
    if (!in) { std::cerr << "cannot open " << argv[2] << "\n"; return 2; }
    ordered_json fixtures;
    try {
        in >> fixtures;
    } catch (const std::exception& e) {
        std::cerr << "fixture JSON error: " << e.what() << "\n";
        return 2;
    }
    if (!fixtures.is_array()) { std::cerr << "fixtures must be an array\n"; return 2; }

    TokenizerBridge tok(model_dir);

    arcaine::diffusion_gemma::DecisionRequestSettings s;
    s.base_seed     = (uint64_t)arg_int(argc, argv, "--seed", 42);
    s.max_seq       = arg_int(argc, argv, "--max-seq", 8192);
    s.canvas_step   = arg_int(argc, argv, "--canvas-step", 16);
    s.canvas_max    = arg_int(argc, argv, "--canvas-max", 256);
    s.vocab_size    = tok.vocab_size();
    {
        int pad = arg_int(argc, argv, "--pad-token-id", -2);
        if (pad == -2) pad = tok.has_token("<pad>") ? tok.token_id("<pad>") : 0;
        s.pad_token_id = pad;
    }
    {
        int tc = arg_int(argc, argv, "--turn-close-id", -2);
        if (tc == -2) {
            if (tok.has_token("<turn|>")) tc = tok.token_id("<turn|>");
            else if (tok.has_token("<end_of_turn>")) tc = tok.token_id("<end_of_turn>");
            else tc = -1;
        }
        s.turn_close_id = tc;
    }

    ordered_json out;
    out["settings"] = {
        {"vocab_size", s.vocab_size},
        {"pad_token_id", s.pad_token_id},
        {"turn_close_id", s.turn_close_id},
        {"max_seq", s.max_seq},
        {"canvas_step", s.canvas_step},
        {"canvas_max", s.canvas_max},
        {"base_seed", s.base_seed},
    };
    ordered_json results = ordered_json::array();
    for (size_t i = 0; i < fixtures.size(); ++i) {
        ordered_json entry;
        entry["index"] = (int)i;
        try {
            auto body = arcaine::diffusion_gemma::parse_json_no_duplicates(fixtures[i].dump());
            auto req = arcaine::diffusion_gemma::parse_decision_request(body, s);
            arcaine::diffusion_gemma::compile_decision_questions(req, tok, s);
            ordered_json qs = ordered_json::array();
            for (const auto& q : req.questions) {
                ordered_json qj;
                qj["external_id"] = q.external_id;
                qj["type"] = (q.type == arcaine::diffusion_gemma::DecisionQuestionType::Noul)
                                 ? "noul"
                                 : (q.type == arcaine::diffusion_gemma::DecisionQuestionType::Choice)
                                       ? "choice" : "score";
                qj["prompt_ids"] = q.prompt_ids;
                qj["prompt_tokens"] = q.prompt_tokens;
                qj["canvas"] = q.templ.canvas;
                qj["slot_position"] = q.templ.slot_position;
                qj["label_ids"] = q.templ.label_ids;
                qj["label_texts"] = q.label_texts;
                qj["option_keys"] = q.option_keys;
                qj["instructions"] = q.instructions;
                ordered_json desc = ordered_json::array();
                for (const auto& d : q.option_descriptions) {
                    if (d.is_string())        desc.push_back(d.get<std::string>());
                    else if (d.is_null())     desc.push_back(std::string());
                    else                      desc.push_back(nlohmann::json(d).dump());
                }
                qj["option_descriptions"] = std::move(desc);
                qj["stream_seed"] = q.stream_seed;
                qj["canonical_content"] = q.canonical_content;
                qs.push_back(std::move(qj));
            }
            entry["questions"] = std::move(qs);
            entry["state_text"] = req.state_text;
        } catch (const std::exception& e) {
            entry["error"] = e.what();
        }
        results.push_back(std::move(entry));
    }
    out["results"] = std::move(results);
    std::cout << out.dump() << std::endl;
    return 0;
}

#pragma once
// Validate requests and build isolated question prompts.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "modeling/diffusion_gemma/model.hpp"

class TokenizerBridge;

namespace arcaine::diffusion_gemma {

// Validation failure in a /v1/systemone request; the route maps it to 422.
struct DecisionSchemaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class DecisionQuestionType { Noul, Choice, Score };

struct DecisionRequestSettings {
    uint64_t    base_seed = 42;    // server-internal seed (--seed)
    int         max_questions = 64;   // local resource limit, not a Jev limit
    int         canvas_step = 16;     // canvas widths round up to this multiple
    int         canvas_max = 256;     // local resource limit
    int         max_seq = 2048;       // KV capacity available to prompts
    int         vocab_size = 0;       // from the loaded model config
    int         pad_token_id = 0;     // from the loaded model config
    int         turn_close_id = -1;   // resolved from the tokenizer (<turn|> or <end_of_turn>)
};

struct DecisionQuestion {
    std::string external_id;         // response key only; never enters inference
    DecisionQuestionType type = DecisionQuestionType::Noul;
    std::string instructions;        // serialized (string or structured) form
    // Label j maps to option_keys[j].
    std::vector<std::string> option_keys;
    // Descriptions aligned with option_keys.
    std::vector<nlohmann::ordered_json> option_descriptions;
    std::vector<std::string> label_texts;   // "A"/"AA"/"yes"/"1"...

    // Compiled artifacts (compile_decision_questions).
    std::vector<int>         prompt_ids;
    CompiledDecisionTemplate templ;
    uint64_t                 stream_seed = 0;
    // Unambiguous canonical form of this question's content. Breaks stream-seed
    // collisions so execution order is determined by content, not by caller order.
    std::string              canonical_content;
    int                      prompt_tokens = 0;
};

struct DecisionRequest {
    std::string                   model;      // as named by the caller
    std::string                   state_text; // deterministically serialized
    std::vector<DecisionQuestion> questions;  // caller declaration order
    bool                          want_diagnostics = false;
};

// Parse JSON and reject duplicate keys.
nlohmann::ordered_json parse_json_no_duplicates(const std::string& body);

// Validate a request. The route checks the model name.
DecisionRequest parse_decision_request(const nlohmann::ordered_json& body,
                                       const DecisionRequestSettings& settings);

// Compile each question prompt and canvas.
void compile_decision_questions(DecisionRequest& req, TokenizerBridge& tokenizer,
                                const DecisionRequestSettings& settings);

// Map reads to a response object. Add diagnostics when requested.
nlohmann::ordered_json map_decision_answer(const DecisionQuestion& q,
                                           const DecisionReadResult& result,
                                           nlohmann::ordered_json* diagnostics);

}  // namespace arcaine::diffusion_gemma

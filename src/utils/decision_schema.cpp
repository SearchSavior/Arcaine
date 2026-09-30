// Schema and canvas compiler.
#include "decision_schema.hpp"
#include "chat.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <unordered_set>

namespace arcaine::diffusion_gemma {
namespace {

using ordered_json = nlohmann::ordered_json;

// Serialize objects with sorted keys.
std::string canonical_dump(const ordered_json& v) {
    return nlohmann::json(v).dump();
}

// Accept text and structured content.
std::string content_text(const ordered_json& v) {
    if (v.is_null()) return {};
    if (v.is_string()) return v.get<std::string>();
    return canonical_dump(v);
}

// FNV-1a-64. The offset basis is the standard 0xcbf29ce484222325.
uint64_t fnv1a64(const void* data, size_t n, uint64_t h) {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
uint64_t fnv1a64(const std::string& s, uint64_t h = 14695981039346656037ULL) {
    return fnv1a64(s.data(), s.size(), h);
}

// Append a length-prefixed field to a canonical string. The decimal length and
// ':' delimiter make the encoding unambiguous even when a value contains '\n',
// '\x1f', or ':'.
void append_canon_field(std::string& out, const std::string& s) {
    out += std::to_string(s.size());
    out += ':';
    out += s;
    out += '\n';
}

// Hash one field with an explicit little-endian 64-bit length prefix. The byte
// order is written by hand, so the stream is identical on any host. The prefix
// makes the byte stream unambiguous: a value that contains a separator cannot
// merge with the next field. Without it, two different questions can hash
// equally. The stream version is "arcaine-systemone-v1".
uint64_t hash_field(uint64_t h, const std::string& s) {
    const uint64_t n = (uint64_t)s.size();
    unsigned char le[8];
    for (int i = 0; i < 8; ++i) le[i] = (unsigned char)((n >> (8 * i)) & 0xffu);
    h = fnv1a64(le, sizeof(le), h);
    return fnv1a64(s.data(), s.size(), h);
}

// Verify answer labels in the canvas context.

std::vector<std::string> letter_labels() {
    std::vector<std::string> out;
    out.reserve(26 + 26 * 26);
    for (char a = 'A'; a <= 'Z'; ++a) out.emplace_back(1, a);
    for (char a = 'A'; a <= 'Z'; ++a)
        for (char b = 'A'; b <= 'Z'; ++b) { out.emplace_back(1, a); out.back() += b; }
    return out;
}

std::vector<std::string> digit_labels() {
    std::vector<std::string> out;
    for (int i = 1; i <= 10; ++i) out.push_back(std::to_string(i));
    return out;
}

// Verify that all alternatives differ at one token.
struct SlotVerification {
    std::vector<int32_t> base;      // full alternative with label 0
    int                  slot = -1;
    std::vector<int32_t> label_ids;
};

SlotVerification verify_slot(TokenizerBridge& tok, const std::string& lead,
                             const std::vector<std::string>& labels,
                             const std::string& what,
                             const std::vector<int32_t>& prefix = {},
                             const std::vector<int32_t>& suffix = {}) {
    SlotVerification v;
    auto build = [&](const std::string& label) {
        // Encode the canvas as a continuation.
        auto enc = tok.encode_raw(lead + label, /*add_bos=*/false,
                                  /*add_prefix_space=*/false);
        std::vector<int32_t> alt;
        alt.reserve(prefix.size() + enc.size() + suffix.size());
        alt.insert(alt.end(), prefix.begin(), prefix.end());
        alt.insert(alt.end(), enc.begin(), enc.end());
        alt.insert(alt.end(), suffix.begin(), suffix.end());
        return alt;
    };
    v.base = build(labels[0]);
    v.label_ids.resize(labels.size());
    for (size_t i = 1; i < labels.size(); ++i) {
        auto e = build(labels[i]);
        if (e.size() != v.base.size())
            throw DecisionSchemaError(what + ": label '" + labels[i] +
                "' is not a single token in the answer template");
        int pos = -1;
        for (size_t t = 0; t < e.size(); ++t) {
            if (e[t] != v.base[t]) {
                if (pos >= 0)
                    throw DecisionSchemaError(what + ": label '" + labels[i] +
                        "' changes more than one template token");
                pos = (int)t;
            }
        }
        if (pos < 0)
            throw DecisionSchemaError(what + ": labels '" + labels[0] + "' and '" +
                labels[i] + "' tokenize identically");
        if (v.slot >= 0 && pos != v.slot)
            throw DecisionSchemaError(what + ": labels do not share one template slot");
        v.slot = pos;
        v.label_ids[i] = e[pos];
    }
    if (v.slot < 0) {
        // The caller requires at least two labels.
        throw DecisionSchemaError(what + ": could not locate the answer slot");
    }
    v.label_ids[0] = v.base[v.slot];
    {
        std::set<int32_t> uniq(v.label_ids.begin(), v.label_ids.end());
        if (uniq.size() != v.label_ids.size())
            throw DecisionSchemaError(what + ": two labels tokenize to the same id");
    }
    // Labels must not be control tokens.
    const int V = tok.vocab_size();
    for (size_t i = 0; i < v.label_ids.size(); ++i) {
        int id = v.label_ids[i];
        if (id < 0 || id >= V)
            throw DecisionSchemaError(what + ": label '" + labels[i] +
                "' resolves to a token id outside the vocabulary");
        if (tok.is_special_token(id))
            throw DecisionSchemaError(what + ": label '" + labels[i] +
                "' resolves to a special/control token id and cannot be an answer label");
    }
    return v;
}

// Select verified labels.
std::vector<std::string> pick_labels(TokenizerBridge& tok, const std::string& lead,
                                     const std::vector<std::string>& candidates,
                                     size_t count, const std::string& what) {
    std::unordered_set<std::string> excluded;
    for (int attempt = 0; attempt < 8; ++attempt) {
        std::vector<std::string> accepted;
        for (const auto& cand : candidates) {
            if (accepted.size() == count) break;
            if (excluded.count(cand)) continue;
            if (accepted.empty()) { accepted.push_back(cand); continue; }
            try {
                verify_slot(tok, lead, {accepted[0], cand}, what);
                accepted.push_back(cand);
            } catch (const DecisionSchemaError&) {
                // Skip this candidate.
            }
        }
        if (accepted.size() >= count) return accepted;
        if (accepted.size() == 1) {
            excluded.insert(accepted[0]);   // anchor may be bad; retry without it
            continue;
        }
        break;   // anchor verified by some pair: genuine codebook shortage
    }
    throw DecisionSchemaError(what + ": the tokenizer cannot provide " +
        std::to_string(count) + " verified single-token labels in the answer "
        "template — a blocking compatibility gap");
}

// Trailing tokens of `full` that are not part of `base`. `base` is the same
// conversation rendered without the assistant generation prompt, so the result
// is the assistant generation suffix alone. The scaffold is validated only in
// this suffix, never in the state or instructions.
std::vector<int32_t> generation_suffix(const std::vector<int>& full,
                                       const std::vector<int>& base,
                                       const std::string& what) {
    size_t n = 0;
    while (n < full.size() && n < base.size() && full[n] == base[n]) ++n;
    if (n != base.size())
        throw DecisionSchemaError(what + ": the chat template's generation prompt "
            "is not a trailing suffix of the rendered conversation; cannot isolate "
            "the assistant suffix");
    return std::vector<int32_t>(full.begin() + (std::ptrdiff_t)n, full.end());
}

// Resolve how the served chat template handled the empty thought scaffold, and
// return the tokens the canvas must add.
//
// Cases, decided from the assistant generation suffix only:
//   complete    - suffix ends with the full scaffold; the canvas adds nothing.
//   absent      - no scaffold token occurs in the suffix; the canvas adds the
//                 full scaffold (the served template leaves it to the model).
//   partial     - suffix ends with the full opening but no close; the missing
//                 close tokens are appended, with no duplication.
//   unsupported - suffix ends with a shorter partial, or a scaffold appears
//                 outside the trailing position; the compiler rejects it.
//
// Only the scaffold token sequence is matched, not the single closing token, so
// quoted channel markers in the state or instructions do not trip the check.
std::vector<int32_t> detect_scaffold_prefix(TokenizerBridge& tok,
                                            const std::vector<int32_t>& suffix,
                                            const std::string& what) {
    static const std::string kScaffold = "<|channel>thought\n<channel|>";
    static const std::string kOpen = "<|channel>thought\n";
    if (!tok.has_token("<channel|>") || !tok.has_token("<|channel>"))
        throw DecisionSchemaError(what + ": tokenizer has no thought-channel "
            "tokens; cannot verify the served chat template's suffix");
    const std::vector<int32_t> full = tok.encode_raw(kScaffold, /*add_bos=*/false);
    const std::vector<int32_t> open = tok.encode_raw(kOpen, /*add_bos=*/false);
    if (full.empty() || open.empty() || open.size() >= full.size() ||
        !std::equal(open.begin(), open.end(), full.begin()))
        throw DecisionSchemaError(what + ": thought scaffold does not tokenize as "
            "a stable prefix; cannot build the canvas");

    auto contains = [](const std::vector<int32_t>& hay, const std::vector<int32_t>& needle,
                       size_t begin, size_t end) {
        return std::search(hay.begin() + (std::ptrdiff_t)begin,
                           hay.begin() + (std::ptrdiff_t)end,
                           needle.begin(), needle.end()) != hay.begin() + (std::ptrdiff_t)end;
    };

    // Complete: the suffix ends with the full scaffold.
    if (suffix.size() >= full.size() &&
        std::equal(full.begin(), full.end(), suffix.end() - full.size())) {
        const size_t head = suffix.size() - full.size();
        if (contains(suffix, full, 0, head) || contains(suffix, open, 0, head))
            throw DecisionSchemaError(what + ": chat template emitted a duplicated "
                "thought scaffold; refusing to guess the canvas prefix");
        return {};
    }

    // Partial: the suffix ends with a proper prefix of the scaffold.
    int k = 0;
    for (int j = (int)full.size() - 1; j >= 1; --j) {
        if (suffix.size() >= (size_t)j &&
            std::equal(full.begin(), full.begin() + j, suffix.end() - j)) { k = j; break; }
    }
    if (k >= (int)open.size()) {
        const size_t head = suffix.size() - (size_t)k;
        if (contains(suffix, full, 0, head) || contains(suffix, open, 0, head))
            throw DecisionSchemaError(what + ": chat template emitted a duplicated "
                "thought scaffold; refusing to guess the canvas prefix");
        return std::vector<int32_t>(full.begin() + k, full.end());   // append the close
    }
    if (k > 0)
        throw DecisionSchemaError(what + ": chat template ended with an unsupported "
            "partial thought scaffold");

    // Absent: no scaffold at the trailing position. Reject one elsewhere in the
    // suffix, but do not scan the state or instructions.
    if (contains(suffix, full, 0, suffix.size()) || contains(suffix, open, 0, suffix.size()))
        throw DecisionSchemaError(what + ": chat template emitted a partial or "
            "duplicated thought scaffold; refusing to guess the canvas prefix");
    return full;
}

std::string question_type_name(DecisionQuestionType t) {
    switch (t) {
    case DecisionQuestionType::Noul:   return "noul";
    case DecisionQuestionType::Choice: return "choice";
    case DecisionQuestionType::Score:  return "score";
    }
    return "?";
}

// Build the question text.
std::string system_text(const DecisionQuestion& q) {
    std::string s =
        "Answer a question about the state the user provides. The question lists "
        "its allowed answers; reply with exactly one label.\n";
    if (!q.instructions.empty()) s += "\nQuestion: " + q.instructions + "\n";
    for (size_t j = 0; j < q.option_keys.size(); ++j) {
        std::string desc = content_text(q.option_descriptions[j]);
        s += "  " + q.label_texts[j] + ":";
        if (q.type == DecisionQuestionType::Noul) {
            if (!desc.empty()) s += " " + desc;
        } else if (q.type == DecisionQuestionType::Choice) {
            s += " " + q.option_keys[j];
            if (!desc.empty()) s += " (" + desc + ")";
        } else {
            if (!desc.empty()) s += " " + desc;
        }
        s += "\n";
    }
    s += "\nReply with exactly one label, formatted as \"answer: label\".";
    return s;
}

}  // namespace

// Parse JSON and reject duplicate keys.

ordered_json parse_json_no_duplicates(const std::string& body) {
    std::vector<std::unordered_set<std::string>> object_stack;
    auto callback = [&](int /*depth*/, ordered_json::parse_event_t event,
                        ordered_json& parsed) -> bool {
        switch (event) {
        case ordered_json::parse_event_t::object_start:
            object_stack.emplace_back();
            return true;
        case ordered_json::parse_event_t::object_end:
            object_stack.pop_back();
            return true;
        case ordered_json::parse_event_t::key: {
            const std::string& key = parsed.get_ref<const std::string&>();
            if (!object_stack.back().insert(key).second)
                throw DecisionSchemaError("duplicate JSON key '" + key + "'");
            return true;
        }
        default:
            return true;
        }
    };
    ordered_json out;
    try {
        out = ordered_json::parse(body, callback);
    } catch (const DecisionSchemaError&) {
        throw;
    } catch (const std::exception& e) {
        throw DecisionSchemaError(std::string("invalid JSON: ") + e.what());
    }
    return out;
}

// Validate the request.

DecisionRequest parse_decision_request(const ordered_json& body,
                                       const DecisionRequestSettings& settings) {
    if (!body.is_object())
        throw DecisionSchemaError("request body must be a JSON object");

    static const std::set<std::string> kTopLevel = {"model", "state", "questions", "arcaine"};
            // Reject unsupported controls.
    static const std::set<std::string> kControls = {
        "seed", "steps", "canvas", "canvas_length", "samples", "temperature",
        "top_p", "top_k", "think", "ask", "depends_on", "ask_if", "chunk_rows",
        "chunk_prompt", "sequential", "auto_max", "auto_threshold", "images"};
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (kTopLevel.count(it.key())) continue;
        if (kControls.count(it.key()))
            throw DecisionSchemaError("unsupported request field '" + it.key() +
                "': canvas, seed, sampling and denoising settings are internal");
        throw DecisionSchemaError("unknown request field '" + it.key() + "'");
    }

    DecisionRequest req;

    if (!body.contains("model") || !body.at("model").is_string())
        throw DecisionSchemaError("model: required string");
    req.model = body.at("model").get<std::string>();

    if (!body.contains("state"))
        throw DecisionSchemaError("state: required");
    const auto& state = body.at("state");
    if (state.is_string()) req.state_text = state.get<std::string>();
    else if (state.is_object() || state.is_array()) req.state_text = canonical_dump(state);
    else throw DecisionSchemaError("state: must be a string, object, or array");

    if (body.contains("arcaine")) {
        const auto& ext = body.at("arcaine");
        if (!ext.is_object())
            throw DecisionSchemaError("arcaine: must be an object of namespaced options");
        for (auto it = ext.begin(); it != ext.end(); ++it) {
            if (it.key() == "diagnostics") {
                if (!it.value().is_boolean())
                    throw DecisionSchemaError("arcaine.diagnostics: must be a boolean");
                req.want_diagnostics = it.value().get<bool>();
            } else {
                throw DecisionSchemaError("unknown arcaine option '" + it.key() + "'");
            }
        }
    }

    if (!body.contains("questions") || !body.at("questions").is_object() ||
        body.at("questions").empty())
        throw DecisionSchemaError("questions: required non-empty map of id -> question");
    const auto& questions = body.at("questions");
    if ((int)questions.size() > settings.max_questions)
        throw DecisionSchemaError("questions: " + std::to_string(questions.size()) +
            " exceeds this server's limit of " + std::to_string(settings.max_questions) +
            " (a local Arcaine resource limit, not a Jev limit)");

    for (auto it = questions.begin(); it != questions.end(); ++it) {
        // Keep the caller ID as a response key only.
        const std::string& qid = it.key();
        const auto& qj = it.value();
        if (!qj.is_object())
            throw DecisionSchemaError("question '" + qid + "': must be an object");
        for (auto qf = qj.begin(); qf != qj.end(); ++qf) {
            if (qf.key() != "type" && qf.key() != "instructions" && qf.key() != "criteria")
                throw DecisionSchemaError("question '" + qid + "': unsupported field '" +
                    qf.key() + "'");
        }
        if (!qj.contains("type") || !qj.at("type").is_string())
            throw DecisionSchemaError("question '" + qid + "': type is required");
        const std::string type = qj.at("type").get<std::string>();

        DecisionQuestion q;
        q.external_id = qid;
        if (qj.contains("instructions")) {
            const auto& ins = qj.at("instructions");
            if (!ins.is_string() && !ins.is_object() && !ins.is_array())
                throw DecisionSchemaError("question '" + qid +
                    "': instructions must be a string or structured content");
            q.instructions = content_text(ins);
        }

        if (type == "noul") {
            q.type = DecisionQuestionType::Noul;
            ordered_json dtrue, dfalse;
            if (qj.contains("criteria")) {
                const auto& crit = qj.at("criteria");
                if (!crit.is_object())
                    throw DecisionSchemaError("question '" + qid +
                        "': noul criteria must be an object with true/false descriptions");
                for (auto cf = crit.begin(); cf != crit.end(); ++cf) {
                    if (cf.key() == "true")       dtrue = cf.value();
                    else if (cf.key() == "false") dfalse = cf.value();
                    else throw DecisionSchemaError("question '" + qid +
                        "': unknown noul criterion '" + cf.key() + "'");
                }
            }
            q.option_keys         = {"yes", "no"};
            q.option_descriptions = {dtrue, dfalse};
        } else if (type == "choice") {
            q.type = DecisionQuestionType::Choice;
            if (!qj.contains("criteria") || !qj.at("criteria").is_object())
                throw DecisionSchemaError("question '" + qid +
                    "': choice criteria must map option names to descriptions");
            const auto& crit = qj.at("criteria");
            if (crit.size() < 2 || crit.size() > 255)
                throw DecisionSchemaError("question '" + qid + "': choice needs 2..255 "
                    "options, got " + std::to_string(crit.size()));
            std::vector<std::string> keys;
            for (auto cf = crit.begin(); cf != crit.end(); ++cf) {
                if (cf.key().empty())
                    throw DecisionSchemaError("question '" + qid +
                        "': option names must be non-empty");
                const auto& d = cf.value();
                if (!d.is_null() && !d.is_string() && !d.is_object() && !d.is_array())
                    throw DecisionSchemaError("question '" + qid + "': description of '" +
                        cf.key() + "' must be a string, null, or structured content");
                keys.push_back(cf.key());
            }
            // Sort keys for stable label mapping.
            std::sort(keys.begin(), keys.end());
            for (const auto& k : keys) {
                q.option_keys.push_back(k);
                q.option_descriptions.push_back(crit.at(k));
            }
        } else if (type == "score") {
            q.type = DecisionQuestionType::Score;
            if (!qj.contains("criteria") || !qj.at("criteria").is_array())
                throw DecisionSchemaError("question '" + qid +
                    "': score criteria must be an ordered list of level descriptions");
            const auto& levels = qj.at("criteria");
            if (levels.size() < 2 || levels.size() > 10)
                throw DecisionSchemaError("question '" + qid + "': score needs 2..10 "
                    "levels, got " + std::to_string(levels.size()));
            for (size_t i = 0; i < levels.size(); ++i) {
                const auto& lv = levels.at(i);
                if (!lv.is_string() && !lv.is_object() && !lv.is_array())
                    throw DecisionSchemaError("question '" + qid + "': score level " +
                        std::to_string(i) + " must be a string or structured content");
                q.option_keys.push_back(std::to_string(i));   // legend index
                q.option_descriptions.push_back(lv);
            }
        } else {
            throw DecisionSchemaError("question '" + qid + "': unknown type '" + type +
                "' (expected noul, choice, or score)");
        }

        // Derive noise from question content, not the external ID. Hash a
        // length-prefixed canonical stream so distinct content cannot collide by
        // concatenation. Variant: FNV-1a-64 over [u64 length][bytes] per field.
        std::string canon;
        append_canon_field(canon, type);
        append_canon_field(canon, q.instructions);
        for (size_t j = 0; j < q.option_keys.size(); ++j) {
            append_canon_field(canon, q.option_keys[j]);
            append_canon_field(canon, canonical_dump(q.option_descriptions[j]));
        }
        uint64_t h = fnv1a64("arcaine-systemone-v1");
        h = hash_field(h, std::to_string(settings.base_seed));
        h = hash_field(h, type);
        h = hash_field(h, req.state_text);
        h = hash_field(h, q.instructions);
        for (size_t j = 0; j < q.option_keys.size(); ++j) {
            h = hash_field(h, q.option_keys[j]);
            h = hash_field(h, canonical_dump(q.option_descriptions[j]));
        }
        q.stream_seed = h;
        q.canonical_content = std::move(canon);

        req.questions.push_back(std::move(q));
    }

    // Run questions in stable content order. The canonical content breaks any
    // stream-seed collision, so caller order cannot change execution order.
    std::stable_sort(req.questions.begin(), req.questions.end(),
                     [](const DecisionQuestion& a, const DecisionQuestion& b) {
                         if (a.stream_seed != b.stream_seed)
                             return a.stream_seed < b.stream_seed;
                         return a.canonical_content < b.canonical_content;
                     });
    return req;
}

// Compile prompts and canvases.

void compile_decision_questions(DecisionRequest& req, TokenizerBridge& tokenizer,
                                const DecisionRequestSettings& settings) {
    const int V = tokenizer.vocab_size();
    if (settings.vocab_size > 0 && settings.vocab_size != V)
        throw DecisionSchemaError("internal: tokenizer vocab (" + std::to_string(V) +
            ") does not match the model config vocab (" +
            std::to_string(settings.vocab_size) + ")");
    if (settings.turn_close_id < 0 || settings.turn_close_id >= V)
        throw DecisionSchemaError("internal: turn-close token not resolved");
    if (!tokenizer.is_special_token(settings.turn_close_id))
        throw DecisionSchemaError("resolved turn-close token " +
            std::to_string(settings.turn_close_id) +
            " is not a special/control token; refusing to build the answer canvas");
    if (settings.pad_token_id < 0 || settings.pad_token_id >= V)
        throw DecisionSchemaError("resolved pad token " +
            std::to_string(settings.pad_token_id) + " is outside the vocabulary");
    const std::vector<int32_t> suffix = {settings.turn_close_id};
    const std::string lead = "answer: ";

    for (auto& q : req.questions) {
        const std::string what = "question '" + q.external_id + "'";
        const size_t n = q.option_keys.size();

        // Select labels, then verify the full canvas.
        if (q.type == DecisionQuestionType::Noul) {
            try {
                q.label_texts = {"yes", "no"};
                verify_slot(tokenizer, lead, q.label_texts, what);
            } catch (const DecisionSchemaError&) {
                q.label_texts = {"true", "false"};
                verify_slot(tokenizer, lead, q.label_texts, what);
            }
        } else if (q.type == DecisionQuestionType::Score) {
            try {
                q.label_texts = pick_labels(tokenizer, lead, digit_labels(), n, what);
            } catch (const DecisionSchemaError&) {
                q.label_texts = pick_labels(tokenizer, lead, letter_labels(), n, what);
            }
        } else {
            q.label_texts = pick_labels(tokenizer, lead, letter_labels(), n, what);
        }

        // Build one prompt for this question. Render it twice so the assistant
        // generation suffix is isolated from the state and instructions.
        const std::vector<ChatTemplateMessage> messages = {
            {"system", system_text(q)},
            {"user", req.state_text}};
        // Render twice (cheap) but tokenize the long shared prefix only once.
        // The generation prompt is appended, so it is a string suffix; tokenize
        // that short suffix separately.
        const std::string full_text = tokenizer.build_prompt_text(messages, true);
        const std::string no_gen_text = tokenizer.build_prompt_text(messages, false);
        q.prompt_ids = tokenizer.encode_raw(full_text, /*add_bos=*/false);
        q.prompt_tokens = (int)q.prompt_ids.size();
        if (q.prompt_ids.empty())
            throw DecisionSchemaError(what + ": chat template produced an empty prompt");
        std::vector<int32_t> gen_suffix;
        if (full_text.size() >= no_gen_text.size() &&
            full_text.compare(0, no_gen_text.size(), no_gen_text) == 0) {
            gen_suffix = tokenizer.encode_raw(full_text.substr(no_gen_text.size()),
                                              /*add_bos=*/false,
                                              /*add_prefix_space=*/false);
            if (gen_suffix.size() > q.prompt_ids.size() ||
                !std::equal(gen_suffix.rbegin(), gen_suffix.rend(), q.prompt_ids.rbegin()))
                throw DecisionSchemaError(what + ": the generation suffix is not a token "
                    "suffix of the rendered prompt");
        } else {
            // The template did not append the generation prompt as a string
            // suffix. Tokenize the no-generation render to locate it.
            const std::vector<int> no_gen = tokenizer.build_prompt_no_generation(messages);
            gen_suffix = generation_suffix(q.prompt_ids, no_gen, what);
        }
        std::vector<int32_t> prefix = detect_scaffold_prefix(tokenizer, gen_suffix, what);

        // Verify labels with the prompt prefix and close token.
        SlotVerification slot = verify_slot(tokenizer, lead, q.label_texts, what,
                                            prefix, suffix);

        // Build and pad the canvas.
        std::vector<int32_t> canvas = slot.base;
        int width = (int)((canvas.size() + settings.canvas_step - 1) /
                          settings.canvas_step) * settings.canvas_step;
        if (width > settings.canvas_max)
            throw DecisionSchemaError(what + ": answer template needs a canvas of " +
                std::to_string(width) + " tokens; this server's limit is " +
                std::to_string(settings.canvas_max) + " (a local Arcaine limit)");
        canvas.resize(width, settings.pad_token_id);

        q.templ.canvas      = std::move(canvas);
        q.templ.slot_position = slot.slot;
        q.templ.label_ids   = std::move(slot.label_ids);

        if ((int64_t)q.prompt_tokens + (int64_t)q.templ.canvas.size() > settings.max_seq)
            throw DecisionSchemaError(what + ": prompt (" +
                std::to_string(q.prompt_tokens) + ") + canvas (" +
                std::to_string(q.templ.canvas.size()) + ") tokens exceeds this "
                "server's limit of " + std::to_string(settings.max_seq) +
                " (a local Arcaine limit)");
    }
}

// Map scores to response objects.

ordered_json map_decision_answer(const DecisionQuestion& q,
                                 const DecisionReadResult& result,
                                 ordered_json* diagnostics) {
    const int K = (int)q.option_keys.size();
    const int N = (int)result.reads.size();
    if (N < 1) throw std::runtime_error("map_decision_answer: no reads");

    // Average conditional label probabilities.
    std::vector<std::vector<double>> per_read(N, std::vector<double>(K));
    std::vector<double> p(K, 0.0);
    for (int r = 0; r < N; ++r) {
        const auto& s = result.reads[r];
        for (int j = 0; j < K; ++j) {
            per_read[r][j] = std::exp((double)s.label_logprobs[j] - (double)s.log_label_mass);
            p[j] += per_read[r][j];
        }
    }
    for (int j = 0; j < K; ++j) p[j] = std::min(1.0, std::max(0.0, p[j] / N));

    // Select the first label on ties.
    int top = 0;
    for (int j = 1; j < K; ++j) if (p[j] > p[top]) top = j;

    ordered_json answer;
    answer["type"] = question_type_name(q.type);

    if (q.type == DecisionQuestionType::Noul) {
        // Noul returns the probability of yes.
        answer["noul"] = p[0];
    } else if (q.type == DecisionQuestionType::Choice) {
        ordered_json probs = ordered_json::object();
        for (int j = 0; j < K; ++j) probs[q.option_keys[j]] = p[j];
        answer["choice"] = q.option_keys[top];
        answer["probabilities"] = std::move(probs);
    } else {
        double score = 0.0;
        ordered_json probs = ordered_json::object();
        ordered_json legend = ordered_json::object();
        for (int j = 0; j < K; ++j) {
            score += (double)j * p[j];
            probs[std::to_string(j)] = p[j];
            legend[std::to_string(j)] = q.option_descriptions[j];
        }
        answer["score"] = score;
        answer["legend"] = std::move(legend);
        answer["probabilities"] = std::move(probs);
    }

    if (q.type != DecisionQuestionType::Noul) {
        // Use the local normalized entropy method.
        double h = 0.0;
        for (int j = 0; j < K; ++j)
            if (p[j] > 0.0) h -= p[j] * std::log(p[j]);
        double conf = 1.0 - h / std::log((double)K);
        answer["confidence"] = std::min(1.0, std::max(0.0, conf));
    }

    if (diagnostics) {
        ordered_json d;
        d["reads"] = N;
        d["canvas_width"] = (int)q.templ.canvas.size();
        d["prompt_tokens"] = result.prompt_tokens;
        d["prefill_calls"] = result.prefill_calls;
        d["decode_calls"] = result.decode_calls;
        d["prefill_ms"] = result.prefill_s * 1e3;
        d["decode_ms"] = result.decode_s * 1e3;
        ordered_json mass = ordered_json::array();
        ordered_json vent = ordered_json::array();
        ordered_json aok = ordered_json::array();
        for (const auto& s : result.reads) {
            mass.push_back(std::exp((double)s.log_label_mass));
            vent.push_back(s.vocab_entropy);
            aok.push_back(std::find(q.templ.label_ids.begin(), q.templ.label_ids.end(),
                                    s.argmax_id) != q.templ.label_ids.end());
        }
        d["label_mass"] = std::move(mass);
        d["vocab_entropy"] = std::move(vent);
        d["argmax_is_label"] = std::move(aok);
        // Report read noise, not factual certainty.
        if (N > 1) {
            double var = 0.0;
            int agree = 0;
            for (int r = 0; r < N; ++r) {
                var += (per_read[r][top] - p[top]) * (per_read[r][top] - p[top]);
                int rtop = 0;
                for (int j = 1; j < K; ++j) if (per_read[r][j] > per_read[r][rtop]) rtop = j;
                if (rtop == top) ++agree;
            }
            d["selected_label_stderr"] = std::sqrt(var / (N - 1) / N);
            d["selected_label_agreement"] = (double)agree / N;
        } else {
            d["selected_label_stderr"] = nullptr;
            d["selected_label_agreement"] = nullptr;
        }
        *diagnostics = std::move(d);
    }
    return answer;
}

}  // namespace arcaine::diffusion_gemma

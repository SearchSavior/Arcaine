#include "tokenizer.hpp"
#include "unicode.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cstdio>

Tokenizer Tokenizer::from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open tokenizer: " + path);
    auto j = nlohmann::json::parse(f);

    Tokenizer tok;

    // Build byte_pieces_ for byte-level fallback (<0xXX> notation)
    tok.byte_pieces_.resize(256);
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tok.byte_pieces_[b] = buf;
    }

    // vocab: {"piece": id}
    auto& model = j["model"];
    auto& vocab_json = model["vocab"];
    int max_id = 0;
    for (auto& [piece, id_j] : vocab_json.items())
        max_id = std::max(max_id, id_j.get<int>());
    tok.id_to_piece_.resize(max_id + 1);
    for (auto& [piece, id_j] : vocab_json.items()) {
        int id = id_j.get<int>();
        tok.vocab_[piece] = id;
        tok.id_to_piece_[id] = piece;
    }

    if (j.contains("added_tokens")) {
        for (auto& added : j["added_tokens"]) {
            std::string content = added["content"].get<std::string>();
            int id = added["id"].get<int>();
            tok.added_token_to_id_[content] = id;
            tok.added_tokens_by_length_.push_back(content);
            tok.vocab_[content] = id;
            if (added.value("special", false))
                tok.special_token_ids_.insert(id);
            if (content == "<bos>") tok.bos_id_ = id;
            if (id >= (int)tok.id_to_piece_.size())
                tok.id_to_piece_.resize(id + 1);
            tok.id_to_piece_[id] = content;
        }
        std::sort(tok.added_tokens_by_length_.begin(), tok.added_tokens_by_length_.end(),
                  [](const std::string& a, const std::string& b) {
                      return a.size() > b.size();
                  });
    }

    // merges: [["piece_a", "piece_b"], ...] — array of 2-element arrays
    auto& merges = model["merges"];
    for (int i = 0; i < (int)merges.size(); ++i) {
        std::string a = merges[i][0].get<std::string>();
        std::string b = merges[i][1].get<std::string>();
        tok.merge_rank_[a + " " + b] = i;
    }

    // Detect GPT-2 / Qwen2 byte-level BPE via a ByteLevel decoder. Gemma's
    // SentencePiece tokenizer has no ByteLevel decoder, so this discriminates
    // the two families and selects the encode/decode path below.
    if (j.contains("decoder") && j["decoder"].value("type", std::string{}) == "ByteLevel") {
        tok.byte_level_ = true;
    }

    return tok;
}

// BPE encode
std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> ids;
    if (add_bos) {
        if (bos_id_ < 0) throw std::runtime_error("Tokenizer BOS token not found");
        ids.push_back(bos_id_);
    }

    if (text.empty()) return ids;

    bool at_text_start = true;
    auto encode_span = [&](const std::string& span, std::vector<int>& out) {
        if (span.empty()) return;

        if (byte_level_) {
            // Qwen2 / GPT-2: regex pre-tokenize (custom Qwen2 scanner ported
            // from llama.cpp), then byte-encode each word and run BPE merges.
            for (const auto& word : byte_level_pretokenize(span)) {
                byte_level_encode(word, out);
            }
            return;
        }

        // Gemma: ▁ (U+2581) sentinel marks text-start / word boundaries.
        std::string t = at_text_start ? ("▁" + span) : span;
        at_text_start = false;
        std::string processed;
        for (char c : t) {
            if (c == ' ') {
                processed += "▁";
            } else {
                processed += c;
            }
        }

        // Split into Unicode characters (UTF-8 aware) and look up each in vocab.
        // Unknown chars use byte-level fallback.
        std::vector<std::string> symbols;
        size_t i = 0;
        while (i < processed.size()) {
            unsigned char c = (unsigned char)processed[i];
            int len = 1;
            if      ((c & 0xF8) == 0xF0) len = 4;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xE0) == 0xC0) len = 2;
            std::string sym = processed.substr(i, len);
            if (vocab_.count(sym)) {
                symbols.push_back(sym);
            } else {
                for (size_t b = 0; b < (size_t)len; ++b)
                    symbols.push_back(byte_pieces_[(unsigned char)processed[i + b]]);
            }
            i += len;
        }

        // BPE merges
        bool changed = true;
        while (changed && symbols.size() > 1) {
            changed = false;
            int best_rank = std::numeric_limits<int>::max();
            int best_pos  = -1;

            for (int j = 0; j + 1 < (int)symbols.size(); ++j) {
                std::string pair = symbols[j] + " " + symbols[j + 1];
                auto it = merge_rank_.find(pair);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos  = j;
                }
            }

            if (best_pos < 0) break;

            symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
            symbols.erase(symbols.begin() + best_pos + 1);
            changed = true;
        }

        // Convert pieces to ids
        for (auto& sym : symbols) {
            auto it = vocab_.find(sym);
            if (it != vocab_.end()) {
                out.push_back(it->second);
            } else {
                for (unsigned char b : sym) {
                    auto bit = vocab_.find(byte_pieces_[b]);
                    if (bit != vocab_.end()) out.push_back(bit->second);
                }
            }
        }
    };

    std::string span;
    for (size_t i = 0; i < text.size();) {
        const std::string* matched = nullptr;
        for (const auto& tok : added_tokens_by_length_) {
            if (!tok.empty() && text.compare(i, tok.size(), tok) == 0) {
                matched = &tok;
                break;
            }
        }
        if (matched) {
            encode_span(span, ids);
            span.clear();
            ids.push_back(added_token_to_id_.at(*matched));
            at_text_start = false;
            i += matched->size();
        } else {
            span.push_back(text[i++]);
        }
    }
    encode_span(span, ids);

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special,
                              bool strip_leading_space) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_piece_.size()) continue;
        if (skip_special && special_token_ids_.count(id)) continue;
        const std::string& piece = id_to_piece_[id];
        if (piece.empty()) continue;

        if (byte_level_) {
            // Special tokens (when kept) are emitted verbatim; normal BPE
            // pieces are mapped char-by-char back to their original bytes via
            // the reverse GPT-2 byte map.
            if (special_token_ids_.count(id)) {
                out += piece;
            } else {
                size_t i = 0;
                while (i < piece.size()) {
                    int len = (int)unicode_len_utf8(piece[i]);
                    if (len < 1 || i + (size_t)len > piece.size()) {
                        out += piece[i];
                        ++i;
                        continue;
                    }
                    std::string ch = piece.substr(i, (size_t)len);
                    try {
                        out.push_back((char)unicode_utf8_to_byte(ch));
                    } catch (...) {
                        out += ch;  // unknown char: emit raw bytes
                    }
                    i += (size_t)len;
                }
            }
        } else {
            // Gemma: replace ▁ (U+2581, 3 bytes E2 96 81) with space.
            std::string p = piece;
            size_t pos = 0;
            while ((pos = p.find("\xe2\x96\x81", pos)) != std::string::npos) {
                p.replace(pos, 3, " ");
                pos += 1;
            }
            out += p;
        }
    }
    // Strip leading space artifact (Gemma ▁ sentinel). Never meaningful for
    // byte-level BPE: a leading space is a real token (Ġ), not an artifact.
    if (strip_leading_space && !byte_level_ && !out.empty() && out[0] == ' ')
        out = out.substr(1);
    return out;
}


bool Tokenizer::has_token(const std::string& token) const {
    return added_token_to_id_.count(token) || vocab_.count(token);
}

int Tokenizer::token_id(const std::string& token) const {
    auto ait = added_token_to_id_.find(token);
    if (ait != added_token_to_id_.end()) return ait->second;
    auto vit = vocab_.find(token);
    if (vit != vocab_.end()) return vit->second;
    throw std::runtime_error("Tokenizer token not found: " + token);
}

// ---- GPT-2 / Qwen2 byte-level BPE ----

std::vector<std::string> Tokenizer::byte_level_pretokenize(const std::string& span) const {
    std::vector<std::string> words;
    if (span.empty()) return words;
    const auto cpts = unicode_cpts_from_utf8(span);
    std::vector<size_t> offsets = { cpts.size() };
    const auto bpe_offsets = unicode_regex_split_custom_qwen2(span, offsets);
    words.reserve(bpe_offsets.size());
    size_t start = 0;
    for (size_t len : bpe_offsets) {
        std::string w;
        for (size_t i = start; i < start + len; ++i) {
            w += unicode_cpt_to_utf8(cpts[i]);
        }
        words.push_back(std::move(w));
        start += len;
    }
    return words;
}

void Tokenizer::byte_level_encode(const std::string& word, std::vector<int>& out) const {
    if (word.empty()) return;
    // Byte-encode: one GPT-2 unicode char per original byte.
    std::vector<std::string> symbols;
    symbols.reserve(word.size());
    for (unsigned char c : word) {
        symbols.push_back(unicode_byte_to_utf8(c));
    }
    // BPE merges: scan for the global minimum-rank pair (leftmost wins on
    // ties), merge, repeat. Equivalent to HF tokenizers' heap-based merge_all
    // and to llama.cpp's bigram queue.
    bool changed = true;
    while (changed && symbols.size() > 1) {
        changed = false;
        int best_rank = std::numeric_limits<int>::max();
        int best_pos  = -1;
        for (int j = 0; j + 1 < (int)symbols.size(); ++j) {
            std::string pair = symbols[j] + " " + symbols[j + 1];
            auto it = merge_rank_.find(pair);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = j;
            }
        }
        if (best_pos < 0) break;
        symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
        changed = true;
    }
    // Emit token ids. Every final symbol should be in vocab (all 256 byte-chars
    // and their learned merges are present); fall back to byte-chars otherwise.
    for (const auto& sym : symbols) {
        auto it = vocab_.find(sym);
        if (it != vocab_.end()) {
            out.push_back(it->second);
        } else {
            for (unsigned char b : sym) {
                std::string bc = unicode_byte_to_utf8(b);
                auto bit = vocab_.find(bc);
                if (bit != vocab_.end()) out.push_back(bit->second);
            }
        }
    }
}

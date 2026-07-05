#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <cstdint>

class Tokenizer {
public:
    static Tokenizer from_json(const std::string& path);

    std::vector<int> encode(const std::string& text, bool add_bos = true) const;
    std::string      decode(const std::vector<int>& ids, bool skip_special = false,
                            bool strip_leading_space = true) const;

    int  token_id(const std::string& token) const;
    bool has_token(const std::string& token) const;
    int  vocab_size() const { return (int)id_to_piece_.size(); }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> added_token_to_id_;
    std::vector<std::string>             added_tokens_by_length_;
    std::unordered_set<int>              special_token_ids_;
    std::vector<std::string>             id_to_piece_;
    std::unordered_map<std::string, int> merge_rank_;

    // Byte-level fallback pieces (256 entries, built at load time)
    std::vector<std::string> byte_pieces_;
    int bos_id_ = -1;

    // GPT-2 / Qwen2 byte-level BPE support. When byte_level_ is true the
    // tokenizer uses the 256-entry byte<->unicode map (space->Ġ, newline->Ċ,
    // ...) instead of the SentencePiece ▁ (U+2581) scheme used by Gemma. The
    // map itself is provided by the ported llama.cpp unicode helpers
    // (unicode_byte_to_utf8 / unicode_utf8_to_byte), so no per-instance state.
    bool byte_level_ = false;

    void byte_level_encode(const std::string& span, std::vector<int>& out) const;
    std::vector<std::string> byte_level_pretokenize(const std::string& s) const;
};

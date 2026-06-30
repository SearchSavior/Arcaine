#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class Tokenizer {
public:
    static Tokenizer from_json(const std::string& path);
    static Tokenizer from_gguf(const std::string& gguf_path);

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
};

#pragma once

#include <string>
#include <vector>

#include "inference/contracts/generation_result.hpp"

namespace arcaine::diffusion_gemma {

// Gemma-family token-boundary accounting (thought vs response channels).
// DiffusionGemma uses the Gemma tokenizer + chat template, so it shares the
// Gemma-native boundary semantics. This is a copy kept inside the diffusion
// module so the model owns its output parsing independently.
struct DiffusionGemmaBoundaryCounts {
    int reasoning_tokens = 0;
    int response_tokens  = 0;
};

struct DiffusionGemmaAssistantOutput {
    std::string                                  content;
    std::vector<arcaine::inference::ParsedToolCall> tool_calls;
};

// Parse Gemma-native assistant output: public, channel-stripped text plus tool
// calls in the Gemma <|tool_call>{...}<tool_call|> syntax.
DiffusionGemmaAssistantOutput parse_assistant_output(const std::string& raw_text);

class DiffusionGemmaBoundaryParser {
public:
    explicit DiffusionGemmaBoundaryParser(const std::string& model_dir);
    DiffusionGemmaBoundaryCounts count(const std::vector<int>& token_ids) const;

private:
    std::vector<int>                 thought_start_ids_;
    std::vector<std::vector<int>>    response_channel_start_ids_;
    std::vector<int>                 ignored_response_ids_;
    int                              channel_start_id_ = -1;
    int                              thought_end_id_   = -1;

    bool has_thought_markers() const;
    bool matches_at(const std::vector<int>& token_ids, size_t pos,
                    const std::vector<int>& pattern) const;
    const std::vector<int>* matching_response_channel_at(
        const std::vector<int>& token_ids, size_t pos) const;
    bool is_ignored_response_id(int token_id) const;
};

}  // namespace arcaine::diffusion_gemma

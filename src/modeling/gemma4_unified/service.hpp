#pragma once

#include <memory>

#include "inference/contracts/model_descriptor.hpp"
#include "inference/contracts/model_load_request.hpp"
#include "inference/service/model_service.hpp"

class Gemma4Model;       // global (modeling/gemma4_unified/model.hpp)
class TokenizerBridge;   // global (utils/chat.hpp)

namespace arcaine::gemma4_unified {

class Gemma4BoundaryParser;

// The public descriptor for the gemma4_unified implementation.
arcaine::inference::ModelDescriptor gemma4_descriptor();

// One loaded gemma4_unified implementation. Owns the architecture engine
// (Gemma4Model), the tokenizer + chat-template bridge, and the boundary
// parser; creates per-request sessions.
class Gemma4Service : public arcaine::inference::ModelService {
public:
    explicit Gemma4Service(const arcaine::inference::ModelLoadRequest& request);
    ~Gemma4Service() override;

    const arcaine::inference::ModelDescriptor& descriptor() const noexcept override {
        return descriptor_;
    }
    std::unique_ptr<arcaine::inference::InferenceSession> create_session(
        const arcaine::inference::SessionCreateRequest& req) override;

    Gemma4Model&          model() noexcept;
    TokenizerBridge&      tokenizer() noexcept;
    Gemma4BoundaryParser& boundary_parser() noexcept;

private:
    std::unique_ptr<Gemma4Model>          model_;
    std::unique_ptr<TokenizerBridge>      tokenizer_;
    std::unique_ptr<Gemma4BoundaryParser> boundary_parser_;
    arcaine::inference::ModelDescriptor   descriptor_;
};

}  // namespace arcaine::gemma4_unified

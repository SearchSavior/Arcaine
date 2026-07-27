#pragma once

#include <memory>

#include "inference/contracts/model_descriptor.hpp"
#include "inference/contracts/model_load_request.hpp"
#include "inference/service/model_service.hpp"

class Qwen35Model;      // global (modeling/qwen3_5/model.hpp)
class TokenizerBridge;  // global (utils/chat.hpp)

namespace arcaine::qwen3_5 {

// The public descriptor for the qwen3_5 implementation.
arcaine::inference::ModelDescriptor qwen35_descriptor();

// One loaded qwen3_5 implementation. Owns the architecture engine
// (Qwen35Model) and the tokenizer + chat-template bridge; creates per-request
// sessions.
class Qwen35Service : public arcaine::inference::ModelService {
public:
    explicit Qwen35Service(const arcaine::inference::ModelLoadRequest& request);
    ~Qwen35Service() override;

    const arcaine::inference::ModelDescriptor& descriptor() const noexcept override {
        return descriptor_;
    }
    std::unique_ptr<arcaine::inference::InferenceSession> create_session(
        const arcaine::inference::SessionCreateRequest& req) override;

    Qwen35Model&     model() noexcept      { return *model_; }
    TokenizerBridge& tokenizer() noexcept  { return *tokenizer_; }

private:
    std::unique_ptr<Qwen35Model>          model_;
    std::unique_ptr<TokenizerBridge>      tokenizer_;
    arcaine::inference::ModelDescriptor   descriptor_;
};

}  // namespace arcaine::qwen3_5

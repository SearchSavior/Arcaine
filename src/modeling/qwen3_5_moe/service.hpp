#pragma once

#include <memory>

#include "inference/contracts/model_descriptor.hpp"
#include "inference/contracts/model_load_request.hpp"
#include "inference/service/model_service.hpp"

class QwenModel;       // global (modeling/qwen3_5_moe/model.hpp)
class TokenizerBridge;  // global (utils/chat.hpp)

namespace arcaine::qwen3_5_moe {

// The public descriptor for the qwen3_5_moe implementation (model_type
// "qwen3_5_moe_text").
arcaine::inference::ModelDescriptor qwen_moe_descriptor();

// One loaded qwen3_5_moe implementation. Owns the architecture engine
// (QwenModel, including its MoE routing and expert execution) and the
// tokenizer + chat-template bridge; creates per-request sessions.
class QwenMoeService : public arcaine::inference::ModelService {
public:
    explicit QwenMoeService(const arcaine::inference::ModelLoadRequest& request);
    ~QwenMoeService() override;

    const arcaine::inference::ModelDescriptor& descriptor() const noexcept override {
        return descriptor_;
    }
    std::unique_ptr<arcaine::inference::InferenceSession> create_session(
        const arcaine::inference::SessionCreateRequest& req) override;

    QwenModel&      model() noexcept     { return *model_; }
    TokenizerBridge& tokenizer() noexcept { return *tokenizer_; }

private:
    std::unique_ptr<QwenModel>           model_;
    std::unique_ptr<TokenizerBridge>     tokenizer_;
    arcaine::inference::ModelDescriptor  descriptor_;
};

}  // namespace arcaine::qwen3_5_moe

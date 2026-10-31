#include "register.hpp"
#include "model.hpp"
#include "../../common/registry.hpp"
#include <memory>

void register_qwen3_5_moe_text(ModelRegistry& reg) {
    auto factory = [](const std::string& model_dir, int max_seq_len) -> std::unique_ptr<Model> {
        return std::make_unique<QwenModel>(model_dir, max_seq_len);
    };
    // "qwen3_5_moe_text" is the NVFP4 text checkpoint's model_type;
    // "qwen3_5_moe" is the AWQ pack-quantized INT4 (Qwen-AgentWorld) variant.
    reg.register_arch("qwen3_5_moe_text", factory);
    reg.register_arch("qwen3_5_moe", factory);
}

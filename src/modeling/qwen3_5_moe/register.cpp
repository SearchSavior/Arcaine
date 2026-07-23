#include "register.hpp"
#include "model.hpp"
#include "../../common/registry.hpp"
#include <memory>

void register_qwen3_5_moe_text(ModelRegistry& reg) {
    reg.register_arch("qwen3_5_moe_text",
        [](const std::string& model_dir, int max_seq_len) -> std::unique_ptr<Model> {
            return std::make_unique<QwenModel>(model_dir, max_seq_len);
        });
}

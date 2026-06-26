#include "register.hpp"
#include "model.hpp"
#include "../../common/registry.hpp"
#include <memory>

void register_gemma4_unified(ModelRegistry& reg) {
    reg.register_arch("gemma4_unified",
        [](const std::string& model_dir, int max_seq_len) -> std::unique_ptr<Model> {
            return std::make_unique<Gemma4Model>(model_dir, max_seq_len);
        });
}

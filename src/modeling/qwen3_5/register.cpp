#include "register.hpp"

#include <memory>

#include "model.hpp"
#include "../../common/registry.hpp"

void register_qwen3_5(ModelRegistry& registry) {
    registry.register_arch("qwen3_5", [](const std::string& directory, int max_seq_len) {
        return std::make_unique<Qwen35Model>(directory, max_seq_len);
    });
}

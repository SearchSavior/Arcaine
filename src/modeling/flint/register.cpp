#include "register.hpp"

#include <memory>
#include <string>

#include "../../common/registry.hpp"
#include "model.hpp"

void register_flint(ModelRegistry& reg) {
    reg.register_arch("flint", [](const std::string& dir, int max_seq_len) {
        return std::make_unique<FlintModel>(dir, max_seq_len);
    });
}

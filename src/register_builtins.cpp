#include "common/registry.hpp"
#include "modeling/gemma4_unified/register.hpp"
#include "modeling/qwen3_5_moe/register.hpp"
#include "modeling/qwen3_5/register.hpp"

// Wires every architecture compiled into this binary into the registry. Add a
// new arch by including its register.hpp and calling its registrar here.
void register_builtin_architectures() {
    ModelRegistry& reg = ModelRegistry::instance();
    register_gemma4_unified(reg);
    register_qwen3_5_moe_text(reg);
    register_qwen3_5(reg);
}

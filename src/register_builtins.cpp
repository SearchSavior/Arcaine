#include "common/registry.hpp"
#include "gemma4_unified/register.hpp"

// Wires every architecture compiled into this binary into the registry. Add a
// new arch by including its register.hpp and calling its registrar here.
void register_builtin_architectures() {
    ModelRegistry& reg = ModelRegistry::instance();
    register_gemma4_unified(reg);
}

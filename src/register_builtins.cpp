#include "common/registry.hpp"
#include "modeling/gemma4_unified/register.hpp"
#include "modeling/qwen3_5_moe/register.hpp"
#include "modeling/qwen3_5/register.hpp"

// Wires every AR architecture compiled into this binary into the registry.
// Each call is gated by the ARCAINE_MODEL_<NAME> define (set from ARCAINE_MODELS
// at configure time), so a single-model build only links the architectures it
// compiled. (DiffusionGemma is not wired here: pre-cutover the server
// constructs it directly.)
void register_builtin_architectures() {
    ModelRegistry& reg = ModelRegistry::instance();
#ifdef ARCAINE_MODEL_GEMMA4_UNIFIED
    register_gemma4_unified(reg);
#endif
#ifdef ARCAINE_MODEL_QWEN3_5_MOE
    register_qwen3_5_moe_text(reg);
#endif
#ifdef ARCAINE_MODEL_QWEN3_5
    register_qwen3_5(reg);
#endif
}

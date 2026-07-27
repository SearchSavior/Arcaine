#pragma once

namespace arcaine::inference {

class ModelRegistry;

// Registers every model implementation compiled into this binary. Each model
// module exposes `namespace <m> { void register_model(ModelRegistry&); }`;
// this function calls only the ones whose ARCAINE_MODEL_<NAME> define is set.
// Explicit registration only — no hidden static-initialization registration.
void register_builtin_models(ModelRegistry& registry);

}  // namespace arcaine::inference

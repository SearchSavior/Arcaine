#include "modeling/gemma4_unified/inference/cache.hpp"
#include "modeling/gemma4_unified/model.hpp"

namespace arcaine::gemma4_unified {

void Gemma4Cache::reset() { model_.reset_cache(); }

}  // namespace arcaine::gemma4_unified

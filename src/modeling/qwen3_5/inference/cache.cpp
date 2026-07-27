#include "modeling/qwen3_5/inference/cache.hpp"
#include "modeling/qwen3_5/model.hpp"

namespace arcaine::qwen3_5 {

void Qwen35Cache::reset() { model_.reset_cache(); }

}  // namespace arcaine::qwen3_5

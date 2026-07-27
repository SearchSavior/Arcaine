#include "modeling/qwen3_5_moe/inference/cache.hpp"
#include "modeling/qwen3_5_moe/model.hpp"

namespace arcaine::qwen3_5_moe {

void QwenMoeCache::reset() { model_.reset_cache(); }

}  // namespace arcaine::qwen3_5_moe

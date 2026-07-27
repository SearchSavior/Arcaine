#pragma once

namespace arcaine::inference {

// Per-request sampling overrides. Sentinel value -1 means "use the model's
// configured default" (the owning model session resolves defaults from its
// config / generation_config.json).
struct SamplingOptions {
    float temperature = -1.0f;
    int   top_k        = -1;
    float top_p        = -1.0f;

    bool overrides_temperature() const { return temperature >= 0.0f; }
    bool overrides_top_k()       const { return top_k       >= 0; }
    bool overrides_top_p()       const { return top_p       >= 0.0f; }
};

}  // namespace arcaine::inference

#pragma once

namespace arcaine::inference {

// Public metric fields. Spelling is part of the API contract and is preserved
// exactly, including the historical "decode_throuput" typo.
struct GenerationMetrics {
    int    input_token        = 0;
    int    new_token          = 0;
    double ttft               = 0.0;   // seconds
    double tpot               = 0.0;   // ms/token after first token
    double prefill_throughput = 0.0;   // tokens/second
    double decode_throuput    = 0.0;   // tokens/second (API spelling preserved)
    double duration           = 0.0;   // seconds
};

}  // namespace arcaine::inference

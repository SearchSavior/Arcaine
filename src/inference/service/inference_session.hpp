#pragma once

#include <memory>

#include "inference/contracts/generation_event.hpp"
#include "inference/contracts/generation_request.hpp"
#include "inference/contracts/generation_result.hpp"
#include "inference/service/generation_sink.hpp"
#include "inference/service/session_create.hpp"

namespace arcaine::inference {

// Owns mutable generation state for one logical generation: KV / DeltaNet
// cache, diffusion canvas, sampling RNG, generated-token state, temporary GPU
// buffers, request metrics, stream state. Model-specific session state is not
// exposed through this interface.
class InferenceSession {
public:
    virtual ~InferenceSession() = default;

    // Run complete generation, emitting events through `sink`. The session
    // owns prefill, decode/denoise loop, sampling, EOS handling, streaming,
    // metrics, and cancellation. Returns the final result.
    virtual GenerationResult generate(
        const GenerationRequest& request,
        GenerationSink& sink) = 0;

    virtual void cancel() = 0;
};

}  // namespace arcaine::inference

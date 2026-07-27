#pragma once

#include "inference/contracts/generation_event.hpp"

namespace arcaine::inference {

// Sink through which a model session emits GenerationEvent objects. Returning
// false from emit() signals the session to stop generation and trigger
// cancellation (e.g. an SSE sink whose client has disconnected).
class GenerationSink {
public:
    virtual ~GenerationSink() = default;
    virtual bool emit(const GenerationEvent& event) = 0;
};

}  // namespace arcaine::inference

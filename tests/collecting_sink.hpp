#pragma once

#include <string>
#include <variant>

#include "inference/contracts/generation_event.hpp"
#include "inference/contracts/generation_metrics.hpp"
#include "inference/service/generation_sink.hpp"

namespace arcaine::test {

// A GenerationSink that collects text deltas and the final metrics/completion
// for non-streaming generation tests.
class CollectingGenerationSink : public arcaine::inference::GenerationSink {
public:
    bool emit(const arcaine::inference::GenerationEvent& event) override {
        using namespace arcaine::inference;
        if (std::holds_alternative<StartedEvent>(event)) {
            started = true;
        } else if (std::holds_alternative<TextDeltaEvent>(event)) {
            text += std::get<TextDeltaEvent>(event).delta;
        } else if (std::holds_alternative<MetricsEvent>(event)) {
            metrics = std::get<MetricsEvent>(event).metrics;
        } else if (std::holds_alternative<CompletedEvent>(event)) {
            completed    = true;
            finish_reason = std::get<CompletedEvent>(event).finish_reason;
        }
        return true;  // never cancel
    }

    bool                               started   = false;
    bool                               completed = false;
    std::string                        text;
    std::string                        finish_reason;
    arcaine::inference::GenerationMetrics metrics;
};

}  // namespace arcaine::test

#include "check.hpp"

#include "inference/contracts/capabilities.hpp"

#include <cstdint>

using namespace arcaine::inference;

int main() {
    // Bitmask composition + queries.
    std::uint32_t in_mask =
        static_cast<std::uint32_t>(InputCapability::PromptText |
                                   InputCapability::Messages |
                                   InputCapability::Tools);
    CHECK(has_input_capability(in_mask, InputCapability::PromptText));
    CHECK(has_input_capability(in_mask, InputCapability::Messages));
    CHECK(has_input_capability(in_mask, InputCapability::Tools));
    CHECK(!has_input_capability(in_mask, InputCapability::Images));
    CHECK(!has_input_capability(in_mask, InputCapability::Audio));
    CHECK(!has_input_capability(0u, InputCapability::PromptText));

    std::uint32_t out_mask =
        static_cast<std::uint32_t>(OutputCapability::Text |
                                   OutputCapability::TextDeltas |
                                   OutputCapability::ToolCalls |
                                   OutputCapability::DraftTokens);
    CHECK(has_output_capability(out_mask, OutputCapability::Text));
    CHECK(has_output_capability(out_mask, OutputCapability::ToolCalls));
    CHECK(has_output_capability(out_mask, OutputCapability::DraftTokens));
    CHECK(!has_output_capability(out_mask, OutputCapability::TokenIds));

    // Mixing uint32_t with capability enum via the overloaded operator.
    std::uint32_t mixed = in_mask | InputCapability::Images;
    CHECK(has_input_capability(mixed, InputCapability::Images));

    RETURN_TESTS();
}

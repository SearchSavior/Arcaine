// Pass-through check for the qwen3_5 output_parser (no GPU). The full
// Qwen3.5-native tool-call parser is deferred; for now the parser returns the
// decoded text as content with no tool calls.
#include "check.hpp"

#include "modeling/qwen3_5/output_parser.hpp"

#include <string>

using namespace arcaine::qwen3_5;

int main() {
    auto out = parse_assistant_output("The capital of France is Paris.  ");
    CHECK(out.tool_calls.empty());
    CHECK_EQ(out.content, "The capital of France is Paris.");
    auto empty = parse_assistant_output("");
    CHECK(empty.tool_calls.empty());
    CHECK(empty.content.empty());
    RETURN_TESTS();
}

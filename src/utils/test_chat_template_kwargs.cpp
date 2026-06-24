// Standalone test for --chat-template-kwargs handling.
//   pure assertions run always; render-merge assertions run only when a model
//   directory is passed as argv[1] (following the test_tokenizer convention).
#include "chat_template_kwargs.hpp"
#include "../common/preprocess/chat_template.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

static int failures = 0;

static void check(bool cond, const char* what) {
    if (cond) {
        std::printf("  ok: %s\n", what);
    } else {
        std::printf("  FAIL: %s\n", what);
        ++failures;
    }
}

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static nlohmann::ordered_json user_messages() {
    return nlohmann::ordered_json::array({
        {{"role", "user"},
         {"content", nlohmann::ordered_json::array({{{"type", "text"}, {"text", "hi"}}})}},
    });
}

// Requires a real model dir (argv[1]) with chat_template.jinja + tokenizer.json.
static void test_render_merges_kwargs(const std::string& model_dir) {
    std::printf("render merge (model=%s):\n", model_dir.c_str());
    const auto msgs = user_messages();
    const auto tools = nlohmann::ordered_json::array();

    // enable_thinking via the bool param vs. via kwargs vs. default off.
    auto off       = build_chat_prompt_json(model_dir, msgs, tools, true, false, {});
    auto on_bool   = build_chat_prompt_json(model_dir, msgs, tools, true, true, {});
    auto on_kwargs = build_chat_prompt_json(model_dir, msgs, tools, true, false,
                                            {{"enable_thinking", true}});

    check(off.tokens != on_bool.tokens,
          "enable_thinking actually changes the rendered prompt");
    check(on_kwargs.tokens == on_bool.tokens,
          "kwargs enable_thinking overrides the bool param (kwargs win)");
}

static void test_parse() {
    std::printf("parse_chat_template_kwargs:\n");

    auto kw = parse_chat_template_kwargs(R"({"enable_thinking": true})");
    check(kw.is_object(), "valid object parses to an object");
    check(kw.contains("enable_thinking") && kw["enable_thinking"] == true,
          "object preserves enable_thinking=true");

    check(throws_runtime_error([] { parse_chat_template_kwargs("{not json"); }),
          "malformed JSON throws");
    check(throws_runtime_error([] { parse_chat_template_kwargs("[1,2]"); }),
          "JSON array (non-object) throws");
    check(throws_runtime_error([] { parse_chat_template_kwargs("5"); }),
          "JSON scalar (non-object) throws");
    check(throws_runtime_error([] { parse_chat_template_kwargs("\"x\""); }),
          "JSON string (non-object) throws");
}

int main(int argc, char** argv) {
    test_parse();

    if (argc > 1) {
        test_render_merges_kwargs(argv[1]);
    } else {
        std::printf("render merge: skipped (pass a model dir as argv[1] to run)\n");
    }

    if (failures) {
        std::printf("\n%d assertion(s) failed\n", failures);
        return 1;
    }
    std::printf("\nall assertions passed\n");
    return 0;
}

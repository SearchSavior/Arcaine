import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSIONGEMMA_MODEL_DIR = Path("/workspace/models/diffusiongemma-26B-A4B-it-NVFP4")
GEMMA4_UNIFIED_MODEL_DIR = Path("/workspace/models/gemma-4-12B-it")

# Parser tests are parameterized over both Gemma4-family model dirs so that
# adding a Gemma4 unified checkpoint automatically exercises the same parser
# paths as DiffusionGemma. Both arches share parse_assistant_output (the parser
# is text-based and uses hardcoded <|tool_call>/<tool_call|> markers), so the
# same inputs are piped through the C++ harness for each model dir.
#
# Note on token layouts: DiffusionGemma stores stc/etc/escape tokens under a
# `model_specific_special_tokens` block (with top-level fallbacks). Gemma4 12B
# it stores them at the top level only. tokenizer_tool_call_tokens() reads
# the nested block when present and falls back to top-level keys otherwise.
MODEL_DIR_PARAMS = [
    pytest.param(DIFFUSIONGEMMA_MODEL_DIR, id="diffusiongemma"),
    pytest.param(GEMMA4_UNIFIED_MODEL_DIR, id="gemma4_unified"),
]


@pytest.fixture(scope="session")
def diffusiongemma_tokenizer():
    return AutoTokenizer.from_pretrained(
        DIFFUSIONGEMMA_MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )


@pytest.fixture(scope="session")
def diffusiongemma_tokenizer_config() -> dict:
    return json.loads((DIFFUSIONGEMMA_MODEL_DIR / "tokenizer_config.json").read_text())


@pytest.fixture(scope="session")
def diffusiongemma_chat_template() -> str:
    return (DIFFUSIONGEMMA_MODEL_DIR / "chat_template.jinja").read_text()


@pytest.fixture(scope="session")
def gemma4_unified_tokenizer():
    if not GEMMA4_UNIFIED_MODEL_DIR.exists():
        pytest.skip(f"gemma4_unified model dir not present: {GEMMA4_UNIFIED_MODEL_DIR}")
    return AutoTokenizer.from_pretrained(
        GEMMA4_UNIFIED_MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )


@pytest.fixture(scope="session")
def gemma4_unified_tokenizer_config() -> dict:
    if not GEMMA4_UNIFIED_MODEL_DIR.exists():
        pytest.skip(f"gemma4_unified model dir not present: {GEMMA4_UNIFIED_MODEL_DIR}")
    return json.loads((GEMMA4_UNIFIED_MODEL_DIR / "tokenizer_config.json").read_text())


@pytest.fixture(scope="session")
def gemma4_unified_chat_template() -> str:
    if not GEMMA4_UNIFIED_MODEL_DIR.exists():
        pytest.skip(f"gemma4_unified model dir not present: {GEMMA4_UNIFIED_MODEL_DIR}")
    return (GEMMA4_UNIFIED_MODEL_DIR / "chat_template.jinja").read_text()


# Parameterized tokenizer_config fixture: yields the tokenizer_config.json dict
# for each available Gemma4-family model dir. Skips entries whose model dir is
# not present on disk.
@pytest.fixture(scope="session", params=MODEL_DIR_PARAMS)
def parser_tokenizer_config_dir(request) -> Path:
    if not request.param.exists():
        pytest.skip(f"model dir not present: {request.param}")
    return request.param


@pytest.fixture(scope="session")
def parser_tokenizer_config(parser_tokenizer_config_dir: Path) -> dict:
    return json.loads((parser_tokenizer_config_dir / "tokenizer_config.json").read_text())


@pytest.fixture(scope="session")
def parser_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("g++") is None:
        pytest.skip("g++ is required for the C++ parser unit harness")

    build_dir = tmp_path_factory.mktemp("gemma4_tool_call_parser")
    harness = build_dir / "gemma4_tool_call_parser_harness.cpp"
    binary = build_dir / "gemma4_tool_call_parser_harness"
    harness.write_text(
        textwrap.dedent(
            r'''
            #include "utils/gemma4_tool_call_parser.hpp"
            #include <nlohmann/json.hpp>
            #include <iostream>
            #include <iterator>
            #include <string>

            int main() {
                std::string input((std::istreambuf_iterator<char>(std::cin)),
                                  std::istreambuf_iterator<char>());
                ParsedAssistantOutput parsed = parse_assistant_output(input);
                nlohmann::ordered_json out;
                out["content"] = parsed.content;
                out["tool_calls"] = nlohmann::ordered_json::array();
                for (const auto& call : parsed.tool_calls) {
                    out["tool_calls"].push_back({
                        {"id", call.id},
                        {"name", call.name},
                        {"arguments", call.arguments},
                    });
                }
                std::cout << out.dump();
                return 0;
            }
            '''
        )
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            str(harness),
            str(REPO_ROOT / "src/utils/gemma4_tool_call_parser.cpp"),
            str(REPO_ROOT / "src/common/preprocess/tokenizer.cpp"),
            # tokenizer.cpp depends on the ported llama.cpp unicode helpers
            # (unicode_len_utf8, unicode_utf8_to_byte, unicode_cpts_from_utf8,
            # unicode_regex_split_custom_qwen2, unicode_cpt_to_utf8,
            # unicode_byte_to_utf8). These are the same UNICODE_SRCS linked
            # into every CMake target that compiles tokenizer.cpp — see
            # CMakeLists.txt:50-53.
            str(REPO_ROOT / "src/common/preprocess/unicode.cpp"),
            str(REPO_ROOT / "src/common/preprocess/unicode-data.cpp"),
            "-I",
            str(REPO_ROOT / "src"),
            "-I",
            str(REPO_ROOT / "third_party"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def parse_with_harness(parser_harness: Path, raw: str) -> dict:
    proc = subprocess.run(
        [str(parser_harness)],
        input=raw,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout)


def tokenizer_tool_call_tokens(config: dict) -> tuple[str, str, str]:
    # DiffusionGemma stores stc/etc/escape under `model_specific_special_tokens`
    # (with top-level fallbacks). Gemma4 12B it stores them at the top level
    # only. Prefer the nested block when present, else read top-level keys.
    if "model_specific_special_tokens" in config:
        special = config["model_specific_special_tokens"]
        return special["stc_token"], special["etc_token"], special["escape_token"]
    return config["stc_token"], config["etc_token"], config["escape_token"]


def test_matches_diffusiongemma_gemma_tokenizer_response_schema(
    diffusiongemma_tokenizer,
    diffusiongemma_tokenizer_config: dict,
    diffusiongemma_chat_template: str,
):
    assert diffusiongemma_tokenizer_config["tokenizer_class"] == "GemmaTokenizer"
    assert diffusiongemma_tokenizer_config["processor_class"] == "Gemma4Processor"
    assert diffusiongemma_tokenizer.name_or_path == str(DIFFUSIONGEMMA_MODEL_DIR)
    assert diffusiongemma_tokenizer.bos_token == diffusiongemma_tokenizer_config["bos_token"]
    assert diffusiongemma_tokenizer.eos_token == diffusiongemma_tokenizer_config["eos_token"]
    assert diffusiongemma_tokenizer.chat_template == diffusiongemma_chat_template

    schema = diffusiongemma_tokenizer_config["response_schema"]
    assert schema["properties"]["tool_calls"]["x-regex-iterator"] == r"<\|tool_call>(.*?)<tool_call\|>"
    assert schema["properties"]["tool_calls"]["items"]["properties"]["function"]["x-regex"] == (
        r"call\:(?P<name>\w+)(?P<arguments>\{.*\})"
    )
    assert "{{- '<|tool_call>call:' + function['name'] + '{' -}}" in diffusiongemma_chat_template
    assert "{{- '}<tool_call|>' -}}" in diffusiongemma_chat_template


def test_matches_gemma4_unified_tokenizer_response_schema(
    gemma4_unified_tokenizer,
    gemma4_unified_tokenizer_config: dict,
    gemma4_unified_chat_template: str,
):
    # The shared parse_assistant_output parser uses the same <|tool_call>...
    # <tool_call|> markers across the Gemma4 family. Gemma4 unified exposes
    # the same response_schema regexes as DiffusionGemma so the same parser
    # is reusable; this test pins that contract. The processor_class differs
    # (Gemma4UnifiedProcessor vs Gemma4Processor) but the tool-call markers
    # and regexes are identical.
    assert gemma4_unified_tokenizer_config["tokenizer_class"] == "GemmaTokenizer"
    assert gemma4_unified_tokenizer_config["processor_class"] == "Gemma4UnifiedProcessor"
    assert gemma4_unified_tokenizer.name_or_path == str(GEMMA4_UNIFIED_MODEL_DIR)
    assert gemma4_unified_tokenizer.bos_token == gemma4_unified_tokenizer_config["bos_token"]
    assert gemma4_unified_tokenizer.eos_token == gemma4_unified_tokenizer_config["eos_token"]
    assert gemma4_unified_tokenizer.chat_template == gemma4_unified_chat_template

    schema = gemma4_unified_tokenizer_config["response_schema"]
    assert schema["properties"]["tool_calls"]["x-regex-iterator"] == r"<\|tool_call>(.*?)<tool_call\|>"
    assert schema["properties"]["tool_calls"]["items"]["properties"]["function"]["x-regex"] == (
        r"call\:(?P<name>\w+)(?P<arguments>\{.*\})"
    )
    assert "{{- '<|tool_call>call:' + function['name'] + '{' -}}" in gemma4_unified_chat_template
    assert "{{- '}<tool_call|>' -}}" in gemma4_unified_chat_template


def test_parses_single_gemma_tokenizer_tool_call(
    parser_harness: Path,
    parser_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(parser_tokenizer_config)
    raw_call = (
        f"{stc_token}call:get_weather{{city:{escape_token}Paris{escape_token},"
        f"unit:{escape_token}celsius{escape_token},days:3,alerts:true}}{etc_token}"
    )

    schema = parser_tokenizer_config["response_schema"]["properties"]["tool_calls"]
    match = re.fullmatch(schema["x-regex-iterator"], raw_call, flags=re.DOTALL)
    assert match is not None
    body = match.group(1)
    function_regex = schema["items"]["properties"]["function"]["x-regex"]
    function_match = re.fullmatch(function_regex, body, flags=re.DOTALL)
    assert function_match is not None
    assert function_match.group("name") == "get_weather"

    parsed = parse_with_harness(
        parser_harness,
        raw_call,
    )

    assert parsed["content"] == ""
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["id"] == "call_0"
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {
        "city": "Paris",
        "unit": "celsius",
        "days": 3,
        "alerts": True,
    }


def test_parses_nested_arrays_and_objects(
    parser_harness: Path,
    parser_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(parser_tokenizer_config)
    parsed = parse_with_harness(
        parser_harness,
        f"{stc_token}call:search{{query:{escape_token}intel gpu{escape_token},"
        f"filters:{{vendor:{escape_token}intel{escape_token},ids:[1,2,3],enabled:false}}}}{etc_token}",
    )

    args = json.loads(parsed["tool_calls"][0]["arguments"])
    assert args == {
        "query": "intel gpu",
        "filters": {"vendor": "intel", "ids": [1, 2, 3], "enabled": False},
    }


def test_parses_tool_call_with_extra_trailing_brace(
    parser_harness: Path,
    parser_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(parser_tokenizer_config)
    parsed = parse_with_harness(
        parser_harness,
        f"{stc_token}call:bash{{command:{escape_token}"
        "uv venv .venv\nsource .venv/bin/activate\npython stability_proof.py"
        f"{escape_token}}}}}{etc_token}",
    )

    assert parsed["content"] == ""
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["name"] == "bash"
    assert json.loads(call["arguments"]) == {
        "command": "uv venv .venv\nsource .venv/bin/activate\npython stability_proof.py",
    }


def test_parses_tool_call_with_duplicated_nested_braces(
    parser_harness: Path,
    parser_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(parser_tokenizer_config)
    parsed = parse_with_harness(
        parser_harness,
        f"{stc_token}call:edit{{edits:[{{{{newText:{escape_token}"
        "dy_noisy = np.diff(y_noisy) / h"
        f"{escape_token},oldText:{escape_token}"
        "dy_noisy = np.diff(y_noisy, h) / h"
        f"{escape_token}}}],path:{escape_token}stability_proof.py{escape_token}}}}}{etc_token}",
    )

    assert parsed["content"] == ""
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["name"] == "edit"
    assert json.loads(call["arguments"]) == {
        "edits": [
            {
                "newText": "dy_noisy = np.diff(y_noisy) / h",
                "oldText": "dy_noisy = np.diff(y_noisy, h) / h",
            }
        ],
        "path": "stability_proof.py",
    }


def test_parses_visible_content_before_malformed_edit_tool_call(
    parser_harness: Path,
    parser_tokenizer_config: dict,
):
    cfg = parser_tokenizer_config
    soc, eoc = cfg["soc_token"], cfg["eoc_token"]
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(parser_tokenizer_config)
    parsed = parse_with_harness(
        parser_harness,
        f"{soc}thought\n{eoc}"
        "The error occurred because np.diff does not take a step size.\n\n"
        f"{stc_token}call:edit{{edits:[{{{{newText:{escape_token}"
        "dy_noisy = np.diff(y_noisy) / h"
        f"{escape_token},oldText:{escape_token}"
        "dy_noisy = np.diff(y_noisy, h) / h"
        f"{escape_token}}}],path:{escape_token}stability_proof.py{escape_token}}}}}{etc_token}",
    )

    assert parsed["content"] == "The error occurred because np.diff does not take a step size."
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["name"] == "edit"


def test_removes_gemma_tokenizer_thinking_and_special_tokens_from_content(
    parser_harness: Path,
    parser_tokenizer_config: dict,
):
    # soc/eoc/eot tokens live at the top level on both layouts (see
    # tokenizer_tool_call_tokens). Read them directly so this test doesn't
    # depend on the model_specific_special_tokens block existing.
    cfg = parser_tokenizer_config
    soc, eoc, eot = cfg["soc_token"], cfg["eoc_token"], cfg["eot_token"]
    parsed = parse_with_harness(
        parser_harness,
        f"{cfg['bos_token']}"
        f"{soc}thought\nhidden chain{eoc}\n"
        f"Final answer.{eot}{cfg['eos_token']}",
    )

    assert parsed == {"content": "Final answer.", "tool_calls": []}


def test_parses_multiple_tool_calls_and_keeps_surrounding_content(
    parser_harness: Path,
    parser_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(parser_tokenizer_config)
    parsed = parse_with_harness(
        parser_harness,
        'I will check.\n'
        f"{stc_token}call:first{{value:1}}{etc_token}\n"
        f"{stc_token}call:second{{name:{escape_token}Ada{escape_token}}}{etc_token}",
    )

    assert parsed["content"] == "I will check."
    assert [call["id"] for call in parsed["tool_calls"]] == ["call_0", "call_1"]
    assert [call["name"] for call in parsed["tool_calls"]] == ["first", "second"]
    assert json.loads(parsed["tool_calls"][0]["arguments"]) == {"value": 1}
    assert json.loads(parsed["tool_calls"][1]["arguments"]) == {"name": "Ada"}


def test_malformed_tool_call_stays_out_of_tool_calls(
    parser_harness: Path,
    parser_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(parser_tokenizer_config)
    parsed = parse_with_harness(
        parser_harness,
        f"Answer before {stc_token}call:broken{{unterminated:{escape_token}oops{etc_token} answer after",
    )

    assert parsed["tool_calls"] == []
    assert "Answer before" in parsed["content"]

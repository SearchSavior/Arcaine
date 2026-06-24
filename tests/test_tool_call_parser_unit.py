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
def parser_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("g++") is None:
        pytest.skip("g++ is required for the C++ parser unit harness")

    build_dir = tmp_path_factory.mktemp("tool_call_parser")
    harness = build_dir / "tool_call_parser_harness.cpp"
    binary = build_dir / "tool_call_parser_harness"
    harness.write_text(
        textwrap.dedent(
            r'''
            #include "utils/tool_call_parser.hpp"
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
            str(REPO_ROOT / "src/utils/tool_call_parser.cpp"),
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
    special = config["model_specific_special_tokens"]
    return special["stc_token"], special["etc_token"], special["escape_token"]


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


def test_parses_single_gemma_tokenizer_tool_call(
    parser_harness: Path,
    diffusiongemma_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(diffusiongemma_tokenizer_config)
    raw_call = (
        f"{stc_token}call:get_weather{{city:{escape_token}Paris{escape_token},"
        f"unit:{escape_token}celsius{escape_token},days:3,alerts:true}}{etc_token}"
    )

    schema = diffusiongemma_tokenizer_config["response_schema"]["properties"]["tool_calls"]
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
    diffusiongemma_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(diffusiongemma_tokenizer_config)
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


def test_removes_gemma_tokenizer_thinking_and_special_tokens_from_content(
    parser_harness: Path,
    diffusiongemma_tokenizer_config: dict,
):
    special = diffusiongemma_tokenizer_config["model_specific_special_tokens"]
    parsed = parse_with_harness(
        parser_harness,
        f"{diffusiongemma_tokenizer_config['bos_token']}"
        f"{special['soc_token']}thought\nhidden chain{special['eoc_token']}\n"
        f"Final answer.{special['eot_token']}{diffusiongemma_tokenizer_config['eos_token']}",
    )

    assert parsed == {"content": "Final answer.", "tool_calls": []}


def test_parses_multiple_tool_calls_and_keeps_surrounding_content(
    parser_harness: Path,
    diffusiongemma_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(diffusiongemma_tokenizer_config)
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
    diffusiongemma_tokenizer_config: dict,
):
    stc_token, etc_token, escape_token = tokenizer_tool_call_tokens(diffusiongemma_tokenizer_config)
    parsed = parse_with_harness(
        parser_harness,
        f"Answer before {stc_token}call:broken{{unterminated:{escape_token}oops{etc_token} answer after",
    )

    assert parsed["tool_calls"] == []
    assert "Answer before" in parsed["content"]

import json
import os

import pytest
from openai import OpenAI


pytestmark = pytest.mark.skipif(
    not os.environ.get("ARCAINE_TEST_OPENAI_BASE_URL")
    or not os.environ.get("ARCAINE_TEST_MODEL"),
    reason="set ARCAINE_TEST_OPENAI_BASE_URL and ARCAINE_TEST_MODEL to test a running diffusion_server",
)


def client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["ARCAINE_TEST_OPENAI_BASE_URL"],
        api_key=os.environ.get("ARCAINE_TEST_API_KEY", "local"),
    )


def model_name() -> str:
    return os.environ["ARCAINE_TEST_MODEL"]


def weather_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }


def test_models_endpoint_is_openai_client_compatible():
    models = client().models.list()
    assert any(model.id == model_name() for model in models.data)


def test_chat_completions_accepts_tools_and_returns_openai_shape():
    response = client().chat.completions.create(
        model=model_name(),
        messages=[
            {
                "role": "user",
                "content": "Use get_weather for Paris. Do not answer directly.",
            }
        ],
        tools=[weather_tool()],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=128,
    )

    choice = response.choices[0]
    assert choice.finish_reason in {"tool_calls", "stop", "length"}
    message = choice.message
    assert message.content is None or isinstance(message.content, str)

    if choice.finish_reason == "tool_calls":
        assert message.tool_calls
        call = message.tool_calls[0]
        assert call.type == "function"
        assert call.function.name == "get_weather"
        parsed_args = json.loads(call.function.arguments)
        assert isinstance(parsed_args, dict)


def test_streaming_tools_does_not_expose_draft_tool_calls_as_content():
    chunks = client().chat.completions.create(
        model=model_name(),
        messages=[
            {
                "role": "user",
                "content": "Use get_weather for Tokyo. Do not answer directly.",
            }
        ],
        tools=[weather_tool()],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=128,
        stream=True,
    )

    saw_final = False
    for chunk in chunks:
        if chunk.choices and chunk.choices[0].finish_reason:
            saw_final = True
            assert chunk.choices[0].finish_reason in {"tool_calls", "stop", "length"}
        if chunk.choices and chunk.choices[0].delta.content:
            assert "<|tool_call>" not in chunk.choices[0].delta.content
            assert "<tool_call|>" not in chunk.choices[0].delta.content

    assert saw_final

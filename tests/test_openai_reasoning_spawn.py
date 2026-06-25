import json
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "diffusiongemma-26B-A4B-it-NVFP4"


pytestmark = pytest.mark.skipif(
    os.environ.get("ARCAINE_TEST_SPAWN_SERVER") != "1",
    reason="set ARCAINE_TEST_SPAWN_SERVER=1 to launch diffusion_server",
)


@dataclass
class SpawnedServer:
    client: OpenAI
    model: str
    base_url: str
    log_path: Path
    proc: subprocess.Popen


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def tail(path: Path, lines: int = 160) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def wait_for_server(client: OpenAI, model: str, proc: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + float(os.environ.get("ARCAINE_TEST_SERVER_READY_TIMEOUT", "600"))
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"diffusion_server exited early with code {proc.returncode}\n\n{tail(log_path)}"
            )
        try:
            models = client.models.list()
            if any(item.id == model for item in models.data):
                return
        except Exception as exc:  # noqa: BLE001 - diagnostics belong in the failure.
            last_error = exc
        time.sleep(1.0)
    raise AssertionError(
        f"diffusion_server did not become ready; last_error={last_error!r}\n\n{tail(log_path)}"
    )


@pytest.fixture(scope="session")
def spawned_server(tmp_path_factory: pytest.TempPathFactory) -> SpawnedServer:
    binary = Path(os.environ.get("ARCAINE_TEST_SERVER_BINARY", REPO_ROOT / "build/diffusion_server"))
    model_dir = Path(os.environ.get("ARCAINE_TEST_MODEL_DIR", REPO_ROOT / "models" / DEFAULT_MODEL))
    model_name = os.environ.get("ARCAINE_TEST_MODEL", model_dir.name)
    api_key = os.environ.get("ARCAINE_TEST_API_KEY", "local")
    host = "127.0.0.1"
    port = int(os.environ.get("ARCAINE_TEST_SERVER_PORT", free_port()))
    base_url = f"http://{host}:{port}/v1"
    log_path = tmp_path_factory.mktemp("arcaine_server") / "diffusion_server.log"

    chat_template_kwargs = os.environ.get(
        "ARCAINE_TEST_CHAT_TEMPLATE_KWARGS",
        '{"enable_thinking": true}',
    )
    cmd = [
        str(binary),
        "--model",
        str(model_dir),
        "--served-model-name",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--chat-template-kwargs",
        chat_template_kwargs,
    ]
    extra_args = os.environ.get("ARCAINE_TEST_SERVER_ARGS")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    env = os.environ.copy()
    env["ARCAINE_API_KEY"] = api_key
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=1200)
    try:
        wait_for_server(client, model_name, proc, log_path)
        yield SpawnedServer(client, model_name, base_url, log_path, proc)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def assert_public_content(text: str, log_path: Path) -> None:
    assert text.strip(), f"model produced no public content\n\nserver log:\n{tail(log_path)}"
    assert "<|channel>" not in text
    assert "<channel|>" not in text
    assert "<|tool_call>" not in text
    assert "<tool_call|>" not in text


def test_reasoning_enabled_non_streaming_returns_public_content(
    spawned_server: SpawnedServer,
    tmp_path: Path,
):
    response = spawned_server.client.chat.completions.create(
        model=spawned_server.model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Think privately if needed. In the final answer, write one short "
                    "sentence that includes the exact marker VISIBLE_REASONING_CHECK."
                ),
            }
        ],
        max_tokens=int(os.environ.get("ARCAINE_TEST_REASONING_MAX_TOKENS", "192")),
        seed=7,
    )
    payload_path = tmp_path / "non_streaming_response.json"
    payload_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    choice = response.choices[0]
    content = choice.message.content or ""
    assert_public_content(content, spawned_server.log_path)


def test_reasoning_enabled_streaming_returns_public_content_after_thought(
    spawned_server: SpawnedServer,
    tmp_path: Path,
):
    chunks = spawned_server.client.chat.completions.create(
        model=spawned_server.model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Use private reasoning if useful, then answer visibly with one "
                    "sentence containing the exact marker STREAM_VISIBLE_CHECK."
                ),
            }
        ],
        max_tokens=int(os.environ.get("ARCAINE_TEST_REASONING_MAX_TOKENS", "192")),
        seed=11,
        stream=True,
    )

    rows: list[dict] = []
    content_parts: list[str] = []
    final_reason = None
    for chunk in chunks:
        row = chunk.model_dump(mode="json")
        rows.append(row)
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta.content:
                content_parts.append(choice.delta.content)
            if choice.finish_reason:
                final_reason = choice.finish_reason

    write_jsonl(tmp_path / "streaming_chunks.jsonl", rows)
    content = "".join(content_parts)
    assert final_reason in {"stop", "length", "tool_calls"}
    assert_public_content(content, spawned_server.log_path)

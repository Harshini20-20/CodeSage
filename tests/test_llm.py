"""Tests for the LLM Integration / Ollama layer (Phase 1G).

Everything in normal test runs uses deterministic fakes and mocks:
no running Ollama server, no network, and no GPU required.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import pytest

from config import Config, settings
from src.llm import (
    DEFAULT_TEMPERATURE,
    LLM,
    LLMConnectionError,
    LLMError,
    LLMInputError,
    LLMResponseError,
    OllamaLLM,
)

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fakes & Helpers
# --------------------------------------------------------------------------- #
class FakeOllamaResponse:
    """Simulates an ollama.GenerateResponse object."""

    def __init__(self, response: str) -> None:
        self.response = response

    def __getitem__(self, item: str) -> Any:
        if item == "response":
            return self.response
        raise KeyError(item)


class FakeOllamaClient:
    """Deterministic fake Ollama client for unit testing."""

    def __init__(
        self,
        response_text: str = "This function calculates the factorial of n recursively.",
        raise_on_generate: Optional[Exception] = None,
        available: bool = True,
    ) -> None:
        self.response_text = response_text
        self.raise_on_generate = raise_on_generate
        self.available = available
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        model: str = "",
        prompt: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append({"model": model, "prompt": prompt, "options": options, "kwargs": kwargs})
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return {"response": self.response_text}

    def list(self) -> Any:
        if not self.available:
            raise ConnectionError("Ollama connection refused")
        return {"models": [{"name": "llama3"}]}


@pytest.fixture
def fake_client() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def llm(fake_client: FakeOllamaClient) -> LLM:
    return LLM(client=fake_client)


# --------------------------------------------------------------------------- #
# 1. Construction and Configuration
# --------------------------------------------------------------------------- #
def test_default_construction() -> None:
    llm = LLM()
    assert llm.model == settings.ollama_model
    assert llm.host == settings.ollama_host
    assert llm.model == "llama3.2:3b"
    assert "11434" in llm.host


def test_construction_with_custom_model() -> None:
    llm = LLM(model="codellama:7b")
    assert llm.model == "codellama:7b"
    assert llm.host == settings.ollama_host


def test_construction_with_custom_host() -> None:
    llm = LLM(host="http://custom-server:11434")
    assert llm.host == "http://custom-server:11434"
    assert llm.model == settings.ollama_model


def test_construction_with_custom_config() -> None:
    custom_cfg = Config(
        ollama_model="mistral:latest",
        ollama_host="http://192.168.1.100:11434",
    )
    llm = LLM(config=custom_cfg)
    assert llm.model == "mistral:latest"


def test_construction_with_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODESAGE_OLLAMA_MODEL", "deepseek-coder")
    monkeypatch.setenv("CODESAGE_OLLAMA_HOST", "http://ai-server:11434")
    from config import get_config

    cfg = get_config()
    llm = LLM(config=cfg)
    assert llm.model == "deepseek-coder"
    assert llm.host == "http://ai-server:11434"


def test_ollamallm_alias_compatibility() -> None:
    assert OllamaLLM is LLM
    instance = OllamaLLM(model="test-model")
    assert isinstance(instance, LLM)
    assert instance.model == "test-model"


def test_client_dependency_injection(fake_client: FakeOllamaClient) -> None:
    llm = LLM(client=fake_client)
    assert llm.client is fake_client

    another_client = FakeOllamaClient(response_text="other")
    llm.client = another_client
    assert llm.client is another_client


# --------------------------------------------------------------------------- #
# 2. Generation Flow & Parameter Passing
# --------------------------------------------------------------------------- #
def test_valid_prompt_returns_str(llm: LLM) -> None:
    res = llm.generate("Explain this function.")
    assert isinstance(res, str)
    assert res == "This function calculates the factorial of n recursively."


def test_ollama_client_called_with_correct_parameters(llm: LLM, fake_client: FakeOllamaClient) -> None:
    prompt = "Summarize src/parser.py"
    llm.generate(prompt)

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["model"] == llm.model
    assert call["prompt"] == prompt
    assert call["options"] == {"temperature": DEFAULT_TEMPERATURE}


def test_temperature_passed_correctly(llm: LLM, fake_client: FakeOllamaClient) -> None:
    llm.generate("Explain this class.", temperature=0.7)

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["options"]["temperature"] == 0.7


def test_custom_model_passed_to_client(fake_client: FakeOllamaClient) -> None:
    custom_llm = LLM(model="qwen2.5-coder:7b", client=fake_client)
    custom_llm.generate("Analyze this code.")

    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["model"] == "qwen2.5-coder:7b"


def test_client_without_options_parameter_supported() -> None:
    class MinimalClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def generate(self, model: str, prompt: str) -> dict[str, str]:
            self.calls.append((model, prompt))
            return {"response": "ok"}

    minimal = MinimalClient()
    llm = LLM(client=minimal)
    out = llm.generate("Test prompt")
    assert out == "ok"
    assert minimal.calls == [(llm.model, "Test prompt")]


# --------------------------------------------------------------------------- #
# 3. Prompt and Temperature Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_prompt", ["", "   ", "\t\n", "\n\n  \t "])
def test_empty_or_whitespace_prompt_raises_input_error(llm: LLM, fake_client: FakeOllamaClient, bad_prompt: str) -> None:
    with pytest.raises(LLMInputError) as exc_info:
        llm.generate(bad_prompt)
    assert "empty or whitespace" in str(exc_info.value)
    # Crucial requirement: client must NOT be contacted when prompt is invalid
    assert len(fake_client.calls) == 0


@pytest.mark.parametrize("invalid_type", [None, 123, 45.6, ["prompt"], {"text": "hi"}])
def test_non_string_prompt_raises_input_error(llm: LLM, fake_client: FakeOllamaClient, invalid_type: Any) -> None:
    with pytest.raises(LLMInputError) as exc_info:
        llm.generate(invalid_type)  # type: ignore[arg-type]
    assert "must be a string" in str(exc_info.value)
    assert len(fake_client.calls) == 0


@pytest.mark.parametrize("bad_temp", [-0.1, -2.0, "high", None, True, False])
def test_invalid_temperature_raises_input_error(llm: LLM, fake_client: FakeOllamaClient, bad_temp: Any) -> None:
    with pytest.raises(LLMInputError):
        llm.generate("Valid prompt", temperature=bad_temp)  # type: ignore[arg-type]
    assert len(fake_client.calls) == 0


def test_input_error_inheritance() -> None:
    assert issubclass(LLMInputError, LLMError)
    assert issubclass(LLMInputError, ValueError)


# --------------------------------------------------------------------------- #
# 4. Response Extraction
# --------------------------------------------------------------------------- #
def test_response_extracted_from_object_attribute() -> None:
    client = FakeOllamaClient()
    client.generate = lambda **kw: FakeOllamaResponse("Response from object attribute")  # type: ignore[assignment]
    llm = LLM(client=client)
    assert llm.generate("hello") == "Response from object attribute"


def test_response_extracted_from_plain_string() -> None:
    client = FakeOllamaClient()
    client.generate = lambda **kw: "Direct plain string response"  # type: ignore[assignment]
    llm = LLM(client=client)
    assert llm.generate("hello") == "Direct plain string response"


def test_response_extracted_from_mapping() -> None:
    client = FakeOllamaClient()
    client.generate = lambda **kw: {"response": "Response from mapping"}  # type: ignore[assignment]
    llm = LLM(client=client)
    assert llm.generate("hello") == "Response from mapping"


def test_empty_response_text_handled_cleanly() -> None:
    client = FakeOllamaClient(response_text="")
    llm = LLM(client=client)
    assert llm.generate("hello") == ""


# --------------------------------------------------------------------------- #
# 5. Error Handling & Diagnostics
# --------------------------------------------------------------------------- #
def test_connection_error_wrapped_in_llm_connection_error() -> None:
    client = FakeOllamaClient(raise_on_generate=ConnectionError("Connection refused by peer"))
    llm = LLM(client=client)

    with pytest.raises(LLMConnectionError) as exc_info:
        llm.generate("hello")

    assert "Could not connect to Ollama server" in str(exc_info.value)
    assert issubclass(LLMConnectionError, LLMError)
    assert issubclass(LLMConnectionError, ConnectionError)


def test_ollama_response_error_wrapped_in_llm_response_error() -> None:
    import ollama

    client = FakeOllamaClient(
        raise_on_generate=ollama.ResponseError("model 'llama3' not found", status_code=404)
    )
    llm = LLM(client=client)

    with pytest.raises(LLMResponseError) as exc_info:
        llm.generate("hello")

    assert "model 'llama3' not found" in str(exc_info.value)
    assert issubclass(LLMResponseError, LLMError)


def test_unexpected_runtime_error_wrapped_in_llm_error() -> None:
    client = FakeOllamaClient(raise_on_generate=RuntimeError("Unexpected client crash"))
    llm = LLM(client=client)

    with pytest.raises(LLMError) as exc_info:
        llm.generate("hello")

    assert "LLM generation failed" in str(exc_info.value)
    assert "Unexpected client crash" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# 6. Availability Check
# --------------------------------------------------------------------------- #
def test_is_available_returns_true_when_server_reachable() -> None:
    client = FakeOllamaClient(available=True)
    llm = LLM(client=client)
    assert llm.is_available() is True


def test_is_available_returns_false_when_server_unreachable() -> None:
    client = FakeOllamaClient(available=False)
    llm = LLM(client=client)
    assert llm.is_available() is False


# --------------------------------------------------------------------------- #
# 7. Architecture & Security (No Code Execution)
# --------------------------------------------------------------------------- #
def test_module_import_does_not_contact_ollama() -> None:
    """Importing src.llm in an isolated process must succeed without Ollama."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; import src.llm; sys.exit(0)",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"Import failed with stderr: {proc.stderr}"


def test_generated_content_never_executed() -> None:
    """Malicious Python payload in LLM response must be returned purely as text."""
    evil_code = (
        "import sys, os\n"
        "os.environ['PWNED_TEST_FLAG'] = 'EXPLOITED'\n"
        "raise SystemExit('Code was executed!')\n"
    )
    client = FakeOllamaClient(response_text=evil_code)
    llm = LLM(client=client)

    result = llm.generate("Write Python code to do something.")
    # The return value must be the raw string and nothing was executed
    assert result == evil_code
    import os

    assert "PWNED_TEST_FLAG" not in os.environ


def test_source_code_contains_no_code_execution_primitives() -> None:
    """src/llm.py must not contain dangerous code execution primitives or imports."""
    import ast

    src = (ROOT / "src" / "llm.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Check that no dangerous functions are called
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "exec" not in called_names
    assert "eval" not in called_names

    # Check that no dangerous modules are imported
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert "subprocess" not in imported_modules
    assert "os.system" not in src


# --------------------------------------------------------------------------- #
# 8. Optional Real Ollama Integration Test (Skipped When Server Offline)
# --------------------------------------------------------------------------- #
def test_real_ollama_integration_if_running() -> None:
    """Optional live Ollama test: executed only if a local server is responsive."""
    llm = LLM()
    if not llm.is_available():
        pytest.skip("Local Ollama server is not running or model unavailable; skipping live test.")

    try:
        reply = llm.generate("Respond with only the word OK.")
        assert isinstance(reply, str)
        assert len(reply.strip()) > 0
    except LLMError as exc:
        pytest.skip(f"Live Ollama server failed to generate: {exc}")

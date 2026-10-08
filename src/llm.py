"""Ollama LLM integration layer for CodeSage (Phase 1G).

Provides a clean interface between CodeSage and a local Ollama model instance:

    Caller (e.g. RAG pipeline) -> LLM.generate(prompt) -> Ollama API

Design notes
------------
* Local Ollama only: no external or cloud LLM APIs are used.
* Ollama is contacted strictly on-demand when :meth:`LLM.generate` is invoked.
  Importing this module never makes network requests or contacts Ollama.
* Dependency injection: an existing client or fake client can be passed into
  :class:`LLM` for testing and offline execution.
* The module strictly handles text generation and never executes generated
  code or runs external shell commands.
* Invalid prompts (empty, whitespace-only, non-string) raise
  :class:`LLMInputError` before contacting Ollama.
* Network and API errors are mapped to meaningful :class:`LLMError` subclasses
  while preserving diagnostic error context.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import ollama

from config import Config, settings
from src.utils import get_logger

logger = get_logger("llm")

DEFAULT_TEMPERATURE: float = 0.2


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Base class for all LLM integration errors in CodeSage."""


class LLMInputError(LLMError, ValueError):
    """Raised when an invalid prompt or parameter is passed to the LLM."""


class LLMConnectionError(LLMError, ConnectionError):
    """Raised when the local Ollama server is unreachable or offline."""


class LLMResponseError(LLMError):
    """Raised when Ollama returns an API or model error."""


# --------------------------------------------------------------------------- #
# LLM Interface
# --------------------------------------------------------------------------- #
class LLM:
    """Interface to a local Ollama model for text generation.

    Args:
        model: Ollama model name (e.g. 'llama3'). Defaults to the configured
            model from :data:`config.settings.ollama_model`.
        host: Ollama server base URL (e.g. 'http://localhost:11434'). Defaults
            to the configured host from :data:`config.settings.ollama_host`.
        client: Optional pre-configured Ollama client or fake client for testing.
            If None, an :class:`ollama.Client` is created with ``host``.
        config: Optional configuration instance overriding global settings.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        client: Optional[Any] = None,
        config: Optional[Config] = None,
    ) -> None:
        cfg = config or settings
        self.model: str = str(model) if model is not None else cfg.ollama_model
        self.host: str = str(host) if host is not None else cfg.ollama_host
        self._client: Optional[Any] = client

    @property
    def client(self) -> Any:
        """Return the underlying Ollama client, creating one on first access."""
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        self._client = value

    def generate(
        self,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Generate a text response for the given prompt using the local Ollama model.

        Args:
            prompt: Non-empty prompt text.
            temperature: Sampling temperature (>= 0.0). Defaults to 0.2.

        Returns:
            The generated response as a plain Python string.

        Raises:
            LLMInputError: If ``prompt`` is not a string, is empty, or consists
                only of whitespace; or if ``temperature`` is invalid.
            LLMConnectionError: If the Ollama server is unreachable or offline.
            LLMResponseError: If Ollama returns an error response.
            LLMError: If any other client-level failure occurs.
        """
        self._validate_prompt(prompt)
        self._validate_temperature(temperature)

        options: dict[str, Any] = {"temperature": float(temperature)}

        try:
            try:
                raw_response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options=options,
                )
            except TypeError as type_err:
                # Support simple fake/mock clients in tests that only accept (model, prompt)
                if "unexpected keyword argument" in str(type_err) or "options" in str(type_err):
                    raw_response = self.client.generate(
                        model=self.model,
                        prompt=prompt,
                    )
                else:
                    raise
        except LLMError:
            # Re-raise already specialized LLM errors directly
            raise
        except (ConnectionError, TimeoutError, OSError, ollama.RequestError) as exc:
            logger.error("Failed to connect to Ollama at %s: %s", self.host, exc)
            raise LLMConnectionError(
                f"Could not connect to Ollama server at '{self.host}': {exc}"
            ) from exc
        except ollama.ResponseError as exc:
            logger.error("Ollama API error for model '%s': %s", self.model, exc)
            raise LLMResponseError(
                f"Ollama server error for model '{self.model}': {exc}"
            ) from exc
        except Exception as exc:
            # Check if exc represents an ollama connection or response failure from duck-typed mocks
            err_name = type(exc).__name__
            if "Connection" in err_name or "RequestError" in err_name:
                raise LLMConnectionError(
                    f"Could not connect to Ollama server at '{self.host}': {exc}"
                ) from exc
            if "ResponseError" in err_name:
                raise LLMResponseError(
                    f"Ollama server error for model '{self.model}': {exc}"
                ) from exc
            # Preserve unexpected client errors under LLMError
            logger.error("Unexpected error during LLM generation: %s", exc)
            raise LLMError(f"LLM generation failed: {exc}") from exc

        return self._extract_text(raw_response)

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable and responsive."""
        try:
            if hasattr(self.client, "list"):
                self.client.list()
            return True
        except Exception:
            return False

    # ----- internals ------------------------------------------------------- #
    @staticmethod
    def _validate_prompt(prompt: Any) -> None:
        if not isinstance(prompt, str):
            raise LLMInputError(
                f"Prompt must be a string, got {type(prompt).__name__}."
            )
        if not prompt.strip():
            raise LLMInputError("Prompt must not be empty or whitespace only.")

    @staticmethod
    def _validate_temperature(temperature: Any) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise LLMInputError(
                f"Temperature must be a float or integer, got {type(temperature).__name__}."
            )
        if temperature < 0.0:
            raise LLMInputError(
                f"Temperature must be non-negative, got {temperature}."
            )

    @staticmethod
    def _extract_text(raw_response: Any) -> str:
        """Extract string text from various Ollama response types safely."""
        if isinstance(raw_response, str):
            return raw_response
        if isinstance(raw_response, Mapping):
            return str(raw_response.get("response", ""))
        if hasattr(raw_response, "response"):
            val = getattr(raw_response, "response")
            return "" if val is None else str(val)
        if hasattr(raw_response, "__getitem__"):
            try:
                return str(raw_response["response"])
            except Exception:
                pass
        return str(raw_response)


# Backwards-compatible alias for the earlier placeholder name
OllamaLLM = LLM

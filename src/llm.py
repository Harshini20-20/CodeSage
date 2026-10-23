"""Ollama LLM integration layer for CodeSage.

Supports Ollama Cloud when OLLAMA_API_KEY is configured,
and local Ollama otherwise.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional

import ollama
import streamlit as st

from config import Config, settings
from src.utils import get_logger

logger = get_logger("llm")

DEFAULT_TEMPERATURE: float = 0.2
OLLAMA_CLOUD_HOST = "https://ollama.com"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Base class for all LLM integration errors in CodeSage."""


class LLMInputError(LLMError, ValueError):
    """Raised when an invalid prompt or parameter is passed to the LLM."""


class LLMConnectionError(LLMError, ConnectionError):
    """Raised when the Ollama server is unreachable or offline."""


class LLMResponseError(LLMError):
    """Raised when Ollama returns an API or model error."""


# --------------------------------------------------------------------------- #
# LLM Interface
# --------------------------------------------------------------------------- #
class LLM:
    """Interface to local Ollama or Ollama Cloud."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        client: Optional[Any] = None,
        config: Optional[Config] = None,
    ) -> None:
        cfg = config or settings

        # Read the API key from environment variables or Streamlit Secrets.
        api_key = os.getenv("OLLAMA_API_KEY", "")
        try:
            api_key = st.secrets.get("OLLAMA_API_KEY", api_key)
        except Exception:
            pass

        self.api_key = str(api_key).strip()

        # Prefer an explicitly supplied host. Otherwise use Cloud if keyed.
        if host is not None:
            self.host = str(host)
        elif self.api_key:
            self.host = OLLAMA_CLOUD_HOST
        else:
            self.host = cfg.ollama_host

        # An optional secret can override the configured model.
        cloud_model = cfg.ollama_model
        try:
            cloud_model = st.secrets.get(
                "CODESAGE_OLLAMA_MODEL", cloud_model
            )
        except Exception:
            pass

        self.model = str(model) if model is not None else str(cloud_model)
        self._client: Optional[Any] = client

    @property
    def client(self) -> Any:
        """Create the Ollama client on first access."""
        if self._client is None:
            if self.api_key and self.host == OLLAMA_CLOUD_HOST:
                self._client = ollama.Client(
                    host=self.host,
                    headers={
                        "Authorization": f"Bearer {self.api_key}"
                    },
                )
            else:
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
        """Generate a text response from the configured Ollama model."""
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
                # Support simple fake/mock clients used in tests.
                if (
                    "unexpected keyword argument" in str(type_err)
                    or "options" in str(type_err)
                ):
                    raw_response = self.client.generate(
                        model=self.model,
                        prompt=prompt,
                    )
                else:
                    raise

        except LLMError:
            raise
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            ollama.RequestError,
        ) as exc:
            logger.error(
                "Failed to connect to Ollama at %s: %s",
                self.host,
                exc,
            )
            raise LLMConnectionError(
                f"Could not connect to Ollama server at "
                f"'{self.host}': {exc}"
            ) from exc
        except ollama.ResponseError as exc:
            logger.error(
                "Ollama API error for model '%s': %s",
                self.model,
                exc,
            )
            raise LLMResponseError(
                f"Ollama server error for model "
                f"'{self.model}': {exc}"
            ) from exc
        except Exception as exc:
            err_name = type(exc).__name__

            if "Connection" in err_name or "RequestError" in err_name:
                raise LLMConnectionError(
                    f"Could not connect to Ollama server at "
                    f"'{self.host}': {exc}"
                ) from exc

            if "ResponseError" in err_name:
                raise LLMResponseError(
                    f"Ollama server error for model "
                    f"'{self.model}': {exc}"
                ) from exc

            logger.error(
                "Unexpected error during LLM generation: %s",
                exc,
            )
            raise LLMError(f"LLM generation failed: {exc}") from exc

        return self._extract_text(raw_response)

    def is_available(self) -> bool:
        """Check whether the configured Ollama endpoint responds."""
        try:
            if hasattr(self.client, "list"):
                self.client.list()
            return True
        except Exception:
            return False

    # ----- Internal validation and response helpers ----------------------- #
    @staticmethod
    def _validate_prompt(prompt: Any) -> None:
        if not isinstance(prompt, str):
            raise LLMInputError(
                f"Prompt must be a string, got {type(prompt).__name__}."
            )

        if not prompt.strip():
            raise LLMInputError(
                "Prompt must not be empty or whitespace only."
            )

    @staticmethod
    def _validate_temperature(temperature: Any) -> None:
        if isinstance(temperature, bool) or not isinstance(
            temperature, (int, float)
        ):
            raise LLMInputError(
                "Temperature must be a float or integer, "
                f"got {type(temperature).__name__}."
            )

        if temperature < 0.0:
            raise LLMInputError(
                f"Temperature must be non-negative, got {temperature}."
            )

    @staticmethod
    def _extract_text(raw_response: Any) -> str:
        """Extract text from supported Ollama response types."""
        if isinstance(raw_response, str):
            return raw_response

        if isinstance(raw_response, Mapping):
            return str(raw_response.get("response", ""))

        if hasattr(raw_response, "response"):
            value = getattr(raw_response, "response")
            return "" if value is None else str(value)

        if hasattr(raw_response, "__getitem__"):
            try:
                return str(raw_response["response"])
            except Exception:
                pass

        return str(raw_response)


# Backwards-compatible alias
OllamaLLM = LLM

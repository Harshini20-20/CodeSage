"""Ollama LLM interface (placeholder - not implemented yet)."""

from __future__ import annotations

from config import settings


class OllamaLLM:
    """Placeholder interface for a local Ollama model."""

    def __init__(self, host: str = settings.ollama_host, model: str = settings.ollama_model) -> None:
        self.host = host
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate an explanation for the given prompt."""
        raise NotImplementedError("Ollama integration is not implemented yet.")

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        raise NotImplementedError("Ollama integration is not implemented yet.")

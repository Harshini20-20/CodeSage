"""Centralised configuration for CodeSage.

Values come from environment variables (optionally loaded from a ``.env``
file) with sensible defaults. All paths are relative to the project root and
built with :mod:`pathlib`, so they work on Windows and Linux.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR: Path = Path(__file__).resolve().parent


def _resolve_path(value: str) -> Path:
    """Resolve a path from env; relative paths are anchored at BASE_DIR."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


@dataclass(frozen=True)
class Config:
    """Immutable application settings."""

    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "CODESAGE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    ollama_host: str = field(
        default_factory=lambda: os.getenv(
            "CODESAGE_OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv(
            "CODESAGE_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        )
    )
    chroma_path: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("CODESAGE_CHROMA_PATH", "chroma_db"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("CODESAGE_LOG_LEVEL", "INFO").upper()
    )
    base_dir: Path = BASE_DIR
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    parsed_dir: Path = BASE_DIR / "data" / "parsed"
    chunks_dir: Path = BASE_DIR / "data" / "chunks"
    logs_dir: Path = BASE_DIR / "logs"

    @property
    def runtime_dirs(self) -> tuple[Path, ...]:
        """Directories the application needs at runtime."""
        return (
            self.uploads_dir,
            self.parsed_dir,
            self.chunks_dir,
            self.chroma_path,
            self.logs_dir,
        )


def get_config() -> Config:
    """Return a fresh :class:`Config` reflecting the current environment."""
    return Config()


settings: Config = get_config()

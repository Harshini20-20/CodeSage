"""Common utilities and logging setup for CodeSage."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import Config, settings

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def ensure_directories(config: Config = settings) -> list[Path]:
    """Create all runtime directories if missing and return them."""
    for directory in config.runtime_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return list(config.runtime_dirs)


def setup_logging(config: Config = settings) -> logging.Logger:
    """Configure root 'codesage' logger (console + rotating file). Idempotent."""
    logger = logging.getLogger("codesage")
    logger.setLevel(getattr(logging, config.log_level, logging.INFO))
    if logger.handlers:
        return logger
    formatter = logging.Formatter(_LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.logs_dir / "codesage.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:  # pragma: no cover - read-only filesystems
        logger.warning("File logging disabled: cannot write to %s", config.logs_dir)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'codesage' namespace."""
    return logging.getLogger(f"codesage.{name}")


def safe_filename(name: str) -> str:
    """Strip directory components and unsafe characters from an uploaded filename."""
    base = Path(name.replace("\\", "/")).name
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)
    return cleaned or "unnamed_file"


def save_uploaded_bytes(filename: str, data: bytes, config: Config = settings) -> Path:
    """Store uploaded bytes under data/uploads as inert data. Never executes the file."""
    config.uploads_dir.mkdir(parents=True, exist_ok=True)
    target = config.uploads_dir / safe_filename(filename)
    target.write_bytes(data)
    return target

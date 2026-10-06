"""ChromaDB vector store interface (placeholder - not implemented yet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import settings
from src.chunker import CodeChunk


class VectorStore:
    """Placeholder interface for ChromaDB storage."""

    def __init__(self, path: Path = settings.chroma_path, collection: str = "codesage") -> None:
        self.path = path
        self.collection = collection

    def add_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings."""
        raise NotImplementedError("Vector storage is not implemented yet.")

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top_k most similar stored chunks."""
        raise NotImplementedError("Vector storage is not implemented yet.")

    def reset(self) -> None:
        """Remove all stored data."""
        raise NotImplementedError("Vector storage is not implemented yet.")

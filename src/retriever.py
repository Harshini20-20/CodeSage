"""Retrieval interface (placeholder - not implemented yet)."""

from __future__ import annotations

from typing import Any

from src.embedder import Embedder
from src.vectordb import VectorStore


class Retriever:
    """Placeholder interface for retrieving relevant code chunks."""

    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 5) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return chunks relevant to the query."""
        raise NotImplementedError("Retrieval is not implemented yet.")

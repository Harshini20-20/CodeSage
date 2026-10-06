"""RAG pipeline orchestration interface (placeholder - not implemented yet).

Official workflow: Upload -> Parse & Chunk -> Embed -> Store -> Query ->
Retrieve -> LLM Explanation -> Display.
"""

from __future__ import annotations

from pathlib import Path


class RAGPipeline:
    """Placeholder for connecting all pipeline stages."""

    def ingest(self, paths: list[Path]) -> int:
        """Parse, chunk, embed and store the given files; return chunk count."""
        raise NotImplementedError("The RAG pipeline is not implemented yet.")

    def ask(self, question: str) -> str:
        """Retrieve relevant chunks and return an LLM explanation."""
        raise NotImplementedError("The RAG pipeline is not implemented yet.")

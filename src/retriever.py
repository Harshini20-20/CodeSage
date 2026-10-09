"""Retrieval engine for CodeSage (Phase 1F).

Connects the embedding layer and the vector store::

    Retriever -> Embedder.embed_query() -> VectorStore.query() -> RetrievalResult

Design notes
------------
* The Retriever never touches ChromaDB; it only talks to the ``Embedder`` and
  ``VectorStore`` objects it is given (dependency injection), so tests can pass
  fakes. The real classes are imported for type checking only, which keeps this
  module free of any ChromaDB / PyTorch import.
* Ordering and distance are exactly what the vector store returned. There is no
  re-ranking and no invented similarity score.
* Retrieved code is only ever treated as text; nothing is executed.
* Only invalid *user input* raises :class:`RetrievalInputError`. Failures from
  the embedder or vector store propagate unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing chromadb
    from src.embedder import Embedder
    from src.vectordb import VectorStore

DEFAULT_TOP_K: int = 5


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class RetrievalError(Exception):
    """Base class for retrieval errors."""


class RetrievalInputError(RetrievalError, ValueError):
    """The query or ``top_k`` passed to the retriever is invalid."""


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved code chunk.

    ``distance`` is the raw vector-store distance (cosine distance for the
    default collection): lower means closer. ``metadata`` is a copy of the full
    metadata dict stored with the chunk. Fields missing from the metadata fall
    back to ``""`` (text) or ``None`` (line numbers).
    """

    chunk_id: str
    content: str
    file_path: str = ""
    chunk_type: str = ""
    name: str = ""
    qualified_name: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    distance: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """Turns a natural-language query into ranked :class:`RetrievalResult` objects.

    Args:
        embedder: Object with ``embed_query(query) -> list[float]``.
        vector_store: Object with ``query(embedding, top_k) -> list[dict]``
            where each dict has ``id``, ``document``, ``metadata``, ``distance``.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        repository_graph: Any = None,
        graph_chunk_lookup: Optional[dict[str, RetrievalResult]] = None,
        graph_expansion_limit: int = 3,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.repository_graph = repository_graph
        self.graph_chunk_lookup = graph_chunk_lookup
        self.graph_expansion_limit = self._validate_graph_expansion_limit(graph_expansion_limit)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Return up to ``top_k`` chunks most relevant to ``query``.

        Results keep the exact order returned by the vector store. An empty
        store yields ``[]``.

        Raises:
            RetrievalInputError: ``query`` is not a non-blank string, or
                ``top_k`` is not a positive integer.
        """
        self._validate_query(query)
        self._validate_top_k(top_k)

        embedding = self.embedder.embed_query(query)
        raw_results = self.vector_store.query(embedding, top_k=top_k)
        results = [self._to_result(raw) for raw in raw_results]
        if self.repository_graph is not None and self.graph_chunk_lookup:
            from src.graph_retrieval import expand_with_graph

            return expand_with_graph(
                results,
                self.repository_graph,
                self.graph_chunk_lookup,
                limit=self.graph_expansion_limit,
            )
        return results

    # ----- internals ------------------------------------------------------- #
    @staticmethod
    def _validate_graph_expansion_limit(limit: Any) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("graph_expansion_limit must be a non-negative integer.")
        return limit

    @staticmethod
    def _validate_query(query: Any) -> None:
        if not isinstance(query, str):
            raise RetrievalInputError(f"query must be a string, got {type(query).__name__}.")
        if not query.strip():
            raise RetrievalInputError("query must not be empty or whitespace only.")

    @staticmethod
    def _validate_top_k(top_k: Any) -> None:
        # bool is a subclass of int; True/False are not meaningful counts.
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise RetrievalInputError(f"top_k must be an integer, got {type(top_k).__name__}.")
        if top_k <= 0:
            raise RetrievalInputError(f"top_k must be greater than zero, got {top_k}.")

    @staticmethod
    def _to_result(raw: dict[str, Any]) -> RetrievalResult:
        """Convert one raw vector-store record into a RetrievalResult."""
        metadata = dict(raw.get("metadata") or {})
        return RetrievalResult(
            chunk_id=raw.get("id", ""),
            content=raw.get("document") or "",
            file_path=str(metadata.get("file_path", "")),
            chunk_type=str(metadata.get("chunk_type", "")),
            name=str(metadata.get("name", "")),
            qualified_name=str(metadata.get("qualified_name", "")),
            start_line=metadata.get("start_line"),
            end_line=metadata.get("end_line"),
            distance=raw.get("distance"),
            metadata=metadata,
        )

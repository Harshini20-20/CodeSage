"""Tests for the Retrieval Engine (Phase 1F).

Everything uses deterministic fakes: no embedding model, no ChromaDB, no internet.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.retriever import (
    DEFAULT_TOP_K,
    RetrievalError,
    RetrievalInputError,
    RetrievalResult,
    Retriever,
)

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return self.vector


class FakeVectorStore:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = records or []
        self.calls: list[tuple[list[float], int]] = []

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        self.calls.append((embedding, top_k))
        return self.records


def make_record(
    chunk_id: str = "c1",
    distance: float = 0.1,
    name: str = "add",
    file_path: str = "src/math_utils.py",
    start_line: int = 10,
    end_line: int = 12,
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "document": f"def {name}(a, b):\n    return a + b",
        "metadata": {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "language": "python",
            "chunk_type": "function",
            "name": name,
            "qualified_name": name,
            "start_line": start_line,
            "end_line": end_line,
        },
        "distance": distance,
    }


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> FakeVectorStore:
    return FakeVectorStore([make_record()])


@pytest.fixture
def retriever(embedder: FakeEmbedder, store: FakeVectorStore) -> Retriever:
    return Retriever(embedder=embedder, vector_store=store)


# --------------------------------------------------------------------------- #
# Construction and the query flow
# --------------------------------------------------------------------------- #
def test_construction_with_keyword_and_positional_args(embedder, store) -> None:
    by_keyword = Retriever(embedder=embedder, vector_store=store)
    by_position = Retriever(embedder, store)
    for r in (by_keyword, by_position):
        assert r.embedder is embedder
        assert r.vector_store is store


def test_valid_query_returns_results(retriever) -> None:
    results = retriever.retrieve("how do I add numbers?")
    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)


def test_embed_query_is_called_with_the_query(retriever, embedder) -> None:
    retriever.retrieve("how do I add numbers?")
    assert embedder.queries == ["how do I add numbers?"]


def test_vector_store_query_receives_the_embedding(retriever, embedder, store) -> None:
    retriever.retrieve("anything")
    assert len(store.calls) == 1
    assert store.calls[0][0] == embedder.vector


def test_default_top_k_is_five(retriever, store) -> None:
    assert DEFAULT_TOP_K == 5
    retriever.retrieve("anything")
    assert store.calls[0][1] == 5


@pytest.mark.parametrize("top_k", [1, 3, 50])
def test_requested_top_k_is_passed_through(retriever, store, top_k) -> None:
    retriever.retrieve("anything", top_k=top_k)
    assert store.calls[0][1] == top_k


# --------------------------------------------------------------------------- #
# Result conversion
# --------------------------------------------------------------------------- #
def test_result_conversion(retriever) -> None:
    result = retriever.retrieve("anything")[0]
    assert result.chunk_id == "c1"
    assert result.content == "def add(a, b):\n    return a + b"
    assert result.chunk_type == "function"
    assert result.name == "add"
    assert result.qualified_name == "add"


def test_multiple_results_and_order_preserved(embedder) -> None:
    # Deliberately not sorted by id or distance: order must be left untouched.
    records = [
        make_record("z", distance=0.30, name="zeta"),
        make_record("a", distance=0.05, name="alpha"),
        make_record("m", distance=0.90, name="mu"),
    ]
    results = Retriever(embedder, FakeVectorStore(records)).retrieve("q", top_k=3)
    assert len(results) == 3
    assert [r.chunk_id for r in results] == ["z", "a", "m"]
    assert [r.name for r in results] == ["zeta", "alpha", "mu"]


def test_metadata_is_preserved(retriever, store) -> None:
    result = retriever.retrieve("anything")[0]
    assert result.metadata == store.records[0]["metadata"]
    assert result.metadata["language"] == "python"
    assert result.metadata["file_name"] == "math_utils.py"


def test_metadata_is_a_copy(retriever, store) -> None:
    result = retriever.retrieve("anything")[0]
    result.metadata["extra"] = 1
    assert "extra" not in store.records[0]["metadata"]


def test_file_path_is_preserved(retriever) -> None:
    assert retriever.retrieve("anything")[0].file_path == "src/math_utils.py"


def test_line_numbers_are_preserved(retriever) -> None:
    result = retriever.retrieve("anything")[0]
    assert result.start_line == 10
    assert result.end_line == 12


def test_distance_is_preserved_exactly(embedder) -> None:
    store = FakeVectorStore([make_record(distance=0.123456789)])
    result = Retriever(embedder, store).retrieve("q")[0]
    assert result.distance == 0.123456789


def test_missing_metadata_is_handled_safely(embedder) -> None:
    raw = [{"id": "x", "document": "code", "metadata": None, "distance": 0.2}]
    result = Retriever(embedder, FakeVectorStore(raw)).retrieve("q")[0]
    assert result.chunk_id == "x"
    assert result.content == "code"
    assert result.file_path == ""
    assert result.start_line is None
    assert result.end_line is None
    assert result.metadata == {}


def test_partial_metadata_is_handled_safely(embedder) -> None:
    raw = [{"id": "x", "document": "code", "metadata": {"file_path": "a.py"}, "distance": 0.2}]
    result = Retriever(embedder, FakeVectorStore(raw)).retrieve("q")[0]
    assert result.file_path == "a.py"
    assert result.name == ""
    assert result.qualified_name == ""
    assert result.chunk_type == ""


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query", ["", "   ", "\n\t", " \n "])
def test_blank_query_raises(retriever, query) -> None:
    with pytest.raises(RetrievalInputError):
        retriever.retrieve(query)


@pytest.mark.parametrize("query", [None, 123, ["a"], b"bytes"])
def test_non_string_query_raises(retriever, query) -> None:
    with pytest.raises(RetrievalInputError):
        retriever.retrieve(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, -100, 2.5, "5", None, True])
def test_invalid_top_k_raises(retriever, top_k) -> None:
    with pytest.raises(RetrievalInputError):
        retriever.retrieve("valid query", top_k=top_k)  # type: ignore[arg-type]


def test_input_error_is_a_value_error_and_retrieval_error() -> None:
    assert issubclass(RetrievalInputError, ValueError)
    assert issubclass(RetrievalInputError, RetrievalError)


def test_embedder_and_store_not_called_for_invalid_query(retriever, embedder, store) -> None:
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(RetrievalInputError):
            retriever.retrieve(bad)
    assert embedder.queries == []
    assert store.calls == []


def test_embedder_and_store_not_called_for_invalid_top_k(retriever, embedder, store) -> None:
    with pytest.raises(RetrievalInputError):
        retriever.retrieve("valid query", top_k=0)
    assert embedder.queries == []
    assert store.calls == []


# --------------------------------------------------------------------------- #
# Empty database and failure propagation
# --------------------------------------------------------------------------- #
def test_empty_vector_store_returns_empty_list(embedder) -> None:
    assert Retriever(embedder, FakeVectorStore([])).retrieve("anything") == []


def test_embedder_failure_is_not_swallowed(store) -> None:
    class BrokenEmbedder:
        def embed_query(self, query: str) -> list[float]:
            raise RuntimeError("embedder exploded")

    with pytest.raises(RuntimeError, match="embedder exploded"):
        Retriever(BrokenEmbedder(), store).retrieve("q")  # type: ignore[arg-type]


def test_vector_store_failure_is_not_swallowed(embedder) -> None:
    class BrokenStore:
        def query(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
            raise RuntimeError("store exploded")

    with pytest.raises(RuntimeError, match="store exploded"):
        Retriever(embedder, BrokenStore()).retrieve("q")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def test_retriever_does_not_depend_on_chromadb() -> None:
    source = (ROOT / "src" / "retriever.py").read_text(encoding="utf-8")
    assert "import chromadb" not in source
    assert "from chromadb" not in source

    # Importing the module in a fresh interpreter must not pull in chromadb.
    code = "import sys, src.retriever; sys.exit(1 if 'chromadb' in sys.modules else 0)"
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert proc.returncode == 0


def test_retriever_never_executes_retrieved_code(embedder) -> None:
    payload = "raise SystemExit('executed!')"
    raw = [{"id": "x", "document": payload, "metadata": {}, "distance": 0.0}]
    result = Retriever(embedder, FakeVectorStore(raw)).retrieve("q")[0]
    assert result.content == payload  # returned as plain text only

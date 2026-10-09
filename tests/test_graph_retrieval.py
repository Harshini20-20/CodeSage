"""Deterministic tests for graph-assisted retrieval (Phase 2C)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.chunker import CodeChunker
from src.graph_retrieval import build_graph_chunk_lookup, expand_with_graph
from src.parser import CodeParser
from src.repository_graph import build_repository_graph
from src.retriever import RetrievalResult, Retriever


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        return self.rows[:top_k]


def _context(tmp_path: Path):
    (tmp_path / "helpers.py").write_text(
        "def validate_order(order):\n    return bool(order)\n", encoding="utf-8"
    )
    (tmp_path / "main.py").write_text(
        "from helpers import validate_order\n\ndef process_order(order):\n    return validate_order(order)\n",
        encoding="utf-8",
    )
    parsed = CodeParser().parse_directory(tmp_path)
    chunks = [chunk for item in parsed for chunk in CodeChunker().chunk(item)]
    graph = build_repository_graph(parsed, root=tmp_path)
    lookup = build_graph_chunk_lookup(graph, chunks)
    by_name = {item.qualified_name: item for item in chunks if item.chunk_type == "function"}
    return graph, lookup, by_name


def _retrieval_result(chunk) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        content=chunk.text,
        file_path=chunk.file_path,
        chunk_type=chunk.chunk_type,
        name=chunk.name,
        qualified_name=chunk.qualified_name,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        distance=0.12,
        metadata=dict(chunk.metadata),
    )


def _raw(chunk) -> dict[str, Any]:
    return {
        "id": chunk.chunk_id,
        "document": chunk.text,
        "metadata": dict(chunk.metadata),
        "distance": 0.12,
    }


def test_build_graph_chunk_lookup_maps_code_elements(tmp_path: Path) -> None:
    graph, lookup, _ = _context(tmp_path)
    assert lookup
    assert all(node_id in graph.nodes for node_id in lookup)
    assert {item.qualified_name for item in lookup.values()} >= {"process_order", "validate_order"}


def test_expansion_appends_resolved_call_target_after_vector_seed(tmp_path: Path) -> None:
    graph, lookup, chunks = _context(tmp_path)
    seed = _retrieval_result(chunks["process_order"])
    expanded = expand_with_graph([seed], graph, lookup, limit=3)
    assert [item.chunk_id for item in expanded][0] == seed.chunk_id
    assert any(item.qualified_name == "validate_order" for item in expanded[1:])
    related = next(item for item in expanded[1:] if item.qualified_name == "validate_order")
    assert related.metadata["graph_relation"] == "calls"
    assert related.distance is None


def test_retriever_uses_graph_context_when_configured(tmp_path: Path) -> None:
    graph, lookup, chunks = _context(tmp_path)
    seed = chunks["process_order"]
    retriever = Retriever(
        FakeEmbedder(),
        FakeVectorStore([_raw(seed)]),
        repository_graph=graph,
        graph_chunk_lookup=lookup,
    )
    results = retriever.retrieve("how is an order validated?", top_k=1)
    assert results[0].qualified_name == "process_order"
    assert any(item.qualified_name == "validate_order" for item in results[1:])


def test_vector_results_unchanged_without_graph_configuration(tmp_path: Path) -> None:
    _, _, chunks = _context(tmp_path)
    seed = chunks["process_order"]
    retriever = Retriever(FakeEmbedder(), FakeVectorStore([_raw(seed)]))
    results = retriever.retrieve("process order", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == seed.chunk_id
    assert "graph_relation" not in results[0].metadata


def test_expansion_respects_limit_and_deduplicates(tmp_path: Path) -> None:
    graph, lookup, chunks = _context(tmp_path)
    seed = _retrieval_result(chunks["process_order"])
    expanded = expand_with_graph([seed, seed], graph, lookup, limit=1)
    assert sum(item.chunk_id == seed.chunk_id for item in expanded) == 2
    assert len({item.chunk_id for item in expanded[2:]}) <= 1


def test_no_expansion_when_graph_context_is_missing(tmp_path: Path) -> None:
    _, _, chunks = _context(tmp_path)
    seed = _retrieval_result(chunks["process_order"])
    assert expand_with_graph([seed], None, None) == [seed]
    assert expand_with_graph([seed], None, {}) == [seed]


def test_graph_relation_is_exposed_in_rag_context(tmp_path: Path) -> None:
    from src.prompts import format_context_chunk

    graph, lookup, chunks = _context(tmp_path)
    seed = _retrieval_result(chunks["process_order"])
    expanded = expand_with_graph([seed], graph, lookup, limit=1)
    related = next(item for item in expanded if item.metadata.get("graph_relation"))
    formatted = format_context_chunk(related)
    assert "Related via repository graph: calls" in formatted

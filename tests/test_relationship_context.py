"""Tests for Phase 2D relationship-aware answer context."""
from __future__ import annotations

from pathlib import Path

from src.chunker import CodeChunker
from src.parser import CodeParser
from src.relationship_context import build_relationship_context
from src.repository_graph import build_repository_graph
from src.retriever import RetrievalResult


def _fixture(tmp_path: Path):
    (tmp_path / "helpers.py").write_text(
        "def validate_order(order):\n"
        "    return bool(order)\n\n"
        "def unused_helper():\n"
        "    return False\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        "from helpers import validate_order\n\n"
        "def process_order(order):\n"
        "    return validate_order(order)\n\n"
        "def external_step(value):\n"
        "    return not_known_here(value)\n",
        encoding="utf-8",
    )
    parsed = CodeParser().parse_directory(tmp_path)
    chunks = [chunk for parsed_file in parsed for chunk in CodeChunker().chunk(parsed_file)]
    graph = build_repository_graph(parsed, root=tmp_path)
    results = {
        chunk.qualified_name: RetrievalResult(
            chunk_id=chunk.chunk_id,
            content=chunk.text,
            file_path=chunk.file_path,
            chunk_type=chunk.chunk_type,
            name=chunk.name,
            qualified_name=chunk.qualified_name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            metadata=dict(chunk.metadata),
        )
        for chunk in chunks
    }
    return graph, results


def test_relationship_context_reports_confirmed_cross_file_call(tmp_path: Path) -> None:
    graph, results = _fixture(tmp_path)
    context = build_relationship_context("How is an order validated?", [results["process_order"]], graph)

    assert "CONFIRMED call" in context
    assert "process_order" in context
    assert "validate_order" in context
    assert "main.py" in context
    assert "helpers.py" in context


def test_relationship_context_labels_unresolved_calls(tmp_path: Path) -> None:
    graph, results = _fixture(tmp_path)
    context = build_relationship_context("What does this call?", [results["external_step"]], graph)

    assert "UNRESOLVED call" in context
    assert "not_known_here" in context
    assert "not resolved to a repository definition" in context


def test_relationship_context_does_not_include_unrelated_functions(tmp_path: Path) -> None:
    graph, results = _fixture(tmp_path)
    context = build_relationship_context("Explain process_order", [results["process_order"]], graph)

    assert "unused_helper" not in context


def test_relationship_context_handles_missing_graph_and_sources() -> None:
    assert "No repository relationship evidence" in build_relationship_context("q", [], None)


def test_relationship_context_missing_matches_is_explicit(tmp_path: Path) -> None:
    graph, _ = _fixture(tmp_path)
    fake = RetrievalResult(chunk_id="missing", content="x", file_path="not-in-repo.py")
    context = build_relationship_context("q", [fake], graph)
    assert "could be matched" in context or "No direct caller" in context


def test_pipeline_includes_graph_relationship_context_in_default_prompt(tmp_path: Path) -> None:
    from src.rag_pipeline import RAGPipeline

    graph, results = _fixture(tmp_path)
    source = results["process_order"]

    class FakeRetriever:
        repository_graph = graph

        def retrieve(self, query: str, top_k: int = 5):
            return [source]

    class FakeLLM:
        prompt = ""

        def generate(self, prompt: str, **kwargs) -> str:
            self.prompt = prompt
            return "It calls validate_order."

    llm = FakeLLM()
    result = RAGPipeline(retriever=FakeRetriever(), llm=llm).answer("How is an order validated?")

    assert result.answer == "It calls validate_order."
    assert "CONFIRMED call" in llm.prompt
    assert "process_order" in llm.prompt
    assert "validate_order" in llm.prompt


def test_relationship_context_finds_callers_when_callee_is_retrieved(tmp_path: Path) -> None:
    graph, results = _fixture(tmp_path)
    context = build_relationship_context("Who calls validate_order?", [results["validate_order"]], graph)

    assert "CONFIRMED call" in context
    assert "process_order" in context
    assert "validate_order" in context


def test_relationship_context_reports_confirmed_inheritance(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text("class Base:\n    pass\n", encoding="utf-8")
    (tmp_path / "child.py").write_text(
        "from base import Base\n\nclass Child(Base):\n    pass\n", encoding="utf-8"
    )
    parsed = CodeParser().parse_directory(tmp_path)
    chunks = [chunk for parsed_file in parsed for chunk in CodeChunker().chunk(parsed_file)]
    graph = build_repository_graph(parsed, root=tmp_path)
    child_chunk = next(chunk for chunk in chunks if chunk.qualified_name == "Child")
    source = RetrievalResult(
        chunk_id=child_chunk.chunk_id,
        content=child_chunk.text,
        file_path=child_chunk.file_path,
        chunk_type=child_chunk.chunk_type,
        name=child_chunk.name,
        qualified_name=child_chunk.qualified_name,
        start_line=child_chunk.start_line,
        end_line=child_chunk.end_line,
    )

    context = build_relationship_context("What does Child inherit from?", [source], graph)
    assert "CONFIRMED inheritance" in context
    assert "Child" in context
    assert "Base" in context

def test_prompt_explicitly_enforces_call_direction() -> None:
    from src.prompts import build_rag_prompt

    prompt = build_rag_prompt(
        question="Which functions call add, and which functions does main call?",
        context="main.py contains main().",
        relationship_context=(
            "CONFIRMED call: function `main` calls function `add`.\n"
            "CONFIRMED call: function `main` calls function `multiply`."
        ),
    )

    assert "TARGET is X" in prompt
    assert "SOURCE is X" in prompt
    assert "Never reverse a call edge" in prompt

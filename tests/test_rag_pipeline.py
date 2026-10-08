"""Tests for the RAG Pipeline (Phase 1H).

Deterministic testing with fakes/mocks only:
No Ollama daemon, no internet, no GPU, no real ChromaDB or embedding model required.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

import pytest

from src.prompts import (
    EXPLANATION_PROMPT_TEMPLATE,
    build_explanation_prompt,
    build_rag_prompt,
    format_context,
    format_context_chunk,
)
from src.rag_pipeline import (
    NO_CONTEXT_MESSAGE,
    RAGInputError,
    RAGPipeline,
    RAGPipelineError,
    RAGResult,
    RAGSource,
)
from src.retriever import DEFAULT_TOP_K, RetrievalResult

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def make_chunk_result(
    chunk_id: str = "chunk-101",
    content: str = "def calculate_discount(price: float) -> float:\n    return price * 0.1",
    file_path: str = "src/pricing.py",
    chunk_type: str = "function",
    name: str = "calculate_discount",
    qualified_name: str = "calculate_discount",
    start_line: int = 15,
    end_line: int = 17,
    distance: float = 0.08,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        file_path=file_path,
        chunk_type=chunk_type,
        name=name,
        qualified_name=qualified_name,
        start_line=start_line,
        end_line=end_line,
        distance=distance,
        metadata={"file_path": file_path, "name": name, "chunk_type": chunk_type},
    )


class FakeRetriever:
    """Deterministic fake retriever for pipeline tests."""

    def __init__(
        self,
        results: Optional[list[RetrievalResult]] = None,
        raise_on_retrieve: Optional[Exception] = None,
    ) -> None:
        self.results = results if results is not None else [make_chunk_result()]
        self.raise_on_retrieve = raise_on_retrieve
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        if self.raise_on_retrieve is not None:
            raise self.raise_on_retrieve
        return self.results


class FakeLLM:
    """Deterministic fake LLM for pipeline tests."""

    def __init__(
        self,
        answer: str = "The `calculate_discount` function takes a price and returns a 10% discount.",
        raise_on_generate: Optional[Exception] = None,
    ) -> None:
        self.answer = answer
        self.raise_on_generate = raise_on_generate
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return self.answer


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(fake_retriever: FakeRetriever, fake_llm: FakeLLM) -> RAGPipeline:
    return RAGPipeline(retriever=fake_retriever, llm=fake_llm)


# --------------------------------------------------------------------------- #
# 1. Pipeline Construction & Initialization
# --------------------------------------------------------------------------- #
def test_pipeline_construction_keyword_and_positional(fake_retriever, fake_llm) -> None:
    pipe1 = RAGPipeline(retriever=fake_retriever, llm=fake_llm)
    pipe2 = RAGPipeline(fake_retriever, fake_llm)
    assert pipe1.retriever is fake_retriever
    assert pipe1.llm is fake_llm
    assert pipe2.retriever is fake_retriever
    assert pipe2.llm is fake_llm


def test_pipeline_unconfigured_raises_error() -> None:
    empty_pipe = RAGPipeline()
    with pytest.raises(RAGPipelineError, match="requires both a retriever and an LLM"):
        empty_pipe.answer("how does discount work?")


# --------------------------------------------------------------------------- #
# 2. Query Flow: Retrieval -> Prompt -> LLM -> Result
# --------------------------------------------------------------------------- #
def test_valid_question_returns_rag_result(pipeline: RAGPipeline) -> None:
    result = pipeline.answer("How is discount calculated?")
    assert isinstance(result, RAGResult)
    assert result.question == "How is discount calculated?"
    assert isinstance(result.answer, str)
    assert "calculate_discount" in result.answer


def test_retriever_called_with_question_and_default_top_k(
    pipeline: RAGPipeline, fake_retriever: FakeRetriever
) -> None:
    q = "Where is the discount logic?"
    pipeline.answer(q)
    assert len(fake_retriever.calls) == 1
    assert fake_retriever.calls[0] == (q, DEFAULT_TOP_K)


def test_retriever_called_with_custom_top_k(
    pipeline: RAGPipeline, fake_retriever: FakeRetriever
) -> None:
    pipeline.answer("Any query", top_k=8)
    assert len(fake_retriever.calls) == 1
    assert fake_retriever.calls[0][1] == 8


def test_llm_called_with_built_prompt(
    pipeline: RAGPipeline, fake_llm: FakeLLM
) -> None:
    pipeline.answer("How is discount calculated?")
    assert len(fake_llm.calls) == 1
    prompt = fake_llm.calls[0]["prompt"]
    assert "How is discount calculated?" in prompt
    assert "calculate_discount" in prompt


def test_temperature_passed_to_llm_when_specified(
    pipeline: RAGPipeline, fake_llm: FakeLLM
) -> None:
    pipeline.answer("Explain the discount logic.", temperature=0.5)
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["kwargs"].get("temperature") == 0.5


# --------------------------------------------------------------------------- #
# 3. Source Metadata Preservation
# --------------------------------------------------------------------------- #
def test_source_information_preserved_in_result(
    pipeline: RAGPipeline, fake_retriever: FakeRetriever
) -> None:
    result = pipeline.answer("Any question")
    assert len(result.sources) == 1
    source = result.sources[0]

    assert source.chunk_id == "chunk-101"
    assert source.file_path == "src/pricing.py"
    assert source.chunk_type == "function"
    assert source.name == "calculate_discount"
    assert source.qualified_name == "calculate_discount"
    assert source.start_line == 15
    assert source.end_line == 17
    assert source.distance == 0.08
    assert "return price * 0.1" in source.content

    # Helper properties
    assert result.has_sources is True
    assert result.source_files == ["src/pricing.py"]


def test_multiple_sources_ordering_preserved(fake_llm: FakeLLM) -> None:
    chunk1 = make_chunk_result(chunk_id="c1", name="foo", file_path="foo.py")
    chunk2 = make_chunk_result(chunk_id="c2", name="bar", file_path="bar.py")
    retriever = FakeRetriever(results=[chunk1, chunk2])
    pipeline = RAGPipeline(retriever=retriever, llm=fake_llm)

    result = pipeline.answer("Question")
    assert len(result.sources) == 2
    assert [s.chunk_id for s in result.sources] == ["c1", "c2"]
    assert result.source_files == ["foo.py", "bar.py"]


# --------------------------------------------------------------------------- #
# 4. Input Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("empty_q", ["", "   ", "\t\n", " \n\r\t "])
def test_empty_question_raises_input_error(
    pipeline: RAGPipeline, fake_retriever: FakeRetriever, fake_llm: FakeLLM, empty_q: str
) -> None:
    with pytest.raises(RAGInputError) as exc_info:
        pipeline.answer(empty_q)
    assert "empty or whitespace" in str(exc_info.value)
    assert len(fake_retriever.calls) == 0
    assert len(fake_llm.calls) == 0


@pytest.mark.parametrize("non_string", [None, 123, 45.6, ["query"], {"q": "hi"}])
def test_non_string_question_raises_input_error(
    pipeline: RAGPipeline, fake_retriever: FakeRetriever, fake_llm: FakeLLM, non_string: Any
) -> None:
    with pytest.raises(RAGInputError) as exc_info:
        pipeline.answer(non_string)  # type: ignore[arg-type]
    assert "must be a string" in str(exc_info.value)
    assert len(fake_retriever.calls) == 0
    assert len(fake_llm.calls) == 0


@pytest.mark.parametrize("bad_top_k", [0, -1, -50, "5", 2.5, True, False, None])
def test_invalid_top_k_raises_input_error(
    pipeline: RAGPipeline, fake_retriever: FakeRetriever, fake_llm: FakeLLM, bad_top_k: Any
) -> None:
    with pytest.raises(RAGInputError):
        pipeline.answer("Valid question", top_k=bad_top_k)  # type: ignore[arg-type]
    assert len(fake_retriever.calls) == 0
    assert len(fake_llm.calls) == 0


def test_input_error_inheritance() -> None:
    assert issubclass(RAGInputError, RAGPipelineError)
    assert issubclass(RAGInputError, ValueError)


# --------------------------------------------------------------------------- #
# 5. Zero Retrieved Chunks Case
# --------------------------------------------------------------------------- #
def test_zero_retrieved_results_handled_cleanly(fake_llm: FakeLLM) -> None:
    empty_retriever = FakeRetriever(results=[])
    pipeline = RAGPipeline(retriever=empty_retriever, llm=fake_llm)

    result = pipeline.answer("Where is quantum_sort implemented?")
    assert isinstance(result, RAGResult)
    assert result.has_sources is False
    assert result.sources == []
    assert result.source_files == []
    assert result.answer == NO_CONTEXT_MESSAGE
    # No wasteful hallucination LLM call when zero context found
    assert len(fake_llm.calls) == 0


# --------------------------------------------------------------------------- #
# 6. Error Propagation / Diagnostics
# --------------------------------------------------------------------------- #
def test_retriever_failure_is_not_swallowed(fake_llm: FakeLLM) -> None:
    broken_retriever = FakeRetriever(raise_on_retrieve=RuntimeError("VectorDB down"))
    pipeline = RAGPipeline(retriever=broken_retriever, llm=fake_llm)

    with pytest.raises(RuntimeError, match="VectorDB down"):
        pipeline.answer("Any question")
    assert len(fake_llm.calls) == 0


def test_llm_failure_is_not_swallowed(fake_retriever: FakeRetriever) -> None:
    broken_llm = FakeLLM(raise_on_generate=ConnectionError("Ollama offline"))
    pipeline = RAGPipeline(retriever=fake_retriever, llm=broken_llm)

    with pytest.raises(ConnectionError, match="Ollama offline"):
        pipeline.answer("Any question")


# --------------------------------------------------------------------------- #
# 7. Prompt Construction & Formatting (src/prompts.py)
# --------------------------------------------------------------------------- #
def test_prompt_contains_user_question_and_context() -> None:
    chunk = make_chunk_result(
        content="def authenticate(token):\n    return token == 'secret'",
        file_path="src/auth.py",
        start_line=10,
        end_line=12,
    )
    prompt = build_rag_prompt("How does authentication work?", [chunk])

    assert "How does authentication work?" in prompt
    assert "src/auth.py" in prompt
    assert "Lines: 10-12" in prompt
    assert "def authenticate(token):" in prompt


def test_prompt_instructs_llm_not_to_invent_facts() -> None:
    chunk = make_chunk_result()
    prompt = build_rag_prompt("Question", [chunk])

    assert "Do not invent or assume" in prompt
    assert "insufficient" in prompt
    assert "Never claim that code was executed" in prompt


def test_format_context_chunk_handles_missing_fields() -> None:
    minimal_chunk = RetrievalResult(
        chunk_id="c_min",
        content="x = 42",
    )
    formatted = format_context_chunk(minimal_chunk, index=1)
    assert "SOURCE 1" in formatted
    assert "x = 42" in formatted
    assert "File:" not in formatted


def test_build_explanation_prompt_backwards_compatible() -> None:
    chunk = make_chunk_result()
    prompt1 = build_rag_prompt("Query", [chunk])
    prompt2 = build_explanation_prompt("Query", [chunk])
    assert prompt1 == prompt2


def test_custom_prompt_builder_used_by_pipeline(fake_retriever, fake_llm) -> None:
    def custom_builder(question: str, context: Any) -> str:
        return f"CUSTOM_PROMPT: {question}"

    pipe = RAGPipeline(retriever=fake_retriever, llm=fake_llm, prompt_builder=custom_builder)
    pipe.answer("Test custom")
    assert fake_llm.calls[0]["prompt"] == "CUSTOM_PROMPT: Test custom"


# --------------------------------------------------------------------------- #
# 8. Security & Foundation Backward Compatibility
# --------------------------------------------------------------------------- #
def test_retrieved_code_never_executed(fake_llm: FakeLLM) -> None:
    malicious_chunk = make_chunk_result(
        content="import os\nos.environ['EXPLOIT_EXECUTED'] = 'YES'\nraise SystemExit('Pwned')",
    )
    retriever = FakeRetriever(results=[malicious_chunk])
    pipe = RAGPipeline(retriever=retriever, llm=fake_llm)

    result = pipe.answer("Explain this chunk")
    import os

    assert "EXPLOIT_EXECUTED" not in os.environ
    assert len(result.sources) == 1
    assert "EXPLOIT_EXECUTED" in result.sources[0].content


def test_no_code_execution_primitives_in_rag_pipeline_and_prompts() -> None:
    for filename in ("rag_pipeline.py", "prompts.py"):
        src = (ROOT / "src" / filename).read_text(encoding="utf-8")
        tree = ast.parse(src)

        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "exec" not in called_names
        assert "eval" not in called_names

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        assert "subprocess" not in imported_modules
        assert "os.system" not in src


def test_foundation_ask_placeholder_still_raises_not_implemented() -> None:
    """Ensure tests/test_foundation.py:test_placeholders_do_not_fake_rag remains satisfied."""
    pipe = RAGPipeline()
    with pytest.raises(NotImplementedError):
        pipe.ask("hi")
    with pytest.raises(NotImplementedError):
        pipe.ingest([Path("some/file.py")])

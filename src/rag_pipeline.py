"""RAG pipeline orchestration layer for CodeSage (Phase 1H).

Connects retrieval and LLM generation:

    User Question
          |
      Retriever -> list[RetrievalResult]
          |
    Prompt Construction (with instructions & formatted code)
          |
         LLM -> Answer
          |
      RAGResult(answer=..., sources=...)

Design notes
------------
* Dependency injection: uses passed-in Retriever and LLM objects. Does not
  directly instantiate ChromaDB, sentence-transformers, or Ollama.
* Validation: questions and parameters are validated before calling Retriever
  or LLM.
* Code safety: retrieved code is strictly treated as text for context; nothing
  is executed.
* Traceability: source metadata from RetrievalResult is preserved in RAGResult.
* Graceful fallback: handles empty retrieval without crashing.
"""

from __future__ import annotations
from src.relationship_answers import answer_relationship_question
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from src.prompts import (
    EXPLANATION_PROMPT_TEMPLATE,
    build_rag_prompt,
    format_context,
)
from src.retriever import DEFAULT_TOP_K, RetrievalResult
from src.relationship_context import build_relationship_context
from src.answer_evidence import validate_source_citations
from src.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from src.llm import LLM
    from src.retriever import Retriever

logger = get_logger("rag_pipeline")

NO_CONTEXT_MESSAGE: str = (
    "No relevant code context was found in the indexed codebase to answer this question."
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class RAGPipelineError(Exception):
    """Base class for all RAG pipeline errors."""


class RAGInputError(RAGPipelineError, ValueError):
    """Raised when an invalid question, top_k, or parameter is provided."""


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RAGResult:
    """Result of a RAG question answering query.

    Attributes:
        answer: Generated explanation or status text from the LLM.
        sources: List of :class:`RetrievalResult` objects retrieved from the
            vector store and used to ground the explanation.
        question: The user question that was answered.
    """

    answer: str
    sources: list[RetrievalResult] = field(default_factory=list)
    question: str = ""
    invalid_source_references: list[int] = field(default_factory=list)

    @property
    def source_files(self) -> list[str]:
        """Return distinct file paths from the retrieved sources."""
        return list(dict.fromkeys(s.file_path for s in self.sources if s.file_path))

    @property
    def has_sources(self) -> bool:
        """Whether any sources were retrieved."""
        return len(self.sources) > 0


# Backwards compatibility alias
RAGSource = RetrievalResult


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class RAGPipeline:
    """Orchestrates retrieval and LLM generation for codebase understanding.

    Args:
        retriever: Component with ``retrieve(query, top_k) -> list[RetrievalResult]``.
        llm: Component with ``generate(prompt, temperature) -> str``.
        prompt_builder: Optional custom prompt builder function.
            Defaults to :func:`src.prompts.build_rag_prompt`.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[LLM] = None,
        prompt_builder: Optional[Callable[..., str]] = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder or build_rag_prompt

    def answer(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        temperature: Optional[float] = None,
    ) -> RAGResult:
        """Answer a user question grounded in retrieved codebase context.

        Args:
            question: Natural language question about the codebase.
            top_k: Maximum number of relevant code chunks to retrieve.
            temperature: Optional sampling temperature for the LLM.

        Returns:
            A :class:`RAGResult` containing the generated answer, list of
            retrieved sources, and the question.

        Raises:
            RAGInputError: If ``question`` is not a non-empty string or
                ``top_k`` is not a positive integer.
            RAGPipelineError: If the pipeline is not configured with a retriever
                and LLM.
        """
        self._validate_question(question)
        self._validate_top_k(top_k)

        if self.retriever is None or self.llm is None:
            raise RAGPipelineError(
                "RAGPipeline requires both a retriever and an LLM instance before answering questions."
            )

        # 1. Retrieve relevant code chunks
        sources = self.retriever.retrieve(question, top_k=top_k)

        # 2. Handle zero retrieved chunks gracefully
        if not sources:
            logger.info("No relevant chunks retrieved for question: %s", question)
            return RAGResult(
                answer=NO_CONTEXT_MESSAGE,
                sources=[],
                question=question,
            )

        # 3. Build relationship facts only from the configured repository graph
        # and the source locations returned by retrieval. Custom prompt builders
        # retain their existing two-argument contract for backward compatibility.
        graph = getattr(self.retriever, "repository_graph", None)
        # Answer explicit caller/callee questions directly from graph evidence.
        # All other questions continue through the existing RAG pipeline.
        if graph is not None:
            relationship_answer = answer_relationship_question(
                question=question,
                graph=graph,
                sources=list(sources),
            )
            if relationship_answer is not None:
                return RAGResult(
                    answer=relationship_answer,
                    sources=list(sources),
                    question=question,
                )
        relationship_context = build_relationship_context(question, sources, graph)

        if self.prompt_builder is build_rag_prompt:
            prompt = self.prompt_builder(
                question=question,
                context=sources,
                relationship_context=relationship_context,
            )
        else:
            prompt = self.prompt_builder(question=question, context=sources)

        # 4. Generate answer with LLM
        gen_kwargs: dict[str, Any] = {}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        generated_answer = self.llm.generate(prompt, **gen_kwargs)

        # Phase 2E: validate explicit source markers against the actual retrieved
        # source list. This is a reference-integrity check, not semantic proof.
        citation_result = validate_source_citations(generated_answer, len(sources))

        return RAGResult(
            answer=citation_result.answer,
            sources=list(sources),
            question=question,
            invalid_source_references=list(citation_result.invalid_source_references),
        )

    # ----- Foundation compatibility methods ------------------------------- #
    def ingest(self, paths: list[Path]) -> int:
        """Placeholder matching the foundation test contract."""
        raise NotImplementedError("The RAG pipeline ingestion is not implemented yet.")

    def ask(self, question: str) -> str:
        """Placeholder matching the foundation test contract."""
        raise NotImplementedError("The RAG pipeline end-to-end ask() is not implemented yet.")

    # ----- internals ------------------------------------------------------- #
    @staticmethod
    def _validate_question(question: Any) -> None:
        if not isinstance(question, str):
            raise RAGInputError(f"question must be a string, got {type(question).__name__}.")
        if not question.strip():
            raise RAGInputError("question must not be empty or whitespace only.")

    @staticmethod
    def _validate_top_k(top_k: Any) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise RAGInputError(f"top_k must be an integer, got {type(top_k).__name__}.")
        if top_k <= 0:
            raise RAGInputError(f"top_k must be greater than zero, got {top_k}.")

"""Prompt templates and builders for CodeSage (Phase 1H).

Formats retrieved code context and user questions into structured prompts
for the local LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

from src.retriever import RetrievalResult

EXPLANATION_PROMPT_TEMPLATE: str = """\
You are CodeSage, an expert AI assistant for codebase understanding.
Your task is to explain and answer questions about a codebase using ONLY the provided code context.

Instructions:
1. Base your answer primarily on the retrieved code context below.
2. Do not invent or assume functions, classes, methods, files, or behaviors not present in the context.
3. If the retrieved context is insufficient to answer the question completely or accurately, state clearly what is missing.
4. Reference specific file names, line numbers, functions, and classes from the context where relevant.
5. Never claim that code was executed or run. Code is analyzed statically as text.
6. Some sources may be included because the repository graph links them to a semantic search result. Treat these as related context, not proof that a call executes at runtime.
7. Provide clear, concise, and technically accurate explanations.

--- CODE CONTEXT ---
{context}

--- USER QUESTION ---
{question}

Answer:"""


def format_context_chunk(result: RetrievalResult, index: int = 1) -> str:
    """Format a single RetrievalResult into a structured context block."""
    lines_str = ""
    if result.start_line is not None and result.end_line is not None:
        lines_str = f"Lines: {result.start_line}-{result.end_line}"
    elif result.start_line is not None:
        lines_str = f"Line: {result.start_line}"

    header_parts = [f"SOURCE {index}"]
    if result.file_path:
        header_parts.append(f"File: {result.file_path}")
    if result.chunk_type:
        header_parts.append(f"Type: {result.chunk_type}")
    name = result.qualified_name or result.name
    if name:
        header_parts.append(f"Name: {name}")
    if lines_str:
        header_parts.append(lines_str)
    graph_relation = (getattr(result, "metadata", {}) or {}).get("graph_relation")
    if graph_relation:
        header_parts.append(f"Related via repository graph: {graph_relation}")

    header = "\n".join(header_parts)
    content = result.content.strip() if hasattr(result, "content") else str(result).strip()
    return f"{header}\n\n```\n{content}\n```"


def format_context(
    context_chunks: Sequence[Union[RetrievalResult, str, Any]]
) -> str:
    """Format a sequence of RetrievalResult items or plain strings into a context block."""
    if not context_chunks:
        return "[No relevant code context was found in the codebase.]"

    formatted: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        if isinstance(chunk, RetrievalResult):
            formatted.append(format_context_chunk(chunk, index=i))
        elif isinstance(chunk, str):
            formatted.append(f"SOURCE {i}\n\n```\n{chunk.strip()}\n```")
        else:
            # Fallback for duck-typed chunk objects
            content = getattr(chunk, "content", str(chunk)).strip()
            formatted.append(f"SOURCE {i}\n\n```\n{content}\n```")

    return "\n\n".join(formatted)


def build_rag_prompt(
    question: str,
    context: Union[str, Sequence[Union[RetrievalResult, str, Any]]],
    template: str = EXPLANATION_PROMPT_TEMPLATE,
) -> str:
    """Build a complete prompt for the LLM from question and context."""
    if isinstance(context, str):
        context_str = context
    else:
        context_str = format_context(context)

    return template.format(context=context_str, question=question.strip())


def build_explanation_prompt(
    question: str,
    context_chunks: Sequence[Union[RetrievalResult, str, Any]],
) -> str:
    """Backwards-compatible wrapper matching earlier placeholder signature."""
    return build_rag_prompt(question=question, context=context_chunks)

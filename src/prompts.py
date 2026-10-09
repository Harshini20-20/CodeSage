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
8. For factual claims grounded in a source, cite its label exactly as [SOURCE 1], [SOURCE 2], etc. Use only labels that appear in CODE CONTEXT.
9. When useful, include the source file and line range shown in the cited source header. Never invent a file path, line number, or source label.
10. If the evidence does not support a claim, state the limitation rather than adding a citation that only looks plausible.
11. Treat repository relationship context as static-analysis evidence, not proof of runtime execution.
12. Only call a relationship confirmed when the supplied relationship context labels it CONFIRMED.
13. Describe unresolved calls as unresolved; never guess their targets or claim a path is complete when evidence is missing.
14. Distinguish direct calls from indirect multi-step paths, and mention uncertainty or missing graph coverage when relevant.
15. Do not infer a relationship merely because two code snippets appear together in the context.
16. For questions asking "which functions call X" or "who calls X", report only confirmed call relationships whose TARGET is X and whose SOURCE is the calling function. Do not include functions that call other functions merely because they appear in the same file or context.
17. For questions asking "which functions does X call", report only confirmed call relationships whose SOURCE is X and whose TARGET is the called function. Never reverse a call edge. If no matching confirmed edge is provided, state that no confirmed caller/callee relationship was found in the supplied graph context rather than guessing.
18. An UNRESOLVED call means the call exists in the analyzed source, but its target could not be resolved to a repository definition. Do not describe it as a nonexistent call. When the source code shows the call expression, acknowledge that the call occurs even if the graph cannot resolve its target.

--- REPOSITORY RELATIONSHIP CONTEXT ---
{relationship_context}

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
    relationship_context: str = "No repository relationship evidence is available for this question.",
) -> str:
    """Build a complete prompt from code evidence and optional graph facts."""
    if isinstance(context, str):
        context_str = context
    else:
        context_str = format_context(context)

    return template.format(
        context=context_str,
        question=question.strip(),
        relationship_context=relationship_context,
    )


def build_explanation_prompt(
    question: str,
    context_chunks: Sequence[Union[RetrievalResult, str, Any]],
) -> str:
    """Backwards-compatible wrapper matching earlier placeholder signature."""
    return build_rag_prompt(question=question, context=context_chunks)

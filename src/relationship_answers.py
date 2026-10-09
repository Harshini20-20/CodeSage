"""Deterministic answers for explicit repository call-relationship questions."""

from __future__ import annotations

import re
from typing import Optional

from src.repository_graph import RepositoryGraph, GraphNode
from src.retriever import RetrievalResult

_ELEMENT_KINDS = {"function", "method"}
_NAME = r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"

_CALLERS_PATTERNS = (
    re.compile(r"\bwhich\s+(?:functions|methods|callers)\s+(?:directly\s+)?(?:call|invoke)\s+" + _NAME, re.I),
    re.compile(r"\bwho\s+(?:directly\s+)?calls?\s+" + _NAME, re.I),
)

_CALLEES_PATTERNS = (
    re.compile(
        r"\bwhich\s+(?:functions|methods)\s+(?:does|do)\s+"
        + _NAME
        + r"\s*\(\s*\)\s+(?:directly\s+)?(?:call|invoke)\b",
        re.I,
    ),
    re.compile(
        r"\bwhat\s+(?:functions|methods)\s+(?:does|do)\s+"
        + _NAME
        + r"\s*\(\s*\)\s+(?:directly\s+)?(?:call|invoke)\b",
        re.I,
    ),
    re.compile(
        r"\bwhat\s+does\s+"
        + _NAME
        + r"\s*\(\s*\)\s+(?:directly\s+)?(?:call|invoke)\b",
        re.I,
    ),
)


def _query_name(patterns: tuple[re.Pattern[str], ...], question: str) -> Optional[str]:
    for pattern in patterns:
        match = pattern.search(question)
        if match:
            return match.group(1).split(".")[-1]
    return None


def _matches_name(node: GraphNode, name: str) -> bool:
    return node.name == name or node.name.endswith("." + name)


def _node_location(node: GraphNode) -> str:
    location = node.file_path
    if node.line is not None:
        location += f":{node.line}"
    return location


def _find_unique_element(graph: RepositoryGraph, name: str) -> Optional[GraphNode]:
    matches = [
        node for node in graph.nodes.values()
        if node.kind in _ELEMENT_KINDS and _matches_name(node, name)
    ]
    return matches[0] if len(matches) == 1 else None


def _source_snippets(
    sources: list[RetrievalResult],
    relevant_names: set[str],
) -> list[str]:
    snippets: list[str] = []
    seen: set[tuple[str, str]] = set()

    for source in sources:
        names = {source.name, source.qualified_name}
        if not relevant_names.intersection(names):
            continue

        key = (source.file_path, source.content)
        if not source.content.strip() or key in seen:
            continue

        seen.add(key)
        label = source.file_path or "Unknown file"
        if source.start_line is not None:
            label += f":{source.start_line}"
        snippets.append(f"{label}\n```python\n{source.content.rstrip()}\n```")

    return snippets


def answer_relationship_question(
    question: str,
    graph: RepositoryGraph,
    sources: list[RetrievalResult],
) -> Optional[str]:
    """Answer clearly phrased direct caller/callee questions from graph edges.

    Returns None when the question is not a supported relationship query or
    when a requested function name is ambiguous or absent.
    """
    caller_target = _query_name(_CALLERS_PATTERNS, question)
    callee_source = _query_name(_CALLEES_PATTERNS, question)

    if caller_target is None and callee_source is None:
        return None

    sections: list[str] = []
    relevant_names: set[str] = set()

    if caller_target is not None:
        target = _find_unique_element(graph, caller_target)
        if target is None:
            return None

        callers: list[str] = []
        for edge in graph.incoming(target.id, kind="calls"):
            caller = graph.nodes.get(edge.source)
            if caller is None or caller.kind not in _ELEMENT_KINDS:
                continue
            if edge.metadata.get("resolved") is not True:
                continue
            label = f"`{caller.name}` ({_node_location(caller)}, line {edge.line})"
            if label not in callers:
                callers.append(label)
                relevant_names.add(caller.name)

        relevant_names.add(target.name)
        if callers:
            sections.append(
                f"**Confirmed direct callers of `{target.name}`:**\n"
                + "\n".join(f"- {item}" for item in callers)
            )
        else:
            sections.append(
                f"No confirmed direct callers of `{target.name}` were found "
                "in the repository's static-analysis graph."
            )

    if callee_source is not None:
        caller = _find_unique_element(graph, callee_source)
        if caller is None:
            return None

        confirmed: list[str] = []
        unresolved: list[str] = []

        for edge in graph.outgoing(caller.id, kind="calls"):
            target = graph.nodes.get(edge.target)
            if target is None:
                continue

            call_name = str(edge.metadata.get("call_name") or target.name)
            line = f"line {edge.line}" if edge.line is not None else "line unknown"

            if edge.metadata.get("resolved") is True and target.kind in _ELEMENT_KINDS:
                label = f"`{target.name}` ({_node_location(target)}, called at {line})"
                if label not in confirmed:
                    confirmed.append(label)
                    relevant_names.add(target.name)
            else:
                label = f"`{call_name}` ({line}; target unresolved)"
                if label not in unresolved:
                    unresolved.append(label)

        relevant_names.add(caller.name)
        details = []
        if confirmed:
            details.append("**Confirmed direct calls:**\n" + "\n".join(f"- {item}" for item in confirmed))
        if unresolved:
            details.append("**Calls with unresolved targets:**\n" + "\n".join(f"- {item}" for item in unresolved))
        if not details:
            details.append("No direct call edges were recorded for this function.")

        sections.append(f"**Direct calls made by `{caller.name}`:**\n" + "\n".join(details))

    answer = "\n\n".join(sections)
    answer += (
        "\n\n*Evidence is based on static analysis of the repository, "
        "not proof of runtime execution.*"
    )

    snippets = _source_snippets(sources, relevant_names)
    if snippets:
        answer += "\n\n**Relevant retrieved source code:**\n\n" + "\n\n".join(snippets)

    return answer


__all__ = ["answer_relationship_question"]

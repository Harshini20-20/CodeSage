
"""Deterministic answers for multi-hop repository call-path questions."""

from __future__ import annotations

import re
from typing import Optional

from src.call_paths import find_call_path
from src.repository_graph import GraphNode, RepositoryGraph


_NAME = r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"

_PATH_PATTERNS = (
    re.compile(
        r"\bhow\s+does\s+"
        + _NAME
        + r"\s*\(\s*\)\s+(?:reach|get\s+to)\s+"
        + _NAME
        + r"\s*\(\s*\)",
        re.I,
    ),
    re.compile(
        r"\b(?:show|trace|find)\s+(?:the\s+)?(?:call\s+)?path\s+from\s+"
        + _NAME
        + r"\s*(?:\(\s*\))?\s+to\s+"
        + _NAME
        + r"\s*(?:\(\s*\))?",
        re.I,
    ),
)


def _unique_element(
    graph: RepositoryGraph, name: str
) -> Optional[GraphNode]:
    """Return an element only when its name identifies exactly one node."""
    matches = [
        node
        for node in graph.nodes.values()
        if node.kind in {"function", "method"} and node.name == name
    ]
    return matches[0] if len(matches) == 1 else None


def _location(node: GraphNode) -> str:
    location = node.file_path
    if node.line is not None:
        location += f":{node.line}"
    return location


def answer_call_path_question(
    question: str,
    graph: RepositoryGraph,
) -> Optional[str]:
    """Answer supported multi-hop call-path questions using graph evidence."""
    requested_path = None

    for pattern in _PATH_PATTERNS:
        match = pattern.search(question)
        if match:
            requested_path = (
                match.group(1).split(".")[-1],
                match.group(2).split(".")[-1],
            )
            break

    if requested_path is None:
        return None

    source_name, target_name = requested_path

    source_node = _unique_element(graph, source_name)
    target_node = _unique_element(graph, target_name)

    if source_node is None or target_node is None:
        return None

    path = find_call_path(graph, source_name, target_name)

    if path is None:
        return (
            f"No confirmed call path from `{source_name}` to `{target_name}` "
            "was found in the repository's static-analysis graph."
            "\n\n*This does not prove that no runtime path exists.*"
        )

    lines = [
        f"**Confirmed call path from `{source_name}` to `{target_name}`:**",
        "",
    ]

    for index, node in enumerate(path.nodes):
        lines.append(f"{index + 1}. `{node.name}` ({_location(node)})")

        if index < len(path.edges):
            edge = path.edges[index]
            next_node = path.nodes[index + 1]
            call_location = (
                f"{node.file_path}:{edge.line}"
                if edge.line is not None
                else f"{node.file_path}:line unknown"
            )
            lines.append(
                f"   - Calls `{next_node.name}` at `{call_location}`"
            )

    lines.append(
        "\n*Evidence is based on confirmed static-analysis call edges, "
        "not proof of runtime execution.*"
    )

    return "\n".join(lines)


__all__ = ["answer_call_path_question"]

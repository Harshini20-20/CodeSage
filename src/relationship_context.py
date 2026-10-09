"""Build bounded, evidence-based relationship context for RAG answers.

Repository relationships are static-analysis evidence, not proof of runtime
execution. This module formats only graph edges connected to retrieved code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.repository_graph import GraphEdge, GraphNode, RepositoryGraph
from src.retriever import RetrievalResult

_ELEMENT_KINDS = {"function", "class", "method"}
_RELATIONSHIP_LIMIT = 16


def _relative_path(file_path: str, root: str) -> str:
    path = Path(file_path)
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix().replace("\\", "/").lstrip("./")


def _node_label(node: GraphNode) -> str:
    location = f"{node.file_path}"
    if node.line is not None:
        location += f":{node.line}"
    return f"{node.kind} `{node.name}` ({location})"


def _edge_description(edge: GraphEdge, graph: RepositoryGraph) -> str:
    source = graph.nodes.get(edge.source)
    target = graph.nodes.get(edge.target)
    if source is None or target is None:
        return ""

    location = f" at line {edge.line}" if edge.line is not None else ""
    if edge.kind == "calls":
        call_name = str(edge.metadata.get("call_name") or target.name)
        if edge.metadata.get("resolved") is True and target.kind in _ELEMENT_KINDS:
            return f"CONFIRMED call: {_node_label(source)} calls {_node_label(target)} (call `{call_name}`{location})."
        return (
            f"UNRESOLVED call: {_node_label(source)} references `{call_name}`{location}; "
            "the target was not resolved to a repository definition."
        )
    if edge.kind == "inherits" and edge.metadata.get("resolved") is True:
        return f"CONFIRMED inheritance: {_node_label(source)} inherits from {_node_label(target)}{location}."
    if edge.kind == "imports":
        return f"FILE IMPORT: `{source.file_path}` imports `{target.file_path}`{location}."
    return ""


def build_relationship_context(
    question: str,
    sources: Iterable[RetrievalResult],
    graph: RepositoryGraph | None,
    limit: int = _RELATIONSHIP_LIMIT,
) -> str:
    """Return relevant graph facts for retrieved elements, or a safe fallback.

    Only direct graph edges touching a retrieved element or its file are
    included. Unresolved call edges are explicitly labelled unresolved.
    """
    del question  # Relationship selection is source-anchored to avoid keyword guesses.
    source_list = list(sources)
    if graph is None or not source_list or limit <= 0:
        return "No repository relationship evidence is available for this question."

    matched_ids: set[str] = set()
    matched_file_paths: set[str] = set()
    for result in source_list:
        rel_path = _relative_path(result.file_path, graph.root) if result.file_path else ""
        if rel_path:
            matched_file_paths.add(rel_path)
        for node in graph.nodes.values():
            if node.kind not in _ELEMENT_KINDS or node.file_path.replace("\\", "/") != rel_path:
                continue
            result_name = result.qualified_name or result.name
            if result_name and node.name != result_name:
                continue
            if result.start_line is not None and node.line != result.start_line:
                continue
            matched_ids.add(node.id)

    if not matched_ids and not matched_file_paths:
        return "No repository relationship evidence could be matched to the retrieved source locations."

    edge_candidates: list[GraphEdge] = []
    for edge in graph.edges:
        source_node = graph.nodes.get(edge.source)
        target_node = graph.nodes.get(edge.target)
        if source_node is None or target_node is None:
            continue
        # Calls/inheritance must touch a retrieved element. File imports are
        # useful only for files represented by retrieved chunks.
        if edge.kind in {"calls", "inherits"}:
            if edge.source in matched_ids or edge.target in matched_ids:
                edge_candidates.append(edge)
        elif edge.kind == "imports":
            if source_node.file_path in matched_file_paths or target_node.file_path in matched_file_paths:
                edge_candidates.append(edge)

    descriptions: list[str] = []
    seen: set[str] = set()
    for edge in edge_candidates:
        description = _edge_description(edge, graph)
        if description and description not in seen:
            seen.add(description)
            descriptions.append(description)
        if len(descriptions) >= limit:
            break

    if not descriptions:
        return (
            "No direct caller, callee, inheritance, or file-import relationships were "
            "found for the retrieved source locations. This does not prove that no "
            "such relationships exist elsewhere or at runtime."
        )

    return "\n".join(f"- {item}" for item in descriptions)


__all__ = ["build_relationship_context"]

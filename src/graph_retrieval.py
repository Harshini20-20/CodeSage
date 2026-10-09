"""Graph-assisted retrieval for CodeSage.

This module expands vector-search seeds with a small, deterministic set of
related code chunks. It never executes source code and never changes the
vector-store ranking of the original results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.repository_graph import RepositoryGraph
from src.retriever import RetrievalResult

_ELEMENT_KINDS = {"function", "class", "method"}
_RELATION_PRIORITY = {"calls": 0, "inherits": 1, "contains": 2, "imports": 3}


def _relative_chunk_path(file_path: str, root: str) -> str:
    """Normalize absolute/relative chunk paths to the graph's POSIX paths."""
    path = Path(file_path)
    root_path = Path(root)
    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix().replace("\\", "/").lstrip("./")


def build_graph_chunk_lookup(
    graph: RepositoryGraph, chunks: Iterable[Any]
) -> dict[str, RetrievalResult]:
    """Map graph element IDs to their corresponding chunks.

    ``chunks`` accepts CodeChunk-like objects, keeping this helper independent
    of the embedding and storage implementations. File-level nodes are not
    mapped because they do not represent a focused source chunk.
    """
    nodes_by_key: dict[tuple[str, str, str, int | None], list[str]] = {}
    for node in graph.nodes.values():
        if node.kind not in _ELEMENT_KINDS:
            continue
        key = (node.file_path.replace("\\", "/"), node.kind, node.name, node.line)
        nodes_by_key.setdefault(key, []).append(node.id)

    lookup: dict[str, RetrievalResult] = {}
    for chunk in chunks:
        file_path = _relative_chunk_path(str(chunk.file_path), graph.root)
        chunk_type = str(chunk.chunk_type)
        if chunk_type not in _ELEMENT_KINDS:
            continue
        key = (file_path, chunk_type, str(chunk.qualified_name), int(chunk.start_line))
        for node_id in nodes_by_key.get(key, []):
            lookup[node_id] = RetrievalResult(
                chunk_id=str(chunk.chunk_id),
                content=str(getattr(chunk, "text", chunk.source)),
                file_path=str(chunk.file_path),
                chunk_type=chunk_type,
                name=str(chunk.name),
                qualified_name=str(chunk.qualified_name),
                start_line=int(chunk.start_line),
                end_line=int(chunk.end_line),
                distance=None,
                metadata=dict(getattr(chunk, "metadata", {}) or {}),
            )
    return lookup


def _nodes_for_result(graph: RepositoryGraph, result: RetrievalResult) -> list[str]:
    rel_path = _relative_chunk_path(result.file_path, graph.root)
    matches = []
    for node in graph.nodes.values():
        if node.kind not in _ELEMENT_KINDS:
            continue
        if node.file_path.replace("\\", "/") != rel_path:
            continue
        if node.name != result.qualified_name:
            continue
        if result.start_line is not None and node.line != result.start_line:
            continue
        matches.append(node.id)
    return sorted(matches)


def expand_with_graph(
    results: list[RetrievalResult],
    graph: RepositoryGraph | None,
    chunk_lookup: dict[str, RetrievalResult] | None,
    limit: int = 3,
) -> list[RetrievalResult]:
    """Append up to ``limit`` connected chunks after the vector-ranked seeds.

    Direct resolved calls and inheritance are preferred, followed by class
    containment and imported-module context. Unresolved symbols are ignored.
    Original results retain their exact order and are never duplicated.
    """
    if not results or graph is None or not chunk_lookup or limit <= 0:
        return list(results)

    existing_ids = {item.chunk_id for item in results}
    seed_node_ids: list[str] = []
    for result in results:
        seed_node_ids.extend(_nodes_for_result(graph, result))

    candidates: dict[str, tuple[int, str]] = {}
    for seed_id in seed_node_ids:
        seed_node = graph.nodes.get(seed_id)
        if seed_node is None:
            continue
        # Calls and inheritance are useful in both directions: dependencies and callers.
        for edge in graph.edges:
            if edge.kind not in {"calls", "inherits", "contains"}:
                continue
            if edge.kind == "calls" and not edge.metadata.get("resolved", False):
                continue
            if edge.source == seed_id:
                neighbor_id = edge.target
            elif edge.target == seed_id:
                neighbor_id = edge.source
            else:
                continue
            neighbor = graph.nodes.get(neighbor_id)
            if neighbor is None or neighbor.kind not in _ELEMENT_KINDS or neighbor_id not in chunk_lookup:
                continue
            if chunk_lookup[neighbor_id].chunk_id in existing_ids:
                continue
            rank = _RELATION_PRIORITY[edge.kind]
            old = candidates.get(neighbor_id)
            relation = edge.kind
            if old is None or rank < old[0]:
                candidates[neighbor_id] = (rank, relation)

        # Imported-module context is a lower-priority fallback. Add only element
        # nodes from directly imported local files; the result limit bounds noise.
        file_id = f"file:{seed_node.file_path}"
        for edge in graph.outgoing(file_id, "imports"):
            target_file = graph.nodes.get(edge.target)
            if target_file is None:
                continue
            for neighbor in graph.nodes.values():
                if neighbor.file_path != target_file.file_path:
                    continue
                if neighbor.kind not in _ELEMENT_KINDS or neighbor.id not in chunk_lookup:
                    continue
                if chunk_lookup[neighbor.id].chunk_id in existing_ids:
                    continue
                candidates.setdefault(neighbor.id, (_RELATION_PRIORITY["imports"], "imports"))

    ordered = sorted(candidates.items(), key=lambda item: (item[1][0], item[0]))
    expanded: list[RetrievalResult] = []
    for node_id, (_, relation) in ordered:
        if len(expanded) >= limit:
            break
        source = chunk_lookup[node_id]
        if source.chunk_id in existing_ids:
            continue
        metadata = dict(source.metadata)
        metadata["graph_relation"] = relation
        metadata["graph_node_id"] = node_id
        expanded.append(
            RetrievalResult(
                chunk_id=source.chunk_id,
                content=source.content,
                file_path=source.file_path,
                chunk_type=source.chunk_type,
                name=source.name,
                qualified_name=source.qualified_name,
                start_line=source.start_line,
                end_line=source.end_line,
                distance=None,
                metadata=metadata,
            )
        )
        existing_ids.add(source.chunk_id)
    return list(results) + expanded


__all__ = ["build_graph_chunk_lookup", "expand_with_graph"]

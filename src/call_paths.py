
from dataclasses import dataclass

from src.repository_graph import GraphEdge, GraphNode, RepositoryGraph


@dataclass(frozen=True)
class CallPath:
    """A call path from a starting function to a target function."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]



def _find_element(graph: RepositoryGraph, name: str) -> GraphNode | None:
    """Resolve a unique function or method by name."""
    matches = [
        node
        for node in graph.nodes.values()
        if node.kind in {"function", "method"} and node.name == name
    ]

    if len(matches) != 1:
        return None

    return matches[0]

def find_call_path(
    graph: RepositoryGraph,
    source: str,
    target: str,
) -> CallPath | None:
    """
    Find the shortest path of confirmed, resolved calls between two functions.

    Returns None if either function is missing or ambiguous, or no path exists.
    """
    source_node = _find_element(graph, source)
    target_node = _find_element(graph, target)

    if source_node is None or target_node is None:
        return None

    if source_node.id == target_node.id:
        return CallPath(nodes=[source_node], edges=[])

    # Breadth-first search finds a shortest call path.
    queue: list[tuple[str, list[GraphNode], list[GraphEdge]]] = [
        (source_node.id, [source_node], [])
    ]
    visited = {source_node.id}

    while queue:
        current_id, path_nodes, path_edges = queue.pop(0)

        outgoing_edges = sorted(
            graph.outgoing(current_id, kind="calls"),
            key=lambda edge: (
                edge.target,
                edge.line if edge.line is not None else -1,
            ),
        )

        for edge in outgoing_edges:
            # Never follow unresolved calls.
            if edge.metadata.get("resolved") is not True:
                continue

            next_node = graph.nodes.get(edge.target)

            if next_node is None or next_node.kind not in {"function", "method"}:
                continue

            # Prevent cycles and repeated visits.
            if next_node.id in visited:
                continue

            next_nodes = [*path_nodes, next_node]
            next_edges = [*path_edges, edge]

            if next_node.id == target_node.id:
                return CallPath(nodes=next_nodes, edges=next_edges)

            visited.add(next_node.id)
            queue.append((next_node.id, next_nodes, next_edges))

    return None

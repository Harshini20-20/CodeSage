
from pathlib import Path

from src.parser import CodeParser
from src.repository_graph import (
    GraphEdge,
    GraphNode,
    RepositoryGraph,
    build_repository_graph,
)
from src.call_paths import find_call_path


def build_graph(tmp_path: Path):
    (tmp_path / "main.py").write_text(
        "def main():\n"
        "    process()\n\n"
        "def process():\n"
        "    calculate()\n\n"
        "def calculate():\n"
        "    return 42\n",
        encoding="utf-8",
    )
    parsed = CodeParser().parse_directory(tmp_path)
    return build_repository_graph(parsed, root=tmp_path)


def test_finds_multi_hop_call_path(tmp_path):
    graph = build_graph(tmp_path)

    path = find_call_path(graph, "main", "calculate")

    assert path is not None
    assert [node.name for node in path.nodes] == [
        "main",
        "process",
        "calculate",
    ]


def test_returns_none_when_no_path_exists(tmp_path):
    graph = build_graph(tmp_path)

    path = find_call_path(graph, "calculate", "main")

    assert path is None


def test_returns_single_node_path_for_same_function(tmp_path):
    graph = build_graph(tmp_path)

    path = find_call_path(graph, "main", "main")

    assert path is not None
    assert [node.name for node in path.nodes] == ["main"]


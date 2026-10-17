
from src.call_paths import find_call_path
from src.repository_graph import GraphEdge, GraphNode, RepositoryGraph
from src.call_path_answers import answer_call_path_question

def build_graph():
    nodes = {
        "main": GraphNode(
            id="main",
            kind="function",
            name="main",
            file_path="main.py",
            line=1,
        ),
        "process": GraphNode(
            id="process",
            kind="function",
            name="process",
            file_path="main.py",
            line=4,
        ),
        "calculate": GraphNode(
            id="calculate",
            kind="function",
            name="calculate",
            file_path="main.py",
            line=7,
        ),
    }

    edges = [
        GraphEdge(
            source="main",
            target="process",
            kind="calls",
            line=2,
            metadata={"resolved": True, "call_name": "process"},
        ),
        GraphEdge(
            source="process",
            target="calculate",
            kind="calls",
            line=5,
            metadata={"resolved": True, "call_name": "calculate"},
        ),
    ]

    return RepositoryGraph(root=".", nodes=nodes, edges=edges)


def test_call_path_contains_ordered_nodes_and_call_lines():
    graph = build_graph()

    path = find_call_path(graph, "main", "calculate")

    assert path is not None
    assert [node.name for node in path.nodes] == [
        "main",
        "process",
        "calculate",
    ]
    assert [edge.line for edge in path.edges] == [2, 5]
def test_answers_how_source_reaches_target():
    answer = answer_call_path_question(
        "How does main() reach calculate()?",
        build_graph(),
    )

    assert answer is not None
    assert "main" in answer
    assert "process" in answer
    assert "calculate" in answer
    assert "main.py:2" in answer
    assert "main.py:5" in answer


def test_reports_when_no_confirmed_path_exists():
    answer = answer_call_path_question(
        "How does calculate() reach main()?",
        build_graph(),
    )

    assert answer is not None
    assert "No confirmed call path" in answer


def test_ignores_unsupported_questions():
    answer = answer_call_path_question(
        "What does this project do?",
        build_graph(),
    )

    assert answer is None
from pathlib import Path

from src.parser import CodeParser
from src.repository_graph import build_repository_graph
from src.retriever import RetrievalResult
from src.relationship_answers import answer_relationship_question


def _build_project(tmp_path: Path):
    (tmp_path / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text(
        "def is_even(number):\n"
        "    return number % 2 == 0\n",
        encoding="utf-8",
    )
    main_code = (
        "from calculator import add, multiply\n"
        "from utils import is_even\n\n"
        "def main():\n"
        "    a = 10\n"
        "    b = 5\n"
        "    total = add(a, b)\n"
        "    product = multiply(a, b)\n"
        '    print("Sum:", total)\n'
        '    print("Product:", product)\n'
        '    print("Sum is even:", is_even(total))\n'
    )
    (tmp_path / "main.py").write_text(main_code, encoding="utf-8")

    parsed = CodeParser().parse_directory(tmp_path)
    graph = build_repository_graph(parsed, root=tmp_path)
    sources = [
        RetrievalResult(
            chunk_id="main",
            content=main_code,
            file_path="main.py",
            name="main",
            qualified_name="main",
            start_line=4,
            end_line=12,
        )
    ]
    return graph, sources


def test_lists_confirmed_direct_calls_and_unresolved_prints(tmp_path):
    graph, sources = _build_project(tmp_path)

    answer = answer_relationship_question(
        "Which functions does main() call?",
        graph,
        sources,
    )

    assert answer is not None
    assert "`add`" in answer
    assert "`multiply`" in answer
    assert "`is_even`" in answer
    assert "`print`" in answer
    assert "target unresolved" in answer
    assert "main.py:4" in answer


def test_caller_query_does_not_reverse_call_direction(tmp_path):
    graph, sources = _build_project(tmp_path)

    answer = answer_relationship_question(
        "Which functions call add()?",
        graph,
        sources,
    )

    assert answer is not None
    assert "`main`" in answer
    assert "`multiply`" not in answer


def test_direct_callee_query_reports_only_edges_from_requested_function(tmp_path):
    graph, sources = _build_project(tmp_path)

    answer = answer_relationship_question(
        "What does multiply() call?",
        graph,
        sources,
    )

    assert answer is not None
    assert "Direct calls made by `multiply`" in answer
    assert "`add`" not in answer
    assert "`is_even`" not in answer


def test_non_relationship_question_returns_none(tmp_path):
    graph, sources = _build_project(tmp_path)

    answer = answer_relationship_question(
        "Explain how the calculator works.",
        graph,
        sources,
    )

    assert answer is None


def test_ambiguous_function_name_falls_back_safely(tmp_path):
    (tmp_path / "a.py").write_text(
        "def helper():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "def helper():\n    pass\n",
        encoding="utf-8",
    )
    graph = build_repository_graph(
        CodeParser().parse_directory(tmp_path),
        root=tmp_path,
    )

    answer = answer_relationship_question(
        "Which functions call helper()?",
        graph,
        [],
    )

    assert answer is None

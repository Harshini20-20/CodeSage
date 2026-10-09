from pathlib import Path

from src.parser import CodeParser
from src.repository_graph import build_repository_graph


def _parse_tree(root: Path):
    return CodeParser().parse_directory(root)


def test_builds_file_and_code_element_nodes_with_containment_edges(tmp_path):
    (tmp_path / "app.py").write_text(
        "def helper():\n    return 1\n\nclass Worker:\n    def run(self):\n        helper()\n",
        encoding="utf-8",
    )
    parsed = _parse_tree(tmp_path)
    graph = build_repository_graph(parsed, root=tmp_path)

    assert len(graph.nodes_of_kind("file")) == 1
    assert len(graph.nodes_of_kind("function")) == 1
    assert len(graph.nodes_of_kind("class")) == 1
    assert len(graph.nodes_of_kind("method")) == 1
    assert len(graph.edges_of_kind("contains")) == 3


def test_links_resolved_calls_and_preserves_unresolved_calls(tmp_path):
    (tmp_path / "calls.py").write_text(
        "def helper():\n    pass\n\ndef run():\n    helper()\n    unknown()\n",
        encoding="utf-8",
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    calls = graph.edges_of_kind("calls")

    assert len(calls) == 2
    assert sum(edge.metadata["resolved"] for edge in calls) == 1
    assert sum(not edge.metadata["resolved"] for edge in calls) == 1
    assert len(graph.nodes_of_kind("symbol")) == 1


def test_links_local_imports_and_unique_inheritance(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "base.py").write_text("class Base:\n    pass\n", encoding="utf-8")
    (package / "child.py").write_text(
        "from pkg.base import Base\n\nclass Child(Base):\n    pass\n",
        encoding="utf-8",
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)

    imports = graph.edges_of_kind("imports")
    inheritance = graph.edges_of_kind("inherits")
    assert len(imports) == 1
    assert graph.nodes[imports[0].target].file_path == "pkg/base.py"
    assert len(inheritance) == 1
    assert graph.nodes[inheritance[0].source].name == "Child"
    assert graph.nodes[inheritance[0].target].name == "Base"


def test_duplicate_class_names_do_not_create_guessed_inheritance_edges(tmp_path):
    (tmp_path / "a.py").write_text("class Base:\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("class Base:\n    pass\nclass Child(Base):\n    pass\n", encoding="utf-8")
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    assert graph.edges_of_kind("inherits") == []


def test_empty_input_returns_empty_graph():
    graph = build_repository_graph([])
    assert graph.nodes == {}
    assert graph.edges == []


def test_resolves_cross_file_function_import_and_alias(tmp_path):
    (tmp_path / "helpers.py").write_text("def validate_order():\n    return True\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "from helpers import validate_order as validate\n\ndef process():\n    validate()\n",
        encoding="utf-8",
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["call_name"] == "validate")
    assert call.metadata["resolved"] is True
    assert graph.nodes[call.target].file_path == "helpers.py"
    assert graph.nodes[call.target].name == "validate_order"


def test_resolves_calls_through_imported_module_alias(tmp_path):
    (tmp_path / "helpers.py").write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "import helpers as h\n\ndef process():\n    h.normalize('x')\n", encoding="utf-8"
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["call_name"] == "h.normalize")
    assert call.metadata["resolved"] is True
    assert graph.nodes[call.target].file_path == "helpers.py"


def test_resolves_imported_class_method_call(tmp_path):
    (tmp_path / "models.py").write_text(
        "class Worker:\n    def run(self):\n        pass\n", encoding="utf-8"
    )
    (tmp_path / "main.py").write_text(
        "from models import Worker as W\n\ndef start():\n    W.run()\n", encoding="utf-8"
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["call_name"] == "W.run")
    assert call.metadata["resolved"] is True
    assert graph.nodes[call.target].file_path == "models.py"
    assert graph.nodes[call.target].name == "Worker.run"


def test_resolves_relative_imports_inside_package(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helpers.py").write_text("def clean():\n    pass\n", encoding="utf-8")
    (package / "service.py").write_text(
        "from .helpers import clean\n\ndef run():\n    clean()\n", encoding="utf-8"
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["call_name"] == "clean")
    assert call.metadata["resolved"] is True
    assert graph.nodes[call.target].file_path == "pkg/helpers.py"


def test_inheritance_resolves_import_alias(tmp_path):
    (tmp_path / "base.py").write_text("class Base:\n    pass\n", encoding="utf-8")
    (tmp_path / "child.py").write_text(
        "from base import Base as Parent\n\nclass Child(Parent):\n    pass\n", encoding="utf-8"
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    edge = graph.edges_of_kind("inherits")[0]
    assert graph.nodes[edge.source].name == "Child"
    assert graph.nodes[edge.target].name == "Base"


def test_ambiguous_cross_file_function_call_stays_unresolved(tmp_path):
    (tmp_path / "a.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def run():\n    helper()\n", encoding="utf-8")
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["call_name"] == "helper")
    assert call.metadata["resolved"] is False
    assert graph.nodes[call.target].kind == "symbol"


def test_graph_query_helpers_and_serialization_are_stable(tmp_path):
    (tmp_path / "app.py").write_text("def helper():\n    pass\n\ndef run():\n    helper()\n", encoding="utf-8")
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    helper = graph.find_nodes("helper", kind="function")[0]
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["resolved"])
    assert graph.incoming(helper.id, kind="calls") == [call]
    payload = graph.to_dict()
    assert payload["nodes"] == sorted(payload["nodes"], key=lambda item: item["id"])
    assert payload["edges"]


def test_resolves_dotted_call_from_unaliased_package_import(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helpers.py").write_text("def normalize():\n    pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "import pkg.helpers\n\ndef run():\n    pkg.helpers.normalize()\n", encoding="utf-8"
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["call_name"] == "pkg.helpers.normalize")
    assert call.metadata["resolved"] is True
    assert graph.nodes[call.target].file_path == "pkg/helpers.py"


def test_duplicate_imported_function_definitions_remain_unresolved(tmp_path):
    (tmp_path / "a.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "from a import helper\nfrom b import helper\n\ndef run():\n    helper()\n", encoding="utf-8"
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["call_name"] == "helper")
    assert call.metadata["resolved"] is False


def test_graph_retains_import_edges_for_local_modules_only(tmp_path):
    (tmp_path / "local.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("import local\nimport os\n", encoding="utf-8")
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    edges = graph.edges_of_kind("imports")
    assert len(edges) == 1
    assert graph.nodes[edges[0].target].file_path == "local.py"


def test_resolves_local_class_constructor_call(tmp_path):
    (tmp_path / "app.py").write_text(
        "class Worker:\n    pass\n\ndef build():\n    Worker()\n", encoding="utf-8"
    )
    graph = build_repository_graph(_parse_tree(tmp_path), root=tmp_path)
    call = next(edge for edge in graph.edges_of_kind("calls") if edge.metadata["call_name"] == "Worker")
    assert call.metadata["resolved"] is True
    assert graph.nodes[call.target].kind == "class"

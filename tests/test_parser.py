"""Tests for src/parser.py (Python AST parsing)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parser import (  # noqa: E402
    CodeParser,
    FileReadError,
    ParsedFile,
    ParserError,
    ParserFileNotFoundError,
    UnsupportedFileTypeError,
)

SAMPLE = '''\
"""Module doc."""
import os
import numpy as np
from pathlib import Path, PurePath as PP
from . import sibling


def greet(name):
    """Say hello."""
    return f"hello {name}"


@staticmethod
async def fetch():
    return 1


class Calculator:
    """A calculator."""

    def __init__(self):
        self.total = 0

    @property
    def value(self):
        return self.total

    async def add(self, x):
        self.total += x
        return self.total
'''


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE, encoding="utf-8")
    return f


@pytest.fixture()
def parsed(sample_file: Path) -> ParsedFile:
    return CodeParser().parse_file(sample_file)


def test_parse_valid_python_file(parsed, sample_file):
    assert parsed.ok and parsed.status == "ok" and parsed.error is None
    assert parsed.language == "python"
    assert parsed.path == sample_file.resolve()
    assert parsed.file_name == "sample.py"
    assert parsed.line_count == len(SAMPLE.splitlines())


def test_detects_function(parsed):
    names = [f.name for f in parsed.functions]
    assert names == ["greet", "fetch"]
    greet = parsed.functions[0]
    assert greet.kind == "function" and greet.parent is None
    assert greet.qualified_name == "greet"
    assert greet.docstring == "Say hello."
    assert greet.language == "python" and greet.file_path == parsed.path
    assert parsed.functions[1].is_async is True


def test_detects_class(parsed):
    assert [c.name for c in parsed.classes] == ["Calculator"]
    cls = parsed.classes[0]
    assert cls.kind == "class" and cls.docstring == "A calculator."


def test_detects_class_methods(parsed):
    cls = parsed.classes[0]
    assert [m.name for m in cls.methods] == ["__init__", "value", "add"]
    add = cls.methods[2]
    assert add.kind == "method" and add.parent == "Calculator"
    assert add.qualified_name == "Calculator.add" and add.is_async is True
    # methods are not reported as top-level functions
    assert "add" not in [f.name for f in parsed.functions]


def test_detects_imports(parsed):
    imports = parsed.imports
    assert [i.start_line for i in imports] == [2, 3, 4, 5]
    os_imp, np_imp, path_imp, rel_imp = imports
    assert (os_imp.kind, os_imp.module, os_imp.names) == ("import", "os", ["os"])
    assert np_imp.aliases == {"numpy": "np"}
    assert (path_imp.kind, path_imp.module, path_imp.names) == ("from", "pathlib", ["Path", "PurePath"])
    assert path_imp.aliases == {"PurePath": "PP"}
    assert rel_imp.level == 1 and rel_imp.module == "" and rel_imp.names == ["sibling"]
    assert path_imp.source == "from pathlib import Path, PurePath as PP"


def test_imports_inside_functions_are_found(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def f():\n    import json\n    return json\n", encoding="utf-8")
    result = CodeParser().parse_file(f)
    assert [i.module for i in result.imports] == ["json"] and result.imports[0].start_line == 2


def test_preserves_source_code(parsed):
    assert parsed.content == SAMPLE
    greet = parsed.functions[0]
    assert greet.source == 'def greet(name):\n    """Say hello."""\n    return f"hello {name}"'
    add = parsed.classes[0].methods[2]
    assert add.source.startswith("    async def add(self, x):")  # indentation preserved
    assert add.source.endswith("return self.total")


def test_line_numbers(parsed):
    lines = SAMPLE.split("\n")
    greet = parsed.functions[0]
    assert (greet.start_line, greet.end_line) == (8, 10)
    fetch = parsed.functions[1]
    assert (fetch.start_line, fetch.end_line) == (13, 15)  # decorator line is included
    assert lines[fetch.start_line - 1] == "@staticmethod"
    cls = parsed.classes[0]
    assert (cls.start_line, cls.end_line) == (18, 30)
    value = cls.methods[1]  # decorated property
    assert lines[value.start_line - 1].strip() == "@property"
    for el in parsed.elements:
        assert el.source == "\n".join(lines[el.start_line - 1:el.end_line])


def test_elements_sorted_by_line(parsed):
    names = [e.qualified_name for e in parsed.elements]
    assert names == ["greet", "fetch", "Calculator", "Calculator.__init__",
                     "Calculator.value", "Calculator.add"]


def test_windows_line_endings(tmp_path):
    f = tmp_path / "crlf.py"
    f.write_bytes(b"import os\r\n\r\ndef f():\r\n    return 1\r\n")
    result = CodeParser().parse_file(f)
    fn = result.functions[0]
    assert (fn.start_line, fn.end_line) == (3, 4)
    assert fn.source == "def f():\n    return 1"


def test_utf8_bom_and_unicode(tmp_path):
    f = tmp_path / "uni.py"
    f.write_bytes(b"\xef\xbb\xbfdef f():\n    return 'h\xc3\xa9llo'\n")
    result = CodeParser().parse_file(f)
    assert result.ok and "héllo" in result.functions[0].source


def test_empty_file(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")
    result = CodeParser().parse_file(f)
    assert result.status == "empty" and not result.ok
    assert result.content == "" and result.line_count == 0
    assert result.imports == [] and result.functions == [] and result.classes == []
    assert result.error


def test_whitespace_only_file_is_empty(tmp_path):
    f = tmp_path / "ws.py"
    f.write_text("  \n\n\t\n", encoding="utf-8")
    assert CodeParser().parse_file(f).status == "empty"


def test_invalid_syntax(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n    pass\n", encoding="utf-8")
    result = CodeParser().parse_file(f)
    assert result.status == "syntax_error" and not result.ok
    assert result.error_line == 1 and "line 1" in result.error
    assert result.content == "def broken(:\n    pass\n"  # content still preserved
    assert result.functions == [] and result.classes == []


def test_null_bytes_reported_as_error(tmp_path):
    f = tmp_path / "nul.py"
    f.write_bytes(b"x = 1\x00\n")
    assert CodeParser().parse_file(f).status == "syntax_error"


def test_missing_file(tmp_path):
    with pytest.raises(ParserFileNotFoundError) as info:
        CodeParser().parse_file(tmp_path / "nope.py")
    assert "nope.py" in str(info.value)
    assert isinstance(info.value, FileNotFoundError) and isinstance(info.value, ParserError)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError) as info:
        CodeParser().parse_file(f)
    assert ".txt" in str(info.value) and ".py" in str(info.value)


def test_unsupported_extension_checked_even_if_missing(tmp_path):
    with pytest.raises(UnsupportedFileTypeError):
        CodeParser().parse_file(tmp_path / "ghost.java")


def test_extension_case_insensitive(tmp_path):
    f = tmp_path / "UP.PY"
    f.write_text("x = 1\n", encoding="utf-8")
    assert CodeParser().parse_file(f).language == "python"


def test_encoding_problem(tmp_path):
    f = tmp_path / "latin.py"
    f.write_bytes(b"# caf\xe9\nx = 1\n")  # invalid UTF-8
    with pytest.raises(FileReadError) as info:
        CodeParser().parse_file(f)
    assert "UTF-8" in str(info.value)


def test_directory_path_is_rejected(tmp_path):
    d = tmp_path / "pkg.py"
    d.mkdir()
    with pytest.raises(FileReadError):
        CodeParser().parse_file(d)


def test_file_too_large(tmp_path, monkeypatch):
    import src.parser as parser_module
    monkeypatch.setattr(parser_module, "MAX_FILE_BYTES", 10)
    f = tmp_path / "big.py"
    f.write_text("x = 1\n" * 10, encoding="utf-8")
    with pytest.raises(FileReadError, match="too large"):
        CodeParser().parse_file(f)


def test_accepts_str_path(sample_file):
    assert CodeParser().parse_file(str(sample_file)).ok


def test_code_is_never_executed(tmp_path):
    marker = tmp_path / "executed.txt"
    f = tmp_path / "evil.py"
    f.write_text(
        f"open({str(marker)!r}, 'w').write('x')\n"
        "def f():\n    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    result = CodeParser().parse_file(f)
    assert result.ok and [fn.name for fn in result.functions] == ["f"]
    assert not marker.exists()


def test_parse_directory(tmp_path):
    (tmp_path / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("def b(:\n", encoding="utf-8")
    (tmp_path / "sub" / "readme.md").write_text("# hi", encoding="utf-8")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "skip.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "latin.py").write_bytes(b"# caf\xe9\n")
    results = CodeParser().parse_directory(tmp_path)
    assert [r.file_name for r in results] == ["a.py", "b.py"]  # latin1 + venv + md skipped
    assert [r.status for r in results] == ["ok", "syntax_error"]
    non_recursive = CodeParser().parse_directory(tmp_path, recursive=False)
    assert [r.file_name for r in non_recursive] == ["a.py"]


def test_parse_directory_errors(tmp_path):
    with pytest.raises(ParserFileNotFoundError):
        CodeParser().parse_directory(tmp_path / "missing")
    f = tmp_path / "f.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(FileReadError):
        CodeParser().parse_directory(f)


def test_is_supported():
    assert CodeParser.is_supported("a.py") and not CodeParser.is_supported("a.js")


def test_extracts_direct_dotted_and_repeated_function_calls(tmp_path):
    f = tmp_path / "calls.py"
    f.write_text(
        "import math\n"
        "def calculate_total(price, tax):\n"
        "    add_tax(price, tax)\n"
        "    add_tax(price, tax)\n"
        "    math.sqrt(price)\n"
        "    return helpers.money.round_total(price)\n",
        encoding="utf-8",
    )
    parsed = CodeParser().parse_file(f)
    assert parsed.functions[0].calls == [
        "add_tax", "add_tax", "math.sqrt", "helpers.money.round_total"
    ]


def test_extracts_method_calls_and_self_calls(tmp_path):
    f = tmp_path / "methods.py"
    f.write_text(
        "class Worker:\n"
        "    def run(self):\n"
        "        self.prepare()\n"
        "        service.execute()\n"
        "    def prepare(self):\n"
        "        initialize()\n",
        encoding="utf-8",
    )
    parsed = CodeParser().parse_file(f)
    run, prepare = parsed.classes[0].methods
    assert run.calls == ["self.prepare", "service.execute"]
    assert prepare.calls == ["initialize"]
    assert parsed.classes[0].calls == []


def test_nested_function_calls_are_not_attributed_to_outer_function(tmp_path):
    f = tmp_path / "nested.py"
    f.write_text(
        "def outer():\n"
        "    before()\n"
        "    def inner():\n"
        "        nested_only()\n"
        "    after()\n",
        encoding="utf-8",
    )
    parsed = CodeParser().parse_file(f)
    assert parsed.functions[0].calls == ["before", "after"]


def test_structured_call_metadata_includes_name_line_and_caller(tmp_path):
    f = tmp_path / "call_metadata.py"
    f.write_text(
        "def calculate():\n"
        "    validate()\n"
        "    math.sqrt(25)\n",
        encoding="utf-8",
    )

    parsed = CodeParser().parse_file(f)
    element = parsed.functions[0]

    assert element.calls == ["validate", "math.sqrt"]
    assert [(call.name, call.line, call.caller) for call in element.call_infos] == [
        ("validate", 2, "calculate"),
        ("math.sqrt", 3, "calculate"),
    ]


def test_structured_call_metadata_uses_qualified_method_name_and_async_body(tmp_path):
    f = tmp_path / "async_calls.py"
    f.write_text(
        "class Worker:\n"
        "    async def run(self):\n"
        "        await self.prepare()\n"
        "    def prepare(self):\n"
        "        initialize()\n",
        encoding="utf-8",
    )

    parsed = CodeParser().parse_file(f)
    run, prepare = parsed.classes[0].methods

    assert [(call.name, call.line, call.caller) for call in run.call_infos] == [
        ("self.prepare", 3, "Worker.run")
    ]
    assert [(call.name, call.line, call.caller) for call in prepare.call_infos] == [
        ("initialize", 5, "Worker.prepare")
    ]


def test_nested_call_expressions_keep_each_call_line(tmp_path):
    f = tmp_path / "nested_calls.py"
    f.write_text(
        "def calculate():\n"
        "    return outer(inner())\n",
        encoding="utf-8",
    )

    element = CodeParser().parse_file(f).functions[0]
    assert [(call.name, call.line, call.caller) for call in element.call_infos] == [
        ("outer", 2, "calculate"),
        ("inner", 2, "calculate"),
    ]


def test_phase2a_resolves_unique_local_function_and_method_targets(tmp_path):
    f = tmp_path / "relationships.py"
    f.write_text(
        "def validate():\n"
        "    pass\n"
        "\n"
        "class Worker(BaseWorker):\n"
        "    def run(self):\n"
        "        validate()\n"
        "        self.prepare()\n"
        "        Worker.finish()\n"
        "    def prepare(self):\n"
        "        pass\n"
        "    @classmethod\n"
        "    def finish(cls):\n"
        "        cls.prepare()\n",
        encoding="utf-8",
    )

    parsed = CodeParser().parse_file(f)
    worker = parsed.classes[0]
    run, prepare, finish = worker.methods

    assert worker.base_classes == ["BaseWorker"]
    assert [(c.name, c.target) for c in run.call_infos] == [
        ("validate", "validate"),
        ("self.prepare", "Worker.prepare"),
        ("Worker.finish", "Worker.finish"),
    ]
    assert [(c.name, c.target) for c in finish.call_infos] == [("cls.prepare", "Worker.prepare")]


def test_phase2a_leaves_ambiguous_and_external_calls_unresolved(tmp_path):
    f = tmp_path / "ambiguous.py"
    f.write_text(
        "def same():\n    pass\n"
        "def same():\n    pass\n"
        "class A:\n    def run(self):\n        same()\n        external.call()\n        dynamic().call()\n",
        encoding="utf-8",
    )

    parsed = CodeParser().parse_file(f)
    calls = parsed.classes[0].methods[0].call_infos
    assert [(c.name, c.target) for c in calls] == [
        ("same", None),
        ("external.call", None),
        ("dynamic", None),
    ]


def test_phase2a_does_not_resolve_inherited_methods_by_guessing(tmp_path):
    f = tmp_path / "inheritance.py"
    f.write_text(
        "class Base:\n    def inherited(self):\n        pass\n"
        "class Child(Base):\n    def run(self):\n        self.inherited()\n",
        encoding="utf-8",
    )

    parsed = CodeParser().parse_file(f)
    child = parsed.classes[1]
    assert child.base_classes == ["Base"]
    assert child.methods[0].call_infos[0].target is None

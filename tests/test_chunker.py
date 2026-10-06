"""Tests for src/chunker.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunker import ChunkerError, CodeChunk, CodeChunker  # noqa: E402
from src.parser import CodeElement, CodeParser, ParsedFile  # noqa: E402

SAMPLE = '''\
"""Calculator module."""
import os
from pathlib import Path

MAX_VALUE = 100


def greet(name):
    """Say hello."""
    return f"hello {name}"


class Calculator:
    """A calculator."""

    def __init__(self):
        self.total = 0

    async def add(self, x):
        self.total += x
        return self.total


def main():
    print(greet("x"))


if __name__ == "__main__":
    main()
'''


def write(tmp_path: Path, text: str, name: str = "sample.py") -> Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


def chunks_for(tmp_path: Path, text: str, name: str = "sample.py") -> list[CodeChunk]:
    parsed = CodeParser().parse_file(write(tmp_path, text, name))
    return CodeChunker().chunk(parsed)


def by_type(chunks, kind):
    return [c for c in chunks if c.chunk_type == kind]


@pytest.fixture()
def sample_chunks(tmp_path):
    return chunks_for(tmp_path, SAMPLE)


# 1. simple file / 15. parser output passed directly
def test_simple_file_produces_chunks(tmp_path):
    chunks = chunks_for(tmp_path, "def f():\n    return 1\n")
    assert len(chunks) == 1 and chunks[0].chunk_type == "function"
    assert all(isinstance(c, CodeChunk) for c in chunks)


def test_parser_output_goes_straight_into_chunker(tmp_path):
    parsed = CodeParser().parse_file(write(tmp_path, SAMPLE))
    assert isinstance(parsed, ParsedFile)
    assert len(CodeChunker().chunk(parsed)) > 0


def test_required_fields_present(sample_chunks):
    for c in sample_chunks:
        for attr in ("chunk_id", "file_path", "language", "chunk_type", "name",
                     "qualified_name", "source", "start_line", "end_line", "metadata"):
            assert getattr(c, attr) not in (None, "") or attr == "metadata"
        assert c.language == "python"


# 2. imports
def test_imports_chunk(sample_chunks):
    (imp,) = by_type(sample_chunks, "imports")
    assert imp.source == "import os\nfrom pathlib import Path"
    assert (imp.start_line, imp.end_line) == (2, 3)
    assert imp.metadata["import_count"] == 2
    assert imp.metadata["imported_modules"] == "os,pathlib"


def test_nested_imports_not_in_imports_chunk(tmp_path):
    chunks = chunks_for(tmp_path, "import os\n\ndef f():\n    import json\n    return json\n")
    (imp,) = by_type(chunks, "imports")
    assert imp.source == "import os"
    (fn,) = by_type(chunks, "function")
    assert "import json" in fn.source


def test_only_imports_file(tmp_path):
    chunks = chunks_for(tmp_path, "import os\nimport sys\n")
    assert [c.chunk_type for c in chunks] == ["imports"]
    assert chunks[0].source == "import os\nimport sys"


def test_scattered_imports_are_gathered(tmp_path):
    chunks = chunks_for(tmp_path, "import os\nX = 1\nimport sys\n")
    (imp,) = by_type(chunks, "imports")
    assert imp.source == "import os\nimport sys" and (imp.start_line, imp.end_line) == (1, 3)
    (mod,) = by_type(chunks, "module_code")
    assert mod.source == "X = 1" and mod.start_line == 2


def test_indented_import_stays_in_module_code(tmp_path):
    text = "try:\n    import json\nexcept ImportError:\n    json = None\n"
    chunks = chunks_for(tmp_path, text)
    assert by_type(chunks, "imports") == []
    (mod,) = by_type(chunks, "module_code")
    assert mod.source == text.rstrip("\n")


# 3. function
def test_top_level_function_chunk(sample_chunks):
    greet = next(c for c in sample_chunks if c.name == "greet")
    assert greet.chunk_type == "function" and greet.qualified_name == "greet"
    assert greet.parent is None
    assert greet.metadata["has_docstring"] is True
    assert greet.metadata["parent"] == ""


# 4. class
def test_class_chunk(sample_chunks):
    cls = next(c for c in sample_chunks if c.chunk_type == "class")
    assert cls.name == "Calculator"
    assert "def __init__" in cls.source and "async def add" in cls.source  # whole class kept
    assert cls.metadata["method_count"] == 2
    assert cls.metadata["method_names"] == "__init__,add"


# 5/6. method + metadata
def test_method_chunk_keeps_class_identity(sample_chunks, tmp_path):
    methods = by_type(sample_chunks, "method")
    assert [m.qualified_name for m in methods] == ["Calculator.__init__", "Calculator.add"]
    add = methods[1]
    assert add.name == "add" and add.parent == "Calculator"
    assert add.metadata["parent"] == "Calculator"
    assert add.metadata["class_name"] == "Calculator"
    assert add.metadata["qualified_name"] == "Calculator.add"
    assert add.metadata["is_async"] is True
    assert add.metadata["file_path"] == add.file_path
    assert add.metadata["file_name"] == "sample.py"
    assert add.file_path.endswith("sample.py") and "\\" not in add.file_path
    assert "# Class: Calculator" in add.text and "# Method: Calculator.add" in add.text
    assert add.text.endswith(add.source)


# 7. source preserved
def test_source_preserved(sample_chunks):
    lines = SAMPLE.split("\n")
    for c in sample_chunks:
        if c.chunk_type != "imports":  # imports are condensed statements
            assert c.source == "\n".join(lines[c.start_line - 1:c.end_line]), c.qualified_name
    add = next(c for c in sample_chunks if c.qualified_name == "Calculator.add")
    assert add.source == "    async def add(self, x):\n        self.total += x\n        return self.total"


# 8. line numbers
def test_line_numbers(sample_chunks):
    spans = {c.qualified_name: (c.start_line, c.end_line) for c in sample_chunks
             if c.chunk_type in ("function", "class", "method")}
    assert spans == {
        "greet": (8, 10),
        "Calculator": (13, 21),
        "Calculator.__init__": (16, 17),
        "Calculator.add": (19, 21),
        "main": (24, 25),
    }
    for c in sample_chunks:
        assert c.metadata["start_line"] == c.start_line and c.metadata["end_line"] == c.end_line
        assert c.metadata["line_count"] == c.end_line - c.start_line + 1


# 9. module-level code is not lost
def test_module_level_code_is_kept(sample_chunks):
    mods = by_type(sample_chunks, "module_code")
    sources = [m.source for m in mods]
    assert '"""Calculator module."""' in sources
    assert "MAX_VALUE = 100" in sources
    assert 'if __name__ == "__main__":\n    main()' in sources


def test_only_module_level_code(tmp_path):
    chunks = chunks_for(tmp_path, "x = 1\nprint(x)\n")
    assert [c.chunk_type for c in chunks] == ["module_code"]
    assert chunks[0].source == "x = 1\nprint(x)" and (chunks[0].start_line, chunks[0].end_line) == (1, 2)


def test_no_code_line_is_lost(sample_chunks):
    covered = set()
    for c in sample_chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    for number, line in enumerate(SAMPLE.split("\n"), start=1):
        if line.strip() and not line.strip().startswith("#"):
            assert number in covered, f"line {number} lost: {line!r}"


# 10. multiple definitions
def test_multiple_definitions(tmp_path):
    text = ("def a():\n    pass\n\n\ndef b():\n    pass\n\n\n"
            "class X:\n    def m(self):\n        pass\n\n\nclass Y:\n    pass\n")
    chunks = chunks_for(tmp_path, text)
    assert [(c.chunk_type, c.qualified_name) for c in chunks] == [
        ("function", "a"), ("function", "b"), ("class", "X"), ("method", "X.m"), ("class", "Y"),
    ]


def test_class_without_methods(tmp_path):
    chunks = chunks_for(tmp_path, "class Empty:\n    x = 1\n")
    assert [c.chunk_type for c in chunks] == ["class"]
    assert chunks[0].metadata["method_count"] == 0 and chunks[0].metadata["method_names"] == ""


def test_decorated_function_includes_decorator(tmp_path):
    chunks = chunks_for(tmp_path, "import functools\n\n@functools.cache\ndef f():\n    return 1\n")
    fn = by_type(chunks, "function")[0]
    assert fn.source.startswith("@functools.cache") and fn.start_line == 3


# 11. empty files
@pytest.mark.parametrize("content", ["", "   \n\n\t\n"])
def test_empty_file_produces_no_chunks(tmp_path, content):
    assert chunks_for(tmp_path, content) == []


# 12. syntax error fallback
def test_syntax_error_fallback_chunk(tmp_path):
    text = "def broken(:\n    pass\n"
    chunks = chunks_for(tmp_path, text, "bad.py")
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_type == "file" and c.source == text
    assert (c.start_line, c.end_line) == (1, 2)
    assert c.name == "bad.py"
    assert c.metadata["parse_status"] == "syntax_error"
    assert c.metadata["fallback_reason"] == "syntax_error"
    assert c.metadata["error_line"] == 1 and "syntax" in c.metadata["error"].lower()


def test_comment_only_file_falls_back_to_whole_file(tmp_path):
    chunks = chunks_for(tmp_path, "# just a note\n# another\n")
    assert len(chunks) == 1 and chunks[0].chunk_type == "file"
    assert chunks[0].metadata["fallback_reason"] == "no_structure"
    assert chunks[0].metadata["parse_status"] == "ok"


# 13/14. IDs
def test_chunk_ids_deterministic(tmp_path):
    first = [c.chunk_id for c in chunks_for(tmp_path, SAMPLE)]
    second = [c.chunk_id for c in chunks_for(tmp_path, SAMPLE)]
    assert first == second and all(len(i) == 32 for i in first)


def test_chunk_ids_stable_across_processes(tmp_path):
    f = write(tmp_path, SAMPLE)
    code = (
        "import sys; sys.path.insert(0, sys.argv[2]);"
        "from src.parser import CodeParser; from src.chunker import CodeChunker;"
        "print(','.join(c.chunk_id for c in CodeChunker().chunk(CodeParser().parse_file(sys.argv[1]))))"
    )
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run([sys.executable, "-c", code, str(f), str(ROOT)],
                             capture_output=True, text=True, env=env, cwd=ROOT, check=True)
        outputs.append(res.stdout.strip().splitlines()[-1])
    assert outputs[0] == outputs[1]
    assert outputs[0].split(",") == [c.chunk_id for c in chunks_for(tmp_path, SAMPLE)]


def test_chunk_ids_are_distinct_within_file(sample_chunks):
    ids = [c.chunk_id for c in sample_chunks]
    assert len(ids) == len(set(ids))


def test_chunk_id_changes_with_content_or_path(tmp_path):
    base = chunks_for(tmp_path, "def f():\n    return 1\n")[0].chunk_id
    edited = chunks_for(tmp_path, "def f():\n    return 2\n")[0].chunk_id
    other_file = chunks_for(tmp_path, "def f():\n    return 1\n", "other.py")[0].chunk_id
    assert len({base, edited, other_file}) == 3


# metadata / ordering / misc
def test_metadata_is_chroma_safe(sample_chunks, tmp_path):
    allchunks = sample_chunks + chunks_for(tmp_path, "def broken(:\n", "bad.py")
    for c in allchunks:
        for key, value in c.metadata.items():
            assert isinstance(value, (str, int, float, bool)), (c.qualified_name, key, value)


def test_chunks_ordered_and_indexed(sample_chunks):
    starts = [c.start_line for c in sample_chunks]
    assert starts == sorted(starts)
    assert [c.metadata["chunk_index"] for c in sample_chunks] == list(range(len(sample_chunks)))
    assert all(c.metadata["total_chunks"] == len(sample_chunks) for c in sample_chunks)


def test_to_dict(sample_chunks):
    d = sample_chunks[0].to_dict()
    assert d["chunk_id"] == sample_chunks[0].chunk_id and isinstance(d["metadata"], dict)


def test_chunker_uses_parser_line_ranges_not_reparsing(tmp_path):
    """Chunk text comes from the parser's CodeElement, proving no independent re-parse."""
    path = tmp_path / "fake.py"
    element = CodeElement(kind="function", name="f", qualified_name="f", start_line=1, end_line=1,
                          source="PARSER_SUPPLIED_SOURCE", file_path=path, language="python")
    parsed = ParsedFile(path=path, language="python", content="def f(): pass\n",
                        functions=[element], line_count=1)
    (chunk,) = CodeChunker().chunk(parsed)
    assert chunk.source == "PARSER_SUPPLIED_SOURCE" and (chunk.start_line, chunk.end_line) == (1, 1)


def test_chunk_many(tmp_path):
    a = CodeParser().parse_file(write(tmp_path, "def a():\n    pass\n", "a.py"))
    b = CodeParser().parse_file(write(tmp_path, "def b():\n    pass\n", "b.py"))
    chunks = CodeChunker().chunk_many([a, b])
    assert [c.name for c in chunks] == ["a", "b"]


def test_rejects_wrong_input():
    with pytest.raises(ChunkerError):
        CodeChunker().chunk("not a parsed file")  # type: ignore[arg-type]


def test_crlf_file(tmp_path):
    f = tmp_path / "crlf.py"
    f.write_bytes(b"import os\r\n\r\nX = 1\r\n\r\ndef f():\r\n    return X\r\n")
    chunks = CodeChunker().chunk(CodeParser().parse_file(f))
    assert [(c.chunk_type, c.source) for c in chunks] == [
        ("imports", "import os"), ("module_code", "X = 1"), ("function", "def f():\n    return X"),
    ]


def test_code_not_executed(tmp_path):
    marker = tmp_path / "ran.txt"
    chunks = chunks_for(tmp_path, f"open({str(marker)!r}, 'w').write('x')\n", "evil.py")
    assert chunks and not marker.exists()

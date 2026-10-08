"""Tests for the Streamlit UI integration layer (Phase 1I).

Validates UI helper functions, business logic, ZIP safety,
Streamlit app initialization, and security guarantees.
Does NOT require a running Ollama server, network access, or GPU.
"""

from __future__ import annotations

import ast
import io
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from app import (
    ALLOWED_TYPES,
    SECTIONS,
    extract_safe_zip,
    format_source_label,
    index_codebase_files,
    main,
    render_ask,
    render_docs,
    render_upload,
    save_uploaded_files,
)
from src.chunker import CodeChunk
from src.retriever import RetrievalResult

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[CodeChunk]] = []

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        self.calls.append(chunks)
        return [[0.1, 0.2, 0.3] for _ in chunks]


class FakeVectorStore:
    def __init__(self) -> None:
        self.cleared = False
        self.stored_chunks: list[CodeChunk] = []
        self.stored_embeddings: list[list[float]] = []

    def clear(self) -> None:
        self.cleared = True
        self.stored_chunks.clear()
        self.stored_embeddings.clear()

    def add_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        self.stored_chunks.extend(chunks)
        self.stored_embeddings.extend(embeddings)

    def count(self) -> int:
        return len(self.stored_chunks)


class FakeUploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


# --------------------------------------------------------------------------- #
# 1. Structure & Import Tests
# --------------------------------------------------------------------------- #
def test_app_imports_and_constants() -> None:
    assert "Upload Code" in SECTIONS
    assert "Ask Code" in SECTIONS
    assert "Documentation" in SECTIONS
    assert "py" in ALLOWED_TYPES
    assert "zip" in ALLOWED_TYPES


def test_required_functions_exist() -> None:
    funcs = [main, render_upload, render_ask, render_docs]
    for func in funcs:
        assert callable(func)


# --------------------------------------------------------------------------- #
# 2. ZIP Extraction & Security (Zip Slip Prevention)
# --------------------------------------------------------------------------- #
def test_extract_safe_zip_extracts_python_files(tmp_path: Path) -> None:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("module/calculator.py", "def add(a, b): return a + b\n")
        zf.writestr("module/readme.txt", "This is documentation.\n")
        zf.writestr("module/run.sh", "#!/bin/bash\necho bad\n")

    extracted = extract_safe_zip(zip_buffer.getvalue(), tmp_path / "extracted")
    assert len(extracted) == 1
    assert extracted[0].name == "calculator.py"
    assert extracted[0].read_text(encoding="utf-8") == "def add(a, b): return a + b\n"
    # Non-Python files must be skipped
    assert not (tmp_path / "extracted" / "module" / "readme.txt").exists()
    assert not (tmp_path / "extracted" / "module" / "run.sh").exists()


def test_extract_safe_zip_blocks_path_traversal(tmp_path: Path) -> None:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Malicious Zip Slip paths
        zf.writestr("../../evil.py", "print('malicious')\n")
        zf.writestr("safe.py", "print('safe')\n")

    target_dir = tmp_path / "safe_dir"
    extracted = extract_safe_zip(zip_buffer.getvalue(), target_dir)

    # Must only extract the file inside target_dir
    assert len(extracted) == 1
    assert extracted[0].name == "safe.py"
    assert not (tmp_path / "evil.py").exists()


# --------------------------------------------------------------------------- #
# 3. File Upload & Inert Storage
# --------------------------------------------------------------------------- #
def test_save_uploaded_files_py_and_zip(tmp_path: Path) -> None:
    py_content = b"def greet(): pass\n"
    py_file = FakeUploadedFile("hello.py", py_content)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("archive_code.py", "x = 10\n")
    zip_file = FakeUploadedFile("code.zip", zip_buffer.getvalue())

    saved = save_uploaded_files([py_file, zip_file], tmp_path)
    file_names = {p.name for p in saved}
    assert "hello.py" in file_names
    assert "archive_code.py" in file_names


# --------------------------------------------------------------------------- #
# 4. Codebase Indexing Flow
# --------------------------------------------------------------------------- #
def test_index_codebase_files_success(tmp_path: Path) -> None:
    code_file = tmp_path / "math_ops.py"
    code_file.write_text(
        "def square(x):\n    '''Square x.'''\n    return x * x\n",
        encoding="utf-8",
    )

    embedder = FakeEmbedder()
    store = FakeVectorStore()
    progress_updates: list[str] = []

    def callback(msg: str, frac: float) -> None:
        progress_updates.append(msg)

    parsed_count, chunk_count = index_codebase_files(
        [code_file],
        embedder=embedder,  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        progress_callback=callback,
    )

    assert parsed_count == 1
    assert chunk_count >= 1
    assert store.cleared is True
    assert store.count() == chunk_count
    assert len(embedder.calls) == 1
    assert any("Parsing" in msg for msg in progress_updates)
    assert any("Storing" in msg for msg in progress_updates)


def test_index_codebase_files_empty_raises(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="No valid Python"):
        index_codebase_files(
            [],
            embedder=embedder,  # type: ignore[arg-type]
            vector_store=store,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# 5. UI Formatting
# --------------------------------------------------------------------------- #
def test_format_source_label() -> None:
    source = RetrievalResult(
        chunk_id="c1",
        content="def auth(): pass",
        file_path="src/auth.py",
        chunk_type="function",
        name="auth",
        start_line=10,
        end_line=12,
    )
    label = format_source_label(source, index=1)
    assert "Source 1: src/auth.py" in label
    assert "function: auth" in label
    assert "Lines 10-12" in label


# --------------------------------------------------------------------------- #
# 6. Streamlit AppTest Execution
# --------------------------------------------------------------------------- #
def test_streamlit_app_renders_main_page() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"))
    at.run()
    assert not at.exception
    # Verify title and sidebar elements
    title_elements = [t.value for t in at.title]
    assert any("CodeSage" in t for t in title_elements)


def test_streamlit_app_navigation_sections() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"))
    at.run()

    # Default is "Upload Code"
    assert any("Upload & Index" in s.value for s in at.subheader)

    # Switch to "Ask Code"
    at.sidebar.radio[0].set_value("Ask Code").run()
    assert not at.exception
    assert any("Ask Codebase" in s.value for s in at.subheader)

    # Switch to "Documentation"
    at.sidebar.radio[0].set_value("Documentation").run()
    assert not at.exception
    assert any("Documentation" in s.value for s in at.subheader)


def test_streamlit_app_ask_without_index_shows_notice() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"))
    at.run()
    at.sidebar.radio[0].set_value("Ask Code").run()

    # Informational notice that codebase is not indexed
    info_messages = [i.value for i in at.info]
    assert any("No codebase indexed yet" in m for m in info_messages)


# --------------------------------------------------------------------------- #
# 7. Security: Inert Code & No Code Execution Primitives
# --------------------------------------------------------------------------- #
def test_no_code_execution_primitives_in_app() -> None:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "exec" not in called_names
    assert "eval" not in called_names

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert "subprocess" not in imported_modules
    assert "os.system" not in src


def test_uploaded_python_code_never_executed(tmp_path: Path) -> None:
    malicious_code = b"import sys, os\nos.environ['PWN_APP_TEST'] = '1'\nraise SystemExit('pwned')\n"
    uploaded = FakeUploadedFile("exploit.py", malicious_code)

    saved = save_uploaded_files([uploaded], tmp_path)
    assert len(saved) == 1
    # Verify file was written as text/bytes only and not imported or executed
    assert saved[0].read_bytes() == malicious_code
    import os

    assert "PWN_APP_TEST" not in os.environ

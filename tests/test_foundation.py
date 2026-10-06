"""Foundation tests: structure, configuration, imports, app validity."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_DIRS = ["src", "data/uploads", "data/parsed", "data/chunks", "chroma_db",
                 "logs", "tests", "docs", "images", "notebooks"]
REQUIRED_FILES = ["app.py", "config.py", "requirements.txt", ".env.example", ".gitignore",
                  "README.md", "src/__init__.py", "tests/__init__.py"] + [
    f"src/{m}.py" for m in ["utils", "parser", "chunker", "embedder", "vectordb",
                            "retriever", "llm", "prompts", "rag_pipeline"]]
MODULES = ["utils", "parser", "chunker", "embedder", "vectordb", "retriever", "llm",
           "prompts", "rag_pipeline"]


@pytest.fixture(scope="module", autouse=True)
def _runtime_dirs():
    """chroma_db/ and logs/ are git-ignored, so create them like the app does."""
    from src.utils import ensure_directories
    ensure_directories()


@pytest.mark.parametrize("rel", REQUIRED_DIRS)
def test_required_directories_exist(rel):
    assert (ROOT / rel).is_dir()


@pytest.mark.parametrize("rel", REQUIRED_FILES)
def test_required_files_exist(rel):
    assert (ROOT / rel).is_file()


def test_config_loads(monkeypatch):
    import config
    cfg = config.get_config()
    assert cfg.embedding_model and cfg.ollama_host and cfg.ollama_model
    assert cfg.chroma_path.is_absolute()
    monkeypatch.setenv("CODESAGE_OLLAMA_MODEL", "custom")
    assert config.get_config().ollama_model == "custom"


@pytest.mark.parametrize("name", MODULES)
def test_source_modules_import(name):
    importlib.import_module(f"src.{name}")


def test_placeholders_do_not_fake_rag():
    from src.rag_pipeline import RAGPipeline
    with pytest.raises(NotImplementedError):
        RAGPipeline().ask("hi")


def test_safe_filename():
    from src.utils import safe_filename
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\x\\a b.py") == "a_b.py"


def test_app_structure_valid():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"main", "render_upload", "render_ask", "render_docs"} <= funcs


def test_gitignore_entries():
    text = (ROOT / ".gitignore").read_text()
    for entry in ["venv/", ".env", "__pycache__/", "*.pyc", "chroma_db/", "logs/"]:
        assert entry in text.splitlines()

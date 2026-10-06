"""Tests for src/embedder.py.

Most tests use a *real* but tiny Sentence Transformers model that is built
locally (a 1-layer BERT with a character vocabulary and fixed random weights).
It goes through the genuine ``SentenceTransformer`` code path, needs no network
and no GPU, but its vectors are not semantically meaningful.

One integration test uses the real default model (``all-MiniLM-L6-v2``); it is
skipped when the model cannot be downloaded/loaded, or when
``CODESAGE_SKIP_MODEL_DOWNLOAD_TESTS=1`` is set.
"""

from __future__ import annotations

import math
import os
import string
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.chunker import CodeChunk, CodeChunker  # noqa: E402
from src.embedder import (  # noqa: E402
    Embedder,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingModelError,
)
from src.parser import CodeParser  # noqa: E402

TINY_DIM = 32

SAMPLE = '''\
import os


def greet(name):
    """Say hello."""
    return f"hello {name}"


class Calculator:
    def add(self, x):
        return x + 1
'''


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> str:
    """Build a tiny real Sentence Transformers model on disk (CPU, offline)."""
    pytest.importorskip("sentence_transformers")
    import torch
    from transformers import BertConfig, BertModel, BertTokenizer

    directory = tmp_path_factory.mktemp("tiny_model")
    chars = [c for c in string.ascii_lowercase + string.digits + string.punctuation]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    (directory / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizer(str(directory / "vocab.txt"), do_lower_case=True).save_pretrained(directory)
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(vocab), hidden_size=TINY_DIM, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
    BertModel(config).save_pretrained(directory)
    return str(directory)


@pytest.fixture(scope="module")
def embedder(tiny_model_dir) -> Embedder:
    return Embedder(model_name=tiny_model_dir)


@pytest.fixture()
def chunks(tmp_path) -> list[CodeChunk]:
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE, encoding="utf-8")
    return CodeChunker().chunk(CodeParser().parse_file(f))


def is_vector(v, dim: int) -> bool:
    return (isinstance(v, list) and len(v) == dim
            and all(isinstance(x, float) and math.isfinite(x) for x in v))


# --------------------------------------------------------------------------- #
# Import / construction (no model needed)
# --------------------------------------------------------------------------- #
def test_module_imports_without_loading_heavy_libraries():
    import subprocess
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import src.embedder as m; "
            "print(all(hasattr(m, n) for n in ('Embedder','EmbeddingError','EmbeddingModelError',"
            "'EmbeddingInputError')), 'torch' in sys.modules, 'sentence_transformers' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True,
                         cwd=ROOT, check=True).stdout.split()[-3:]
    assert out == ["True", "False", "False"]


def test_create_embedder_defaults_and_lazy_loading():
    e = Embedder()
    assert e.model_name == settings.embedding_model
    assert e.device == "cpu" and e.batch_size == 32 and e.normalize is True
    assert not e.is_loaded  # nothing loaded until first use


def test_model_name_comes_from_environment(monkeypatch):
    monkeypatch.setenv("CODESAGE_EMBEDDING_MODEL", "some/other-model")
    import config
    assert config.get_config().embedding_model == "some/other-model"
    assert Embedder(model_name=config.get_config().embedding_model).model_name == "some/other-model"


@pytest.mark.parametrize("bad", ["", "   "])
def test_invalid_model_name_rejected(bad):
    with pytest.raises(EmbeddingInputError):
        Embedder(model_name=bad)


def test_invalid_batch_size_rejected():
    with pytest.raises(EmbeddingInputError):
        Embedder(batch_size=0)


# --------------------------------------------------------------------------- #
# Empty / invalid input (no model needed, and the model must not be loaded)
# --------------------------------------------------------------------------- #
def test_empty_chunk_list_returns_empty_without_loading():
    e = Embedder(model_name="never-loaded")
    assert e.embed_chunks([]) == [] and e.embed_texts([]) == [] and e.embed_chunks(()) == []
    assert not e.is_loaded


@pytest.mark.parametrize("bad", ["", "   \n\t", None, 123, b"bytes"])
def test_invalid_single_text(bad):
    e = Embedder(model_name="never-loaded")
    with pytest.raises(EmbeddingInputError):
        e.embed_text(bad)  # type: ignore[arg-type]
    assert not e.is_loaded  # validation happens before any model load


def test_blank_item_in_list_names_its_index():
    e = Embedder(model_name="never-loaded")
    with pytest.raises(EmbeddingInputError, match=r"texts\[1\]"):
        e.embed_texts(["fine", "  "])
    with pytest.raises(EmbeddingInputError, match=r"texts\[0\]"):
        e.embed_texts([5])  # type: ignore[list-item]


@pytest.mark.parametrize("bad", ["a plain string", None, {"a": 1}, 42])
def test_non_sequence_inputs_rejected(bad):
    e = Embedder(model_name="never-loaded")
    with pytest.raises(EmbeddingInputError):
        e.embed_texts(bad)  # type: ignore[arg-type]
    with pytest.raises(EmbeddingInputError):
        e.embed_chunks(bad)  # type: ignore[arg-type]


def test_non_chunk_items_rejected(chunks):
    e = Embedder(model_name="never-loaded")
    with pytest.raises(EmbeddingInputError, match=r"chunks\[1\]"):
        e.embed_chunks([chunks[0], "not a chunk"])  # type: ignore[list-item]
    with pytest.raises(EmbeddingInputError):
        e.embed_chunk("not a chunk")  # type: ignore[arg-type]


def test_chunk_with_blank_text_rejected():
    class BlankTextChunk(CodeChunk):
        @property
        def text(self) -> str:  # type: ignore[override]
            return "   "

    chunk = BlankTextChunk(chunk_id="x", file_path="f.py", language="python", chunk_type="file",
                           name="f", qualified_name="f", source="", start_line=1, end_line=1)
    with pytest.raises(EmbeddingInputError):
        Embedder(model_name="never-loaded").embed_chunk(chunk)


# --------------------------------------------------------------------------- #
# Model loading errors
# --------------------------------------------------------------------------- #
def test_load_failure_gives_clear_error(tmp_path):
    empty_dir = tmp_path / "not_a_model"
    empty_dir.mkdir()
    e = Embedder(model_name=str(empty_dir))
    with pytest.raises(EmbeddingModelError) as info:
        e.embed_text("hello")
    assert str(empty_dir) in str(info.value) and "CODESAGE_EMBEDDING_MODEL" in str(info.value)
    assert not e.is_loaded


def test_load_failure_wraps_library_exception(monkeypatch):
    pytest.importorskip("sentence_transformers")
    import sentence_transformers

    def boom(*args, **kwargs):
        raise OSError("simulated network failure")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", boom)
    with pytest.raises(EmbeddingModelError, match="simulated network failure") as info:
        Embedder(model_name="some/model").load()
    assert isinstance(info.value.__cause__, OSError)


def test_missing_sentence_transformers_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingModelError, match="pip install -r requirements.txt"):
        Embedder(model_name="some/model").load()


# --------------------------------------------------------------------------- #
# Real (tiny) Sentence Transformers model
# --------------------------------------------------------------------------- #
def test_runs_on_cpu(embedder):
    assert embedder.model.device.type == "cpu"


def test_dimension_discovered_from_model(embedder):
    assert embedder.dimension == TINY_DIM


def test_embed_single_chunk(embedder, chunks):
    vector = embedder.embed_chunk(chunks[0])
    assert is_vector(vector, TINY_DIM)
    assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-4)  # normalised


def test_embed_multiple_chunks(embedder, chunks):
    assert len(chunks) >= 4
    vectors = embedder.embed_chunks(chunks)
    assert len(vectors) == len(chunks)
    assert all(is_vector(v, TINY_DIM) for v in vectors)
    assert len({tuple(v) for v in vectors}) == len(vectors)  # different chunks, different vectors


def test_chunk_order_is_preserved(embedder, chunks):
    batch = embedder.embed_chunks(chunks)
    singles = [embedder.embed_chunk(c) for c in chunks]
    for b, s in zip(batch, singles):
        assert b == pytest.approx(s, abs=1e-5)
    reversed_batch = embedder.embed_chunks(list(reversed(chunks)))
    for b, r in zip(batch, reversed(reversed_batch)):
        assert b == pytest.approx(r, abs=1e-5)


def test_order_preserved_with_many_items_and_small_batches(tiny_model_dir):
    texts = [f"item number {i} " + "x" * (i * 7 % 40) for i in range(25)]
    small = Embedder(model_name=tiny_model_dir, batch_size=4)
    batch = small.embed_texts(texts)
    for text, vector in zip(texts, batch):
        assert vector == pytest.approx(small.embed_text(text), abs=1e-5)


def test_embedding_uses_chunk_text_not_source(embedder, chunks):
    chunk = next(c for c in chunks if c.chunk_type == "method")
    assert embedder.embed_chunk(chunk) == embedder.embed_text(chunk.text)
    assert embedder.embed_chunk(chunk) != embedder.embed_text(chunk.source)


def test_deterministic_for_same_text(embedder, tiny_model_dir):
    text = "def add(a, b):\n    return a + b"
    first = embedder.embed_text(text)
    assert embedder.embed_text(text) == first
    assert embedder.embed_texts([text, text])[0] == pytest.approx(first, abs=1e-6)
    fresh = Embedder(model_name=tiny_model_dir)  # separate instance, same model
    assert fresh.embed_text(text) == pytest.approx(first, abs=1e-6)


def test_embed_query_matches_embed_text(embedder):
    assert embedder.embed_query("how does add work?") == embedder.embed_text("how does add work?")


def test_long_text_is_truncated_not_fatal(embedder):
    assert is_vector(embedder.embed_text("word " * 2000), TINY_DIM)


def test_unicode_text(embedder):
    assert is_vector(embedder.embed_text("# héllo wörld ✓ 日本語"), TINY_DIM)


def test_model_is_loaded_only_once(tiny_model_dir, monkeypatch, chunks):
    import sentence_transformers
    real = sentence_transformers.SentenceTransformer
    calls: list[str] = []

    def counting(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", counting)
    e = Embedder(model_name=tiny_model_dir)
    assert calls == [] and not e.is_loaded
    e.embed_text("one")
    e.embed_chunks(chunks)
    e.embed_query("two")
    _ = e.dimension, e.model, e.load()
    assert len(calls) == 1 and e.is_loaded
    assert e.model is e.model


def test_end_to_end_parse_chunk_embed(embedder, tmp_path):
    f = tmp_path / "calc.py"
    f.write_text(SAMPLE, encoding="utf-8")
    chunks = CodeChunker().chunk(CodeParser().parse_file(f))
    vectors = embedder.embed_chunks(chunks)
    assert len(vectors) == len(chunks) and all(is_vector(v, TINY_DIM) for v in vectors)


def test_syntax_error_fallback_chunk_can_be_embedded(embedder, tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n", encoding="utf-8")
    (chunk,) = CodeChunker().chunk(CodeParser().parse_file(f))
    assert is_vector(embedder.embed_chunk(chunk), TINY_DIM)


def test_code_is_never_executed(embedder, tmp_path):
    marker = tmp_path / "ran.txt"
    code = f"open({str(marker)!r}, 'w').write('x')"
    f = tmp_path / "evil.py"
    f.write_text(code + "\n", encoding="utf-8")
    embedder.embed_chunks(CodeChunker().chunk(CodeParser().parse_file(f)))
    assert not marker.exists()


# --------------------------------------------------------------------------- #
# Optional integration test with the real default model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_embedder() -> Embedder:
    if os.getenv("CODESAGE_SKIP_MODEL_DOWNLOAD_TESTS") == "1":
        pytest.skip("CODESAGE_SKIP_MODEL_DOWNLOAD_TESTS=1")
    e = Embedder()  # settings.embedding_model, CPU
    try:
        e.load()
    except EmbeddingModelError as exc:
        pytest.skip(f"Default model '{e.model_name}' unavailable here: {exc}")
    return e


def test_real_default_model_integration(real_embedder, chunks):
    if real_embedder.model_name != "sentence-transformers/all-MiniLM-L6-v2":
        pytest.skip("Dimension check only applies to the default model.")
    assert real_embedder.dimension == 384
    vectors = real_embedder.embed_chunks(chunks)
    assert len(vectors) == len(chunks) and all(is_vector(v, 384) for v in vectors)
    assert real_embedder.embed_chunk(chunks[0]) == pytest.approx(vectors[0], abs=1e-5)
    assert real_embedder.embed_chunks(chunks)[0] == real_embedder.embed_chunks(chunks)[0]

from pathlib import Path

import pytest

from src.chunker import CodeChunk
from src.vectordb import VectorStore, VectorStoreInputError


def make_chunk(
    chunk_id: str = "chunk-1",
    source: str = "def add(a, b):\n    return a + b",
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        file_path="example.py",
        language="python",
        chunk_type="function",
        name="add",
        qualified_name="add",
        source=source,
        start_line=1,
        end_line=2,
        metadata={
            "file_name": "example.py",
            "language": "python",
            "chunk_type": "function",
            "name": "add",
        },
    )


def make_embedding(value: float = 1.0) -> list[float]:
    return [value, 0.0, 0.0]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(
        path=tmp_path / "chroma_db",
        collection="test_codesage",
    )


def test_store_initializes(store: VectorStore) -> None:
    assert store.count() == 0


def test_add_one_chunk(store: VectorStore) -> None:
    chunk = make_chunk()

    store.add_chunks([chunk], [make_embedding()])

    assert store.count() == 1


def test_add_multiple_chunks(store: VectorStore) -> None:
    chunks = [
        make_chunk("chunk-1"),
        make_chunk("chunk-2", "def sub(a, b):\n    return a - b"),
    ]

    store.add_chunks(
        chunks,
        [
            make_embedding(1.0),
            make_embedding(2.0),
        ],
    )

    assert store.count() == 2


def test_get_by_ids(store: VectorStore) -> None:
    chunk = make_chunk()

    store.add_chunks([chunk], [make_embedding()])

    result = store.get_by_ids(["chunk-1"])

    assert result["ids"] == ["chunk-1"]
    assert result["documents"] == [chunk.text]
    assert result["metadatas"][0]["file_name"] == "example.py"


def test_exists(store: VectorStore) -> None:
    store.add_chunks(
        [make_chunk()],
        [make_embedding()],
    )

    result = store.exists(["chunk-1", "missing"])

    assert result["chunk-1"] is True
    assert result["missing"] is False


def test_duplicate_id_uses_upsert(store: VectorStore) -> None:
    chunk = make_chunk()

    store.add_chunks([chunk], [make_embedding(1.0)])

    updated = make_chunk(
        "chunk-1",
        "def add(a, b):\n    return b + a",
    )

    store.add_chunks([updated], [make_embedding(2.0)])

    assert store.count() == 1

    result = store.get_by_ids(["chunk-1"])

    assert result["documents"] == [updated.text]


def test_delete(store: VectorStore) -> None:
    store.add_chunks(
        [make_chunk()],
        [make_embedding()],
    )

    store.delete(["chunk-1"])

    assert store.count() == 0


def test_clear(store: VectorStore) -> None:
    chunks = [
        make_chunk("chunk-1"),
        make_chunk("chunk-2"),
    ]

    store.add_chunks(
        chunks,
        [
            make_embedding(1.0),
            make_embedding(2.0),
        ],
    )

    store.clear()

    assert store.count() == 0


def test_reset_is_alias_for_clear(store: VectorStore) -> None:
    store.add_chunks(
        [make_chunk()],
        [make_embedding()],
    )

    store.reset()

    assert store.count() == 0


def test_empty_add_is_safe(store: VectorStore) -> None:
    store.add_chunks([], [])

    assert store.count() == 0


def test_empty_get_is_safe(store: VectorStore) -> None:
    result = store.get_by_ids([])

    assert result["ids"] == []


def test_empty_delete_is_safe(store: VectorStore) -> None:
    store.delete([])

    assert store.count() == 0


def test_mismatched_chunk_embeddings_raise(store: VectorStore) -> None:
    with pytest.raises(VectorStoreInputError):
        store.add_chunks(
            [make_chunk()],
            [],
        )


def test_mismatched_record_lengths_raise(store: VectorStore) -> None:
    with pytest.raises(VectorStoreInputError):
        store.add_records(
            ids=["chunk-1"],
            documents=["code"],
            embeddings=[],
            metadatas=[{"language": "python"}],
        )


def test_invalid_embedding_raises(store: VectorStore) -> None:
    with pytest.raises(VectorStoreInputError):
        store.add_records(
            ids=["chunk-1"],
            documents=["code"],
            embeddings=[[]],
            metadatas=[{"language": "python"}],
        )


def test_metadata_is_preserved(store: VectorStore) -> None:
    chunk = make_chunk()

    store.add_chunks([chunk], [make_embedding()])

    result = store.get_by_ids(["chunk-1"])

    metadata = result["metadatas"][0]

    assert metadata["file_name"] == "example.py"
    assert metadata["language"] == "python"
    assert metadata["chunk_type"] == "function"


def test_custom_database_path_is_isolated(tmp_path: Path) -> None:
    store_a = VectorStore(
        path=tmp_path / "db_a",
        collection="codesage",
    )
    store_b = VectorStore(
        path=tmp_path / "db_b",
        collection="codesage",
    )

    store_a.add_chunks(
        [make_chunk()],
        [make_embedding()],
    )

    assert store_a.count() == 1
    assert store_b.count() == 0


def test_database_persists_after_reopening(tmp_path: Path) -> None:
    db_path = tmp_path / "persistent_db"

    store_a = VectorStore(
        path=db_path,
        collection="codesage",
    )

    store_a.add_chunks(
        [make_chunk()],
        [make_embedding()],
    )

    store_b = VectorStore(
        path=db_path,
        collection="codesage",
    )

    assert store_b.count() == 1

    result = store_b.get_by_ids(["chunk-1"])

    assert result["ids"] == ["chunk-1"]
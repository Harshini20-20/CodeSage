"""ChromaDB vector store for CodeSage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from config import settings
from src.chunker import CodeChunk


class VectorStoreError(Exception):
    """Base exception for vector store errors."""


class VectorStoreInputError(VectorStoreError):
    """Raised when vector store input is invalid."""


class VectorStore:
    """Persistent ChromaDB vector store for CodeSage code chunks."""

    DEFAULT_COLLECTION = "codesage"

    def __init__(
        self,
        path: Path = settings.chroma_path,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection

        try:
            self.path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(self.path),
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize ChromaDB at {self.path}: {exc}"
            ) from exc

    def add_chunks(
        self,
        chunks: list[CodeChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store CodeChunk objects and their corresponding embeddings."""
        if len(chunks) != len(embeddings):
            raise VectorStoreInputError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return

        self.add_records(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def add_records(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert raw records into ChromaDB."""
        lengths = {
            len(ids),
            len(documents),
            len(embeddings),
            len(metadatas),
        }

        if len(lengths) != 1:
            raise VectorStoreInputError(
                "IDs, documents, embeddings, and metadata must have "
                "the same length."
            )

        if not ids:
            return

        if any(not isinstance(item, str) or not item for item in ids):
            raise VectorStoreInputError("All IDs must be non-empty strings.")

        if any(
            not isinstance(embedding, list)
            or not embedding
            or any(not isinstance(value, (int, float)) for value in embedding)
            for embedding in embeddings
        ):
            raise VectorStoreInputError(
                "Each embedding must be a non-empty list of numbers."
            )

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to store records in ChromaDB: {exc}"
            ) from exc

    def get_by_ids(self, ids: list[str]) -> dict[str, Any]:
        """Return stored records matching the supplied IDs."""
        if not ids:
            return {
                "ids": [],
                "documents": [],
                "embeddings": [],
                "metadatas": [],
            }

        try:
            return self.collection.get(
                ids=ids,
                include=["documents", "embeddings", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to retrieve records from ChromaDB: {exc}"
            ) from exc

    def exists(self, ids: list[str]) -> dict[str, bool]:
        """Return whether each supplied ID exists in the collection."""
        if not ids:
            return {}

        result = self.get_by_ids(ids)
        existing_ids = set(result["ids"])

        return {
            chunk_id: chunk_id in existing_ids
            for chunk_id in ids
        }

    def delete(self, ids: list[str]) -> None:
        """Delete records by their IDs."""
        if not ids:
            return

        try:
            self.collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete records from ChromaDB: {exc}"
            ) from exc

    def count(self) -> int:
        """Return the number of stored chunks."""
        try:
            return self.collection.count()
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to count ChromaDB records: {exc}"
            ) from exc

    def clear(self) -> None:
        """Delete all records from the current collection."""
        try:
            result = self.collection.get()

            if result["ids"]:
                self.collection.delete(ids=result["ids"])
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to clear ChromaDB collection: {exc}"
            ) from exc

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top_k most similar stored chunks."""
        if not embedding:
            raise VectorStoreInputError(
                "Query embedding must not be empty."
            )

        if top_k <= 0:
            raise VectorStoreInputError(
                "top_k must be greater than zero."
            )

        try:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to query ChromaDB: {exc}"
            ) from exc

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {
                "id": chunk_id,
                "document": document,
                "metadata": metadata,
                "distance": distance,
            }
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
            )
        ]

    def reset(self) -> None:
        """Backward-compatible alias for clearing the collection."""
        self.clear()
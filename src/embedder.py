"""Embedding generation module for CodeSage.

Turns :class:`~src.chunker.CodeChunk` objects (or plain text) into numerical
vectors using a Sentence Transformers model. Vectors are returned as plain
Python ``list[float]`` so they can be handed to ChromaDB later without conversion.

Design notes
------------
* The text embedded for a chunk is always ``chunk.text`` (the chunker's
  embedding-ready form: a short file/class header plus the source). It is never
  rebuilt here.
* The model is loaded lazily on first use, **once per** :class:`Embedder`
  **instance**, then reused. ``sentence_transformers`` (and PyTorch) are only
  imported at that point, so importing this module stays cheap.
* Runs on CPU by default; no GPU or CUDA is required.
* Vectors are L2-normalised by default, which keeps cosine similarity and
  Euclidean distance consistent for retrieval.
* Uploaded code is only ever treated as text; nothing is executed.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from config import settings
from src.chunker import CodeChunk
from src.utils import get_logger

logger = get_logger("embedder")

Vector = list[float]

DEFAULT_BATCH_SIZE: int = 32


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class EmbeddingError(Exception):
    """Base class for all embedding errors."""


class EmbeddingModelError(EmbeddingError):
    """The embedding model could not be loaded."""


class EmbeddingInputError(EmbeddingError, ValueError):
    """The input to embed is invalid (wrong type, empty text, ...)."""


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #
class Embedder:
    """Generates embeddings with a Sentence Transformers model.

    Args:
        model_name: Hugging Face model id or a local model directory. Defaults
            to ``settings.embedding_model`` (``CODESAGE_EMBEDDING_MODEL``).
        device: Torch device to run on. Defaults to ``"cpu"``.
        batch_size: Batch size passed to Sentence Transformers' ``encode``.
        normalize: L2-normalise the returned vectors.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise EmbeddingInputError("model_name must be a non-empty string.")
        if batch_size < 1:
            raise EmbeddingInputError("batch_size must be at least 1.")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Optional[Any] = None

    # ----- model management ------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return self._model is not None

    @property
    def model(self) -> Any:
        """The underlying ``SentenceTransformer`` (loaded on first access)."""
        return self.load()

    def load(self) -> Any:
        """Load the model if it is not loaded yet and return it.

        Raises:
            EmbeddingModelError: Sentence Transformers is missing or the model
                cannot be loaded (e.g. unknown name, no internet on first use).
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(
                "The 'sentence-transformers' package is not installed or failed to import. "
                "Install the project requirements with: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading embedding model '%s' on %s ...", self.model_name, self.device)
        started = time.perf_counter()
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:  # library raises many different types
            raise EmbeddingModelError(
                f"Could not load embedding model '{self.model_name}'. Check the model name "
                "(CODESAGE_EMBEDDING_MODEL) and, on first use, your internet connection, "
                "because the model is downloaded once and then cached. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        logger.info("Embedding model ready in %.1fs (dimension %d)",
                    time.perf_counter() - started, self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector size, read from the loaded model (loads it if needed)."""
        model = self.load()
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        dim = getter()
        if not isinstance(dim, int) or dim < 1:
            raise EmbeddingModelError(f"Model '{self.model_name}' did not report an embedding dimension.")
        return dim

    # ----- public API ------------------------------------------------------ #
    def embed_chunk(self, chunk: CodeChunk) -> Vector:
        """Embed one chunk (using ``chunk.text``)."""
        return self.embed_chunks([chunk])[0]

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> list[Vector]:
        """Embed several chunks; ``result[i]`` belongs to ``chunks[i]``.

        An empty sequence returns an empty list without loading the model.

        Raises:
            EmbeddingInputError: an item is not a ``CodeChunk`` or has no text.
        """
        self._require_sequence(chunks, "chunks")
        texts: list[str] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, CodeChunk):
                raise EmbeddingInputError(
                    f"chunks[{index}] must be a CodeChunk, got {type(chunk).__name__}."
                )
            texts.append(chunk.text)
        return self.embed_texts(texts)

    def embed_text(self, text: str) -> Vector:
        """Embed one piece of raw text."""
        return self.embed_texts([text])[0]

    def embed_query(self, query: str) -> Vector:
        """Embed a user query (same model and settings as the chunk embeddings)."""
        return self.embed_text(query)

    def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        """Embed several raw texts; ``result[i]`` belongs to ``texts[i]``.

        An empty sequence returns an empty list without loading the model.

        Raises:
            EmbeddingInputError: ``texts`` is a bare string, or an item is not a
                string or is empty / whitespace only.
            EmbeddingModelError: the model cannot be loaded.
            EmbeddingError: the model returned an unusable result.
        """
        self._require_sequence(texts, "texts")
        if len(texts) == 0:
            return []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingInputError(f"texts[{index}] must be a string, got {type(text).__name__}.")
            if not text.strip():
                raise EmbeddingInputError(f"texts[{index}] is empty or whitespace only.")

        model = self.load()
        import numpy as np  # installed together with sentence-transformers

        try:
            matrix = model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {type(exc).__name__}: {exc}") from exc

        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise EmbeddingError(
                f"Model returned shape {tuple(matrix.shape)} for {len(texts)} text(s)."
            )
        if not np.isfinite(matrix).all():
            raise EmbeddingError("Model returned non-finite values (NaN or infinity).")
        logger.debug("Embedded %d text(s) into %d-dim vectors", len(texts), matrix.shape[1])
        return matrix.tolist()

    # ----- internals ------------------------------------------------------- #
    @staticmethod
    def _require_sequence(value: Any, label: str) -> None:
        """Reject strings/None/non-sequences passed where a list is expected."""
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise EmbeddingInputError(
                f"{label} must be a list or tuple, got {type(value).__name__}."
            )

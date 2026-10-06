"""Code chunking module for CodeSage.

Turns a :class:`~src.parser.ParsedFile` into :class:`CodeChunk` objects that are
ready for embedding and storage in ChromaDB. Nothing is executed and the source
is not re-parsed: function/class/method boundaries come from the parser's line
ranges, and the text of those elements is the parser's own ``CodeElement.source``.

Chunking strategy (Python files)
--------------------------------
* ``function``    - one chunk per top-level function (never split further).
* ``class``       - one chunk per class holding the *whole* class source.
* ``method``      - one extra chunk per method, carrying its class in metadata.
  A method's text therefore appears in both its class chunk and its own chunk,
  so both "what does this class do" and "what does this method do" retrieve well.
* ``imports``     - one chunk with the file's top-level (column 0) import
  statements. Indented module-level imports (e.g. inside ``try:``) stay in
  module code so their surrounding block is not broken apart.
* ``module_code`` - every run of remaining top-level lines that holds code
  (docstring, constants, ``if __name__ == "__main__":`` ...). Exact line ranges.
* ``file``        - fallback: whole file in one chunk, used for syntax-error
  files and for files with only comments.

Empty files produce no chunks. Comment-only gaps between definitions are not
emitted as separate chunks.

Chunk IDs are SHA-256 based, so they are stable across runs and machines for the
same file path, content and chunk boundaries.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence, Union

from src.parser import CodeElement, ParsedFile
from src.utils import get_logger

logger = get_logger("chunker")

ChunkType = Literal["imports", "function", "class", "method", "module_code", "file"]
MetadataValue = Union[str, int, float, bool]

_NEWLINE_RE = re.compile(r"\r\n|\r|\n")
_TYPE_LABELS: dict[str, str] = {
    "imports": "Imports",
    "function": "Function",
    "class": "Class",
    "method": "Method",
    "module_code": "Module-level code",
    "file": "File",
}


class ChunkerError(Exception):
    """Raised when the chunker receives input it cannot work with."""


@dataclass
class CodeChunk:
    """One retrievable piece of code.

    ``source`` is the original code text. ``metadata`` is flat (str / int /
    float / bool values only, no ``None``) so it can be stored in ChromaDB as is.
    Line numbers are 1-based and inclusive.
    """

    chunk_id: str
    file_path: str                   # POSIX-style path string
    language: str
    chunk_type: ChunkType
    name: str
    qualified_name: str
    source: str
    start_line: int
    end_line: int
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    @property
    def parent(self) -> Optional[str]:
        """Owning class name for methods, otherwise ``None``."""
        value = self.metadata.get("parent")
        return str(value) if value else None

    @property
    def text(self) -> str:
        """Text to embed: a short context header followed by the source.

        The header tells the embedding model which file and class the code
        belongs to, so a method on its own is still understood in context.
        """
        file_name = str(self.metadata.get("file_name", self.file_path))
        header = [f"# File: {file_name}"]
        if self.parent:
            header.append(f"# Class: {self.parent}")
        header.append(f"# {_TYPE_LABELS.get(self.chunk_type, self.chunk_type)}: {self.qualified_name}")
        return "\n".join(header) + "\n" + self.source

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (everything JSON-serialisable)."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "source": self.source,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": dict(self.metadata),
        }


@dataclass
class _FileInfo:
    """Per-file values shared by all chunks of that file."""

    path: str
    file_name: str
    language: str
    file_hash: str
    status: str
    lines: list[str]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CodeChunker:
    """Splits parsed files into meaningful chunks."""

    # ----- public API ------------------------------------------------------ #
    def chunk(self, parsed: ParsedFile) -> list[CodeChunk]:
        """Split one parsed file into chunks, ordered by position in the file.

        Raises:
            ChunkerError: ``parsed`` is not a :class:`ParsedFile`.
        """
        if not isinstance(parsed, ParsedFile):
            raise ChunkerError(f"Expected ParsedFile, got {type(parsed).__name__}")

        info = _FileInfo(
            path=parsed.path.as_posix(),
            file_name=parsed.path.name,
            language=parsed.language,
            file_hash=_sha256(parsed.content),
            status=parsed.status,
            lines=_NEWLINE_RE.split(parsed.content),
        )

        if parsed.status == "empty" or not parsed.content.strip():
            logger.info("No chunks for empty file %s", info.file_name)
            return []
        if parsed.status == "syntax_error":
            logger.warning("Fallback chunk for %s: %s", info.file_name, parsed.error)
            return [self._fallback_chunk(parsed, info, "syntax_error")]

        chunks = self._structural_chunks(parsed, info)
        if not chunks:  # e.g. a file containing only comments
            chunks = [self._fallback_chunk(parsed, info, "no_structure")]

        chunks.sort(key=lambda c: (c.start_line, c.end_line, c.chunk_type))
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["total_chunks"] = len(chunks)
        logger.info("Chunked %s into %d chunk(s)", info.file_name, len(chunks))
        return chunks

    def chunk_many(self, parsed_files: Sequence[ParsedFile]) -> list[CodeChunk]:
        """Chunk several parsed files, preserving their order."""
        chunks: list[CodeChunk] = []
        for parsed in parsed_files:
            chunks.extend(self.chunk(parsed))
        return chunks

    # ----- structural chunking --------------------------------------------- #
    def _structural_chunks(self, parsed: ParsedFile, info: _FileInfo) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        covered: set[int] = set()
        spans: list[tuple[int, int]] = []

        for func in parsed.functions:
            chunks.append(self._element_chunk(func, info))
            spans.append((func.start_line, func.end_line))
        for cls in parsed.classes:
            chunks.append(self._element_chunk(cls, info))
            spans.append((cls.start_line, cls.end_line))
            for method in cls.methods:
                chunks.append(self._element_chunk(method, info))
        for start, end in spans:
            covered.update(range(start, end + 1))

        import_chunk, import_lines = self._imports_chunk(parsed, info, spans)
        if import_chunk is not None:
            chunks.append(import_chunk)
            covered.update(import_lines)

        chunks.extend(self._module_code_chunks(info, covered))
        return chunks

    def _element_chunk(self, element: CodeElement, info: _FileInfo) -> CodeChunk:
        extra: dict[str, MetadataValue] = {
            "parent": element.parent or "",
            "class_name": element.parent or (element.name if element.kind == "class" else ""),
            "is_async": element.is_async,
            "has_docstring": element.docstring is not None,
        }
        if element.kind == "class":
            extra["method_count"] = len(element.methods)
            extra["method_names"] = ",".join(m.name for m in element.methods)
        return self._make_chunk(
            info, element.kind, element.name, element.qualified_name,
            element.source, element.start_line, element.end_line, extra,
        )

    def _imports_chunk(
        self, parsed: ParsedFile, info: _FileInfo, spans: list[tuple[int, int]]
    ) -> tuple[Optional[CodeChunk], set[int]]:
        """Build the imports chunk from top-level, column-0 import statements."""
        kept: dict[tuple[int, int], list[str]] = {}
        for imp in parsed.imports:
            start, end = imp.start_line, imp.end_line
            if not 1 <= start <= end <= len(info.lines):
                continue
            if any(s <= start and end <= e for s, e in spans):
                continue  # nested inside a function/class: belongs to that chunk
            if info.lines[start - 1][:1].isspace():
                continue  # indented (e.g. inside try:): stays in module code
            modules = kept.setdefault((start, end), [])
            if imp.kind == "import":
                modules.extend(imp.names)
            else:
                modules.append("." * imp.level + imp.module)
        if not kept:
            return None, set()

        ordered = sorted(kept)
        source = "\n".join("\n".join(info.lines[s - 1:e]) for s, e in ordered)
        modules_seen: list[str] = []
        for key in ordered:
            for module in kept[key]:
                if module not in modules_seen:
                    modules_seen.append(module)
        lines_used: set[int] = set()
        for s, e in ordered:
            lines_used.update(range(s, e + 1))
        extra: dict[str, MetadataValue] = {
            "parent": "",
            "class_name": "",
            "is_async": False,
            "has_docstring": False,
            "import_count": len(ordered),
            "imported_modules": ",".join(modules_seen),
        }
        chunk = self._make_chunk(
            info, "imports", "imports", "imports", source, ordered[0][0], ordered[-1][1], extra
        )
        return chunk, lines_used

    def _module_code_chunks(self, info: _FileInfo, covered: set[int]) -> list[CodeChunk]:
        """One chunk per run of uncovered top-level lines that contains code."""
        lines = info.lines
        runs: list[tuple[int, int]] = []
        run_start: Optional[int] = None
        for number in range(1, len(lines) + 1):
            if number in covered:
                if run_start is not None:
                    runs.append((run_start, number - 1))
                    run_start = None
            elif run_start is None:
                run_start = number
        if run_start is not None:
            runs.append((run_start, len(lines)))

        chunks: list[CodeChunk] = []
        for start, end in runs:
            while start <= end and not lines[start - 1].strip():
                start += 1
            while end >= start and not lines[end - 1].strip():
                end -= 1
            if start > end:
                continue
            block = lines[start - 1:end]
            if all(not ln.strip() or ln.strip().startswith("#") for ln in block):
                continue  # blank / comment-only gap, not code
            extra: dict[str, MetadataValue] = {
                "parent": "", "class_name": "", "is_async": False, "has_docstring": False,
            }
            chunks.append(self._make_chunk(
                info, "module_code", "module_code", "module_code", "\n".join(block), start, end, extra
            ))
        return chunks

    def _fallback_chunk(self, parsed: ParsedFile, info: _FileInfo, reason: str) -> CodeChunk:
        """Whole-file chunk used when no structure is available."""
        extra: dict[str, MetadataValue] = {
            "parent": "", "class_name": "", "is_async": False, "has_docstring": False,
            "fallback_reason": reason,
            "chunk_index": 0,
            "total_chunks": 1,
        }
        if parsed.error:
            extra["error"] = parsed.error
        if parsed.error_line is not None:
            extra["error_line"] = parsed.error_line
        end = max(parsed.line_count, 1)
        return self._make_chunk(
            info, "file", info.file_name, info.file_name, parsed.content, 1, end, extra
        )

    # ----- shared helpers -------------------------------------------------- #
    @staticmethod
    def make_chunk_id(file_path: str, chunk_type: str, qualified_name: str,
                      start_line: int, end_line: int, source: str) -> str:
        """Deterministic chunk ID (first 32 hex chars of a SHA-256 digest)."""
        key = "\x1f".join(
            [file_path, chunk_type, qualified_name, str(start_line), str(end_line), _sha256(source)]
        )
        return _sha256(key)[:32]

    def _make_chunk(
        self,
        info: _FileInfo,
        chunk_type: ChunkType,
        name: str,
        qualified_name: str,
        source: str,
        start_line: int,
        end_line: int,
        extra: dict[str, MetadataValue],
    ) -> CodeChunk:
        metadata: dict[str, MetadataValue] = {
            "file_path": info.path,
            "file_name": info.file_name,
            "language": info.language,
            "chunk_type": chunk_type,
            "name": name,
            "qualified_name": qualified_name,
            "start_line": start_line,
            "end_line": end_line,
            "line_count": end_line - start_line + 1,
            "char_count": len(source),
            "parse_status": info.status,
            "file_hash": info.file_hash,
        }
        metadata.update(extra)
        return CodeChunk(
            chunk_id=self.make_chunk_id(info.path, chunk_type, qualified_name, start_line, end_line, source),
            file_path=info.path,
            language=info.language,
            chunk_type=chunk_type,
            name=name,
            qualified_name=qualified_name,
            source=source,
            start_line=start_line,
            end_line=end_line,
            metadata=metadata,
        )

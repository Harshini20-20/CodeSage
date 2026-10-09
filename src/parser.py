"""Code parsing module for CodeSage.

Reads source files as *text only* and extracts structure. Uploaded code is
never imported, compiled or executed: Python files are analysed statically with
the standard-library :mod:`ast` module, which only builds a syntax tree.

Output of :class:`CodeParser` is a :class:`ParsedFile`, which the chunker
consumes. Line numbers are 1-based and inclusive, matching editors and Python
tracebacks.

Currently supported languages: Python (``.py``). Other extensions raise
:class:`UnsupportedFileTypeError`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

from src.utils import get_logger

logger = get_logger("parser")

PathLike = Union[str, Path]

#: Map of supported file extensions to language names.
SUPPORTED_EXTENSIONS: dict[str, str] = {".py": "python"}

#: Files larger than this are refused rather than loaded into memory.
MAX_FILE_BYTES: int = 5 * 1024 * 1024

#: Directory names skipped when scanning a directory.
IGNORED_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", "venv", ".venv", "env", "node_modules", ".idea", ".vscode",
     ".pytest_cache", ".mypy_cache"}
)

ParseStatus = Literal["ok", "empty", "syntax_error"]

_NEWLINE_RE = re.compile(r"\r\n|\r|\n")


def _count_lines(content: str, lines: list[str]) -> int:
    """Number of lines in the file (a trailing newline does not add a line)."""
    if not content:
        return 0
    return len(lines) - 1 if lines[-1] == "" else len(lines)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ParserError(Exception):
    """Base class for all parser errors."""


class ParserFileNotFoundError(ParserError, FileNotFoundError):
    """The requested file or directory does not exist."""


class UnsupportedFileTypeError(ParserError, ValueError):
    """The file extension is not supported by the parser."""


class FileReadError(ParserError):
    """The file could not be read (permissions, encoding, size, not a file...)."""


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #
@dataclass
class ImportInfo:
    """One ``import`` / ``from ... import`` statement."""

    kind: Literal["import", "from"]
    module: str                      # "os.path" for ``import os.path`` / ``from os.path import x``;
                                     # "" for ``from . import x`` and multi-name ``import a, b``
    names: list[str]                 # imported names (or the module for plain imports)
    aliases: dict[str, str]          # name -> alias, only where ``as`` is used
    level: int                       # leading dots for relative imports, else 0
    start_line: int
    end_line: int
    source: str                      # statement text


@dataclass
class CallInfo:
    """Static metadata for one function or method call expression.

    ``caller`` is the qualified name of the function/method whose body contains
    this call. ``line`` is the 1-based source line of the call expression.
    ``target`` is populated only when a unique same-module definition can be
    identified statically; unresolved and dynamic targets remain ``None``.
    """

    name: str
    line: int
    caller: str
    target: Optional[str] = None  # resolved qualified name, when unambiguous


@dataclass
class CodeElement:
    """A function, class or method found in a file."""

    kind: Literal["function", "class", "method"]
    name: str
    qualified_name: str              # "Class.method" for methods, otherwise name
    start_line: int                  # includes decorators
    end_line: int
    source: str                      # exact text of lines start_line..end_line
    file_path: Path
    language: str
    parent: Optional[str] = None     # owning class name for methods
    docstring: Optional[str] = None
    is_async: bool = False
    methods: list["CodeElement"] = field(default_factory=list)  # classes only
    calls: list[str] = field(default_factory=list)  # backwards-compatible call names
    call_infos: list[CallInfo] = field(default_factory=list)  # structured call metadata
    base_classes: list[str] = field(default_factory=list)  # classes only; statically visible bases


@dataclass
class ParsedFile:
    """Structured result of parsing one source file.

    ``status`` is ``"ok"`` for a parsed file, ``"empty"`` for a file with no
    content, and ``"syntax_error"`` when the code is not valid Python. In the
    last two cases ``content`` is still preserved and ``error`` explains why, so
    later stages can decide how to treat the file.
    """

    path: Path
    language: str
    content: str
    imports: list[ImportInfo] = field(default_factory=list)
    functions: list[CodeElement] = field(default_factory=list)   # top-level only
    classes: list[CodeElement] = field(default_factory=list)     # each holds .methods
    line_count: int = 0
    status: ParseStatus = "ok"
    error: Optional[str] = None
    error_line: Optional[int] = None

    @property
    def ok(self) -> bool:
        """True when the file was parsed without problems."""
        return self.status == "ok"

    @property
    def file_name(self) -> str:
        """Base name of the file."""
        return self.path.name

    @property
    def elements(self) -> list[CodeElement]:
        """All functions, classes and methods, ordered by start line."""
        items: list[CodeElement] = [*self.functions]
        for cls in self.classes:
            items.append(cls)
            items.extend(cls.methods)
        return sorted(items, key=lambda e: (e.start_line, e.end_line))


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
class CodeParser:
    """Static source-code parser (no code is ever executed)."""

    # ----- public API ------------------------------------------------------ #
    def parse_file(self, path: PathLike) -> ParsedFile:
        """Parse a single source file.

        Raises:
            ParserFileNotFoundError: the file does not exist.
            UnsupportedFileTypeError: the extension is not supported.
            FileReadError: not a regular file, too large, unreadable or not
                valid UTF-8.

        Invalid Python syntax and empty files do not raise; they are returned
        with ``status`` set to ``"syntax_error"`` / ``"empty"``.
        """
        file_path = Path(path)
        language = self._language_for(file_path)
        if not file_path.exists():
            raise ParserFileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise FileReadError(f"Not a regular file: {file_path}")

        content = self._read_text(file_path)
        resolved = file_path.resolve()
        logger.debug("Parsing %s (%s)", resolved, language)
        return self._parse_python(resolved, content, language)

    def parse_directory(self, directory: PathLike, recursive: bool = True) -> list[ParsedFile]:
        """Parse every supported file in a directory.

        Unsupported files and ignored folders (``venv``, ``.git``, ...) are
        skipped. Files that cannot be read are logged and skipped so one bad file
        does not stop the rest. Results are sorted by path.

        Raises:
            ParserFileNotFoundError: the directory does not exist.
            FileReadError: the path is not a directory.
        """
        root = Path(directory)
        if not root.exists():
            raise ParserFileNotFoundError(f"Directory not found: {root}")
        if not root.is_dir():
            raise FileReadError(f"Not a directory: {root}")

        results: list[ParsedFile] = []
        candidates = root.rglob("*") if recursive else root.glob("*")
        for candidate in sorted(candidates):
            relative_parts = candidate.relative_to(root).parts[:-1]
            if any(part in IGNORED_DIRS for part in relative_parts):
                continue
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                results.append(self.parse_file(candidate))
            except ParserError as exc:
                logger.warning("Skipping %s: %s", candidate, exc)
        logger.info("Parsed %d file(s) from %s", len(results), root)
        return results

    @staticmethod
    def is_supported(path: PathLike) -> bool:
        """Whether the file's extension is supported."""
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    # ----- internals ------------------------------------------------------- #
    @staticmethod
    def _language_for(path: Path) -> str:
        suffix = path.suffix.lower()
        try:
            return SUPPORTED_EXTENSIONS[suffix]
        except KeyError:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{suffix or '(none)'}' for {path.name}. "
                f"Supported: {supported}"
            ) from None

    @staticmethod
    def _read_text(path: Path) -> str:
        """Read a file as UTF-8 text (BOM tolerated), with safe error handling."""
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise FileReadError(
                    f"File too large ({size} bytes, limit {MAX_FILE_BYTES}): {path.name}"
                )
            raw = path.read_bytes()
        except FileReadError:
            raise
        except OSError as exc:
            raise FileReadError(f"Cannot read {path}: {exc}") from exc
        try:
            content = raw.decode("utf-8-sig")
            return _NEWLINE_RE.sub("\n", content)
        except UnicodeDecodeError as exc:
            raise FileReadError(
                f"{path.name} is not valid UTF-8 text (byte {exc.start}: {exc.reason})"
            ) from exc

    def _parse_python(self, path: Path, content: str, language: str) -> ParsedFile:
        lines = _NEWLINE_RE.split(content)
        parsed = ParsedFile(
            path=path,
            language=language,
            content=content,
            line_count=_count_lines(content, lines),
        )
        if not content.strip():
            parsed.status = "empty"
            parsed.error = "File is empty or contains only whitespace."
            logger.info("Empty file: %s", path.name)
            return parsed

        try:
            tree = ast.parse(content, filename=str(path))  # parse only, never exec
        except SyntaxError as exc:
            parsed.status = "syntax_error"
            parsed.error_line = exc.lineno
            parsed.error = f"Invalid Python syntax at line {exc.lineno}: {exc.msg}"
            logger.warning("%s: %s", path.name, parsed.error)
            return parsed
        except (ValueError, RecursionError) as exc:  # e.g. null bytes, absurd nesting
            parsed.status = "syntax_error"
            parsed.error = f"Cannot parse Python source: {exc}"
            logger.warning("%s: %s", path.name, parsed.error)
            return parsed

        parsed.imports = self._extract_imports(tree, lines)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parsed.functions.append(self._make_element(node, "function", path, language, lines))
            elif isinstance(node, ast.ClassDef):
                parsed.classes.append(self._make_class(node, path, language, lines))
        self._resolve_local_call_targets(parsed)
        logger.info(
            "Parsed %s: %d import(s), %d function(s), %d class(es)",
            path.name, len(parsed.imports), len(parsed.functions), len(parsed.classes),
        )
        return parsed

    @staticmethod
    def _span(node: ast.AST) -> tuple[int, int]:
        """Start/end lines of a node, including decorators."""
        start = node.lineno  # type: ignore[attr-defined]
        for deco in getattr(node, "decorator_list", []):
            start = min(start, deco.lineno)
        end = getattr(node, "end_lineno", None) or start
        return start, end

    def _make_element(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
        kind: Literal["function", "class", "method"],
        path: Path,
        language: str,
        lines: list[str],
        parent: Optional[str] = None,
    ) -> CodeElement:
        start, end = self._span(node)
        qualified_name = f"{parent}.{node.name}" if parent else node.name
        call_infos = (
            self._extract_call_infos(node, qualified_name)
            if kind in ("function", "method")
            else []
        )
        base_classes = (
            [name for base in node.bases if (name := self._call_name(base)) is not None]
            if kind == "class" and isinstance(node, ast.ClassDef)
            else []
        )
        return CodeElement(
            kind=kind,
            name=node.name,
            qualified_name=qualified_name,
            start_line=start,
            end_line=end,
            source="\n".join(lines[start - 1:end]),
            file_path=path,
            language=language,
            parent=parent,
            docstring=ast.get_docstring(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            calls=[info.name for info in call_infos],
            call_infos=call_infos,
            base_classes=base_classes,
        )

    @staticmethod
    def _call_name(func: ast.expr) -> Optional[str]:
        """Return a statically visible call name, or ``None`` if dynamic."""
        if isinstance(func, ast.Name):
            return func.id
        if not isinstance(func, ast.Attribute):
            return None

        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
            return ".".join(reversed(parts))
        return None

    @staticmethod
    def _extract_call_infos(
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef], caller: str
    ) -> list[CallInfo]:
        """Return structured call metadata in source traversal order.

        Repeated calls are preserved. Nested function, async-function, class,
        and lambda bodies are excluded so calls are attributed to their own
        element/scope. Calls inside async function bodies are still collected.
        Only syntactically identifiable names are recorded; no code is run.
        """
        class CallVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.calls: list[CallInfo] = []

            def visit_Call(self, call: ast.Call) -> None:
                name = CodeParser._call_name(call.func)
                if name is not None:
                    self.calls.append(CallInfo(name=name, line=call.lineno, caller=caller))
                self.generic_visit(call)

            def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, child: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, child: ast.Lambda) -> None:
                return

        visitor = CallVisitor()
        for statement in node.body:
            visitor.visit(statement)
        return visitor.calls

    @staticmethod
    def _extract_calls(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> list[str]:
        """Backward-compatible helper returning only call names."""
        return [info.name for info in CodeParser._extract_call_infos(node, node.name)]

    @staticmethod
    def _resolve_local_call_targets(parsed: ParsedFile) -> None:
        """Resolve only unambiguous calls to definitions in the same module.

        This is deliberately conservative: imported symbols, dynamic receivers,
        inheritance dispatch, and duplicate definitions are not inferred. The
        resolver records possible static relationships; it never executes code.
        """
        functions_by_name: dict[str, list[CodeElement]] = {}
        classes_by_name: dict[str, list[CodeElement]] = {}
        for function in parsed.functions:
            functions_by_name.setdefault(function.name, []).append(function)
        for cls in parsed.classes:
            classes_by_name.setdefault(cls.name, []).append(cls)

        for element in parsed.elements:
            for call in element.call_infos:
                parts = call.name.split(".")
                target: Optional[str] = None
                if len(parts) == 1:
                    candidates = functions_by_name.get(parts[0], [])
                    if len(candidates) == 1:
                        target = candidates[0].qualified_name
                elif len(parts) == 2 and parts[0] in {"self", "cls"} and element.kind == "method" and element.parent:
                    owner_matches = classes_by_name.get(element.parent, [])
                    if len(owner_matches) == 1:
                        method_candidates = [m for m in owner_matches[0].methods if m.name == parts[1]]
                        if len(method_candidates) == 1:
                            target = method_candidates[0].qualified_name
                elif len(parts) == 2 and parts[0] in classes_by_name:
                    class_candidates = classes_by_name[parts[0]]
                    if len(class_candidates) == 1:
                        method_candidates = [m for m in class_candidates[0].methods if m.name == parts[1]]
                        if len(method_candidates) == 1:
                            target = method_candidates[0].qualified_name
                call.target = target

    def _make_class(self, node: ast.ClassDef, path: Path, language: str, lines: list[str]) -> CodeElement:
        cls = self._make_element(node, "class", path, language, lines)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cls.methods.append(
                    self._make_element(child, "method", path, language, lines, parent=node.name)
                )
        return cls

    @staticmethod
    def _extract_imports(tree: ast.AST, lines: list[str]) -> list[ImportInfo]:
        """Collect every import statement in the file (including nested ones)."""
        found: list[ImportInfo] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            start, end = node.lineno, getattr(node, "end_lineno", None) or node.lineno
            aliases = {a.name: a.asname for a in node.names if a.asname}
            if isinstance(node, ast.Import):
                info = ImportInfo("import", "", [a.name for a in node.names], aliases, 0,
                                  start, end, "\n".join(lines[start - 1:end]))
                info.module = info.names[0] if len(info.names) == 1 else ""
            else:
                info = ImportInfo("from", node.module or "", [a.name for a in node.names], aliases,
                                  node.level, start, end, "\n".join(lines[start - 1:end]))
            found.append(info)
        return sorted(found, key=lambda i: i.start_line)

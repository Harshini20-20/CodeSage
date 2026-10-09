"""Build a conservative, static relationship graph for Python repositories.

The graph consumes ``ParsedFile`` objects from :mod:`src.parser`. Analysed
source is parsed, never imported or executed. Relationships that cannot be
resolved uniquely are represented as unresolved symbol nodes or omitted when
there is no useful static reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional
import os

from src.parser import CodeElement, ImportInfo, ParsedFile

NodeKind = Literal["file", "function", "class", "method", "symbol"]
EdgeKind = Literal["contains", "calls", "imports", "inherits"]


@dataclass(frozen=True)
class GraphNode:
    """A file, code element, or unresolved symbol in a repository."""

    id: str
    kind: NodeKind
    name: str
    file_path: str
    line: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """A directed relationship between two graph nodes."""

    source: str
    target: str
    kind: EdgeKind
    line: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryGraph:
    """In-memory graph with deterministic nodes and ordered edges."""

    root: str
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    def nodes_of_kind(self, kind: NodeKind) -> list[GraphNode]:
        """Return nodes of a given kind in stable ID order."""
        return sorted((n for n in self.nodes.values() if n.kind == kind), key=lambda n: n.id)

    def edges_of_kind(self, kind: EdgeKind) -> list[GraphEdge]:
        """Return edges of a given kind in insertion order."""
        return [edge for edge in self.edges if edge.kind == kind]

    def outgoing(self, node_id: str, kind: Optional[EdgeKind] = None) -> list[GraphEdge]:
        """Return outgoing edges, optionally filtered by relationship kind."""
        return [e for e in self.edges if e.source == node_id and (kind is None or e.kind == kind)]

    def incoming(self, node_id: str, kind: Optional[EdgeKind] = None) -> list[GraphEdge]:
        """Return incoming edges, optionally filtered by relationship kind."""
        return [e for e in self.edges if e.target == node_id and (kind is None or e.kind == kind)]

    def find_nodes(self, name: str, kind: Optional[NodeKind] = None) -> list[GraphNode]:
        """Find nodes by exact name or qualified name, deterministically."""
        return sorted(
            (n for n in self.nodes.values() if n.name == name and (kind is None or n.kind == kind)),
            key=lambda n: n.id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable ordering."""
        return {
            "root": self.root,
            "nodes": [
                {"id": n.id, "kind": n.kind, "name": n.name, "file_path": n.file_path,
                 "line": n.line, "metadata": dict(n.metadata)}
                for n in sorted(self.nodes.values(), key=lambda item: item.id)
            ],
            "edges": [
                {"source": e.source, "target": e.target, "kind": e.kind,
                 "line": e.line, "metadata": dict(e.metadata)}
                for e in self.edges
            ],
        }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.name


def _module_name(relative_path: str) -> str:
    path = Path(relative_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _element_kind(element: CodeElement) -> Literal["function", "class", "method"]:
    return element.kind


def _element_id(relative_path: str, element: CodeElement) -> str:
    return f"{element.kind}:{relative_path}::{element.qualified_name}@{element.start_line}"


def _resolve_relative_module(current_module: str, imported: ImportInfo) -> str:
    """Resolve a from-import's module name against its package context."""
    parts = imported.module.split(".") if imported.module else []
    if imported.level:
        package = current_module.split(".")[:-1]
        # A module whose path ends in __init__.py is represented by its package.
        remove = max(0, imported.level - 1)
        if remove:
            package = package[: max(0, len(package) - remove)]
        parts = package + parts
    return ".".join(part for part in parts if part)


def _unique(values: list[str]) -> Optional[str]:
    return values[0] if len(values) == 1 else None


def build_repository_graph(
    parsed_files: Iterable[ParsedFile], root: str | Path | None = None
) -> RepositoryGraph:
    """Build a graph of files, Python elements, calls, imports, and inheritance.

    Cross-file call resolution is deliberately limited to unambiguous targets
    visible through explicit imports (including aliases) or unique local module
    definitions. Dynamic dispatch, arbitrary attribute chains, and duplicate
    definitions remain unresolved rather than being guessed.
    """
    files = sorted(list(parsed_files), key=lambda item: str(item.path))
    if root is not None:
        root_path = Path(root).resolve()
    elif files:
        try:
            root_path = Path(os.path.commonpath([str(Path(f.path).resolve().parent) for f in files])).resolve()
        except (ValueError, OSError):
            root_path = Path(files[0].path).resolve().parent
    else:
        root_path = Path.cwd().resolve()

    graph = RepositoryGraph(root=str(root_path))
    parsed_by_rel: dict[str, ParsedFile] = {}
    module_to_rel: dict[str, str] = {}
    file_node_ids: dict[str, str] = {}
    element_ids: dict[tuple[str, str], str] = {}
    elements_by_node: dict[str, CodeElement] = {}
    classes_by_name: dict[str, list[str]] = {}
    functions_by_module_name: dict[tuple[str, str], list[str]] = {}
    classes_by_module_name: dict[tuple[str, str], list[str]] = {}

    # Pass 1: establish stable file/element identities and containment.
    for parsed in files:
        rel = _relative_path(Path(parsed.path), root_path)
        parsed_by_rel[rel] = parsed
        module = _module_name(rel)
        if module:
            module_to_rel[module] = rel
        file_id = f"file:{rel}"
        file_node_ids[rel] = file_id
        graph.nodes[file_id] = GraphNode(
            id=file_id, kind="file", name=Path(rel).name, file_path=rel,
            metadata={"module": module, "status": parsed.status, "language": parsed.language},
        )
        for element in parsed.elements:
            node_id = _element_id(rel, element)
            element_ids[(rel, element.qualified_name)] = node_id
            elements_by_node[node_id] = element
            graph.nodes[node_id] = GraphNode(
                id=node_id, kind=_element_kind(element), name=element.qualified_name,
                file_path=rel, line=element.start_line,
                metadata={"end_line": element.end_line, "parent": element.parent,
                          "is_async": element.is_async, "docstring": element.docstring,
                          "base_classes": list(element.base_classes)},
            )
            graph.edges.append(GraphEdge(file_id, node_id, "contains", element.start_line))
            if element.kind == "function":
                functions_by_module_name.setdefault((module, element.name), []).append(node_id)
            elif element.kind == "class":
                classes_by_name.setdefault(element.name, []).append(node_id)
                classes_by_module_name.setdefault((module, element.name), []).append(node_id)

    # Build each file's import binding table. Keys are names used by source code;
    # values identify a local module or imported definition when statically known.
    module_bindings: dict[str, dict[str, str]] = {}
    symbol_bindings: dict[str, dict[str, tuple[str, str]]] = {}
    ambiguous_module_bindings: dict[str, set[str]] = {}
    ambiguous_symbol_bindings: dict[str, set[str]] = {}
    for rel, parsed in parsed_by_rel.items():
        current_module = _module_name(rel)
        module_bindings[rel] = {}
        symbol_bindings[rel] = {}
        ambiguous_module_bindings[rel] = set()
        ambiguous_symbol_bindings[rel] = set()
        for imported in parsed.imports:
            if imported.kind == "import":
                for original in imported.names:
                    has_alias = original in imported.aliases
                    local = imported.aliases.get(original, original.split(".")[0])
                    # With an explicit alias bind the full module; without an
                    # alias Python binds only the first package component.
                    bound_module = original if has_alias else original.split(".")[0]
                    if local not in ambiguous_module_bindings[rel]:
                        previous = module_bindings[rel].get(local)
                        if previous is not None and previous != bound_module:
                            module_bindings[rel].pop(local, None)
                            ambiguous_module_bindings[rel].add(local)
                        else:
                            module_bindings[rel][local] = bound_module
                    # Preserve the import edge to the represented module.
                    target_rel = module_to_rel.get(original) or module_to_rel.get(bound_module)
                    if target_rel:
                        graph.edges.append(GraphEdge(
                            file_node_ids[rel], file_node_ids[target_rel], "imports", imported.start_line,
                            {"module": _module_name(target_rel), "source": imported.source,
                            "alias": imported.aliases.get(original)},
                        ))
            else:
                base_module = _resolve_relative_module(current_module, imported)
                candidate_modules = [base_module] if base_module else []
                candidate_modules.extend(
                    f"{base_module}.{name}" for name in imported.names
                    if base_module and name != "*"
                )
                target_rels = {module_to_rel[m] for m in candidate_modules if m in module_to_rel}
                for target_rel in sorted(target_rels):
                    graph.edges.append(GraphEdge(
                        file_node_ids[rel], file_node_ids[target_rel], "imports", imported.start_line,
                        {"module": _module_name(target_rel), "source": imported.source,
                        "imported_names": list(imported.names), "level": imported.level},
                    ))
                for original in imported.names:
                    if original == "*":
                        continue
                    local = imported.aliases.get(original, original)
                    if f"{base_module}.{original}" in module_to_rel:
                        # `from package import module` binds the submodule.
                        bound_module = f"{base_module}.{original}"
                        if local not in ambiguous_module_bindings[rel]:
                            previous = module_bindings[rel].get(local)
                            if previous is not None and previous != bound_module:
                                module_bindings[rel].pop(local, None)
                                ambiguous_module_bindings[rel].add(local)
                            else:
                                module_bindings[rel][local] = bound_module
                    elif base_module in module_to_rel:
                        binding = (base_module, original)
                        if local not in ambiguous_symbol_bindings[rel]:
                            previous = symbol_bindings[rel].get(local)
                            if previous is not None and previous != binding:
                                symbol_bindings[rel].pop(local, None)
                                ambiguous_symbol_bindings[rel].add(local)
                            else:
                                symbol_bindings[rel][local] = binding

    def find_definition(module: str, name: str, method: Optional[str] = None) -> Optional[str]:
        if method:
            class_ids = classes_by_module_name.get((module, name), [])
            if len(class_ids) != 1:
                return None
            class_node = class_ids[0]
            class_element = elements_by_node[class_node]
            matches = [m for m in class_element.methods if m.name == method]
            if len(matches) != 1:
                return None
            return element_ids.get((_node_file(class_node), matches[0].qualified_name))
        return _unique(functions_by_module_name.get((module, name), [])) or _unique(
            classes_by_module_name.get((module, name), [])
        )

    def _node_file(node_id: str) -> str:
        return graph.nodes[node_id].file_path

    # Pass 2: resolve inheritance using same-module names and imported class names.
    for rel, parsed in parsed_by_rel.items():
        module = _module_name(rel)
        for element in parsed.elements:
            if element.kind != "class":
                continue
            source_id = element_ids.get((rel, element.qualified_name))
            if not source_id:
                continue
            for base in element.base_classes:
                parts = base.split(".")
                target: Optional[str] = None
                if len(parts) == 1:
                    # Unqualified class names are resolved only when unique
                    # across the repository; duplicate names stay conservative.
                    if len(classes_by_name.get(parts[0], [])) == 1:
                        target = _unique(classes_by_module_name.get((module, parts[0]), []))
                    if (target is None and parts[0] in symbol_bindings[rel]
                            and parts[0] not in ambiguous_symbol_bindings[rel]):
                        imported_module, imported_name = symbol_bindings[rel][parts[0]]
                        target = _unique(classes_by_module_name.get((imported_module, imported_name), []))
                elif parts[0] in module_bindings[rel]:
                    target_module = module_bindings[rel][parts[0]]
                    target = _unique(classes_by_module_name.get((target_module, parts[-1]), []))
                if target and target != source_id:
                    graph.edges.append(GraphEdge(source_id, target, "inherits", element.start_line,
                                                 {"base_name": base, "resolved": True}))

    # Pass 3: resolve calls within the module and through known import bindings.
    for rel, parsed in parsed_by_rel.items():
        module = _module_name(rel)
        for element in parsed.elements:
            if element.kind not in ("function", "method"):
                continue
            source_id = element_ids.get((rel, element.qualified_name))
            if not source_id:
                continue
            for call in element.call_infos:
                target_id: Optional[str] = None
                parts = call.name.split(".")
                # Trust parser's same-file resolution only if it points to a real element.
                if call.target:
                    target_id = element_ids.get((rel, call.target))
                if target_id is None and len(parts) == 1:
                    target_id = _unique(functions_by_module_name.get((module, parts[0]), []))
                    if target_id is None:
                        target_id = _unique(classes_by_module_name.get((module, parts[0]), []))
                    if (target_id is None and parts[0] in symbol_bindings[rel]
                            and parts[0] not in ambiguous_symbol_bindings[rel]):
                        imported_module, imported_name = symbol_bindings[rel][parts[0]]
                        target_id = find_definition(imported_module, imported_name)
                elif target_id is None and len(parts) >= 2:
                    first, *tail = parts
                    if first in {"self", "cls"} and element.kind == "method" and element.parent:
                        owner_id = _unique(classes_by_module_name.get((module, element.parent), []))
                        if owner_id:
                            owner = elements_by_node[owner_id]
                            method_matches = [m for m in owner.methods if m.name == tail[-1]]
                            if len(method_matches) == 1:
                                target_id = element_ids.get((rel, method_matches[0].qualified_name))
                    elif first in module_bindings[rel] and first not in ambiguous_module_bindings[rel]:
                        imported_module = module_bindings[rel][first]
                        # Resolve a call through an imported module alias. For
                        # `import pkg.mod` (without alias), `pkg.mod.func()` is
                        # handled by matching the longest imported module prefix.
                        suffix = parts[1:]
                        for split_at in range(len(suffix), 0, -1):
                            module_suffix = ".".join(suffix[:split_at])
                            candidate_module = f"{imported_module}.{module_suffix}".strip(".")
                            if candidate_module in module_to_rel and split_at < len(suffix):
                                target_id = find_definition(candidate_module, suffix[-1])
                                if target_id:
                                    break
                        if target_id is None:
                            target_id = find_definition(imported_module, tail[-1]) if len(tail) == 1 else None
                        if target_id is None and len(tail) == 2:
                            target_id = find_definition(imported_module, tail[0], tail[1])
                    elif first in symbol_bindings[rel] and first not in ambiguous_symbol_bindings[rel]:
                        imported_module, imported_name = symbol_bindings[rel][first]
                        if len(tail) == 1:
                            target_id = find_definition(imported_module, imported_name, tail[0])
                resolved = target_id is not None
                if target_id is None:
                    symbol_id = f"symbol:{rel}::{call.name}"
                    if symbol_id not in graph.nodes:
                        graph.nodes[symbol_id] = GraphNode(
                            id=symbol_id, kind="symbol", name=call.name, file_path=rel,
                            metadata={"resolution": "unresolved"},
                        )
                    target_id = symbol_id
                graph.edges.append(GraphEdge(
                    source_id, target_id, "calls", call.line,
                    {"call_name": call.name, "resolved": resolved, "caller": call.caller},
                ))

    # Stable edge de-duplication (imports can legitimately be seen through two
    # equivalent resolution paths). Preserve distinct call sites by line number.
    unique_edges: list[GraphEdge] = []
    seen: set[tuple[Any, ...]] = set()
    for edge in graph.edges:
        # Metadata values can include lists; use a repr-based key for hashability.
        key = (edge.source, edge.target, edge.kind, edge.line, repr(sorted(edge.metadata.items())))
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)
    graph.edges = unique_edges
    return graph


__all__ = ["GraphNode", "GraphEdge", "RepositoryGraph", "build_repository_graph"]

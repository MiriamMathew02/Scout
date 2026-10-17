"""
Symbol-level graph: functions, schema fields, routes (JS-first).

Complements the file dependency graph. Used to:
- expand execute context to related handlers
- detect schema fields that were added but never plumbed through writers
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

# --- regexes (JS / mongoose-oriented) ---
_EXPORTS_FN_RE = re.compile(
    r"exports\.([A-Za-z_$][\w$]*)\s*=",
)
_SCHEMA_START_RE = re.compile(r"(?:mongoose\.)?Schema\s*\(\s*\{")
_SCHEMA_KEY_RE = re.compile(r"""\s*['"]?([A-Za-z_$][\w$]*)['"]?\s*:""")
_JS_COMMENT_RE = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
_ROUTE_RE = re.compile(
    r"""app\.(get|post|put|patch|delete)\s*\(\s*['"]([^'"]+)['"]\s*,\s*([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)""",
    re.IGNORECASE,
)
_NEW_MODEL_RE = re.compile(
    r"new\s+([A-Za-z_$][\w$]*)\s*\(\s*\{([\s\S]*?)\}\s*\)",
)
_OBJ_KEY_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*:")
_FIND_UPDATE_RE = re.compile(
    r"(?:findByIdAndUpdate|findOneAndUpdate|updateOne|updateMany)\s*\(\s*[^,]+,\s*\{([\s\S]*?)\}",
)
_PROP_IN_QUERY_RE = re.compile(
    r"""['"]([A-Za-z_$][\w$]*)['"]\s*:|\b([A-Za-z_$][\w$]*)\s*:""",
)
_PY_DEF_RE = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)
_PY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)\s*[\(:]", re.MULTILINE)


@dataclass
class Symbol:
    id: str  # e.g. "exports.create@app/controllers/note.controller.js"
    kind: str  # export | schema_field | route | function | class
    name: str
    file: str
    detail: str = ""


@dataclass
class SymbolGraph:
    """Nodes by id; edges: id -> list of related ids."""

    symbols: dict[str, Symbol] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)
    # schema field name -> list of symbol ids that write it
    field_writers: dict[str, list[str]] = field(default_factory=dict)
    # schema field name -> list of symbol ids that read it
    field_readers: dict[str, list[str]] = field(default_factory=dict)
    # schema fields declared per model file
    schema_fields: dict[str, list[str]] = field(default_factory=dict)

    def related(self, seed_ids: list[str], hops: int = 1) -> list[str]:
        undirected: dict[str, set[str]] = defaultdict(set)
        for a, bs in self.edges.items():
            for b in bs:
                undirected[a].add(b)
                undirected[b].add(a)
        seen: set[str] = set()
        order: list[str] = []
        q: deque[tuple[str, int]] = deque()
        for s in seed_ids:
            if s in self.symbols and s not in seen:
                seen.add(s)
                order.append(s)
                q.append((s, 0))
        while q:
            node, dist = q.popleft()
            if dist >= hops:
                continue
            for nb in undirected.get(node, ()):
                if nb not in seen:
                    seen.add(nb)
                    order.append(nb)
                    q.append((nb, dist + 1))
        return order

    def files_for_symbols(self, symbol_ids: list[str]) -> list[str]:
        files: list[str] = []
        for sid in symbol_ids:
            sym = self.symbols.get(sid)
            if sym and sym.file not in files:
                files.append(sym.file)
        return files


def build_symbol_graph(repo_root: Path | str, files: list[str]) -> SymbolGraph:
    """Extract symbols and coarse relationships from candidate files."""
    root = Path(repo_root).resolve()
    g = SymbolGraph()

    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        suffix = path.suffix.lower()
        if suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}:
            _extract_js(g, rel, text)
        elif suffix == ".py":
            _extract_py(g, rel, text)

    _link_routes_to_exports(g)
    return g


def check_schema_field_plumbing(
    graph: SymbolGraph,
    *,
    added_fields: list[str] | None = None,
    require_writers: bool = True,
) -> list[str]:
    """
    Return human-readable issues when schema fields lack create/update writers.

    If added_fields is None, check all schema fields that look "extra" beyond
    common defaults — better to pass explicit fields detected from the diff.
    """
    issues: list[str] = []
    fields = added_fields
    if fields is None:
        # all non-core fields across schemas
        core = {"title", "content", "name", "id", "_id", "createdAt", "updatedAt"}
        fields = []
        for flist in graph.schema_fields.values():
            for f in flist:
                if f not in core and f not in fields:
                    fields.append(f)

    writer_names = {"create", "update", "store", "save", "add", "insert", "patch"}
    for field_name in fields:
        writers = graph.field_writers.get(field_name, [])
        # Also scan export bodies that should be writers
        export_writers = [
            sid
            for sid, sym in graph.symbols.items()
            if sym.kind == "export" and sym.name.lower() in writer_names
        ]
        # Field is plumbed if any writer export's file content was marked
        if require_writers and not writers:
            # Check if any create/update export exists at all
            if export_writers:
                issues.append(
                    f"Schema field '{field_name}' is defined but not written in "
                    f"create/update-style handlers ({', '.join(graph.symbols[s].name for s in export_writers[:6])}). "
                    f"Thread req.body.{field_name} (or equivalent) through those functions."
                )
            else:
                issues.append(
                    f"Schema field '{field_name}' is defined but no create/update handler "
                    f"appears to set it."
                )
    return issues


def detect_added_schema_fields(
    repo_root: Path | str,
    backups: dict[str, str | None],
    touched_files: list[str],
) -> list[str]:
    """Compare backups vs current content to find new schema field names."""
    root = Path(repo_root).resolve()
    added: list[str] = []
    for rel in touched_files:
        if not rel.endswith((".js", ".ts", ".mjs", ".cjs")):
            continue
        path = root / rel
        if not path.is_file():
            continue
        try:
            current = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        old = backups.get(rel) or ""
        old_fields = set(_schema_fields_in_text(old)) if old else set()
        new_fields = set(_schema_fields_in_text(current))
        for f in sorted(new_fields - old_fields):
            if f not in added:
                added.append(f)
    return added


def _schema_fields_in_text(text: str) -> list[str]:
    fields: list[str] = []
    for m in _SCHEMA_START_RE.finditer(text):
        for name in _top_level_keys(text, m.end()):
            if name not in fields:
                fields.append(name)
    return fields


def _top_level_keys(text: str, start: int) -> list[str]:
    """Keys of the object literal whose body begins at text[start], just after its '{'.

    Counts brackets so nested options like `{ type: String, default: Date }` are skipped.
    Brackets and commas inside comments and strings are ignored.
    """
    keys: list[str] = []

    def add(segment: str) -> None:
        m = _SCHEMA_KEY_RE.match(_JS_COMMENT_RE.sub("", segment))
        if m:
            keys.append(m.group(1))

    depth = 0
    seg_start = i = start
    quote = ""
    while i < len(text):
        ch = text[i]
        if quote:  # inside a string: only its closing quote matters
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = ""
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end == -1 else end
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end == -1 else end + 2
            continue
        elif ch in "'\"`":
            quote = ch
        elif ch in "{[(":
            depth += 1
        elif ch in "}])":
            if depth == 0:  # end of the schema object
                add(text[seg_start:i])
                break
            depth -= 1
        elif ch == "," and depth == 0:
            add(text[seg_start:i])
            seg_start = i + 1
        i += 1
    return keys


def _sid(kind: str, name: str, file: str) -> str:
    return f"{kind}:{name}@{file}"


def _add(g: SymbolGraph, sym: Symbol) -> None:
    g.symbols[sym.id] = sym
    g.edges.setdefault(sym.id, [])


def _edge(g: SymbolGraph, a: str, b: str) -> None:
    g.edges.setdefault(a, [])
    if b not in g.edges[a]:
        g.edges[a].append(b)


def _extract_js(g: SymbolGraph, rel: str, text: str) -> None:
    # exports
    for m in _EXPORTS_FN_RE.finditer(text):
        name = m.group(1)
        sid = _sid("export", name, rel)
        _add(g, Symbol(id=sid, kind="export", name=name, file=rel))

    # schema fields
    fields = _schema_fields_in_text(text)
    if fields:
        g.schema_fields[rel] = fields
        for f in fields:
            sid = _sid("schema_field", f, rel)
            _add(g, Symbol(id=sid, kind="schema_field", name=f, file=rel))

    # routes
    for m in _ROUTE_RE.finditer(text):
        method, path, alias, handler = m.group(1), m.group(2), m.group(3), m.group(4)
        rid = _sid("route", f"{method.upper()} {path}", rel)
        _add(
            g,
            Symbol(
                id=rid,
                kind="route",
                name=f"{method.upper()} {path}",
                file=rel,
                detail=f"{alias}.{handler}",
            ),
        )
        # will link to export after all files parsed
        g.edges.setdefault(rid, [])
        # temporary detail for linking
        g.symbols[rid].detail = f"{handler}"

    # field writes in new Model({ ... }) and updates
    for m in _NEW_MODEL_RE.finditer(text):
        body = m.group(2)
        keys = _OBJ_KEY_RE.findall(body)
        owner_export = _enclosing_export(text, m.start())
        for k in keys:
            g.field_writers.setdefault(k, [])
            if owner_export:
                sid = _sid("export", owner_export, rel)
                if sid not in g.field_writers[k]:
                    g.field_writers[k].append(sid)
            # link schema field if known later
            for sfile, flist in g.schema_fields.items():
                if k in flist:
                    fsid = _sid("schema_field", k, sfile)
                    if owner_export:
                        esid = _sid("export", owner_export, rel)
                        _edge(g, esid, fsid)

    for m in _FIND_UPDATE_RE.finditer(text):
        body = m.group(1)
        keys = _OBJ_KEY_RE.findall(body)
        owner_export = _enclosing_export(text, m.start())
        for k in keys:
            g.field_writers.setdefault(k, [])
            if owner_export:
                sid = _sid("export", owner_export, rel)
                if sid not in g.field_writers[k]:
                    g.field_writers[k].append(sid)

    # field reads: property names in export bodies that match schema fields later
    for m in _EXPORTS_FN_RE.finditer(text):
        name = m.group(1)
        # body until next exports. or EOF
        start = m.end()
        next_m = _EXPORTS_FN_RE.search(text, start)
        end = next_m.start() if next_m else len(text)
        body = text[start:end]
        for fm in re.finditer(r"\b([A-Za-z_$][\w$]*)\b", body):
            tok = fm.group(1)
            if tok in {"req", "res", "then", "catch", "find", "send", "status", "message"}:
                continue
            # mark as potential reader; refined against schema_fields at end
            g.field_readers.setdefault(tok, [])
            sid = _sid("export", name, rel)
            if sid not in g.field_readers[tok]:
                g.field_readers[tok].append(sid)


def _enclosing_export(text: str, pos: int) -> str | None:
    last = None
    for m in _EXPORTS_FN_RE.finditer(text):
        if m.start() > pos:
            break
        last = m.group(1)
    return last


def _link_routes_to_exports(g: SymbolGraph) -> None:
    exports_by_name: dict[str, list[str]] = defaultdict(list)
    for sid, sym in g.symbols.items():
        if sym.kind == "export":
            exports_by_name[sym.name].append(sid)

    for sid, sym in list(g.symbols.items()):
        if sym.kind != "route":
            continue
        handler = sym.detail  # handler name only
        for esid in exports_by_name.get(handler, []):
            _edge(g, sid, esid)

    # prune field_readers to only schema fields
    all_fields = set()
    for flist in g.schema_fields.values():
        all_fields.update(flist)
    g.field_readers = {
        k: v for k, v in g.field_readers.items() if k in all_fields
    }
    g.field_writers = {
        k: v for k, v in g.field_writers.items() if k in all_fields or True
    }
    # keep writers even if field only in one schema - filter keys that appear in schema
    g.field_writers = {
        k: v for k, v in g.field_writers.items() if k in all_fields or not all_fields
    }


def _extract_py(g: SymbolGraph, rel: str, text: str) -> None:
    for m in _PY_DEF_RE.finditer(text):
        name = m.group(1)
        sid = _sid("function", name, rel)
        _add(g, Symbol(id=sid, kind="function", name=name, file=rel))
    for m in _PY_CLASS_RE.finditer(text):
        name = m.group(1)
        sid = _sid("class", name, rel)
        _add(g, Symbol(id=sid, kind="class", name=name, file=rel))

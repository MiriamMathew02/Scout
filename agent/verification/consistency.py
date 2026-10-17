"""
Post-edit consistency checks over changed files and graph neighbors.

Catches cross-file mistakes that per-file syntax checks miss, e.g.
routes calling notes.search while the controller exports searchNotes.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.discovery.graph import build_dependency_graph, neighbors
from agent.discovery.snippets import read_line_numbered
from agent.llm import chat_json

# require / import path → local binding
_REQUIRE_RE = re.compile(
    r"""(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)"""
)
# property access: notes.create, notes.searchNotes
_PROP_USE_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)\b")
# exports.NAME = ...
_EXPORTS_RE = re.compile(r"\bexports\.([A-Za-z_$][\w$]*)\s*=")
# module.exports = { a, b: fn, c }
_MODULE_EXPORTS_OBJ_RE = re.compile(
    r"module\.exports\s*=\s*\{([^}]*)\}",
    re.DOTALL,
)
_OBJ_KEY_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*:")
_OBJ_SHORTHAND_RE = re.compile(r"(?:^|,)\s*([A-Za-z_$][\w$]*)\s*(?=,|$)")


@dataclass
class ConsistencyIssue:
    kind: str
    message: str
    file: str = ""
    related_file: str = ""
    symbol: str = ""

    def as_text(self) -> str:
        parts = [f"[{self.kind}] {self.message}"]
        if self.file:
            parts.append(f"  file: {self.file}")
        if self.related_file:
            parts.append(f"  related: {self.related_file}")
        if self.symbol:
            parts.append(f"  symbol: {self.symbol}")
        return "\n".join(parts)


@dataclass
class ConsistencyResult:
    ok: bool
    issues: list[ConsistencyIssue] = field(default_factory=list)
    scope_files: list[str] = field(default_factory=list)
    review_error: str | None = None  # set when the LLM review was asked for but did not complete

    def as_text(self) -> str:
        note = f"\nLLM consistency review did not complete: {self.review_error}" if self.review_error else ""
        if self.ok:
            scope = ", ".join(self.scope_files[:12]) or "(none)"
            return f"PASS\nConsistency OK over: {scope}{note}"
        lines = ["FAIL", f"Checked files: {', '.join(self.scope_files)}"]
        for issue in self.issues:
            lines.append(issue.as_text())
        return "\n".join(lines) + note


def files_in_consistency_scope(
    touched: list[str],
    graph: dict[str, list[str]],
    *,
    hops: int = 1,
) -> list[str]:
    """Changed files plus undirected graph neighbors."""
    if not touched:
        return []
    return neighbors(graph, touched, hops=hops)


def check_consistency(
    repo_root: Path | str,
    touched: list[str],
    graph: dict[str, list[str]],
    *,
    hops: int = 1,
    use_llm: bool = True,
    model: str | None = None,
    task: str = "",
    client: Any = None,
    deleted: list[str] | None = None,
) -> ConsistencyResult:
    """
    Static checks first (JS require/export wiring); optional LLM review
    of the same file scope for residual mistakes. `deleted` lists files this
    run removed; any file that still imports one of them is an issue.
    """
    root = Path(repo_root).resolve()
    scope = files_in_consistency_scope(touched, graph, hops=hops)
    # Always include touched even if graph is empty
    for t in touched:
        if t not in scope:
            scope.append(t)

    issues: list[ConsistencyIssue] = []
    issues.extend(_check_js_require_exports(root, scope))
    issues.extend(_check_python_imports(root, scope))
    if deleted:
        files = list(dict.fromkeys(list(graph.keys()) + scope))
        issues.extend(check_deleted_references(root, deleted, files))

    if use_llm and not issues:
        # Only spend tokens if static checks are clean but we still want a pass
        # Actually: always run LLM if use_llm and we have scope — catches semantic issues
        # Token efficiency: run LLM only when static found nothing? User wants mistakes caught.
        # Run LLM when static is clean OR always for small scope.
        # Prefer always for small scopes (<= 8 files) to catch name mismatches static might miss
        # in unusual patterns; skip LLM if static already failed (enough signal to re-execute).
        pass

    review_error = None
    if use_llm and not issues and scope:
        llm_issues, review_error = _llm_consistency_review(
            root,
            scope,
            touched=touched,
            task=task,
            model=model,
            client=client,
        )
        issues.extend(llm_issues)
    elif use_llm and issues:
        # Static already failed — skip LLM, return static issues for repair
        pass

    return ConsistencyResult(
        ok=len(issues) == 0, issues=issues, scope_files=scope, review_error=review_error
    )


def _check_js_require_exports(root: Path, scope: list[str]) -> list[ConsistencyIssue]:
    """If A requires B and uses alias.prop, prop must be exported by B."""
    issues: list[ConsistencyIssue] = []
    scope_set = set(scope)

    # Precompute exports for each JS file in scope
    exports_map: dict[str, set[str]] = {}
    texts: dict[str, str] = {}
    for rel in scope:
        path = root / rel
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".js", ".mjs", ".cjs"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        texts[rel] = text
        exports_map[rel] = _extract_js_exports(text)

    for rel, text in texts.items():
        for m in _REQUIRE_RE.finditer(text):
            alias, spec = m.group(1), m.group(2)
            if not spec.startswith("."):
                continue
            target = _resolve_js_require(rel, spec, scope_set, root)
            if not target:
                # try resolve even outside scope if file exists
                target = _resolve_js_require(rel, spec, None, root)
            if not target or target not in exports_map:
                continue

            exported = exports_map[target]
            if not exported:
                # CommonJS module.exports = (app) => ... factory — skip property check
                continue

            used_props = _props_used_on_alias(text, alias)
            # Framework noise / methods not from our module
            skip = {
                "then",
                "catch",
                "finally",
                "toString",
                "valueOf",
                "call",
                "apply",
                "bind",
                "length",
                "name",
                "prototype",
                "constructor",
                "exports",
                "log",
                "status",
                "send",
                "json",
                "params",
                "body",
                "query",
                "message",
            }
            for prop in used_props:
                if prop in skip:
                    continue
                if prop not in exported:
                    issues.append(
                        ConsistencyIssue(
                            kind="missing_export",
                            message=(
                                f"'{rel}' uses {alias}.{prop} but "
                                f"'{target}' does not export '{prop}'. "
                                f"Exports found: {', '.join(sorted(exported)) or '(none)'}."
                            ),
                            file=rel,
                            related_file=target,
                            symbol=prop,
                        )
                    )
    return issues


def check_deleted_references(
    root: Path, deleted: list[str], files: list[str]
) -> list[ConsistencyIssue]:
    """Files that still import a module this run deleted."""
    gone = set(deleted)
    # Import edges only: a test file sitting next to a deleted module is not a reference
    graph = build_dependency_graph(root, list(dict.fromkeys(files + deleted)), structural=False)
    issues: list[ConsistencyIssue] = []
    for src, deps in graph.items():
        if src in gone or not (root / src).is_file():
            continue
        for dep in deps:
            if dep in gone:
                issues.append(
                    ConsistencyIssue(
                        kind="deleted_module_in_use",
                        message=(
                            f"'{src}' still imports '{dep}', which this run deleted. "
                            f"Update '{src}' or keep '{dep}'."
                        ),
                        file=src,
                        related_file=dep,
                    )
                )
    return issues


def _extract_js_exports(text: str) -> set[str]:
    names: set[str] = set()
    for m in _EXPORTS_RE.finditer(text):
        names.add(m.group(1))

    for m in _MODULE_EXPORTS_OBJ_RE.finditer(text):
        body = m.group(1)
        for km in _OBJ_KEY_RE.finditer(body):
            names.add(km.group(1))
        # shorthand { create, update }
        for part in body.split(","):
            part = part.strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", part):
                names.add(part)

    # module.exports = something (single binding) — treat as opaque
    if re.search(r"module\.exports\s*=\s*(?!\{)", text) and not names:
        return set()  # unknown shape (e.g. factory function)
    return names


def _props_used_on_alias(text: str, alias: str) -> set[str]:
    props: set[str] = set()
    for m in _PROP_USE_RE.finditer(text):
        if m.group(1) == alias:
            props.add(m.group(2))
    return props


def _resolve_js_require(
    from_file: str,
    spec: str,
    scope_set: set[str] | None,
    root: Path,
) -> str | None:
    base = Path(from_file).parent
    raw = (base / spec).as_posix()
    parts: list[str] = []
    for p in raw.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    normalized = "/".join(parts)
    candidates = [
        normalized,
        normalized + ".js",
        normalized + ".mjs",
        normalized + ".cjs",
        normalized + "/index.js",
    ]
    for c in candidates:
        if scope_set is not None and c not in scope_set:
            # still accept if file exists on disk
            if (root / c).is_file():
                return c
            continue
        if (root / c).is_file():
            return c
    return None


_PY_IMPORT_RE = re.compile(
    r"^\s*from\s+([\w.]+)\s+import\s+(.+)$|^\s*import\s+([\w.]+)",
    re.MULTILINE,
)


def _check_python_imports(root: Path, scope: list[str]) -> list[ConsistencyIssue]:
    """
    Lightweight check: from package.mod import name — if the module file is in
    scope and defines names, flag imports that are clearly missing.
    Kept conservative to avoid false positives on re-exports.
    """
    issues: list[ConsistencyIssue] = []
    py_files = [f for f in scope if f.endswith(".py")]
    if len(py_files) < 2:
        return issues

    def_names: dict[str, set[str] | None] = {}
    trees: dict[str, ast.Module] = {}
    for rel in py_files:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            trees[rel] = ast.parse(text)
        except SyntaxError:
            continue  # reported by the verify phase
        def_names[rel] = _python_top_level_names(trees[rel])

    typing_skip = {
        "Optional",
        "Any",
        "List",
        "Dict",
        "Tuple",
        "Set",
        "Union",
        "Callable",
        "Iterable",
        "Mapping",
        "Sequence",
        "TYPE_CHECKING",
    }

    for rel, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                # skip relative imports (harder to resolve reliably)
                continue
            mod = node.module
            target = _resolve_py_module(mod, set(def_names), root)
            if not target or target not in def_names:
                continue
            defined = def_names[target]
            if not defined:
                continue
            package_dir = (root / target).parent if target.endswith("__init__.py") else None
            for alias in node.names:
                name = alias.name
                if name == "*" or name in typing_skip or name.startswith("_"):
                    continue
                if package_dir and (
                    (package_dir / f"{name}.py").is_file()
                    or (package_dir / name / "__init__.py").is_file()
                ):
                    continue  # `from pkg import submodule`
                if name not in defined:
                    issues.append(
                        ConsistencyIssue(
                            kind="missing_python_symbol",
                            message=(
                                f"'{rel}' imports '{name}' from '{mod}' "
                                f"but '{target}' does not define it. "
                                f"Found: {', '.join(sorted(defined)[:20])}"
                            ),
                            file=rel,
                            related_file=target,
                            symbol=name,
                        )
                    )
    return issues


def _python_top_level_names(tree: ast.Module) -> set[str] | None:
    """Names a module binds at top level, or None when that can't be known statically."""
    names: set[str] = set()
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return None
                names.add((alias.asname or alias.name).split(".")[0])
        else:
            # module-level if / try / with blocks can bind names too
            pending.extend(
                child
                for child in ast.iter_child_nodes(node)
                if isinstance(child, (ast.stmt, ast.excepthandler))
            )
    if "__getattr__" in names:
        return None
    return names


def _resolve_py_module(
    module: str,
    known: set[str],
    root: Path,
) -> str | None:
    pathish = module.replace(".", "/")
    for c in (pathish + ".py", pathish + "/__init__.py"):
        if c in known or (root / c).is_file():
            return c
    return None


CONSISTENCY_SYSTEM = """You are a code reviewer checking for cross-file consistency bugs
after automated edits. Focus on:
- Handlers/routes calling functions that do not exist / wrong names
- Imports of missing symbols
- Model fields used in controllers but not defined on the schema
- Duplicate conflicting definitions

Respond with JSON only:
{
  "issues": [
    {
      "kind": "name_mismatch" | "missing_symbol" | "schema_mismatch" | "other",
      "message": "clear description of the bug and how to fix it",
      "file": "path",
      "related_file": "path or empty",
      "symbol": "name or empty"
    }
  ]
}
If everything is consistent, return {"issues": []}.
Do not invent style nits. Only real breakages.
"""


def _llm_consistency_review(
    root: Path,
    scope: list[str],
    *,
    touched: list[str],
    task: str,
    model: str | None,
    client: Any,
) -> tuple[list[ConsistencyIssue], str | None]:
    """Return (issues, error); error says why the review did not complete."""
    # Cap files for tokens
    files_to_show = list(dict.fromkeys(touched + scope))[:10]
    parts: list[str] = []
    for rel in files_to_show:
        path = root / rel
        if not path.is_file():
            continue
        try:
            body = read_line_numbered(path)
        except OSError:
            continue
        # truncate large files
        lines = body.splitlines()
        if len(lines) > 200:
            body = "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
        parts.append(f"===== {rel} =====\n{body}")

    if not parts:
        return [], None

    user = (
        f"Task that was implemented:\n{task}\n\n"
        f"Changed files: {', '.join(touched)}\n"
        f"Review scope (changed + neighbors):\n{', '.join(scope)}\n\n"
        + "\n\n".join(parts)
    )

    try:
        from agent.llm import DEFAULT_MODEL

        data = chat_json(
            system=CONSISTENCY_SYSTEM,
            user=user,
            model=model or DEFAULT_MODEL,
            client=client,
            max_tokens=2000,
        )
    except Exception as exc:  # noqa: BLE001 — soft: static checks already ran
        return [], f"model request failed: {exc}"

    raw = data.get("issues")
    if not isinstance(raw, list):
        return [], "reply had no 'issues' list"
    out: list[ConsistencyIssue] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        msg = str(item.get("message") or "").strip()
        if not msg:
            continue
        out.append(
            ConsistencyIssue(
                kind=str(item.get("kind") or "other"),
                message=msg,
                file=str(item.get("file") or ""),
                related_file=str(item.get("related_file") or ""),
                symbol=str(item.get("symbol") or ""),
            )
        )
    return out, None

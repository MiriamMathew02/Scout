"""Best-effort multi-language dependency graph."""

from __future__ import annotations

import ast
import re
from collections import defaultdict, deque
from pathlib import Path

# path -> list of dependency paths (also relative posix)
Graph = dict[str, list[str]]


def build_dependency_graph(repo_root: Path | str, files: list[str]) -> Graph:
    """
    Build adjacency map: file -> files it depends on (within the candidate set).

    Unresolved imports are dropped. Includes structural co-location edges.
    """
    root = Path(repo_root).resolve()
    file_set = set(files)
    index = _path_index(files)
    graph: dict[str, set[str]] = defaultdict(set)

    for rel in files:
        full = root / rel
        if not full.is_file():
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        suffix = Path(rel).suffix.lower()
        deps: set[str] = set()
        if suffix == ".py":
            deps |= _python_deps(rel, text, file_set, index, root)
        elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            deps |= _js_deps(rel, text, file_set, index)
        elif suffix == ".go":
            deps |= _go_deps(rel, text, file_set, index)
        elif suffix == ".rs":
            deps |= _rust_deps(rel, text, file_set, index)
        elif suffix in {".java", ".kt", ".kts"}:
            deps |= _jvm_deps(rel, text, file_set, index)
        elif suffix in {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx"}:
            deps |= _c_include_deps(rel, text, file_set)
        elif suffix in {".rb"}:
            deps |= _ruby_deps(rel, text, file_set, index)
        elif suffix == ".php":
            deps |= _php_deps(rel, text, file_set, index)

        deps |= _structural_neighbors(rel, file_set)
        deps.discard(rel)
        graph[rel] |= deps

    # Ensure all nodes exist
    for f in files:
        graph.setdefault(f, set())

    return {k: sorted(v) for k, v in graph.items()}


def neighbors(graph: Graph, seeds: list[str], hops: int = 1) -> list[str]:
    """Expand seed files by undirected hops through the graph."""
    if hops <= 0:
        return list(dict.fromkeys(seeds))

    undirected: dict[str, set[str]] = defaultdict(set)
    for src, dests in graph.items():
        for d in dests:
            undirected[src].add(d)
            undirected[d].add(src)

    seen: set[str] = set()
    order: list[str] = []
    q: deque[tuple[str, int]] = deque()
    for s in seeds:
        if s not in seen:
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


def topological_edit_order(graph: Graph, targets: list[str]) -> list[str]:
    """
    Order targets so dependencies tend to come first (Kahn on induced subgraph).

    Cycles fall back to original relative order.
    """
    target_set = set(targets)
    indeg: dict[str, int] = {t: 0 for t in targets}
    adj: dict[str, list[str]] = {t: [] for t in targets}

    for t in targets:
        for dep in graph.get(t, []):
            if dep in target_set:
                # edge dep -> t (dep before t)
                adj[dep].append(t)
                indeg[t] += 1

    q = deque([t for t in targets if indeg[t] == 0])
    result: list[str] = []
    while q:
        n = q.popleft()
        result.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)

    if len(result) < len(targets):
        # cycle — append remaining in original order
        remaining = [t for t in targets if t not in set(result)]
        result.extend(remaining)
    return result


# --- language extractors ---


def _path_index(files: list[str]) -> dict[str, list[str]]:
    """Map stem / basename keys to candidate relative paths."""
    idx: dict[str, list[str]] = defaultdict(list)
    for f in files:
        p = Path(f)
        idx[p.name].append(f)
        idx[p.stem].append(f)
        # package-style: a/b/c.py -> a.b.c
        if p.suffix == ".py":
            mod = f[:-3].replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            idx[mod].append(f)
    return idx


def _python_deps(
    rel: str,
    text: str,
    file_set: set[str],
    index: dict[str, list[str]],
    root: Path,
) -> set[str]:
    deps: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return deps

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                deps |= _resolve_python_module(alias.name, rel, file_set, index)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod or node.level:
                deps |= _resolve_python_module(mod, rel, file_set, index, level=node.level)
            # from pkg import b  → also try pkg.b / pkg/b.py
            for alias in node.names:
                if alias.name == "*":
                    continue
                if mod:
                    sub = f"{mod}.{alias.name}"
                elif node.level:
                    sub = alias.name
                else:
                    continue
                deps |= _resolve_python_module(sub, rel, file_set, index, level=node.level)
    return deps


def _resolve_python_module(
    module: str,
    from_file: str,
    file_set: set[str],
    index: dict[str, list[str]],
    level: int = 0,
) -> set[str]:
    found: set[str] = set()
    parts = Path(from_file).parts
    if level > 0:
        # relative import
        base_parts = list(parts[:-1])
        up = level - 1
        if up:
            base_parts = base_parts[:-up] if up <= len(base_parts) else []
        if module:
            base_parts = base_parts + module.split(".")
        candidates = [
            "/".join(base_parts) + ".py",
            "/".join(base_parts) + "/__init__.py",
        ]
        for c in candidates:
            if c in file_set:
                found.add(c)
        return found

    if not module:
        return found

    # absolute within repo
    dotted = module
    for key in (dotted, dotted + ".__init__"):
        for cand in index.get(key, []):
            if cand in file_set:
                found.add(cand)
    # a.b -> a/b.py or a/b/__init__.py
    pathish = dotted.replace(".", "/")
    for c in (pathish + ".py", pathish + "/__init__.py"):
        if c in file_set:
            found.add(c)
    return found


_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:[^'"]*\s+from\s+)?|export\s+[^'"]*\s+from\s+|require\s*\(\s*)['"]([^'"]+)['"]""",
    re.MULTILINE,
)


def _js_deps(rel: str, text: str, file_set: set[str], index: dict[str, list[str]]) -> set[str]:
    deps: set[str] = set()
    for m in _JS_IMPORT_RE.finditer(text):
        spec = m.group(1)
        if not spec.startswith("."):
            continue
        resolved = _resolve_relative(rel, spec, file_set, js=True)
        if resolved:
            deps.add(resolved)
    return deps


def _go_deps(rel: str, text: str, file_set: set[str], index: dict[str, list[str]]) -> set[str]:
    deps: set[str] = set()
    # same-package files: all .go in same directory
    parent = str(Path(rel).parent).replace("\\", "/")
    if parent == ".":
        parent = ""
    for f in file_set:
        if f.endswith(".go") and f != rel:
            fp = str(Path(f).parent).replace("\\", "/")
            if fp == ".":
                fp = ""
            if fp == parent:
                deps.add(f)
    return deps


_RUST_MOD_RE = re.compile(r"^\s*(?:pub\s+)?mod\s+(\w+)\s*;", re.MULTILINE)
_RUST_USE_RE = re.compile(r"^\s*use\s+(crate|super|self)::([\w:]+)", re.MULTILINE)


def _rust_deps(rel: str, text: str, file_set: set[str], index: dict[str, list[str]]) -> set[str]:
    deps: set[str] = set()
    parent = Path(rel).parent
    for m in _RUST_MOD_RE.finditer(text):
        name = m.group(1)
        for cand in (
            (parent / f"{name}.rs").as_posix(),
            (parent / name / "mod.rs").as_posix(),
        ):
            if cand in file_set:
                deps.add(cand)
    return deps


_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([\w.]+)\s*;", re.MULTILINE)


def _jvm_deps(rel: str, text: str, file_set: set[str], index: dict[str, list[str]]) -> set[str]:
    deps: set[str] = set()
    for m in _JAVA_IMPORT_RE.finditer(text):
        pkg = m.group(1)
        # com.foo.Bar -> com/foo/Bar.java
        pathish = pkg.replace(".", "/")
        for ext in (".java", ".kt"):
            cand = pathish + ext
            if cand in file_set:
                deps.add(cand)
            # also try under src/main/java
            for prefix in ("src/main/java/", "src/main/kotlin/", "src/"):
                c2 = prefix + cand
                if c2 in file_set:
                    deps.add(c2)
    return deps


_C_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def _c_include_deps(rel: str, text: str, file_set: set[str]) -> set[str]:
    deps: set[str] = set()
    for m in _C_INCLUDE_RE.finditer(text):
        resolved = _resolve_relative(rel, m.group(1), file_set, js=False)
        if resolved:
            deps.add(resolved)
    return deps


_RUBY_REQ_RE = re.compile(
    r"""require(?:_relative)?\s+['"]([^'"]+)['"]""",
)


def _ruby_deps(rel: str, text: str, file_set: set[str], index: dict[str, list[str]]) -> set[str]:
    deps: set[str] = set()
    for m in _RUBY_REQ_RE.finditer(text):
        spec = m.group(1)
        if "require_relative" in m.group(0) or spec.startswith("."):
            for ext in ("", ".rb"):
                resolved = _resolve_relative(rel, spec + ext if not spec.endswith(".rb") else spec, file_set)
                if resolved:
                    deps.add(resolved)
        else:
            for cand in index.get(Path(spec).name, []):
                if cand.endswith(".rb") and cand in file_set:
                    deps.add(cand)
    return deps


_PHP_REQ_RE = re.compile(
    r"""(?:require|include)(?:_once)?\s*\(?\s*['"]([^'"]+)['"]""",
)


def _php_deps(rel: str, text: str, file_set: set[str], index: dict[str, list[str]]) -> set[str]:
    deps: set[str] = set()
    for m in _PHP_REQ_RE.finditer(text):
        resolved = _resolve_relative(rel, m.group(1), file_set)
        if resolved:
            deps.add(resolved)
    return deps


def _resolve_relative(
    from_file: str,
    spec: str,
    file_set: set[str],
    *,
    js: bool = False,
) -> str | None:
    base = Path(from_file).parent
    # normalize
    target = (base / spec).as_posix()
    # collapse .. and .
    parts: list[str] = []
    for p in target.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    normalized = "/".join(parts)

    candidates = [normalized]
    if js:
        for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
            candidates.append(normalized + ext)
        candidates.append(normalized + "/index.ts")
        candidates.append(normalized + "/index.js")
    else:
        if not Path(normalized).suffix:
            candidates.append(normalized + ".rb")
            candidates.append(normalized + ".php")
            candidates.append(normalized + ".h")
            candidates.append(normalized + ".c")

    for c in candidates:
        if c in file_set:
            return c
    return None


def _structural_neighbors(rel: str, file_set: set[str]) -> set[str]:
    """Co-located tests and same-stem files."""
    deps: set[str] = set()
    p = Path(rel)
    stem = p.stem
    parent = p.parent.as_posix()
    if parent == ".":
        parent = ""

    patterns = [
        f"{stem}_test{p.suffix}",
        f"test_{stem}{p.suffix}",
        f"{stem}.test{p.suffix}",
        f"{stem}.spec{p.suffix}",
        f"{stem}Test{p.suffix}",
        f"{stem}Tests{p.suffix}",
    ]
    for name in patterns:
        cand = f"{parent}/{name}" if parent else name
        if cand in file_set:
            deps.add(cand)

    # reverse: if this is a test file, link to implementation
    for prefix in ("test_",):
        if stem.startswith(prefix):
            impl = stem[len(prefix) :]
            cand = f"{parent}/{impl}{p.suffix}" if parent else f"{impl}{p.suffix}"
            if cand in file_set:
                deps.add(cand)
    for suffix in (".test", ".spec", "_test", "Test", "Tests"):
        if stem.endswith(suffix):
            impl = stem[: -len(suffix)]
            cand = f"{parent}/{impl}{p.suffix}" if parent else f"{impl}{p.suffix}"
            if cand in file_set:
                deps.add(cand)

    return deps

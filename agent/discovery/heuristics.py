"""Heuristic filters and ranking to shrink candidate sets without an LLM."""

from __future__ import annotations

import re
from pathlib import Path

SOURCE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".kts",
    ".cs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hxx",
    ".swift",
    ".scala",
    ".vue",
    ".svelte",
    ".dart",
    ".lua",
    ".r",
    ".jl",
    ".ex",
    ".exs",
    ".hs",
    ".ml",
    ".mli",
    ".zig",
    ".sh",
    ".bash",
    ".ps1",
    ".sql",
    ".graphql",
    ".gql",
}

CONFIG_NAMES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "tsconfig.json",
    "jsconfig.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "poetry.lock",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "Cargo.lock",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "Gemfile",
    "composer.json",
    "CMakeLists.txt",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    ".env.example",
    "vite.config.ts",
    "vite.config.js",
    "webpack.config.js",
    "next.config.js",
    "next.config.mjs",
    "nuxt.config.ts",
    "manage.py",
    "main.go",
    "main.rs",
    "index.js",
    "index.ts",
    "app.py",
    "main.py",
}

MEDIA_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".o",
    ".a",
    ".pyc",
    ".pyo",
    ".pickle",
    ".pkl",
}

LOCKFILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
    "Gemfile.lock",
}

DEFAULT_MAX_BYTES = 200_000
DEFAULT_MAX_CANDIDATES = 400


def filter_candidates(
    paths: list[str],
    repo_root: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    include_lockfiles: bool = False,
) -> list[str]:
    """
    Filter scanned paths to source/config/docs suitable for agent context.

    Returns relative paths (posix), sorted by usefulness then path.
    """
    root = Path(repo_root).resolve()
    kept: list[tuple[int, str]] = []

    for rel in paths:
        name = Path(rel).name
        suffix = Path(rel).suffix.lower()

        if suffix in MEDIA_EXTENSIONS:
            continue
        if name.endswith(".min.js") or name.endswith(".min.css"):
            continue
        if name in LOCKFILE_NAMES and not include_lockfiles:
            continue
        if name.startswith(".") and name not in {".env.example", ".gitignore"}:
            # keep common useful dotfiles only
            if name not in {".gitignore", ".dockerignore", ".eslintrc", ".prettierrc"}:
                if not name.startswith(".env"):
                    pass

        is_source = suffix in SOURCE_EXTENSIONS
        is_config = name in CONFIG_NAMES or name.startswith("README")
        is_text_config = suffix in {
            ".json",
            ".toml",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
            ".md",
            ".txt",
            ".xml",
            ".html",
            ".css",
            ".scss",
            ".less",
        }

        if not (is_source or is_config or is_text_config):
            continue

        full = root / rel
        try:
            if not full.is_file():
                continue
            size = full.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            continue
        if size == 0 and not is_config:
            continue

        # Prefer source and important configs
        score = 0
        if is_source:
            score += 10
        if name in CONFIG_NAMES or name.startswith("README"):
            score += 8
        if any(part in {"src", "lib", "app", "pkg", "internal"} for part in Path(rel).parts):
            score += 3
        kept.append((score, rel))

    kept.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, p in kept[:max_candidates]]


def rank_by_task_signals(paths: list[str], task: str) -> list[str]:
    """
    Re-rank candidate paths using simple keyword overlap with the task.

    Pure heuristic — no LLM. Tokens in the task are matched against path segments.
    """
    tokens = _task_tokens(task)
    if not tokens:
        return list(paths)

    scored: list[tuple[int, str]] = []
    for path in paths:
        path_l = path.lower()
        score = 0
        for tok in tokens:
            if tok in path_l:
                score += 5
            # basename match is stronger
            if tok in Path(path).stem.lower():
                score += 8
        # slight boost for entry-ish names
        name = Path(path).name.lower()
        if name in {"main.py", "app.py", "index.js", "index.ts", "main.go", "main.rs"}:
            score += 2
        if "test" in path_l or "spec" in path_l:
            score += 1
        scored.append((score, path))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, p in scored]


def _task_tokens(task: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", task.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "add",
        "make",
        "create",
        "update",
        "implement",
        "should",
        "must",
        "will",
        "please",
        "file",
        "code",
        "function",
        "method",
        "class",
        "when",
        "where",
        "which",
        "using",
        "return",
        "have",
        "been",
        "also",
        "just",
        "need",
        "want",
    }
    return [t for t in raw if t not in stop]

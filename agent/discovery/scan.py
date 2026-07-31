"""Scan repository file structure."""

from __future__ import annotations

import os
from pathlib import Path

# Directories skipped during walk (never entered).
SKIP_DIRS = {
    ".git",
    ".svn",
    ".hg",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "vendor",
    "target",
    "dist",
    "build",
    "out",
    "bin",
    "obj",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
    ".next",
    ".nuxt",
    ".turbo",
    "coverage",
    "htmlcov",
    ".cache",
    "eggs",
    ".eggs",
}


def scan_file_tree(
    repo_root: Path | str,
    *,
    max_tree_entries: int = 400,
) -> tuple[list[str], str]:
    """
    Walk the repository and return (relative_paths, tree_text).

    Paths use forward slashes and are relative to repo_root.
    Does not apply extension/size filters — use filter_candidates for that.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path is not a directory: {root}")

    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if d not in SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for name in sorted(filenames):
            if name in {".DS_Store", "Thumbs.db"}:
                continue
            full = Path(dirpath) / name
            try:
                rel = full.resolve().relative_to(root)
            except ValueError:
                continue
            paths.append(rel.as_posix())

    paths.sort()
    tree_text = _format_tree(paths, max_entries=max_tree_entries)
    return paths, tree_text


def _format_tree(paths: list[str], *, max_entries: int) -> str:
    """Compact path list suitable for prompts."""
    if not paths:
        return "(empty repository)"
    shown = paths[:max_entries]
    lines = shown[:]
    if len(paths) > max_entries:
        lines.append(f"... ({len(paths) - max_entries} more files omitted)")
    return "\n".join(lines)

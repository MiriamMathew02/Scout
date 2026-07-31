"""Language-agnostic atomic line-range edits with rollback support."""

from __future__ import annotations

from pathlib import Path

from agent.state import EditHunk


class EditError(Exception):
    """Raised when a hunk cannot be applied safely."""


def apply_hunk(
    repo_root: Path | str,
    hunk: EditHunk,
    *,
    dry_run: bool = False,
    backups: dict[str, str] | None = None,
    allow_create: bool = True,
) -> str:
    """
    Replace lines [start_line, end_line] inclusive with new_text.

    Conventions:
    - start_line/end_line are 1-based.
    - If end_line < start_line, insert new_text before start_line (no deletion).
    - If end_line == 0 and start_line == 1 and file missing, create file.
    - Empty new_text with a valid range deletes those lines.

    Returns a small unified-diff-like summary string.
    """
    root = Path(repo_root).resolve()
    path = _safe_path(root, hunk.path)
    rel = path.relative_to(root).as_posix()

    if not path.exists():
        if not allow_create and hunk.end_line >= hunk.start_line:
            raise EditError(f"File does not exist: {rel}")
        original = ""
        lines: list[str] = []
    else:
        original = path.read_text(encoding="utf-8", errors="replace")
        lines = original.splitlines(keepends=True)

    if backups is not None and rel not in backups:
        backups[rel] = original

    # Normalize line endings in new_text to match file (default \n)
    new_block = hunk.new_text
    if new_block and not new_block.endswith("\n") and "\n" in new_block:
        # keep as-is for partial mid-line replacements; splitlines handles content
        pass
    new_lines = _to_lines(new_block)

    n = len(lines)
    start = hunk.start_line
    end = hunk.end_line

    if path.exists() and n == 0 and start == 1 and end == 0:
        # create/write into empty file
        result_lines = new_lines
    elif not path.exists():
        result_lines = new_lines
    elif end < start:
        # pure insert before `start`
        if start < 1 or start > n + 1:
            raise EditError(f"{rel}: insert line {start} out of range (file has {n} lines)")
        idx = start - 1
        result_lines = lines[:idx] + new_lines + lines[idx:]
    else:
        if start < 1 or end > n or start > end:
            raise EditError(
                f"{rel}: range {start}-{end} out of bounds for file with {n} lines"
            )
        # replace [start, end]
        result_lines = lines[: start - 1] + new_lines + lines[end:]

    new_content = "".join(result_lines)
    # Ensure trailing newline for non-empty text files (common convention)
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"

    summary = _diff_summary(rel, start, end, hunk.new_text)

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content, encoding="utf-8")

    return summary


def apply_edits(
    repo_root: Path | str,
    hunks: list[EditHunk],
    *,
    dry_run: bool = False,
    backups: dict[str, str] | None = None,
) -> list[str]:
    """
    Apply multiple hunks. Within each file, apply bottom-to-top so line
    numbers stay valid relative to the original content for that batch.
    """
    # Group by path preserving reverse line order
    by_path: dict[str, list[EditHunk]] = {}
    for h in hunks:
        by_path.setdefault(h.path.replace("\\", "/"), []).append(h)

    summaries: list[str] = []
    for path, group in by_path.items():
        ordered = sorted(group, key=lambda h: (h.start_line, h.end_line), reverse=True)
        for h in ordered:
            normalized = EditHunk(
                path=path,
                start_line=h.start_line,
                end_line=h.end_line,
                new_text=h.new_text,
                rationale=h.rationale,
            )
            summaries.append(
                apply_hunk(
                    repo_root,
                    normalized,
                    dry_run=dry_run,
                    backups=backups,
                )
            )
    return summaries


def rollback_edits(repo_root: Path | str, backups: dict[str, str]) -> list[str]:
    """Restore files from in-memory backups. Empty string means delete created file."""
    root = Path(repo_root).resolve()
    restored: list[str] = []
    for rel, content in backups.items():
        path = _safe_path(root, rel)
        if content == "":
            if path.exists():
                path.unlink()
                # clean empty parents optionally — skip for safety
                restored.append(f"deleted created file {rel}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            restored.append(f"restored {rel}")
    return restored


def _safe_path(root: Path, rel: str) -> Path:
    rel_norm = rel.replace("\\", "/").lstrip("/")
    if rel_norm.startswith("..") or "/../" in f"/{rel_norm}/":
        raise EditError(f"Path escapes repository: {rel}")
    full = (root / rel_norm).resolve()
    try:
        full.relative_to(root)
    except ValueError as exc:
        raise EditError(f"Path escapes repository: {rel}") from exc
    return full


def _to_lines(text: str) -> list[str]:
    if text == "":
        return []
    # splitlines(keepends) drops final empty; rebuild with \n
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
        return [p + "\n" for p in parts]
    # no trailing newline in input — still write lines with \n except we add file-level later
    return [p + "\n" for p in parts]


def _diff_summary(rel: str, start: int, end: int, new_text: str) -> str:
    n_new = 0 if new_text == "" else len(new_text.splitlines())
    if end < start:
        return f"{rel}: insert {n_new} line(s) before {start}"
    deleted = end - start + 1
    return f"{rel}: replace lines {start}-{end} ({deleted} line(s)) with {n_new} line(s)"

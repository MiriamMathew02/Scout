"""Read files with line numbers and extract prompt-sized excerpts."""

from __future__ import annotations

from pathlib import Path


def read_line_numbered(path: Path | str, *, start: int = 1, end: int | None = None) -> str:
    """
    Return file content with `N|` line prefixes (1-based).

    If end is set, only that inclusive range is included.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    s = max(1, start)
    e = len(lines) if end is None else min(end, len(lines))
    out: list[str] = []
    for i in range(s, e + 1):
        out.append(f"{i}|{lines[i - 1]}")
    return "\n".join(out)


def excerpt_for_prompt(
    path: Path | str,
    *,
    max_lines: int = 200,
    focus_line: int | None = None,
    window: int = 80,
) -> str:
    """
    Build a token-efficient excerpt.

    For small files, return the full line-numbered content (capped).
    For large files, return a window around focus_line (or the head).
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    n = len(lines)

    if n <= max_lines:
        return read_line_numbered(p)

    if focus_line is not None:
        mid = max(1, min(focus_line, n))
        start = max(1, mid - window)
        end = min(n, mid + window)
        header = f"... (lines 1-{start - 1} omitted)\n" if start > 1 else ""
        body = read_line_numbered(p, start=start, end=end)
        footer = f"\n... (lines {end + 1}-{n} omitted)" if end < n else ""
        return header + body + footer

    # Head + note
    body = read_line_numbered(p, start=1, end=max_lines)
    return body + f"\n... ({n - max_lines} more lines omitted)"


def read_raw(path: Path | str) -> str:
    """Read full file text as UTF-8."""
    return Path(path).read_text(encoding="utf-8", errors="replace")

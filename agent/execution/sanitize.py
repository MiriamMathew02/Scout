"""Pre-apply guards: line bounds, duplicate top-level JS requires, obvious bad hunks."""

from __future__ import annotations

import re
from pathlib import Path

from agent.discovery.scan import resolve_inside
from agent.execution.atomic_edit import EditError
from agent.state import EditHunk

# Top-level (unindented) declarations only: names inside functions may repeat freely.
_DECL_RE = re.compile(
    r"^(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=",
    re.MULTILINE,
)
_REQUIRE_RE = re.compile(
    r"""^(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*require\s*\(""",
    re.MULTILINE,
)
_FUNC_RE = re.compile(
    r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
    re.MULTILINE,
)


def validate_and_sanitize_hunks(
    repo_root: Path | str,
    hunks: list[EditHunk],
) -> list[EditHunk]:
    """
    Validate line ranges against current files and strip duplicate
    redeclarations from new_text (common LLM failure mode).
    """
    root = Path(repo_root).resolve()
    cleaned: list[EditHunk] = []

    for h in hunks:
        path = resolve_inside(root, h.path)
        if path is None:
            raise EditError(f"{h.path}: path is outside the repository; use a path inside it")
        if not path.exists():
            # create path — only allow create-style ranges
            if h.start_line == 1 and h.end_line == 0:
                cleaned.append(h)
                continue
            if h.end_line < h.start_line:
                cleaned.append(h)
                continue
            raise EditError(
                f"{h.path}: file does not exist; use start_line=1, end_line=0 to create"
            )

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        n = len(lines)
        start, end = h.start_line, h.end_line

        if end < start:
            if start < 1 or start > n + 1:
                raise EditError(
                    f"{h.path}: insert line {start} out of bounds "
                    f"(file currently has {n} lines)"
                )
        else:
            if start < 1 or end > n or start > end:
                raise EditError(
                    f"{h.path}: range {start}-{end} out of bounds "
                    f"(file currently has {n} lines). "
                    f"Re-read the file and use valid 1-{n} line numbers."
                )

        new_text = h.new_text
        if path.suffix.lower() in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}:
            new_text = _strip_duplicate_js_decls(text, lines, start, end, new_text)

        cleaned.append(
            EditHunk(
                path=h.path.replace("\\", "/"),
                start_line=start,
                end_line=end,
                new_text=new_text,
                rationale=h.rationale,
            )
        )

    # Drop no-ops (empty replace of empty / identical)
    result: list[EditHunk] = []
    for h in cleaned:
        if h.end_line >= h.start_line and h.new_text == "":
            # pure delete — keep
            result.append(h)
            continue
        if h.new_text.strip() == "" and h.end_line < h.start_line:
            # empty insert — drop
            continue
        result.append(h)
    return result


def _strip_duplicate_js_decls(
    full_text: str,
    lines: list[str],
    start: int,
    end: int,
    new_text: str,
) -> str:
    """
    Remove top-level require lines from new_text that re-require a name already
    declared at top level outside the replaced region (prevents 'Identifier X has
    already been declared'). Other duplicates are left for `node --check` to report.
    """
    if end < start:
        kept_outside = lines[: start - 1] + lines[start - 1 :]
    else:
        kept_outside = lines[: start - 1] + lines[end:]

    outside_text = "\n".join(kept_outside)
    existing = _declared_names(outside_text)
    if not existing:
        return new_text

    out_lines: list[str] = []
    for line in new_text.splitlines(keepends=True):
        bare = line.rstrip("\n\r")
        names = _names_declared_on_line(bare)
        if names and any(n in existing for n in names):
            # skip re-declaration
            continue
        out_lines.append(line if line.endswith("\n") else line + "\n")

    # If we stripped everything but original had content, keep original new_text
    # only when all lines were decls — empty is OK for "only duplicate requires"
    result = "".join(out_lines)
    # preserve trailing newline convention of input
    if new_text and not new_text.endswith("\n") and result.endswith("\n"):
        result = result[:-1]
    return result


def _declared_names(text: str) -> set[str]:
    names: set[str] = set()
    for rx in (_DECL_RE, _REQUIRE_RE, _FUNC_RE):
        for m in rx.finditer(text):
            names.add(m.group(1))
    return names


def _names_declared_on_line(line: str) -> set[str]:
    # Only re-added requires are stripped. Any other duplicate fails `node --check`,
    # which rolls the step back and retries with the error message.
    m = _REQUIRE_RE.match(line)
    return {m.group(1)} if m else set()

"""Propose and apply edits for a plan step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.discovery.heuristics import _task_tokens
from agent.discovery.scan import clean_rel_path, resolve_inside
from agent.discovery.snippets import read_line_numbered
from agent.execution.atomic_edit import EditError, apply_edits, read_exact, rollback_edits
from agent.execution.sanitize import validate_and_sanitize_hunks
from agent.llm import chat_json
from agent.planning.plan import is_ops_step
from agent.prompts import EXECUTE_SYSTEM, execute_user_prompt
from agent.state import AgentState, EditHunk, PlanStep
from agent.verification.verify import verify_changes

MAX_HUNK_FRACTION = 0.70
_WINDOW = 90  # lines shown on each side of a focus line in long files


def _parse_hunks(
    raw_edits: list,
    *,
    root: Path,
    step: PlanStep,
) -> list[EditHunk]:
    hunks: list[EditHunk] = []
    for item in raw_edits:
        if not isinstance(item, dict):
            continue
        path = clean_rel_path(str(item.get("path", "")))
        if not path:
            continue
        try:
            start = int(item.get("start_line", 1))
            end = int(item.get("end_line", start))
        except (TypeError, ValueError) as exc:
            raise EditError(f"Invalid line numbers in edit for {path}") from exc
        new_text = item.get("new_text", "")
        if new_text is None:
            new_text = ""
        if not isinstance(new_text, str):
            new_text = str(new_text)
        rationale = str(item.get("rationale", "") or "")

        full = resolve_inside(root, path)
        if full is None:
            raise EditError(f"{path}: path is outside the repository; use a path inside it")
        if full.is_file() and step.action == "modify":
            n = len(full.read_text(encoding="utf-8", errors="replace").splitlines())
            if end >= start and n > 0:
                span = end - start + 1
                if span / n > MAX_HUNK_FRACTION and n > 80:
                    raise EditError(
                        f"Rejected oversized hunk for {path}: lines {start}-{end} "
                        f"cover {span}/{n} of file. Prefer smaller edits."
                    )

        hunks.append(
            EditHunk(
                path=path,
                start_line=start,
                end_line=end,
                new_text=new_text,
                rationale=rationale,
            )
        )
    return hunks


def _guess_focus_line(lines: list[str], hints: str) -> int | None:
    """1-based line mentioning the most words from the step/task, if any line does."""
    tokens = set(_task_tokens(hints))
    best, best_score = None, 0
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        score = sum(1 for t in tokens if t in low)
        if score > best_score:
            best, best_score = i, score
    return best


def _merge_ranges(ranges: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for a, b in sorted((max(1, a), min(n, b)) for a, b in ranges):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _load_file_for_prompt(
    root: Path,
    rel: str,
    *,
    hints: str = "",
    ranges: list[tuple[int, int]] | None = None,
) -> tuple[str, int, list[tuple[int, int]] | None]:
    """Return (prompt text, line count, shown line ranges or None if the whole file is shown)."""
    full = resolve_inside(root, rel)
    if full is None:
        return "(outside the repository; not shown)", 0, None
    if not full.is_file():
        return "(file does not exist yet)", 0, None
    try:
        raw = full.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(unreadable: {exc})", 0, None
    lines = raw.splitlines()
    n = len(lines)
    if n <= 250:
        return read_line_numbered(full), n, None
    if not ranges:
        focus = _guess_focus_line(lines, hints)
        ranges = [(1, 180)] if focus is None else [(focus - _WINDOW, focus + _WINDOW)]
    shown = _merge_ranges(ranges, n)
    parts: list[str] = []
    prev = 0
    for a, b in shown:
        if a > prev + 1:
            parts.append(f"... (lines {prev + 1}-{a - 1} omitted)")
        parts.append("\n".join(f"{i}|{lines[i - 1]}" for i in range(a, b + 1)))
        prev = b
    if prev < n:
        parts.append(f"... (lines {prev + 1}-{n} omitted)")
    return "\n".join(parts), n, shown


def _within(hunk: EditHunk, shown: list[tuple[int, int]]) -> bool:
    for first, last in shown:
        if hunk.end_line < hunk.start_line:  # insert before start_line
            if first <= hunk.start_line <= last + 1:
                return True
        elif first <= hunk.start_line and hunk.end_line <= last:
            return True
    return False


def _related_files_for_step(state: AgentState, step: PlanStep, *, limit: int = 4) -> list[str]:
    """File-graph neighbors + symbol-related files for execute context."""
    from agent.discovery.graph import neighbors

    related: list[str] = []
    seeds = list(step.files)
    if state.graph and seeds:
        for n in neighbors(state.graph, seeds, hops=1):
            if n not in seeds and n not in related:
                related.append(n)

    # Symbol graph: files of related exports / schema
    sg = state.symbol_graph
    if sg is not None and hasattr(sg, "symbols"):
        seed_ids = [
            sid
            for sid, sym in sg.symbols.items()  # type: ignore[attr-defined]
            if sym.file in step.files
        ]
        if seed_ids and hasattr(sg, "related"):
            for sid in sg.related(seed_ids, hops=1)[:20]:  # type: ignore[attr-defined]
                sym = sg.symbols.get(sid)  # type: ignore[attr-defined]
                if sym and sym.file not in step.files and sym.file not in related:
                    related.append(sym.file)

    # Prefer model/controller/routes ordering
    def rank(p: str) -> tuple[int, str]:
        pl = p.lower()
        if "model" in pl:
            return (0, p)
        if "controller" in pl:
            return (1, p)
        if "route" in pl:
            return (2, p)
        return (3, p)

    related = sorted(related, key=rank)
    return related[:limit]


def propose_edits(
    state: AgentState,
    step: PlanStep,
    *,
    client: Any = None,
    force: bool = False,
    bound_error: str | None = None,
    ranges: dict[str, list[tuple[int, int]]] | None = None,
) -> list[EditHunk]:
    """Ask the LLM for EditHunks for one plan step. Retries once if empty/invalid."""
    root = Path(state.repo_path)
    files_content: dict[str, str] = {}
    line_counts: dict[str, int] = {}
    partial: dict[str, list[tuple[int, int]]] = {}

    primary = list(step.files)
    related = _related_files_for_step(state, step)
    for rel in primary + related:
        if rel in files_content:
            continue
        content, n, shown = _load_file_for_prompt(
            root,
            rel,
            hints=f"{step.description} {state.task}",
            ranges=(ranges or {}).get(rel),
        )
        files_content[rel] = content
        if n:
            line_counts[rel] = n
        if shown:
            partial[rel] = shown

    counts_note = ", ".join(f"{p}={n} lines" for p, n in line_counts.items())
    extra_parts: list[str] = []
    if counts_note:
        extra_parts.append(
            f"Line counts for current files: {counts_note}. "
            "Never use start_line/end_line beyond these counts."
        )
    if partial:
        shown_note = "; ".join(
            f"{p} lines " + ", ".join(f"{a}-{b}" for a, b in rs) for p, rs in partial.items()
        )
        extra_parts.append(
            f"Only part of some files is shown: {shown_note}. Edit only lines that are shown."
        )
    if state.last_error:
        extra_parts.append(state.last_error)
    if bound_error:
        extra_parts.append(bound_error)
    if force:
        extra_parts.append(
            "Your previous response had zero usable edits or invalid line ranges. "
            "Return at least one concrete code edit with valid line numbers. "
            "Do NOT re-declare const/let/var/require identifiers that already exist "
            "outside the lines you replace."
        )
    extra_error = "\n".join(extra_parts) if extra_parts else None

    related_note = None
    if related:
        related_note = (
            "Related files included for context (via dependency/symbol graph): "
            + ", ".join(related)
            + ". If you add a schema field, update create/update writers here too."
        )

    user = execute_user_prompt(
        task=state.task,
        step_description=step.description,
        action=step.action,
        files_content=files_content,
        last_error=extra_error,
        related_note=related_note,
    )

    data = chat_json(
        system=EXECUTE_SYSTEM,
        user=user,
        model=state.model,
        client=client,
        max_tokens=6000,
    )

    raw_edits = data.get("edits") or data.get("items") or []
    if not isinstance(raw_edits, list):
        raise EditError("LLM returned non-list edits")

    hunks = _parse_hunks(raw_edits, root=root, step=step)

    unseen = [h for h in hunks if h.path in partial and not _within(h, partial[h.path])]
    if unseen:
        where = ", ".join(f"{h.path} lines {h.start_line}-{h.end_line}" for h in unseen)
        msg = f"Some edits target lines you were not shown ({where})."
        if ranges is None:
            # Keep what was shown, add every region the model asked about, and ask once more
            wider = {p: list(rs) for p, rs in partial.items()}
            for h in unseen:
                wider[h.path].append((h.start_line - _WINDOW, max(h.start_line, h.end_line) + _WINDOW))
            return propose_edits(
                state,
                step,
                client=client,
                force=True,
                bound_error=msg + " Those parts are shown now; base the edits on them.",
                ranges=wider,
            )
        raise EditError(msg)

    if not hunks and not force:
        return propose_edits(state, step, client=client, force=True)

    return hunks


def run_plan_step(
    state: AgentState,
    step: PlanStep,
    *,
    client: Any = None,
) -> list[str]:
    """Propose edits for a step, sanitize, apply, then syntax-check."""
    root = Path(state.repo_path)

    if is_ops_step(step.description):
        step.status = "skipped"
        return [f"step {step.id}: skipped non-code/ops step"]

    if step.action == "delete":
        summaries: list[str] = []
        for rel in step.files:
            path = resolve_inside(root, rel)
            if path is None or not rel or (root / rel).is_symlink():
                raise EditError(f"{rel}: refusing to delete a path outside the repository or a symlink")
            if not path.exists():
                summaries.append(f"{rel}: already absent")
                continue
            if state.backups is not None and rel not in state.backups:
                state.backups[rel] = read_exact(path)
            if not state.dry_run:
                path.unlink()
            summaries.append(f"{rel}: deleted")
            state.applied_edits.append(
                EditHunk(path=rel, start_line=1, end_line=0, new_text="", rationale="delete")
            )
        return summaries

    hunks = propose_edits(state, step, client=client)
    if not hunks:
        raise EditError(
            f"step {step.id}: model proposed no code edits for "
            f"'{step.description}'. Need concrete line-level changes."
        )

    try:
        hunks = validate_and_sanitize_hunks(root, hunks)
    except EditError as exc:
        # Re-prompt once with the bounds / sanitize error
        hunks = propose_edits(
            state, step, client=client, force=True, bound_error=str(exc)
        )
        hunks = validate_and_sanitize_hunks(root, hunks)

    if not hunks:
        raise EditError(
            f"step {step.id}: all proposed edits were invalid or empty after sanitization"
        )

    touched = list(dict.fromkeys(h.path for h in hunks))
    # Snapshot pre-step content so a bad step does not wipe earlier good steps
    pre_step: dict[str, str | None] = {}
    for rel in touched:
        path = root / rel
        pre_step[rel] = read_exact(path) if path.is_file() else None

    try:
        summaries = apply_edits(
            root,
            hunks,
            dry_run=state.dry_run,
            backups=state.backups,
        )
    except EditError:
        if not state.dry_run:
            rollback_edits(root, pre_step)  # leave no half-applied step behind
        raise
    state.applied_edits.extend(hunks)

    if not state.dry_run:
        result = verify_changes(root, touched, state.stack_hints, run_tests=False)
        if not result.ok:
            rollback_edits(root, pre_step)
            keep = len(state.applied_edits) - len(hunks)
            state.applied_edits = state.applied_edits[:keep]
            raise EditError(
                f"step {step.id} broke syntax after apply:\n{result.as_text()}\n"
                "Do not re-add existing const/require declarations; "
                "use line numbers from the latest file content."
            )

    return summaries

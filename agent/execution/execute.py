"""Propose and apply edits for a plan step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.discovery.snippets import excerpt_for_prompt, read_line_numbered
from agent.execution.atomic_edit import EditError, apply_edits, rollback_edits
from agent.execution.sanitize import validate_and_sanitize_hunks
from agent.llm import chat_json
from agent.prompts import EXECUTE_SYSTEM, execute_user_prompt
from agent.state import AgentState, EditHunk, PlanStep
from agent.verification.verify import verify_changes

MAX_HUNK_FRACTION = 0.70
_SKIP_STEP_RE = (
    "restart",
    "reboot",
    "install package",
    "npm install",
    "yarn install",
    "pip install",
    "docker",
    "deploy",
    "manual",
)


def _is_noop_ops_step(step: PlanStep) -> bool:
    text = f"{step.description} {' '.join(step.files)}".lower()
    return any(k in text for k in _SKIP_STEP_RE)


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
        path = str(item.get("path", "")).replace("\\", "/").lstrip("./")
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
        if new_text.strip() and not _looks_like_code(new_text) and len(new_text) < 40:
            continue
        rationale = str(item.get("rationale", "") or "")

        full = root / path
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


def _looks_like_code(text: str) -> bool:
    code_signals = (
        "{",
        "}",
        ";",
        "=",
        "function",
        "const ",
        "let ",
        "var ",
        "def ",
        "class ",
        "import ",
        "require(",
        "module.exports",
        "exports.",
        "return ",
        "=>",
        "/*",
        "//",
    )
    return any(s in text for s in code_signals)


def propose_edits(
    state: AgentState,
    step: PlanStep,
    *,
    client: Any = None,
    force: bool = False,
    bound_error: str | None = None,
) -> list[EditHunk]:
    """Ask the LLM for EditHunks for one plan step. Retries once if empty/invalid."""
    root = Path(state.repo_path)
    files_content: dict[str, str] = {}
    line_counts: dict[str, int] = {}

    for rel in step.files:
        full = root / rel
        if full.is_file():
            try:
                raw = full.read_text(encoding="utf-8", errors="replace")
                size_lines = len(raw.splitlines())
                line_counts[rel] = size_lines
                if size_lines > 250:
                    files_content[rel] = excerpt_for_prompt(full, max_lines=220)
                else:
                    files_content[rel] = read_line_numbered(full)
            except OSError as exc:
                files_content[rel] = f"(unreadable: {exc})"
        else:
            files_content[rel] = "(file does not exist yet)"

    counts_note = ", ".join(f"{p}={n} lines" for p, n in line_counts.items())
    extra_parts: list[str] = []
    if counts_note:
        extra_parts.append(
            f"Line counts for current files: {counts_note}. "
            "Never use start_line/end_line beyond these counts."
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

    user = execute_user_prompt(
        task=state.task,
        step_description=step.description,
        action=step.action,
        files_content=files_content,
        last_error=extra_error,
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

    if _is_noop_ops_step(step):
        step.status = "skipped"
        return [f"step {step.id}: skipped non-code/ops step"]

    if step.action == "delete":
        summaries: list[str] = []
        for rel in step.files:
            path = root / rel
            if not path.exists():
                summaries.append(f"{rel}: already absent")
                continue
            if state.backups is not None and rel not in state.backups:
                state.backups[rel] = path.read_text(encoding="utf-8", errors="replace")
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
    pre_step: dict[str, str] = {}
    for rel in touched:
        path = root / rel
        if path.is_file():
            pre_step[rel] = path.read_text(encoding="utf-8", errors="replace")
        else:
            pre_step[rel] = ""

    summaries = apply_edits(
        root,
        hunks,
        dry_run=state.dry_run,
        backups=state.backups,
    )
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

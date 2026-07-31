"""Phase-based agent loop (simple state machine, no LangGraph)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent.discovery.explore import explore_with_llm
from agent.discovery.graph import build_dependency_graph, neighbors
from agent.discovery.heuristics import filter_candidates, rank_by_task_signals
from agent.discovery.scan import scan_file_tree
from agent.discovery.stack import detect_stack
from agent.execution.atomic_edit import EditError, rollback_edits
from agent.execution.execute import run_plan_step
from agent.planning.plan import create_plan
from agent.state import AgentState, Phase
from agent.summary.summarize import build_summary
from agent.verification.verify import verify_changes


Handler = Callable[[AgentState], None]


def _safe_rollback(state: AgentState) -> None:
    """Restore any backed-up files; clear edit bookkeeping."""
    if state.backups and not state.dry_run:
        try:
            rollback_edits(state.repo_path, state.backups)
            state.note(f"rolled back {len(state.backups)} file(s)")
        except Exception as exc:  # noqa: BLE001
            state.note(f"rollback warning: {exc}")
    state.backups.clear()
    state.applied_edits.clear()


def run_agent(state: AgentState, *, client: Any = None) -> AgentState:
    """Run the discover → plan → execute → verify → summarize loop."""
    handlers: dict[Phase, Handler] = {
        Phase.DISCOVER: lambda s: handle_discover(s, client=client),
        Phase.PLAN: lambda s: handle_plan(s, client=client),
        Phase.EXECUTE: lambda s: handle_execute(s, client=client),
        Phase.VERIFY: lambda s: handle_verify(s),
        Phase.SUMMARIZE: lambda s: handle_summarize(s),
    }

    while state.phase not in (Phase.DONE, Phase.FAILED):
        state.iteration += 1
        if state.iteration > state.max_iterations:
            state.last_error = (
                "max iterations exceeded — stopping to avoid further damage. "
                "Partial edits were rolled back when possible."
            )
            _safe_rollback(state)
            state.phase = Phase.FAILED
            state.summary = build_summary(state)
            break

        state.note(f"phase={state.phase.value} iteration={state.iteration}")
        handler = handlers.get(state.phase)
        if handler is None:
            state.last_error = f"no handler for phase {state.phase}"
            _safe_rollback(state)
            state.phase = Phase.FAILED
            break

        try:
            handler(state)
        except Exception as exc:  # noqa: BLE001
            state.note(f"handler error: {exc}")
            state.last_error = str(exc)
            if state.phase == Phase.EXECUTE and state.retry_count < state.max_execute_retries:
                state.retry_count += 1
                _safe_rollback(state)
                for step in state.plan:
                    if step.status != "skipped":
                        step.status = "pending"
                state.note(f"retry execute ({state.retry_count}/{state.max_execute_retries})")
                continue
            if state.phase in (Phase.PLAN, Phase.DISCOVER) and state.replan_count < state.max_replans:
                state.replan_count += 1
                _safe_rollback(state)
                state.phase = Phase.DISCOVER if state.phase == Phase.DISCOVER else Phase.PLAN
                continue
            _safe_rollback(state)
            state.phase = Phase.FAILED
            state.summary = build_summary(state)
            break

    if state.phase == Phase.FAILED:
        _safe_rollback(state)

    if state.phase == Phase.DONE and not state.summary:
        state.summary = build_summary(state)
    if state.phase == Phase.FAILED and not state.summary:
        state.summary = build_summary(state)
    return state


def handle_discover(state: AgentState, *, client: Any = None) -> None:
    root = Path(state.repo_path)
    state.note("scanning repository")
    all_files, tree = scan_file_tree(root)
    state.all_files = all_files
    state.tree_text = tree

    state.candidates = filter_candidates(all_files, root)
    state.candidates = rank_by_task_signals(state.candidates, state.task)
    state.stack_hints = detect_stack(root, state.candidates)
    state.note(
        f"candidates={len(state.candidates)} languages={state.stack_hints.languages}"
    )

    # Graph on a capped set for speed
    graph_files = state.candidates[:250]
    state.graph = build_dependency_graph(root, graph_files)
    state.note(f"graph nodes={len(state.graph)}")

    ranked, summary = explore_with_llm(state, tree, client=client)
    state.relevant_files = neighbors(state.graph, ranked, hops=1)
    # Prefer candidates order stability for unknowns
    state.repo_summary = summary
    state.note(f"relevant_files={len(state.relevant_files)}")
    state.phase = Phase.PLAN


def handle_plan(state: AgentState, *, client: Any = None) -> None:
    state.note("creating plan")
    state.plan = create_plan(state, client=client)
    for step in state.plan:
        step.status = "pending"
    state.note(f"plan steps={len(state.plan)}")
    state.phase = Phase.EXECUTE


def handle_execute(state: AgentState, *, client: Any = None) -> None:
    state.note("executing plan")
    pending = [s for s in state.plan if s.status == "pending"]
    if not pending:
        # re-run all if retry after verify
        for s in state.plan:
            if s.status in {"failed", "done"}:
                s.status = "pending"
        pending = [s for s in state.plan if s.status == "pending"]

    edits_before = len(state.applied_edits)
    try:
        for step in pending:
            if step.status == "skipped":
                continue
            state.note(f"step {step.id}: {step.description}")
            summaries = run_plan_step(state, step, client=client)
            for line in summaries:
                state.note(line)
            if step.status != "skipped":
                step.status = "done"
        # Must actually change code for a successful implement run
        if len(state.applied_edits) == edits_before and not any(
            s.status == "done" and s.action == "delete" for s in state.plan
        ):
            # All steps skipped or empty — treat as failure to replan
            raise EditError(
                "No code edits were applied for the task. "
                "Plan steps produced no file changes."
            )
        state.last_error = None
        state.phase = Phase.VERIFY
    except EditError as exc:
        state.last_error = str(exc)
        state.note(f"edit error: {exc}")
        # mark current pending as failed
        for step in state.plan:
            if step.status == "pending":
                step.status = "failed"
                break
        if state.retry_count < state.max_execute_retries:
            state.retry_count += 1
            _safe_rollback(state)
            for step in state.plan:
                if step.status != "skipped":
                    step.status = "pending"
            state.note(f"retrying execute ({state.retry_count})")
            return
        if state.replan_count < state.max_replans:
            state.replan_count += 1
            _safe_rollback(state)
            state.note("replanning after edit failures")
            state.phase = Phase.PLAN
            return
        state.phase = Phase.FAILED


def handle_verify(state: AgentState) -> None:
    state.note("verifying changes")
    touched = list(dict.fromkeys(h.path for h in state.applied_edits))

    if not touched and not state.dry_run:
        state.verify_result = "FAIL\nNo files were modified; task not implemented."
        state.note("FAIL — no files modified")
        state.last_error = state.verify_result
        if state.replan_count < state.max_replans:
            state.replan_count += 1
            for step in state.plan:
                if step.status != "skipped":
                    step.status = "pending"
            state.note("replanning because no files changed")
            state.phase = Phase.PLAN
            return
        state.phase = Phase.FAILED
        return

    result = verify_changes(
        state.repo_path,
        touched,
        state.stack_hints,
        run_tests=state.run_tests,
    )
    state.verify_result = result.as_text()
    state.note(state.verify_result.splitlines()[0] if state.verify_result else "verify done")

    if result.ok:
        state.phase = Phase.SUMMARIZE
        return

    state.last_error = state.verify_result
    if state.retry_count < state.max_execute_retries:
        state.retry_count += 1
        _safe_rollback(state)
        for step in state.plan:
            if step.status != "skipped":
                step.status = "pending"
        state.note(f"verify failed; retry execute ({state.retry_count})")
        state.phase = Phase.EXECUTE
        return

    if state.replan_count < state.max_replans:
        state.replan_count += 1
        _safe_rollback(state)
        state.note("verify failed; replanning")
        state.phase = Phase.PLAN
        return

    state.phase = Phase.FAILED


def handle_summarize(state: AgentState) -> None:
    # Mark DONE first so the report status is SUCCESS
    state.phase = Phase.DONE
    state.summary = build_summary(state)
    state.note("summarized")

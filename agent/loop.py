"""Phase-based agent loop (simple state machine, no LangGraph)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent.discovery.explore import explore_with_llm
from agent.discovery.graph import build_dependency_graph, neighbors
from agent.discovery.heuristics import filter_candidates, rank_by_task_signals
from agent.discovery.scan import scan_file_tree
from agent.discovery.stack import detect_stack
from agent.discovery.symbols import build_symbol_graph
from agent.execution.atomic_edit import EditError, rollback_edits
from agent.execution.execute import run_plan_step
from agent.planning.plan import create_plan
from agent.state import AgentState, Phase, PlanStep
from agent.summary.summarize import build_summary
from agent.verification.behavior import check_behavior
from agent.verification.consistency import check_consistency
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
    # repair steps only make sense on top of the edits that were just undone
    state.plan = [s for s in state.plan if not s.id.startswith("repair")]


def _queue_repair_step(state: AgentState, files: list[str], check: str) -> None:
    """Add one step that fixes what a check reported; finished steps keep their edits."""
    root = Path(state.repo_path)
    state.plan.append(
        PlanStep(
            id=f"repair-{state.retry_count}",
            description=f"Fix the problems reported by the {check} check",
            files=[f for f in dict.fromkeys(files) if f and (root / f).is_file()],
            action="modify",
        )
    )


def run_agent(state: AgentState, *, client: Any = None) -> AgentState:
    """Run discover → plan → execute → verify → consistency → behavior → summarize."""
    try:
        return _run_phases(state, client=client)
    except BaseException:
        # Ctrl+C (KeyboardInterrupt is not an Exception) or a crash outside the
        # phase handlers: backups live only in memory, so restore files now
        _safe_rollback(state)
        raise


def _run_phases(state: AgentState, *, client: Any = None) -> AgentState:
    handlers: dict[Phase, Handler] = {
        Phase.DISCOVER: lambda s: handle_discover(s, client=client),
        Phase.PLAN: lambda s: handle_plan(s, client=client),
        Phase.EXECUTE: lambda s: handle_execute(s, client=client),
        Phase.VERIFY: lambda s: handle_verify(s),
        Phase.CONSISTENCY: lambda s: handle_consistency(s, client=client),
        Phase.BEHAVIOR: lambda s: handle_behavior(s, client=client),
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

    graph_files = state.candidates[:250]
    state.graph = build_dependency_graph(root, graph_files)
    state.note(f"file graph nodes={len(state.graph)}")

    state.symbol_graph = build_symbol_graph(root, graph_files)
    n_sym = len(getattr(state.symbol_graph, "symbols", {}))
    state.note(f"symbol graph nodes={n_sym}")

    if state.fast:
        # Skip explore LLM — use heuristic shortlist only
        ranked = state.candidates[:15]
        state.repo_summary = (
            f"Fast mode (no explore LLM). Languages: "
            f"{', '.join(state.stack_hints.languages)}. "
            f"Top candidates: {', '.join(ranked[:8])}"
        )
        state.note("explore skipped (--fast)")
    else:
        ranked, summary = explore_with_llm(state, tree, client=client)
        state.repo_summary = summary

    state.relevant_files = neighbors(state.graph, ranked, hops=1)
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
        # Prefer re-running only previously failed step when set
        if state.failed_step_id:
            for s in state.plan:
                if s.id == state.failed_step_id:
                    s.status = "pending"
                    pending = [s]
                    break
        if not pending:
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
            try:
                summaries = run_plan_step(state, step, client=client)
            except EditError:
                state.failed_step_id = step.id
                raise
            for line in summaries:
                state.note(line)
            if step.status != "skipped":
                step.status = "done"
        state.failed_step_id = None
        if len(state.applied_edits) == edits_before and not any(
            s.status == "done" and s.action == "delete" for s in state.plan
        ):
            # Allow zero new edits on repair rounds if files already changed earlier
            if not state.applied_edits and not state.backups:
                raise EditError(
                    "No code edits were applied for the task. "
                    "Plan steps produced no file changes."
                )
        state.last_error = None
        state.phase = Phase.VERIFY
    except EditError as exc:
        state.last_error = str(exc)
        state.note(f"edit error: {exc}")
        for step in state.plan:
            if step.status == "pending":
                step.status = "failed"
                state.failed_step_id = step.id
                break
        if state.retry_count < state.max_execute_retries:
            state.retry_count += 1
            if state.failed_step_id:
                # Step-local retry: run_plan_step leaves nothing behind from the failed
                # step, so finished steps keep their edits and only the failed step reruns
                for step in state.plan:
                    if step.status == "failed":
                        step.status = "pending"
            else:
                _safe_rollback(state)
                for step in state.plan:
                    if step.status != "skipped":
                        step.status = "pending"
            state.note(
                f"retrying execute ({state.retry_count}) "
                f"(focus step={state.failed_step_id or 'all'})"
            )
            return
        if state.replan_count < state.max_replans:
            state.replan_count += 1
            _safe_rollback(state)
            state.failed_step_id = None
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
        run_tests=False,  # tests deferred to BEHAVIOR when --run-tests
    )
    state.verify_result = result.as_text()
    state.note(state.verify_result.splitlines()[0] if state.verify_result else "verify done")

    if result.ok:
        state.phase = Phase.CONSISTENCY
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


def handle_consistency(state: AgentState, *, client: Any = None) -> None:
    """Cross-file wiring: changed files + file-graph neighbors."""
    state.note("consistency check (changed files + graph neighbors)")
    touched = list(dict.fromkeys(h.path for h in state.applied_edits))
    root = Path(state.repo_path)
    deleted = [t for t in touched if state.backups.get(t) is not None and not (root / t).exists()]

    scope_seeds = list(dict.fromkeys(touched + state.relevant_files))[:250]
    if scope_seeds:
        try:
            fresh = build_dependency_graph(
                root,
                list(dict.fromkeys(list(state.graph.keys()) + scope_seeds))[:400],
            )
            state.graph = fresh
            state.symbol_graph = build_symbol_graph(
                root,
                list(dict.fromkeys(list(state.candidates[:200]) + scope_seeds))[:400],
            )
        except Exception as exc:  # noqa: BLE001
            state.note(f"graph refresh skipped: {exc}")

    # LLM consistency only with --deep-review (speed default: static only)
    use_llm = bool(state.deep_review) and not state.fast
    result = check_consistency(
        root,
        touched,
        state.graph,
        hops=1,
        use_llm=use_llm,
        model=state.model,
        task=state.task,
        client=client,
        deleted=deleted,
    )
    state.consistency_result = result.as_text()
    state.note(
        "consistency PASS"
        if result.ok
        else f"consistency FAIL ({len(result.issues)} issue(s))"
    )
    for issue in result.issues[:8]:
        state.note(issue.message)

    if result.ok:
        state.phase = Phase.BEHAVIOR
        return

    state.last_error = (
        "Cross-file consistency check failed. Fix these without breaking other behavior:\n"
        + result.as_text()
    )
    if state.retry_count < state.max_execute_retries:
        state.retry_count += 1
        issue_files = [f for i in result.issues for f in (i.file, i.related_file)]
        _queue_repair_step(state, issue_files + touched, "consistency")
        state.note(
            f"consistency failed; retry execute with fix context "
            f"({state.retry_count}/{state.max_execute_retries})"
        )
        state.phase = Phase.EXECUTE
        return

    if state.replan_count < state.max_replans:
        state.replan_count += 1
        state.note("consistency failed; replanning")
        state.phase = Phase.PLAN
        return

    state.phase = Phase.FAILED


def handle_behavior(state: AgentState, *, client: Any = None) -> None:
    """Field plumbing + optional acceptance LLM + optional project tests."""
    state.note("behavior check (field plumbing / acceptance)")
    touched = list(dict.fromkeys(h.path for h in state.applied_edits))
    use_llm = bool(state.deep_review) and not state.fast

    result = check_behavior(
        state.repo_path,
        task=state.task,
        touched=touched,
        backups=state.backups,
        candidates=state.candidates,
        stack_hints=state.stack_hints,
        run_tests=state.run_tests,
        use_llm=use_llm,
        model=state.model,
        client=client,
        symbol_graph=state.symbol_graph,  # type: ignore[arg-type]
    )
    state.behavior_result = result.as_text()
    state.note("behavior PASS" if result.ok else "behavior FAIL")
    for line in result.messages[:10]:
        state.note(line)

    if result.ok:
        state.phase = Phase.SUMMARIZE
        return

    state.last_error = (
        "Behavior / acceptance check failed. Fix these functional gaps:\n"
        + result.as_text()
    )
    if state.retry_count < state.max_execute_retries:
        state.retry_count += 1
        _queue_repair_step(state, touched, "behavior")
        state.note(
            f"behavior failed; retry execute with fix context "
            f"({state.retry_count}/{state.max_execute_retries})"
        )
        state.phase = Phase.EXECUTE
        return

    if state.replan_count < state.max_replans:
        state.replan_count += 1
        state.note("behavior failed; replanning")
        state.phase = Phase.PLAN
        return

    state.phase = Phase.FAILED


def handle_summarize(state: AgentState) -> None:
    state.phase = Phase.DONE
    state.summary = build_summary(state)
    state.note("summarized")

"""Create and validate execution plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.discovery.graph import topological_edit_order
from agent.discovery.scan import clean_rel_path, resolve_inside
from agent.llm import chat_json
from agent.prompts import PLAN_SYSTEM, plan_user_prompt
from agent.state import AgentState, PlanStep


def create_plan(state: AgentState, *, client: Any = None) -> list[PlanStep]:
    """LLM produces a brief plan; validate and lightly reorder by graph."""
    # Induced subgraph for relevant files
    graph_slice = {
        f: [d for d in state.graph.get(f, []) if d in set(state.relevant_files)]
        for f in state.relevant_files
    }

    user = plan_user_prompt(
        task=state.task,
        repo_summary=state.repo_summary,
        relevant_files=state.relevant_files,
        graph_edges=graph_slice,
        last_error=state.last_error,
    )

    data = chat_json(
        system=PLAN_SYSTEM,
        user=user,
        model=state.model,
        client=client,
    )

    raw_steps = data.get("steps") or data.get("items") or []
    if not isinstance(raw_steps, list):
        raise ValueError("Plan JSON missing steps list")

    steps: list[PlanStep] = []
    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        files = item.get("files") or []
        if isinstance(files, str):
            files = [files]
        files = [clean_rel_path(str(f)) for f in files]
        action = str(item.get("action", "modify")).lower()
        if action not in {"modify", "create", "delete"}:
            action = "modify"
        steps.append(
            PlanStep(
                id=str(item.get("id") or i),
                description=str(item.get("description") or f"Step {i}"),
                files=files,
                action=action,
                status="pending",
            )
        )

    steps = validate_plan(state, steps)

    # Reorder steps by primary file dependency order when possible
    primary = []
    for s in steps:
        primary.append(s.files[0] if s.files else "")
    order_map = {
        f: i
        for i, f in enumerate(
            topological_edit_order(state.graph, [f for f in primary if f])
        )
    }
    steps.sort(key=lambda s: order_map.get(s.files[0], 999) if s.files else 999)
    # re-id for clarity
    for i, s in enumerate(steps, start=1):
        s.id = str(i)
    return steps


# Phrases that mean "run an operation" rather than "edit a file". Kept specific on
# purpose: a bare "docker" or "deploy" also matches real edits like "Update the Dockerfile".
_OPS_PHRASES = (
    "restart the server",
    "restart the app",
    "restart the service",
    "reboot",
    "install dependenc",
    "install package",
    "npm install",
    "yarn install",
    "pnpm install",
    "pip install",
    "run the server",
    "start the server",
    "run docker",
    "docker compose up",
    "docker-compose up",
    "deploy to production",
    "deploy the app",
    "redeploy",
    "manual step",
    "manually test",
    "manual testing",
)


def is_ops_step(description: str) -> bool:
    """True when a step asks for an operation (install, restart, deploy), not an edit."""
    text = description.lower()
    return any(p in text for p in _OPS_PHRASES)


def validate_plan(state: AgentState, steps: list[PlanStep]) -> list[PlanStep]:
    """Drop empty/ops steps; warn on missing modify targets; cap length."""
    root = Path(state.repo_path)
    valid: list[PlanStep] = []
    for step in steps[:10]:
        if not step.description.strip():
            continue
        if is_ops_step(step.description):
            continue
        step.files = [f for f in step.files if resolve_inside(root, f) is not None]
        if not step.files and step.action != "create":
            continue
        if step.action == "modify":
            # keep even if missing — execute may fail and replan
            missing = [f for f in step.files if not (root / f).exists()]
            if missing and len(missing) == len(step.files):
                # all missing — convert to create if intentional paths
                step.action = "create"
        valid.append(step)

    if not valid:
        # Fallback single step over top relevant files
        files = state.relevant_files[:5] or state.candidates[:5]
        valid = [
            PlanStep(
                id="1",
                description=f"Implement: {state.task}",
                files=files,
                action="modify",
            )
        ]
    return valid

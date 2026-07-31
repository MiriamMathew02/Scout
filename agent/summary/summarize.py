"""Build a human-readable summary of the agent run (no LLM required)."""

from __future__ import annotations

from agent.state import AgentState, Phase


def build_summary(state: AgentState) -> str:
    lines: list[str] = []
    # SUMMARIZE is the success path before phase flips to DONE
    status = (
        "SUCCESS"
        if state.phase in (Phase.DONE, Phase.SUMMARIZE)
        else "FAILED"
    )
    lines.append(f"# Agent run: {status}")
    lines.append("")
    lines.append(f"**Task:** {state.task}")
    lines.append(f"**Repo:** {state.repo_path}")
    lines.append(f"**Model:** {state.model}")
    if state.dry_run:
        lines.append("**Mode:** dry-run (no files written)")
    lines.append("")

    if state.repo_summary:
        lines.append("## Repository understanding")
        lines.append(state.repo_summary)
        lines.append("")

    if state.relevant_files:
        lines.append("## Relevant files")
        for f in state.relevant_files:
            lines.append(f"- `{f}`")
        lines.append("")

    if state.plan:
        lines.append("## Plan")
        for step in state.plan:
            files = ", ".join(f"`{f}`" for f in step.files) or "(no files)"
            lines.append(
                f"- [{step.status}] **{step.id}** ({step.action}): {step.description} — {files}"
            )
        lines.append("")

    if state.applied_edits:
        lines.append("## Edits applied")
        for h in state.applied_edits:
            span = f"{h.start_line}-{h.end_line}"
            why = f" — {h.rationale}" if h.rationale else ""
            lines.append(f"- `{h.path}` lines {span}{why}")
        lines.append("")
    else:
        lines.append("## Edits applied")
        lines.append("None.")
        lines.append("")

    if state.verify_result:
        lines.append("## Verification (syntax)")
        lines.append("```")
        lines.append(state.verify_result.strip())
        lines.append("```")
        lines.append("")

    if state.consistency_result:
        lines.append("## Consistency (changed + neighbors)")
        lines.append("```")
        lines.append(state.consistency_result.strip())
        lines.append("```")
        lines.append("")

    if state.last_error and state.phase == Phase.FAILED:
        lines.append("## Error")
        lines.append(state.last_error)
        lines.append("")

    if state.log:
        lines.append("## Log (tail)")
        for entry in state.log[-20:]:
            lines.append(f"- {entry}")

    return "\n".join(lines)

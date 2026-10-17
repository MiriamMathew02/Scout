"""LLM-assisted relevance ranking for discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.discovery.scan import clean_rel_path
from agent.discovery.snippets import excerpt_for_prompt
from agent.llm import chat_json
from agent.prompts import EXPLORE_SYSTEM, explore_user_prompt
from agent.state import AgentState


def explore_with_llm(
    state: AgentState,
    tree_text: str,
    *,
    shortlist_size: int = 40,
    snippet_count: int = 12,
    client: Any = None,
) -> tuple[list[str], str]:
    """
    Ask the LLM which files are relevant; return (ranked_paths, repo_summary).

    Falls back to top shortlist if the LLM call fails.
    """
    root = Path(state.repo_path)
    shortlist = state.candidates[:shortlist_size]

    snippets: dict[str, str] = {}
    for rel in shortlist[:snippet_count]:
        full = root / rel
        if full.is_file():
            try:
                snippets[rel] = excerpt_for_prompt(full, max_lines=80)
            except OSError:
                continue

    state.file_snippets.update(snippets)

    user = explore_user_prompt(
        task=state.task,
        tree_text=tree_text,
        candidates=shortlist,
        snippets=snippets,
        stack_hints=state.stack_hints,
    )

    try:
        data = chat_json(
            system=EXPLORE_SYSTEM,
            user=user,
            model=state.model,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001 — fallback for offline/rate limits
        state.note(f"explore LLM failed, using heuristic shortlist: {exc}")
        summary = (
            f"Heuristic discovery (LLM unavailable). "
            f"Languages: {', '.join(state.stack_hints.languages) if state.stack_hints else 'unknown'}. "
            f"Top candidates: {', '.join(shortlist[:8])}"
        )
        return shortlist[:15], summary

    ranked = data.get("relevant_files") or data.get("files") or []
    if not isinstance(ranked, list):
        ranked = []

    valid: list[str] = []
    candidate_set = set(state.candidates) | set(state.all_files)
    for item in ranked:
        if not isinstance(item, str):
            continue
        path = clean_rel_path(item)
        if path in candidate_set or (root / path).is_file():
            if path not in valid:
                valid.append(path)

    if not valid:
        valid = shortlist[:15]

    summary = data.get("repo_summary") or data.get("summary") or ""
    if not isinstance(summary, str) or not summary.strip():
        summary = f"Relevant files: {', '.join(valid[:10])}"

    return valid, summary.strip()

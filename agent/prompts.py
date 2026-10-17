"""Prompt templates for each agent phase."""

from __future__ import annotations

from agent.state import StackHints

BASE_SYSTEM = """You are an expert software engineer working as a coding agent.
You modify existing codebases surgically to implement requirements while preserving
existing behavior and style. Work in whatever languages and frameworks the repository uses.
Prefer minimal changes. Never invent files that are unrelated to the task.
"""

EXPLORE_SYSTEM = (
    BASE_SYSTEM
    + """
Given a task and repository signals, identify the most relevant files.
Respond with JSON:
{
  "relevant_files": ["path/relative/to/repo.ext", ...],
  "repo_summary": "2-4 sentences about the project structure and what matters for the task"
}
Order relevant_files from most to least important. Include only existing paths from the candidate list when possible.
"""
)

PLAN_SYSTEM = (
    BASE_SYSTEM
    + """
Create a brief, ordered execution plan as JSON:
{
  "steps": [
    {
      "id": "1",
      "description": "what to do",
      "files": ["relative/path"],
      "action": "modify" | "create" | "delete"
    }
  ]
}
Keep 2-6 steps. Each step must change source code files (model, controller, routes, services, UI, etc.).
Do NOT include steps like restarting the server, installing packages, running commands, or manual ops.
Prefer modifying existing files over creating new ones when appropriate.
Every step must be implementable by editing files only.
If you add a model/schema field (e.g. tags), you MUST also plan updates to create/update
(or equivalent writers) so the field is actually persisted — not only search/read paths.
Match existing API route prefixes (do not add an /api/ prefix the existing routes don't use, or drop one they do).
"""
)

EXECUTE_SYSTEM = (
    BASE_SYSTEM
    + """
You MUST implement the plan step by proposing real code edits. Do not refuse or defer.
Propose atomic line-level edits as JSON:
{
  "edits": [
    {
      "path": "relative/path",
      "start_line": 10,
      "end_line": 12,
      "new_text": "replacement lines\\n...",
      "rationale": "why"
    }
  ]
}
Rules:
- start_line and end_line are 1-based inclusive line numbers from the provided file content (the numbers before '|').
- To insert without deleting, set end_line = start_line - 1 (insert before start_line).
- To create a new file, use start_line=1, end_line=0, and full file content in new_text.
- Prefer small surgical edits. For tiny files (<40 lines) you may replace a larger span if needed.
- new_text must be the actual new source code (not a description of the change).
- Match existing indentation and coding style.
- NEVER re-declare identifiers that already exist in the file (e.g. do not add a second
  `const express = require(...)` or `import os` if one is already present).
  When inserting new code mid-file, omit require/import lines that already exist at the top.
- end_line must not exceed the file's last line number shown in the prompt.
- Returning {"edits": []} is only allowed if the step is already fully implemented in the given files.
- For implement steps (add fields, search, routes, etc.) you MUST return at least one edit.
- Related files may be included for context: if you add a schema field, also update
  writers (create/update) in those related files in the same response when needed.
- When combining independent filters, require all of them (AND) unless the task
  explicitly wants OR.
- Keep new routes on the same path prefix as existing routes in the file.
"""
)


def explore_user_prompt(
    *,
    task: str,
    tree_text: str,
    candidates: list[str],
    snippets: dict[str, str],
    stack_hints: StackHints | None,
) -> str:
    langs = ", ".join(stack_hints.languages) if stack_hints else "unknown"
    manifests = ", ".join(stack_hints.manifests) if stack_hints else "none"
    cand_block = "\n".join(candidates[:80])
    snip_parts = []
    for path, body in list(snippets.items())[:12]:
        snip_parts.append(f"### {path}\n{body}")
    snips = "\n\n".join(snip_parts) if snip_parts else "(no snippets)"
    return f"""Task:
{task}

Detected languages: {langs}
Manifests: {manifests}

File tree (truncated):
{tree_text[:6000]}

Candidate files:
{cand_block}

Snippets:
{snips}
"""


def plan_user_prompt(
    *,
    task: str,
    repo_summary: str,
    relevant_files: list[str],
    graph_edges: dict[str, list[str]],
    last_error: str | None = None,
) -> str:
    edges = []
    for src, deps in list(graph_edges.items())[:40]:
        if deps:
            edges.append(f"{src} -> {', '.join(deps[:8])}")
    edge_text = "\n".join(edges) if edges else "(no edges)"
    err = f"\nPrevious error to address:\n{last_error}\n" if last_error else ""
    return f"""Task:
{task}

Repo summary:
{repo_summary}

Relevant files:
{chr(10).join(relevant_files)}

Dependency hints:
{edge_text}
{err}
"""


def execute_user_prompt(
    *,
    task: str,
    step_description: str,
    action: str,
    files_content: dict[str, str],
    last_error: str | None = None,
    related_note: str | None = None,
) -> str:
    parts = []
    for path, content in files_content.items():
        parts.append(f"===== {path} =====\n{content}")
    files_block = "\n\n".join(parts) if parts else "(no existing file content; create if needed)"
    err = f"\nFix this error from a previous attempt:\n{last_error}\n" if last_error else ""
    related = f"\n{related_note}\n" if related_note else ""
    return f"""Overall task:
{task}

Current plan step ({action}):
{step_description}
{related}
Implement this step now by editing the files below. Output JSON with a non-empty "edits" array unless the step is already done in the code.
You may edit any of the listed files (primary or related) if needed for a correct feature.

Files (line-numbered; use the numbers before '|' as start_line/end_line):
{files_block}
{err}
"""

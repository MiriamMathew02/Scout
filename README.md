# AI Coding Agent

A Python CLI agent that explores a codebase, plans changes, applies **atomic line-level edits**, verifies them, and summarizes the result.

**Priorities:** accuracy first, token efficiency second.  
**Orchestration:** simple phase loop + in-memory classes (no LangGraph).  
**LLM:** Groq free tier (`llama-3.3-70b-versatile` by default).

## Features

1. **Discover** — scan file tree, heuristic filter, multi-language dependency graph, LLM relevance ranking  
2. **Plan** — short ordered execution plan  
3. **Execute** — surgical line-range edits with rollback  
4. **Verify** — soft multi-language syntax checks; optional project tests  
5. **Consistency** — re-check **changed files + graph neighbors** for cross-file mistakes (e.g. route handler name ≠ controller export)  
6. **Summarize** — human-readable report  

Works across languages and stacks: edits are text-based; dependency edges are best-effort per language.

## Setup

```powershell
# From the project root
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GROQ_API_KEY from https://console.groq.com
```

## Usage

```powershell
python -m agent --repo path\to\target\repo --task "Add a health check endpoint that returns 200"

# Dry-run (no writes)
python -m agent --repo path\to\repo --task "..." --dry-run

# Optional: run detected project tests during verify
python -m agent --repo path\to\repo --task "..." --run-tests

# Smaller/faster Groq model
python -m agent --repo path\to\repo --task "..." --model llama-3.1-8b-instant
```

Exit code `0` = success (`DONE`), `1` = failure (`FAILED`).

## Architecture

```
DISCOVER → PLAN → EXECUTE → VERIFY → CONSISTENCY → SUMMARIZE → DONE
                ↑              │           │
                └──── retry ◄──┴───────────┘
```

`CONSISTENCY` expands the dependency graph around every touched file and runs static
checks (JS `require` → `exports` wiring, light Python imports) plus a small LLM review.
Failures return to **EXECUTE** with a concrete error (no silent SUCCESS on name mismatches).


Importable building blocks (used by the loop, also callable on their own):

| Function | Module |
|----------|--------|
| `scan_file_tree` | `agent.discovery.scan` |
| `filter_candidates`, `rank_by_task_signals` | `agent.discovery.heuristics` |
| `detect_stack` | `agent.discovery.stack` |
| `build_dependency_graph`, `neighbors`, `topological_edit_order` | `agent.discovery.graph` |
| `read_line_numbered`, `excerpt_for_prompt` | `agent.discovery.snippets` |
| `apply_hunk`, `apply_edits`, `rollback_edits` | `agent.execution.atomic_edit` |
| `create_plan` | `agent.planning.plan` |
| `verify_changes` | `agent.verification.verify` |
| `build_summary` | `agent.summary.summarize` |
| `run_agent` | `agent.loop` |

## Tests

```powershell
pytest -q
```

Fixtures live under `fixtures/mini_py` and `fixtures/mini_js`.

## Environment

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | API key from [console.groq.com](https://console.groq.com) |

Default model: `llama-3.3-70b-versatile` (OpenAI-compatible endpoint `https://api.groq.com/openai/v1`).

"""
Behavior / acceptance checks after syntax + consistency.

Tier 1: static field plumbing + optional LLM checklist review of the diff.
Tier 2: optional project tests (when enabled).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.discovery.symbols import (
    SymbolGraph,
    build_symbol_graph,
    check_schema_field_plumbing,
    detect_added_schema_fields,
)
from agent.llm import chat_json
from agent.verification.verify import verify_changes


@dataclass
class BehaviorResult:
    ok: bool
    messages: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        body = "\n".join(self.messages) if self.messages else "(no details)"
        return f"{status}\n{body}"


BEHAVIOR_SYSTEM = """You review whether code changes satisfy a product task.
Return JSON only:
{
  "gaps": [
    "short description of a missing or incorrect behavior"
  ],
  "ok": true/false
}
Only report real functional gaps (missing field plumbing, wrong combine of filters,
wrong route prefix vs rest of API, search not matching requirements). No style nits.
If the task is reasonably satisfied, return {"ok": true, "gaps": []}.
"""


def check_behavior(
    repo_root: Path | str,
    *,
    task: str,
    touched: list[str],
    backups: dict[str, str | None],
    candidates: list[str],
    stack_hints: Any = None,
    run_tests: bool = False,
    use_llm: bool = False,
    model: str | None = None,
    client: Any = None,
    symbol_graph: SymbolGraph | None = None,
) -> BehaviorResult:
    root = Path(repo_root).resolve()
    messages: list[str] = []
    hard_fail = False

    files_for_graph = list(dict.fromkeys(candidates[:200] + touched))[:300]
    graph = symbol_graph or build_symbol_graph(root, files_for_graph)

    added_fields = detect_added_schema_fields(root, backups, touched)
    if added_fields:
        messages.append(f"Detected new schema field(s): {', '.join(added_fields)}")
        # Rebuild writers from current files
        graph = build_symbol_graph(root, files_for_graph)
        plumbing = check_schema_field_plumbing(graph, added_fields=added_fields)
        for issue in plumbing:
            messages.append(issue)
            hard_fail = True
    else:
        messages.append("No new schema fields detected in diff (static).")

    # Route prefix consistency: new routes on /x when existing ones use /api/x (or reverse)
    route_issue = _check_route_prefix_consistency(root, touched, backups, candidates)
    if route_issue:
        messages.append(route_issue)
        hard_fail = True

    if use_llm and not hard_fail:
        gaps = _llm_behavior_gaps(
            root,
            task=task,
            touched=touched,
            model=model,
            client=client,
        )
        if gaps:
            hard_fail = True
            messages.append("LLM acceptance review found gaps:")
            messages.extend(f"- {g}" for g in gaps)
        else:
            messages.append("LLM acceptance review: no gaps reported.")

    if run_tests and stack_hints is not None:
        # Reuse verify's test runner path via touched=[] skipped — call full verify with tests
        tr = verify_changes(root, touched, stack_hints, run_tests=True)
        if not tr.ok:
            hard_fail = True
            messages.append("Project tests failed:")
            messages.extend(tr.messages)
        else:
            messages.append("Project tests: OK (or skipped if unavailable).")

    return BehaviorResult(ok=not hard_fail, messages=messages)


_ROUTE_DEF_RE = re.compile(
    r"""\b(?:app|router)\.(?:get|post|put|patch|delete)\s*\(\s*['"]([^'"]+)['"]""",
    re.I,
)


def _route_paths(text: str) -> list[str]:
    return [m.group(1) for m in _ROUTE_DEF_RE.finditer(text)]


def _prefix_bucket(p: str) -> str:
    parts = [x for x in p.split("/") if x]
    if not parts:
        return "/"
    if parts[0] == "api" and len(parts) > 1:
        return "/api/" + parts[1]
    return "/" + parts[0]


def _check_route_prefix_consistency(
    root: Path,
    touched: list[str],
    backups: dict[str, str | None],
    candidates: list[str],
) -> str | None:
    """Flag routes added in this run on '/x' when other routes use '/api/x', or the reverse.

    Only new routes are judged, so an app that already serves both prefixes on purpose
    (pages on /users, JSON on /api/users) is left alone.
    """
    old_buckets: set[str] = set()
    new_paths: list[str] = []
    scan = list(dict.fromkeys(touched + [c for c in candidates if "route" in c.lower()]))
    for rel in scan:
        path = root / rel
        if not path.is_file() or path.suffix not in {".js", ".ts", ".mjs", ".cjs"}:
            continue
        try:
            current = _route_paths(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        before = _route_paths(backups.get(rel) or "") if rel in touched else current
        old_buckets |= {_prefix_bucket(p) for p in before}
        new_paths += [p for p in current if p not in before]

    all_buckets = old_buckets | {_prefix_bucket(p) for p in new_paths}
    for p in new_paths:
        bucket = _prefix_bucket(p)
        twin = bucket[len("/api"):] if bucket.startswith("/api/") else "/api" + bucket
        if twin in all_buckets and bucket not in old_buckets:
            return (
                f"Route prefix inconsistency: new route '{p}' uses '{bucket}', but other "
                f"routes use '{twin}'. Prefer one API prefix matching existing routes."
            )
    return None


def _llm_behavior_gaps(
    root: Path,
    *,
    task: str,
    touched: list[str],
    model: str | None,
    client: Any,
) -> list[str]:
    from agent.discovery.snippets import read_line_numbered
    from agent.llm import DEFAULT_MODEL

    parts: list[str] = []
    for rel in touched[:8]:
        path = root / rel
        if not path.is_file():
            continue
        try:
            body = read_line_numbered(path)
        except OSError:
            continue
        lines = body.splitlines()
        if len(lines) > 180:
            body = "\n".join(lines[:180]) + "\n..."
        parts.append(f"===== {rel} =====\n{body}")
    if not parts:
        return []

    user = f"Task:\n{task}\n\nChanged files:\n" + "\n\n".join(parts)
    try:
        data = chat_json(
            system=BEHAVIOR_SYSTEM,
            user=user,
            model=model or DEFAULT_MODEL,
            client=client,
            max_tokens=1500,
        )
    except Exception:  # noqa: BLE001
        return []

    if data.get("ok") is True and not data.get("gaps"):
        return []
    gaps = data.get("gaps") or []
    if not isinstance(gaps, list):
        return []
    return [str(g) for g in gaps if str(g).strip()]

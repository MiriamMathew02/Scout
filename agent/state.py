"""In-memory agent state and shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    DISCOVER = "discover"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    CONSISTENCY = "consistency"  # cross-file check: changed + graph neighbors
    SUMMARIZE = "summarize"
    DONE = "done"
    FAILED = "failed"


@dataclass
class StackHints:
    languages: list[str] = field(default_factory=list)
    manifests: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)


@dataclass
class PlanStep:
    id: str
    description: str
    files: list[str]
    action: str = "modify"  # modify | create | delete
    status: str = "pending"  # pending | done | failed | skipped


@dataclass
class EditHunk:
    path: str
    start_line: int  # 1-based inclusive
    end_line: int  # 1-based inclusive; end < start means insert before start
    new_text: str
    rationale: str = ""


@dataclass
class AgentState:
    repo_path: str
    task: str
    phase: Phase = Phase.DISCOVER
    iteration: int = 0
    max_iterations: int = 20
    max_execute_retries: int = 3
    max_replans: int = 2

    model: str = "groq/openai/gpt-oss-120b"
    dry_run: bool = False
    run_tests: bool = False

    all_files: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    tree_text: str = ""
    graph: dict[str, list[str]] = field(default_factory=dict)
    relevant_files: list[str] = field(default_factory=list)
    file_snippets: dict[str, str] = field(default_factory=dict)
    repo_summary: str = ""
    stack_hints: StackHints | None = None

    plan: list[PlanStep] = field(default_factory=list)
    applied_edits: list[EditHunk] = field(default_factory=list)
    backups: dict[str, str] = field(default_factory=dict)
    last_error: str | None = None
    retry_count: int = 0
    replan_count: int = 0

    verify_result: str = ""
    consistency_result: str = ""
    summary: str = ""
    log: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.log.append(message)

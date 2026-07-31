"""Command-line entrypoint for the coding agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.llm import DEFAULT_MODEL
from agent.loop import run_agent
from agent.state import AgentState, Phase


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent",
        description="AI coding agent: discover, plan, edit, verify, summarize.",
    )
    p.add_argument(
        "--repo",
        required=True,
        help="Path to the target repository",
    )
    p.add_argument(
        "--task",
        required=True,
        help="Product requirement / task description",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Groq model id (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Propose and log edits without writing files",
    )
    p.add_argument(
        "--run-tests",
        action="store_true",
        help="Run detected project tests during verification",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max phase-loop iterations (default: 20)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        print(f"Error: repo is not a directory: {repo}", file=sys.stderr)
        return 2

    state = AgentState(
        repo_path=str(repo),
        task=args.task,
        model=args.model,
        dry_run=args.dry_run,
        run_tests=args.run_tests,
        max_iterations=args.max_iterations,
    )

    print(f"Starting agent on {repo}")
    print(f"Task: {args.task}")
    if args.dry_run:
        print("Dry-run mode: no files will be written")
    print("---")

    try:
        run_agent(state)
    except RuntimeError as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        return 2

    print(state.summary or "(no summary)")
    return 0 if state.phase == Phase.DONE else 1


if __name__ == "__main__":
    raise SystemExit(main())

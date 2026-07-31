"""Multi-language soft verification of changed files."""

from __future__ import annotations

import ast
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from agent.state import StackHints


@dataclass
class VerifyResult:
    ok: bool
    messages: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        body = "\n".join(self.messages) if self.messages else "(no details)"
        return f"{status}\n{body}"


def verify_changes(
    repo_root: Path | str,
    touched_files: list[str],
    stack_hints: StackHints | None = None,
    *,
    run_tests: bool = False,
) -> VerifyResult:
    """
    Soft checks: existence/encoding, language syntax when tools available.
    Missing toolchains do not fail the run.
    """
    root = Path(repo_root).resolve()
    messages: list[str] = []
    hard_fail = False

    if not touched_files:
        messages.append("No files were modified.")
        return VerifyResult(ok=True, messages=messages)

    for rel in touched_files:
        path = root / rel
        if not path.exists():
            # deleted files are ok
            messages.append(f"{rel}: not present (deleted or never created)")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            messages.append(f"{rel}: not valid UTF-8")
            hard_fail = True
            continue
        except OSError as exc:
            messages.append(f"{rel}: unreadable ({exc})")
            hard_fail = True
            continue

        if text.strip() == "":
            messages.append(f"{rel}: warning empty file")

        suffix = path.suffix.lower()
        if suffix == ".py":
            ok, msg = _check_python(path, text)
            messages.append(msg)
            if not ok:
                hard_fail = True
        elif suffix in {".js", ".mjs", ".cjs"}:
            ok, msg = _check_node_syntax(path)
            messages.append(msg)
            if not ok and msg.startswith("FAIL"):
                hard_fail = True
        elif suffix == ".go":
            ok, msg = _run_soft(["go", "vet", str(path)], root)
            messages.append(msg)
            if not ok and msg.startswith("FAIL"):
                hard_fail = True
        elif suffix == ".rs":
            # cargo check is project-level; skip per-file hard fail
            messages.append(f"{rel}: rust syntax check skipped (use cargo test if needed)")
        else:
            messages.append(f"{rel}: basic read OK")

    if run_tests and stack_hints and stack_hints.test_commands:
        cmd = stack_hints.test_commands[0]
        ok, msg = _run_shell(cmd, root, timeout=120)
        messages.append(msg)
        if not ok:
            hard_fail = True

    return VerifyResult(ok=not hard_fail, messages=messages)


def _check_python(path: Path, text: str) -> tuple[bool, str]:
    try:
        ast.parse(text, filename=str(path))
        return True, f"{path.name}: python syntax OK"
    except SyntaxError as exc:
        return False, f"FAIL {path.name}: python syntax error: {exc}"


def _check_node_syntax(path: Path) -> tuple[bool, str]:
    if not shutil.which("node"):
        return True, f"{path.name}: node not installed; skipped JS syntax check"
    try:
        proc = subprocess.run(
            ["node", "--check", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return True, f"{path.name}: node check skipped ({exc})"
    if proc.returncode == 0:
        return True, f"{path.name}: JS syntax OK"
    err = (proc.stderr or proc.stdout or "unknown error").strip()
    return False, f"FAIL {path.name}: {err[:500]}"


def _run_soft(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    if not shutil.which(cmd[0]):
        return True, f"{cmd[0]} not installed; skipped"
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return True, f"{' '.join(cmd)} skipped ({exc})"
    if proc.returncode == 0:
        return True, f"{' '.join(cmd)}: OK"
    err = (proc.stderr or proc.stdout or "").strip()
    return False, f"FAIL {' '.join(cmd)}: {err[:500]}"


def _run_shell(cmd: str, cwd: Path, timeout: int = 120) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"FAIL tests: {exc}"
    if proc.returncode == 0:
        return True, f"tests OK: {cmd}"
    err = (proc.stderr or proc.stdout or "").strip()
    return False, f"FAIL tests ({cmd}): {err[:800]}"

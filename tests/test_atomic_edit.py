from pathlib import Path

from agent.execution.atomic_edit import apply_edits, apply_hunk, rollback_edits
from agent.state import EditHunk


def test_replace_lines(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    hunk = EditHunk(path="a.py", start_line=2, end_line=2, new_text="TWO\n")
    apply_hunk(tmp_path, hunk)
    assert f.read_text(encoding="utf-8") == "one\nTWO\nthree\n"


def test_insert_before(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("one\ntwo\n", encoding="utf-8")
    hunk = EditHunk(path="a.py", start_line=2, end_line=1, new_text="mid\n")
    apply_hunk(tmp_path, hunk)
    assert f.read_text(encoding="utf-8") == "one\nmid\ntwo\n"


def test_multi_hunk_bottom_to_top(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")
    hunks = [
        EditHunk(path="a.py", start_line=1, end_line=1, new_text="A\n"),
        EditHunk(path="a.py", start_line=3, end_line=3, new_text="C\n"),
    ]
    apply_edits(tmp_path, hunks)
    assert f.read_text(encoding="utf-8") == "A\nb\nC\nd\n"


def test_rollback(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("original\n", encoding="utf-8")
    backups: dict[str, str] = {}
    apply_hunk(
        tmp_path,
        EditHunk(path="a.py", start_line=1, end_line=1, new_text="changed\n"),
        backups=backups,
    )
    assert f.read_text(encoding="utf-8") == "changed\n"
    rollback_edits(tmp_path, backups)
    assert f.read_text(encoding="utf-8") == "original\n"


def test_create_file(tmp_path: Path):
    apply_hunk(
        tmp_path,
        EditHunk(path="new.py", start_line=1, end_line=0, new_text="print(1)\n"),
    )
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print(1)\n"

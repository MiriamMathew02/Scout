from pathlib import Path

import pytest

from agent.execution.atomic_edit import (
    EditError,
    apply_edits,
    apply_hunk,
    read_exact,
    rollback_edits,
)
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


def test_edit_keeps_crlf_line_endings(tmp_path: Path):
    f = tmp_path / "w.py"
    f.write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    backups: dict[str, str] = {}
    apply_hunk(
        tmp_path,
        EditHunk(path="w.py", start_line=2, end_line=2, new_text="b = 20\n"),
        backups=backups,
    )
    assert f.read_bytes() == b"a = 1\r\nb = 20\r\nc = 3\r\n"
    rollback_edits(tmp_path, backups)
    assert f.read_bytes() == b"a = 1\r\nb = 2\r\nc = 3\r\n"


def test_edit_keeps_lf_line_endings(tmp_path: Path):
    # On Windows, write_text used to turn every \n into \r\n
    f = tmp_path / "a.py"
    f.write_bytes(b"one\ntwo\n")
    apply_hunk(tmp_path, EditHunk(path="a.py", start_line=2, end_line=2, new_text="TWO\n"))
    assert f.read_bytes() == b"one\nTWO\n"


def test_refuses_to_rewrite_non_utf8_file(tmp_path: Path):
    f = tmp_path / "l.py"
    data = "a = 1\nname = 'Jos\xe9'\n".encode("latin-1")
    f.write_bytes(data)
    with pytest.raises(EditError, match="UTF-8"):
        apply_hunk(tmp_path, EditHunk(path="l.py", start_line=1, end_line=1, new_text="a = 2\n"))
    assert f.read_bytes() == data


def test_rollback_restores_exact_bytes(tmp_path: Path):
    f = tmp_path / "l.py"
    data = "name = 'Jos\xe9'\r\n".encode("latin-1")
    f.write_bytes(data)
    snapshot = {"l.py": read_exact(f)}
    f.write_text("x = 1\n", encoding="utf-8")
    rollback_edits(tmp_path, snapshot)
    assert f.read_bytes() == data


def test_rollback_keeps_existing_empty_file(tmp_path: Path):
    f = tmp_path / "__init__.py"
    f.write_text("", encoding="utf-8")
    backups: dict[str, str | None] = {}
    apply_hunk(tmp_path, EditHunk(path="__init__.py", start_line=1, end_line=0, new_text="x = 1\n"),
               backups=backups)
    rollback_edits(tmp_path, backups)
    assert f.exists() and f.read_bytes() == b""


def test_rollback_deletes_file_the_run_created(tmp_path: Path):
    backups: dict[str, str | None] = {}
    apply_hunk(tmp_path, EditHunk(path="new.py", start_line=1, end_line=0, new_text="x = 1\n"),
               backups=backups)
    assert backups == {"new.py": None}
    rollback_edits(tmp_path, backups)
    assert not (tmp_path / "new.py").exists()


def test_overlapping_edits_are_rejected_before_writing(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("a = 1\nb = 2\n", encoding="utf-8")
    hunks = [
        EditHunk(path="a.py", start_line=1, end_line=2, new_text="a = 10\nb = 20\n"),
        EditHunk(path="a.py", start_line=2, end_line=2, new_text="b = 200\n"),
    ]
    with pytest.raises(EditError, match="overlap"):
        apply_edits(tmp_path, hunks)
    assert f.read_text(encoding="utf-8") == "a = 1\nb = 2\n"


def test_insert_inside_a_replaced_range_is_rejected(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    hunks = [
        EditHunk(path="a.py", start_line=1, end_line=3, new_text="A\nB\nC\n"),
        EditHunk(path="a.py", start_line=2, end_line=1, new_text="mid\n"),
    ]
    with pytest.raises(EditError, match="falls inside"):
        apply_edits(tmp_path, hunks)


def test_inserts_at_the_same_line_keep_their_order(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("a\nb\n", encoding="utf-8")
    apply_edits(tmp_path, [
        EditHunk(path="a.py", start_line=2, end_line=1, new_text="x\n"),
        EditHunk(path="a.py", start_line=2, end_line=1, new_text="y\n"),
    ])
    assert f.read_text(encoding="utf-8") == "a\nx\ny\nb\n"

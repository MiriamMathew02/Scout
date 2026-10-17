from pathlib import Path

import pytest

from agent.execution.atomic_edit import EditError
from agent.execution.sanitize import validate_and_sanitize_hunks
from agent.state import EditHunk


def test_rejects_out_of_bounds(tmp_path: Path):
    f = tmp_path / "a.js"
    f.write_text("const x = 1;\nconst y = 2;\n", encoding="utf-8")
    hunk = EditHunk(path="a.js", start_line=10, end_line=10, new_text="const z = 3;\n")
    with pytest.raises(EditError, match="out of bounds"):
        validate_and_sanitize_hunks(tmp_path, [hunk])


def test_strips_duplicate_require(tmp_path: Path):
    f = tmp_path / "a.js"
    f.write_text(
        "const mongoose = require('mongoose');\n\nconst schema = {};\n",
        encoding="utf-8",
    )
    # Insert before line 3, but LLM re-includes mongoose require
    hunk = EditHunk(
        path="a.js",
        start_line=3,
        end_line=2,
        new_text=(
            "const mongoose = require('mongoose');\n"
            "const extra = 1;\n"
        ),
    )
    cleaned = validate_and_sanitize_hunks(tmp_path, [hunk])
    assert len(cleaned) == 1
    assert "mongoose" not in cleaned[0].new_text
    assert "extra" in cleaned[0].new_text


def test_allows_valid_replace(tmp_path: Path):
    f = tmp_path / "a.js"
    f.write_text("const x = 1;\nconst y = 2;\n", encoding="utf-8")
    hunk = EditHunk(path="a.js", start_line=2, end_line=2, new_text="const y = 3;\n")
    cleaned = validate_and_sanitize_hunks(tmp_path, [hunk])
    assert cleaned[0].new_text == "const y = 3;\n"


def test_keeps_local_variable_in_new_function(tmp_path: Path):
    f = tmp_path / "c.js"
    f.write_text(
        "async function list(req, res) {\n"
        "  const result = await Note.find();\n"
        "  res.json(result);\n"
        "}\n",
        encoding="utf-8",
    )
    new_fn = (
        "async function one(req, res) {\n"
        "  const result = await Note.findById(req.params.id);\n"
        "  res.json(result);\n"
        "}\n"
    )
    hunk = EditHunk(path="c.js", start_line=5, end_line=4, new_text=new_fn)
    cleaned = validate_and_sanitize_hunks(tmp_path, [hunk])
    assert cleaned[0].new_text == new_fn


def test_leaves_other_duplicates_for_syntax_check(tmp_path: Path):
    f = tmp_path / "a.js"
    f.write_text("const PORT = 3000;\n", encoding="utf-8")
    hunk = EditHunk(path="a.js", start_line=2, end_line=1, new_text="const PORT = 4000;\n")
    cleaned = validate_and_sanitize_hunks(tmp_path, [hunk])
    assert cleaned[0].new_text == "const PORT = 4000;\n"

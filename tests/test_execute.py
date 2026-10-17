from pathlib import Path

from agent.execution.execute import _load_file_for_prompt, _parse_hunks, propose_edits
from agent.state import AgentState, PlanStep


def _big_file(root: Path) -> None:
    (root / "big.py").write_text("".join(f"x{i} = {i}\n" for i in range(1, 301)), encoding="utf-8")


def test_long_file_shows_the_lines_the_step_mentions(tmp_path: Path):
    _big_file(tmp_path)
    text, n, shown = _load_file_for_prompt(tmp_path, "big.py", hints="set x290 to zero")
    assert n == 300
    assert "290|x290 = 290" in text
    assert shown[0] <= 290 <= shown[1]


def test_edit_outside_shown_lines_reprompts_with_that_region(tmp_path: Path, fake_llm):
    _big_file(tmp_path)
    state = AgentState(repo_path=str(tmp_path), task="t")
    step = PlanStep(id="1", description="set x290 to zero", files=["big.py"])
    llm = fake_llm(lambda prompt: {"edits": [
        {"path": "big.py", "start_line": 20, "end_line": 20, "new_text": "x20 = 0\n"}
    ]})
    hunks = propose_edits(state, step, client=llm)
    assert len(llm.prompts) == 2
    assert "\n20|x20 = 20\n" not in llm.prompts[0]
    assert "\n20|x20 = 20\n" in llm.prompts[1]
    assert [(h.start_line, h.end_line) for h in hunks] == [(20, 20)]


def test_edit_to_dotfile_keeps_its_path(tmp_path: Path):
    (tmp_path / ".eslintrc.js").write_text("module.exports = {};\n", encoding="utf-8")
    step = PlanStep(id="1", description="d", files=[".eslintrc.js"])
    hunks = _parse_hunks(
        [{"path": "./.eslintrc.js", "start_line": 1, "end_line": 1, "new_text": "module.exports = { root: true };\n"}],
        root=tmp_path,
        step=step,
    )
    assert hunks[0].path == ".eslintrc.js"


def test_short_python_edit_is_kept(tmp_path: Path):
    (tmp_path / "m.py").write_text("def f():\n    pass\n", encoding="utf-8")
    step = PlanStep(id="1", description="d", files=["m.py"])
    hunks = _parse_hunks(
        [{"path": "m.py", "start_line": 2, "end_line": 2, "new_text": "    print('hi')\n"}],
        root=tmp_path,
        step=step,
    )
    assert [h.new_text for h in hunks] == ["    print('hi')\n"]

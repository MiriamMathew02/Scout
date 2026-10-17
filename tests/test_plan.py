from pathlib import Path

from agent.planning.plan import is_ops_step, validate_plan
from agent.state import AgentState, PlanStep


def test_file_edits_are_not_ops_steps():
    assert not is_ops_step("Update the base image in the Dockerfile")
    assert not is_ops_step("Add a deploy button to the header")
    assert not is_ops_step("Write the user manual page")


def test_operations_are_ops_steps():
    assert is_ops_step("Restart the server to apply changes")
    assert is_ops_step("Run npm install")


def test_validate_plan_keeps_deploy_button_step(tmp_path: Path):
    (tmp_path / "header.js").write_text("const x = 1;\n", encoding="utf-8")
    state = AgentState(repo_path=str(tmp_path), task="t")
    step = PlanStep(id="1", description="Add a deploy button", files=["header.js"])
    assert [s.description for s in validate_plan(state, [step])] == ["Add a deploy button"]

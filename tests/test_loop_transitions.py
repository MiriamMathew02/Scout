from agent.loop import handle_summarize, handle_verify
from agent.state import AgentState, EditHunk, Phase, PlanStep


def test_summarize_goes_to_done():
    state = AgentState(repo_path=".", task="t")
    state.phase = Phase.SUMMARIZE
    handle_summarize(state)
    assert state.phase == Phase.DONE
    assert "Agent run" in state.summary
    assert "SUCCESS" in state.summary


def test_verify_no_edits_is_failure(tmp_path):
    state = AgentState(repo_path=str(tmp_path), task="t", max_replans=0)
    state.phase = Phase.VERIFY
    state.applied_edits = []
    handle_verify(state)
    assert state.phase == Phase.FAILED
    assert "No files were modified" in (state.verify_result or "")


def test_verify_pass_to_summarize(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("x = 1\n", encoding="utf-8")
    state = AgentState(repo_path=str(tmp_path), task="t")
    state.phase = Phase.VERIFY
    state.applied_edits = [
        EditHunk(path="ok.py", start_line=1, end_line=1, new_text="x = 1\n")
    ]
    handle_verify(state)
    assert state.phase == Phase.SUMMARIZE
    assert "PASS" in state.verify_result


def test_verify_fail_retries_execute(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(\n", encoding="utf-8")
    state = AgentState(repo_path=str(tmp_path), task="t", max_execute_retries=3)
    state.phase = Phase.VERIFY
    state.plan = [PlanStep(id="1", description="fix", files=["bad.py"])]
    state.plan[0].status = "done"
    state.applied_edits = [
        EditHunk(path="bad.py", start_line=1, end_line=1, new_text="def broken(\n")
    ]
    state.backups = {"bad.py": "x = 1\n"}
    handle_verify(state)
    assert state.phase == Phase.EXECUTE
    assert state.retry_count == 1
    # rollback restored original
    assert f.read_text(encoding="utf-8") == "x = 1\n"

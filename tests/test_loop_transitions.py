from agent.loop import handle_consistency, handle_execute, handle_summarize, handle_verify
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


def test_verify_pass_goes_to_consistency(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("x = 1\n", encoding="utf-8")
    state = AgentState(repo_path=str(tmp_path), task="t")
    state.phase = Phase.VERIFY
    state.applied_edits = [
        EditHunk(path="ok.py", start_line=1, end_line=1, new_text="x = 1\n")
    ]
    handle_verify(state)
    assert state.phase == Phase.CONSISTENCY
    assert "PASS" in state.verify_result


def test_consistency_pass_goes_to_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    f = tmp_path / "ok.py"
    f.write_text("x = 1\n", encoding="utf-8")
    state = AgentState(repo_path=str(tmp_path), task="t")
    state.phase = Phase.CONSISTENCY
    state.applied_edits = [
        EditHunk(path="ok.py", start_line=1, end_line=1, new_text="x = 1\n")
    ]
    state.graph = {"ok.py": []}
    handle_consistency(state, client=None)
    assert state.phase == Phase.BEHAVIOR
    assert "PASS" in state.consistency_result


def test_consistency_fail_retries_execute(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ctrl = tmp_path / "c"
    rte = tmp_path / "r"
    ctrl.mkdir()
    rte.mkdir()
    (ctrl / "ctrl.js").write_text("exports.foo = 1;\n", encoding="utf-8")
    (rte / "routes.js").write_text(
        "const notes = require('../c/ctrl.js');\nnotes.bar();\n",
        encoding="utf-8",
    )
    state = AgentState(
        repo_path=str(tmp_path),
        task="t",
        max_execute_retries=3,
    )
    state.phase = Phase.CONSISTENCY
    state.plan = [PlanStep(id="1", description="fix", files=["r/routes.js"])]
    state.plan[0].status = "done"
    state.applied_edits = [
        EditHunk(path="r/routes.js", start_line=1, end_line=1, new_text="x")
    ]
    state.graph = {
        "r/routes.js": ["c/ctrl.js"],
        "c/ctrl.js": [],
    }
    handle_consistency(state, client=None)
    assert state.phase == Phase.EXECUTE
    assert state.retry_count == 1
    assert "bar" in (state.last_error or "")



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


def test_repair_round_keeps_finished_edits(tmp_path, fake_llm):
    (tmp_path / "c").mkdir()
    (tmp_path / "r").mkdir()
    ctrl = tmp_path / "c" / "ctrl.js"
    ctrl.write_text("exports.foo = 1;\n", encoding="utf-8")
    routes = tmp_path / "r" / "routes.js"
    routes.write_text("const notes = require('../c/ctrl.js');\nnotes.bar();\n", encoding="utf-8")
    state = AgentState(repo_path=str(tmp_path), task="add bar", max_execute_retries=3)
    state.phase = Phase.CONSISTENCY
    state.plan = [PlanStep(id="1", description="call bar", files=["r/routes.js"], status="done")]
    state.applied_edits = [
        EditHunk(path="r/routes.js", start_line=2, end_line=1, new_text="notes.bar();\n")
    ]
    state.backups = {"r/routes.js": "const notes = require('../c/ctrl.js');\n"}
    state.graph = {"r/routes.js": ["c/ctrl.js"], "c/ctrl.js": []}
    handle_consistency(state, client=None)

    def reply(prompt):
        if "call bar" in prompt:
            return {"edits": []}  # finished step: nothing left to change
        return {"edits": [{"path": "c/ctrl.js", "start_line": 2, "end_line": 1,
                           "new_text": "exports.bar = () => {};\n"}]}

    handle_execute(state, client=fake_llm(reply))
    assert state.phase == Phase.VERIFY
    assert state.plan[0].status == "done"
    assert routes.read_text(encoding="utf-8").endswith("notes.bar();\n")
    assert "exports.bar" in ctrl.read_text(encoding="utf-8")


def test_failed_step_retry_keeps_earlier_steps(tmp_path, fake_llm):
    a = tmp_path / "a.py"
    a.write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    state = AgentState(repo_path=str(tmp_path), task="t")
    state.phase = Phase.EXECUTE
    state.plan = [
        PlanStep(id="1", description="set x", files=["a.py"]),
        PlanStep(id="2", description="set y", files=["b.py"]),
    ]

    def reply(prompt):
        if "set x" in prompt:
            return {"edits": [{"path": "a.py", "start_line": 1, "end_line": 1, "new_text": "x = 2\n"}]}
        # out-of-range lines: step 2 fails every time
        return {"edits": [{"path": "b.py", "start_line": 5, "end_line": 6, "new_text": "y = 2\n"}]}

    handle_execute(state, client=fake_llm(reply))
    assert a.read_text(encoding="utf-8") == "x = 2\n"
    assert [s.status for s in state.plan] == ["done", "pending"]
    assert state.retry_count == 1

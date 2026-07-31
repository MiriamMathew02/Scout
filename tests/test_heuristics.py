from pathlib import Path

from agent.discovery.heuristics import filter_candidates, rank_by_task_signals


def test_filter_candidates_keeps_source_drops_media(tmp_path: Path):
    (tmp_path / "app.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "bundle.min.js").write_text("var a=1;\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")

    paths = ["app.py", "photo.png", "bundle.min.js", "README.md"]
    out = filter_candidates(paths, tmp_path)
    assert "app.py" in out
    assert "README.md" in out
    assert "photo.png" not in out
    assert "bundle.min.js" not in out


def test_rank_by_task_signals_boosts_matching_names():
    paths = ["src/utils.py", "src/health.py", "README.md"]
    ranked = rank_by_task_signals(paths, "Add a health check endpoint")
    assert ranked[0] == "src/health.py"

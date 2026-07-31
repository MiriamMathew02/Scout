from pathlib import Path

from agent.discovery.scan import scan_file_tree


def test_scan_skips_venv_and_node_modules(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "x.py").write_text("secret\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("1\n", encoding="utf-8")

    paths, tree = scan_file_tree(tmp_path)
    assert "src/app.py" in paths
    assert not any(p.startswith(".venv") for p in paths)
    assert not any(p.startswith("node_modules") for p in paths)
    assert "src/app.py" in tree

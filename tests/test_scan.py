from pathlib import Path

from agent.discovery.scan import clean_rel_path, resolve_inside, scan_file_tree


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


def test_clean_rel_path_keeps_dot_names():
    assert clean_rel_path(".github/workflows/ci.yml") == ".github/workflows/ci.yml"
    assert clean_rel_path("./.eslintrc.js") == ".eslintrc.js"
    assert clean_rel_path("/src/app.py") == "src/app.py"
    assert clean_rel_path("..\\..\\etc\\passwd") == "../../etc/passwd"


def test_resolve_inside_rejects_paths_that_leave_the_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (repo / "link").symlink_to(tmp_path / "outside", target_is_directory=True)
    assert resolve_inside(repo, "sub/x.py") == (repo / "sub" / "x.py").resolve()
    assert resolve_inside(repo, "sub/../../outside.txt") is None
    assert resolve_inside(repo, "../outside.txt") is None
    assert resolve_inside(repo, "/etc/passwd") is None
    assert resolve_inside(repo, "link/secret.txt") is None

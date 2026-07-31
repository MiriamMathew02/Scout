from pathlib import Path

from agent.discovery.graph import build_dependency_graph, neighbors, topological_edit_order


def test_python_import_graph(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("from pkg import b\n", encoding="utf-8")
    (pkg / "b.py").write_text("x = 1\n", encoding="utf-8")
    files = ["pkg/__init__.py", "pkg/a.py", "pkg/b.py"]
    g = build_dependency_graph(tmp_path, files)
    assert "pkg/b.py" in g.get("pkg/a.py", [])


def test_js_relative_import(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "index.js").write_text('const m = require("./math");\n', encoding="utf-8")
    (src / "math.js").write_text("module.exports = {};\n", encoding="utf-8")
    files = ["src/index.js", "src/math.js"]
    g = build_dependency_graph(tmp_path, files)
    assert "src/math.js" in g.get("src/index.js", [])


def test_c_include(tmp_path: Path):
    (tmp_path / "main.c").write_text('#include "util.h"\nint main(){return 0;}\n', encoding="utf-8")
    (tmp_path / "util.h").write_text("int f(void);\n", encoding="utf-8")
    files = ["main.c", "util.h"]
    g = build_dependency_graph(tmp_path, files)
    assert "util.h" in g.get("main.c", [])


def test_neighbors_and_topo():
    graph = {
        "a.py": ["b.py"],
        "b.py": [],
        "c.py": ["a.py"],
    }
    n = neighbors(graph, ["a.py"], hops=1)
    assert "a.py" in n and "b.py" in n and "c.py" in n
    order = topological_edit_order(graph, ["c.py", "a.py", "b.py"])
    assert order.index("b.py") < order.index("a.py")

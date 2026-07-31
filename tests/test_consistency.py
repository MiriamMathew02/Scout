from pathlib import Path

from agent.discovery.graph import build_dependency_graph
from agent.verification.consistency import check_consistency, files_in_consistency_scope


def test_detects_missing_export_on_required_module(tmp_path: Path):
    ctrl = tmp_path / "controllers"
    routes = tmp_path / "routes"
    ctrl.mkdir()
    routes.mkdir()
    (ctrl / "note.controller.js").write_text(
        "exports.create = (req, res) => {};\n"
        "exports.searchNotes = (req, res) => {};\n",
        encoding="utf-8",
    )
    (routes / "note.routes.js").write_text(
        "module.exports = (app) => {\n"
        "  const notes = require('../controllers/note.controller.js');\n"
        "  app.get('/notes/search', notes.search);\n"
        "};\n",
        encoding="utf-8",
    )
    files = [
        "controllers/note.controller.js",
        "routes/note.routes.js",
    ]
    graph = build_dependency_graph(tmp_path, files)
    result = check_consistency(
        tmp_path,
        touched=["routes/note.routes.js"],
        graph=graph,
        use_llm=False,
    )
    assert not result.ok
    assert any(i.symbol == "search" for i in result.issues)
    assert any("searchNotes" in i.message for i in result.issues)


def test_pass_when_export_matches(tmp_path: Path):
    ctrl = tmp_path / "controllers"
    routes = tmp_path / "routes"
    ctrl.mkdir()
    routes.mkdir()
    (ctrl / "note.controller.js").write_text(
        "exports.create = () => {};\nexports.searchNotes = () => {};\n",
        encoding="utf-8",
    )
    (routes / "note.routes.js").write_text(
        "module.exports = (app) => {\n"
        "  const notes = require('../controllers/note.controller.js');\n"
        "  app.get('/notes/search', notes.searchNotes);\n"
        "};\n",
        encoding="utf-8",
    )
    files = [
        "controllers/note.controller.js",
        "routes/note.routes.js",
    ]
    graph = build_dependency_graph(tmp_path, files)
    result = check_consistency(
        tmp_path,
        touched=["controllers/note.controller.js", "routes/note.routes.js"],
        graph=graph,
        use_llm=False,
    )
    assert result.ok


def test_scope_includes_neighbors():
    graph = {
        "a.js": ["b.js"],
        "b.js": [],
        "c.js": ["a.js"],
    }
    scope = files_in_consistency_scope(["a.js"], graph, hops=1)
    assert "a.js" in scope
    assert "b.js" in scope
    assert "c.js" in scope

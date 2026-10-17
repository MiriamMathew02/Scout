from pathlib import Path

from agent.verification.behavior import check_behavior


def test_behavior_fails_when_tags_not_in_create(tmp_path: Path):
    (tmp_path / "model.js").write_text(
        "const mongoose = require('mongoose');\n"
        "const NoteSchema = mongoose.Schema({\n"
        "  title: String,\n"
        "  content: String,\n"
        "  tags: [String]\n"
        "});\n",
        encoding="utf-8",
    )
    (tmp_path / "ctrl.js").write_text(
        "exports.create = (req, res) => {\n"
        "  const note = new Note({ title: req.body.title, content: req.body.content });\n"
        "};\n",
        encoding="utf-8",
    )
    old_model = (
        "const NoteSchema = mongoose.Schema({\n"
        "  title: String,\n"
        "  content: String\n"
        "});\n"
    )
    result = check_behavior(
        tmp_path,
        task="Add tags and search",
        touched=["model.js", "ctrl.js"],
        backups={"model.js": old_model},
        candidates=["model.js", "ctrl.js"],
        use_llm=False,
        run_tests=False,
    )
    assert not result.ok
    assert any("tags" in m for m in result.messages)


def test_route_prefix_inconsistency(tmp_path: Path):
    routes = tmp_path / "app" / "routes"
    routes.mkdir(parents=True)
    (routes / "note.routes.js").write_text(
        "module.exports = (app) => {\n"
        "  app.get('/notes', notes.findAll);\n"
        "  app.get('/api/notes/search', notes.searchNotes);\n"
        "};\n",
        encoding="utf-8",
    )
    result = check_behavior(
        tmp_path,
        task="search",
        touched=["app/routes/note.routes.js"],
        backups={},
        candidates=["app/routes/note.routes.js"],
        use_llm=False,
    )
    assert not result.ok
    assert any("prefix" in m.lower() for m in result.messages)


def _route_check(tmp_path: Path, old: str, new: str):
    (tmp_path / "routes.js").write_text(new, encoding="utf-8")
    return check_behavior(
        tmp_path,
        task="t",
        touched=["routes.js"],
        backups={"routes.js": old},
        candidates=["routes.js"],
        use_llm=False,
    )


def test_route_prefix_check_works_for_any_resource(tmp_path: Path):
    old = "router.get('/tasks', tasks.list);\n"
    result = _route_check(tmp_path, old, old + "router.get('/api/tasks/search', tasks.search);\n")
    assert not result.ok
    assert any("/api/tasks" in m for m in result.messages)


def test_route_prefix_check_allows_existing_dual_prefixes(tmp_path: Path):
    old = "app.get('/users', pages.users);\napp.get('/api/users', api.users);\n"
    result = _route_check(tmp_path, old, old + "app.get('/users/:id', pages.user);\n")
    assert result.ok, result.messages


def test_route_prefix_check_allows_moving_every_route(tmp_path: Path):
    old = "app.get('/notes', notes.list);\napp.post('/notes', notes.create);\n"
    new = "app.get('/api/notes', notes.list);\napp.post('/api/notes', notes.create);\n"
    result = _route_check(tmp_path, old, new)
    assert result.ok, result.messages

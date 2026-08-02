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

from pathlib import Path

from agent.discovery.symbols import (
    _schema_fields_in_text,
    build_symbol_graph,
    check_schema_field_plumbing,
    detect_added_schema_fields,
)


def test_extract_exports_schema_and_routes(tmp_path: Path):
    (tmp_path / "model.js").write_text(
        "const mongoose = require('mongoose');\n"
        "const NoteSchema = mongoose.Schema({\n"
        "  title: String,\n"
        "  content: String,\n"
        "  tags: [String]\n"
        "});\n"
        "module.exports = mongoose.model('Note', NoteSchema);\n",
        encoding="utf-8",
    )
    (tmp_path / "ctrl.js").write_text(
        "const Note = require('./model.js');\n"
        "exports.create = (req, res) => {\n"
        "  const note = new Note({ title: req.body.title, content: req.body.content });\n"
        "  note.save().then(d => res.send(d));\n"
        "};\n"
        "exports.searchNotes = (req, res) => {\n"
        "  Note.find({ tags: req.query.tags }).then(n => res.send(n));\n"
        "};\n",
        encoding="utf-8",
    )
    (tmp_path / "routes.js").write_text(
        "module.exports = (app) => {\n"
        "  const notes = require('./ctrl.js');\n"
        "  app.post('/notes', notes.create);\n"
        "  app.get('/notes/search', notes.searchNotes);\n"
        "};\n",
        encoding="utf-8",
    )
    files = ["model.js", "ctrl.js", "routes.js"]
    g = build_symbol_graph(tmp_path, files)
    assert any(s.kind == "schema_field" and s.name == "tags" for s in g.symbols.values())
    assert any(s.kind == "export" and s.name == "searchNotes" for s in g.symbols.values())
    assert any(s.kind == "route" and "search" in s.name for s in g.symbols.values())
    # tags written only if create sets it — create does not set tags
    issues = check_schema_field_plumbing(g, added_fields=["tags"])
    assert issues and "tags" in issues[0]


def test_detect_added_schema_fields(tmp_path: Path):
    path = tmp_path / "model.js"
    old = (
        "const NoteSchema = mongoose.Schema({\n"
        "  title: String,\n"
        "  content: String\n"
        "});\n"
    )
    new = (
        "const NoteSchema = mongoose.Schema({\n"
        "  title: String,\n"
        "  content: String,\n"
        "  tags: [String]\n"
        "});\n"
    )
    path.write_text(new, encoding="utf-8")
    added = detect_added_schema_fields(
        tmp_path,
        backups={"model.js": old},
        touched_files=["model.js"],
    )
    assert added == ["tags"]


def test_field_plumbing_ok_when_create_sets_tags(tmp_path: Path):
    (tmp_path / "model.js").write_text(
        "const NoteSchema = mongoose.Schema({ title: String, tags: [String] });\n",
        encoding="utf-8",
    )
    (tmp_path / "ctrl.js").write_text(
        "exports.create = (req, res) => {\n"
        "  const note = new Note({ title: req.body.title, tags: req.body.tags || [] });\n"
        "};\n"
        "exports.update = (req, res) => {\n"
        "  Note.findByIdAndUpdate(id, { title: req.body.title, tags: req.body.tags });\n"
        "};\n",
        encoding="utf-8",
    )
    g = build_symbol_graph(tmp_path, ["model.js", "ctrl.js"])
    issues = check_schema_field_plumbing(g, added_fields=["tags"])
    assert issues == []


def test_schema_fields_after_a_nested_object():
    text = "new Schema({ title: { type: String, required: true }, tags: [String], body: String })"
    assert _schema_fields_in_text(text) == ["title", "tags", "body"]


def test_schema_option_keys_and_comments_are_not_fields():
    text = (
        "mongoose.Schema({\n"
        "  // when it was posted\n"
        "  postedAt: { type: Date, default: Date.now },\n"
        "  'author': String\n"
        "}, { timestamps: true })"
    )
    assert _schema_fields_in_text(text) == ["postedAt", "author"]

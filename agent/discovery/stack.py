"""Detect languages and toolchains from manifests and extensions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from agent.state import StackHints

MANIFEST_LANGUAGE: dict[str, str] = {
    "package.json": "javascript",
    "tsconfig.json": "typescript",
    "pyproject.toml": "python",
    "setup.py": "python",
    "requirements.txt": "python",
    "Pipfile": "python",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "Gemfile": "ruby",
    "composer.json": "php",
    "CMakeLists.txt": "cpp",
}

EXT_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
    ".dart": "dart",
}


def detect_stack(repo_root: Path | str, candidates: list[str]) -> StackHints:
    """Infer languages, manifests, and possible test commands from the repo."""
    root = Path(repo_root).resolve()
    manifests: list[str] = []
    languages: list[str] = []
    ext_counts: Counter[str] = Counter()

    for rel in candidates:
        name = Path(rel).name
        if name in MANIFEST_LANGUAGE:
            manifests.append(rel)
            lang = MANIFEST_LANGUAGE[name]
            if lang not in languages:
                languages.append(lang)
        suffix = Path(rel).suffix.lower()
        if suffix in EXT_LANGUAGE:
            ext_counts[EXT_LANGUAGE[suffix]] += 1

    # Dominant extensions fill in languages not seen via manifests
    for lang, _ in ext_counts.most_common():
        if lang not in languages:
            languages.append(lang)

    # package.json implies possible typescript if .ts files present
    if "javascript" in languages and ext_counts.get("typescript", 0) > 0:
        if "typescript" not in languages:
            languages.insert(0, "typescript")

    test_commands = _infer_test_commands(root, manifests, languages)
    return StackHints(
        languages=languages,
        manifests=manifests,
        test_commands=test_commands,
    )


def _infer_test_commands(
    root: Path,
    manifests: list[str],
    languages: list[str],
) -> list[str]:
    cmds: list[str] = []
    names = {Path(m).name for m in manifests}

    if "package.json" in names:
        cmds.append("npm test")
    if any(n in names for n in ("pyproject.toml", "setup.py", "requirements.txt", "Pipfile")):
        cmds.append("pytest")
    if "go.mod" in names:
        cmds.append("go test ./...")
    if "Cargo.toml" in names:
        cmds.append("cargo test")
    if "pom.xml" in names:
        cmds.append("mvn test")
    if any(n.startswith("build.gradle") for n in names):
        cmds.append("gradle test")
    if "Gemfile" in names:
        cmds.append("bundle exec rspec")
    if "composer.json" in names:
        cmds.append("composer test")

    # Fallback by language if no manifest-driven command
    if not cmds:
        if "python" in languages:
            cmds.append("pytest")
        if "go" in languages:
            cmds.append("go test ./...")
        if "rust" in languages:
            cmds.append("cargo test")
        if "javascript" in languages or "typescript" in languages:
            cmds.append("npm test")

    return cmds

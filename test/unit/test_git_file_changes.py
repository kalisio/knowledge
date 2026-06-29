"""Unit tests for Git-based incremental file discovery."""

from pathlib import Path
from types import SimpleNamespace

import ingestion.git_file_changes as git_file_changes


def test_candidate_files_is_none_on_first_ingestion(tmp_path):
    repo = tmp_path / "kano"
    repo.mkdir()
    candidates, stale = git_file_changes.find_file_changes([repo], None)
    assert candidates is None
    assert stale == {}


def test_candidate_files_groups_git_hits_by_repository(tmp_path, monkeypatch):
    repo_a = tmp_path / "kano"
    repo_b = tmp_path / "kdk"
    repo_a.mkdir()
    repo_b.mkdir()

    def fake_git_changed(repo_dir, since):
        assert since == "2026-06-19T10:35:00Z"
        if Path(repo_dir).name == "kano":
            return {"src/store/layers.js", "src/main.js"}, set()
        return set(), set()

    monkeypatch.setattr(
        git_file_changes, "_file_changes_since", fake_git_changed)

    candidates, stale = git_file_changes.find_file_changes(
        [repo_a, repo_b], "2026-06-19T10:35:00Z")
    assert candidates == {
        "kano": {"src/store/layers.js", "src/main.js"},
    }
    assert stale == {}


def test_git_changed_files_since_deduplicates_and_normalizes(monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout="src\\store\\layers.js\nsrc/store/layers.js\n\nREADME.md\n",
    )
    monkeypatch.setattr(git_file_changes.subprocess, "run",
                        lambda *args, **kwargs: completed)

    candidates, stale = git_file_changes.find_file_changes(
        [Path("/tmp/kano")], "2026-06-19T10:35:00Z")
    assert candidates == {"kano": {
        "src/store/layers.js",
        "README.md",
    }}
    assert stale == {}

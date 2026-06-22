"""Unit tests for incremental-ingestion helpers in main.py."""

from pathlib import Path
from types import SimpleNamespace

import ingestion.main as ingestion_main


def test_candidate_files_is_none_on_first_ingestion(tmp_path):
    repo = tmp_path / "kano"
    repo.mkdir()
    assert ingestion_main._candidate_files([repo], None) is None


def test_candidate_files_groups_git_hits_by_repository(tmp_path, monkeypatch):
    repo_a = tmp_path / "kano"
    repo_b = tmp_path / "kdk"
    repo_a.mkdir()
    repo_b.mkdir()

    def fake_git_changed(repo_dir, since):
        assert since == "2026-06-19T10:35:00Z"
        if Path(repo_dir).name == "kano":
            return {"src/store/layers.js", "src/main.js"}
        return set()

    monkeypatch.setattr(
        ingestion_main, "_git_changed_files_since", fake_git_changed)

    assert ingestion_main._candidate_files(
        [repo_a, repo_b], "2026-06-19T10:35:00Z") == {
            "kano": {"src/store/layers.js", "src/main.js"},
        }


def test_git_changed_files_since_deduplicates_and_normalizes(monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout="src\\store\\layers.js\nsrc/store/layers.js\n\nREADME.md\n",
    )
    monkeypatch.setattr(ingestion_main.subprocess, "run",
                        lambda *args, **kwargs: completed)

    assert ingestion_main._git_changed_files_since(
        Path("/tmp/kano"), "2026-06-19T10:35:00Z") == {
            "src/store/layers.js",
            "README.md",
        }

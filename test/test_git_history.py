"""Tests for ``ingestion_job/git_history.py``.

Spins up a throwaway git repo in a tmp dir and verifies the filter rules,
``--follow`` rename tracing, and the limit/order guarantees.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ingestion_job.git_history import is_significant, significant_commits


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def commit(repo: Path, file_rel: str, content: str, message: str) -> None:
    (repo / file_rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / file_rel).write_text(content)
    git(repo, "add", file_rel)
    git(repo, "commit", "-m", message)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "t@t.t")
    git(tmp_path, "config", "user.name", "Tester")
    git(tmp_path, "config", "commit.gpgsign", "false")
    return tmp_path


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("feat: real feature", True),
        ("fix: real fix", True),
        ("refactor: cleanup", True),
        ("chore: deps", False),
        ("chore(deps): bump", False),
        ("bump: 1.2.3", False),
        ("release: 1.0.0", False),
        ("Merge pull request #42", False),
        ("Merge branch 'main'", False),
        ("fix lint", False),
        ("fmt", False),
        ("style: prettier", False),
        ("", False),
        ("   ", False),
    ],
)
def test_is_significant(subject: str, expected: bool) -> None:
    assert is_significant(subject) is expected


def test_returns_empty_for_nonexistent_path(repo: Path) -> None:
    assert significant_commits(repo, "does/not/exist.js") == []


def test_returns_empty_when_repo_path_invalid(tmp_path: Path) -> None:
    assert significant_commits(tmp_path / "no-such-dir", "x.js") == []


def test_filters_noise_and_keeps_signal(repo: Path) -> None:
    commit(repo, "src/a.js", "v1", "feat: add A")
    commit(repo, "src/a.js", "v2", "chore: deps bump")
    commit(repo, "src/a.js", "v3", "fix: nullable A")
    commit(repo, "src/a.js", "v4", "fmt")
    commit(repo, "src/a.js", "v5", "refactor: rename A")
    history = significant_commits(repo, "src/a.js")
    assert history == [
        "refactor: rename A",
        "fix: nullable A",
        "feat: add A",
    ]


def test_limit_caps_output(repo: Path) -> None:
    for i in range(15):
        commit(repo, "src/b.js", f"v{i}", f"feat: change {i}")
    history = significant_commits(repo, "src/b.js", limit=3)
    assert len(history) == 3
    assert history[0] == "feat: change 14"
    assert history[-1] == "feat: change 12"


def test_follow_traces_renames(repo: Path) -> None:
    commit(repo, "src/old.js", "v1", "feat: initial impl")
    git(repo, "mv", "src/old.js", "src/new.js")
    git(repo, "commit", "-m", "refactor: rename old.js -> new.js")
    commit(repo, "src/new.js", "v2", "fix: post-rename bugfix")
    history = significant_commits(repo, "src/new.js")
    assert history == [
        "fix: post-rename bugfix",
        "refactor: rename old.js -> new.js",
        "feat: initial impl",
    ]

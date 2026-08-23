"""Filling the workspace with the repositories to index.

k-clone resolves everything through $KALISIO_DEVELOPMENT_DIR: the kli file
it reads, the kli runner it calls. A developer's workspace already holds the
tooling; a fresh cluster volume holds nothing, so the copy shipped in the
image is linked in first -- without it the run dies on "k-clone: No such
file or directory" before scanning a single file.
"""

import subprocess

import pytest

from ingestion.pipeline import workspace_clone


# A workspace root, plus the tooling as the image installs it.
@pytest.fixture
def image(tmp_path, monkeypatch):
    tooling = tmp_path / "opt"
    for name in workspace_clone.TOOLED_DIRECTORIES:
        (tooling / name).mkdir(parents=True)
    monkeypatch.setattr(workspace_clone, "TOOLING_DIR", str(tooling))
    monkeypatch.setenv("KALISIO_GITHUB_TOKEN", "a-token")
    return tooling


# Records the commands instead of running them.
@pytest.fixture
def commands(monkeypatch):
    calls = []

    def run(command, *args, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(workspace_clone.subprocess, "run", run)
    return calls


def test_a_fresh_volume_gets_the_tooling_linked(tmp_path, image, commands):
    workspace_clone.clone_workspace(tmp_path / "kalisio", "kalisio", "libs")

    for name in workspace_clone.TOOLED_DIRECTORIES:
        link = tmp_path / "kalisio" / name
        assert link.is_symlink()
        assert link.resolve() == (image / name).resolve()


def test_k_clone_is_called_with_the_organisation_and_the_workspace(
        tmp_path, image, commands):
    workspace_clone.clone_workspace(tmp_path / "kalisio", "irsn", "planet")

    assert commands == [["bash", "k-clone", "irsn", "planet"]]


def test_a_workspace_that_has_its_own_tooling_is_left_alone(
        tmp_path, image, commands):
    # A developer bind-mounts a real workspace; overwriting their checkout
    # with a link into the image would be a surprise.
    own = tmp_path / "kalisio" / "development"
    own.mkdir(parents=True)

    workspace_clone.clone_workspace(tmp_path / "kalisio", "kalisio", "libs")

    assert not own.is_symlink()


def test_the_links_survive_a_second_run(tmp_path, image, commands):
    workspace_clone.clone_workspace(tmp_path / "kalisio", "kalisio", "libs")
    workspace_clone.clone_workspace(tmp_path / "kalisio", "kalisio", "libs")

    assert (tmp_path / "kalisio" / "development").is_symlink()
    assert len(commands) == 2


def test_a_tooling_missing_from_the_image_is_reported(
        tmp_path, image, commands, knowledge_logs):
    (image / "kli").rmdir()

    workspace_clone.clone_workspace(tmp_path / "kalisio", "kalisio", "libs")

    assert "kli is missing from the image" in knowledge_logs.text


def test_a_missing_github_token_is_reported(
        tmp_path, image, commands, monkeypatch, knowledge_logs):
    # Without a token the tooling falls back to ssh, and the container has
    # no key: the run would fail on an authentication error instead.
    monkeypatch.delenv("KALISIO_GITHUB_TOKEN", raising=False)

    workspace_clone.clone_workspace(tmp_path / "kalisio", "kalisio", "libs")

    assert "KALISIO_GITHUB_TOKEN is not set" in knowledge_logs.text


def test_an_irsn_workspace_looks_for_the_gitlab_token(
        tmp_path, image, commands, monkeypatch, knowledge_logs):
    monkeypatch.delenv("GITLAB_IRSN_TOKEN", raising=False)

    workspace_clone.clone_workspace(tmp_path / "kalisio", "irsn", "planet")

    assert "GITLAB_IRSN_TOKEN is not set" in knowledge_logs.text


def test_a_failed_clone_is_raised(tmp_path, image, monkeypatch):
    # Nothing downstream can work without the repositories, so this one has
    # to travel up to the run.
    def failing(command, *args, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(workspace_clone.subprocess, "run", failing)

    with pytest.raises(subprocess.CalledProcessError):
        workspace_clone.clone_workspace(tmp_path / "kalisio", "kalisio", "libs")

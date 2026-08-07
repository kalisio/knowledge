import os
import subprocess

import pytest

import ingestion.config as ingestion_config


# The environment IngestionConfig requires. A test overrides only the
# variables it actually exercises.
_BASE_ENV = {
    "DEVELOPMENT_DIR": "/tmp/development",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_COLLECTION_CODE": "knowledge_test_code",
    "QDRANT_COLLECTION_METADATA": "knowledge_test_metadata",
    "EMBEDDING_MODEL": "test-model",
}


# One initialised git repository inside a workspace, as k-clone leaves it.
@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "kdk"
    path.mkdir()
    repository = _Repository(path)
    repository.git("init", "-q", "-b", "main")
    repository.git("config", "user.email", "test@kalisio.com")
    repository.git("config", "user.name", "test")
    return repository


# Build an IngestionConfig from a known environment. The module cache is reset
# first, so a test gets its own config rather than the first one built in the
# session; monkeypatch restores both the environment and the cache afterwards.
@pytest.fixture
def ingestion_env(monkeypatch):
    def build(**overrides):
        for name, value in {**_BASE_ENV, **overrides}.items():
            monkeypatch.setenv(name, str(value))
        monkeypatch.setattr(ingestion_config, "_config", None)
        return ingestion_config.get_ingestion_config()
    return build


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# A git repository the tests drive: commits at a chosen date, deletions, and
# the paths find_file_changes is expected to report.
class _Repository:
    def __init__(self, path):
        self.path = path
        self.workspace = path.parent

    # Run a git command in the repository, optionally at a fixed date so a
    # commit lands on a chosen side of the cursor.
    def git(self, *args, date=None):
        env = dict(os.environ)
        if date:
            env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
        return subprocess.run(
            ["git", "-C", str(self.path), *args],
            capture_output=True, text=True, check=True, env=env)

    # Write a repo-relative file and commit it at `date`.
    def commit(self, source_path, text, date, message="change"):
        path = self.file(source_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message, date=date)

    # Delete a repo-relative file and commit the deletion at `date`.
    def remove(self, source_path, date, message="remove"):
        self.git("rm", "-q", source_path)
        self.git("commit", "-q", "-m", message, date=date)

    # Absolute path of a repo-relative file.
    def file(self, source_path):
        return self.path / source_path
